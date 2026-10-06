"""Nodes cho Agentic RAG (CRAG + Query Decomposition) — Buổi 6.

Mỗi node là một hàm thuần: nhận GraphState, trả về phần state cần cập nhật.
QUAN TRỌNG: mọi lời gọi bên trong đều dùng NATIVE SDK đã xây từ các buổi trước —
  - completion.chat / chat_parsed  (Buổi 1)
  - retriever.retrieve              (Buổi 5)
  - tools_web.web_search            (Buổi 6, Tavily native)
LangGraph chỉ lo control flow (node nào chạy sau node nào), không gọi LangChain.
"""

from __future__ import annotations

from langgraph.types import Send
from pydantic import BaseModel, Field

from app.agent.state import GradedChunk, GraphState, RetrieveTask
from app.agent.tools_web import web_search
from app.config import settings
from app.guardrails.checks import check_input, check_output
from app.llm import completion
from app.llm.params import GenerationParams
from app.monitoring.tracing import trace_step
from app.retrieval.retriever import RetrievedChunk, retrieve

# ── Guardrails (Buổi 7) ────────────────────────────────────────────────────────
#
# Nối cùng logic dùng trong pipeline.answer(): check_input chạy TRƯỚC graph (raise
# GuardrailViolation nếu injection — LangGraph không catch, exception nổi thẳng
# lên caller, giống hệt cách FastAPI exception handler bắt nó ở /chat thường).
# check_output chạy SAU generate, KHÔNG raise — ghi đè `generation` bằng fallback
# nếu phát hiện vấn đề (câu quá ngắn, số liệu không xuất hiện trong context).

def guardrail_input_node(state: GraphState) -> dict:
    """Raise GuardrailViolation nếu câu hỏi chứa prompt injection — dừng graph
    trước khi tốn bất kỳ lời gọi LLM/retrieval nào."""
    span = state.get("_trace_span")
    with trace_step(span, "guardrail_input", input=state["question"]):
        check_input(state["question"])
    return {}


def guardrail_output_node(state: GraphState) -> dict:
    """Kiểm tra câu trả lời cuối cùng so với context đã dùng để generate."""
    span = state.get("_trace_span")
    context = [c.text for c in state.get("documents", [])]
    with trace_step(span, "guardrail_output", input=state.get("generation", "")) as t:
        result = check_output(state.get("generation", ""), context)
        t["output"] = {"answer": result.answer, "issues": result.issues}
    return {"generation": result.answer, "output_issues": result.issues}


# ── Query Decomposition ───────────────────────────────────────────────────────

class _SubQuestions(BaseModel):
    sub_questions: list[str] = Field(
        description="2-4 câu hỏi con, mỗi câu trả lời độc lập được"
    )


def decompose_query(state: GraphState) -> dict:
    """Chia câu hỏi phức tạp thành sub-questions (structured output).

    Chỉ decompose khi câu hỏi đủ dài (heuristic AGENT_DECOMPOSE_MIN_CHARS) —
    tránh tốn 1 lời gọi LLM cho câu hỏi đơn giản.
    """
    span = state.get("_trace_span")
    question = state["question"]

    with trace_step(span, "decompose_query", input=question) as t:
        if len(question) < settings.agent_decompose_min_chars:
            t["output"] = [question]
            return {"sub_questions": [question]}

        messages = [
            {
                "role": "system",
                "content": (
                    "Chia câu hỏi pháp lý phức tạp sau thành 2-4 sub-questions đơn giản "
                    "hơn, mỗi câu trả lời độc lập được bằng tra cứu luật. Nếu câu hỏi đã "
                    "đơn giản, trả về chính nó trong danh sách 1 phần tử."
                ),
            },
            {"role": "user", "content": question},
        ]
        result = completion.chat_parsed(
            messages, _SubQuestions, GenerationParams(temperature=0.0)
        )
        sub_qs = result.sub_questions[: settings.agent_max_sub_questions] or [question]
        t["output"] = sub_qs
    return {"sub_questions": sub_qs}


# ── Retrieve song song theo sub-question (LangGraph Send API) ───────────────
#
# Fan-out: send_retrieve là một CONDITIONAL EDGE (không phải node) — trả về list
# Send(...), mỗi phần tử là một "task" gọi node parallel_retrieve với state RIÊNG
# {"sub_question": sq}. LangGraph chạy các task này song song, mỗi task chỉ thấy
# đúng payload của Send, không thấy toàn bộ GraphState.
#
# Fan-in: mọi nhánh song song đều trả {"raw_documents": [...]}. Vì
# GraphState.raw_documents khai báo Annotated[list, operator.add] (xem state.py),
# LangGraph NỐI list của các nhánh lại (a + b + c) khi hội tụ — không ghi đè. Nếu
# thiếu reducer này, chỉ nhánh "thắng" cuối cùng còn lại → mất dữ liệu, đây là lỗi
# phổ biến nhất khi mới dùng Send.
#
# dedupe_documents chạy SAU khi hội tụ, đọc `raw_documents` (đã gộp bởi reducer)
# và ghi ra `documents` (field KHÔNG có reducer, ghi đè bình thường). Bắt buộc phải
# tách 2 field: nếu dedupe ghi thẳng vào `documents` mà field đó cũng có reducer
# operator.add, kết quả đã dedupe sẽ bị CỘNG THÊM vào list cũ mỗi lần chạy → nhân
# bản dữ liệu, thay vì thay thế.

def send_retrieve(state: GraphState) -> list[Send]:
    """Router cho add_conditional_edges: sinh 1 Send task cho mỗi sub-question.

    Truyền `_trace_span` vào mỗi payload — nhánh Send CHỈ thấy payload này, không
    thấy GraphState đầy đủ, nên parallel_retrieve cần span được gửi kèm ở đây để
    tạo nested span đúng cha (xem RetrieveTask trong state.py)."""
    sub_questions = state.get("sub_questions") or [state["question"]]
    span = state.get("_trace_span")
    return [
        Send("parallel_retrieve", RetrieveTask(sub_question=sq, _trace_span=span))
        for sq in sub_questions
    ]


def parallel_retrieve(state: GraphState) -> dict:
    """Chạy trong một nhánh Send — state ở đây CHỈ có `sub_question` + `_trace_span`."""
    sub_question = state.get("sub_question", "")
    span = state.get("_trace_span")
    with trace_step(span, "parallel_retrieve", input=sub_question) as t:
        chunks = retrieve(sub_question)
        t["output"] = f"{len(chunks)} chunks"
    return {"raw_documents": chunks}


def dedupe_documents(state: GraphState) -> dict:
    """Sau khi các nhánh song song hội tụ: khử trùng theo text, giữ bản score cao nhất.

    Đọc từ `raw_documents` (đã gộp qua reducer), GHI RA `documents` (không reducer).
    """
    span = state.get("_trace_span")
    with trace_step(span, "dedupe_documents", input=f"{len(state.get('raw_documents', []))} raw chunks") as t:
        seen: dict[str, RetrievedChunk] = {}
        for chunk in state.get("raw_documents", []):
            existing = seen.get(chunk.text)
            if existing is None or chunk.score > existing.score:
                seen[chunk.text] = chunk

        documents = sorted(seen.values(), key=lambda c: c.score, reverse=True)
        t["output"] = f"{len(documents)} deduped chunks"
    return {"documents": documents}


# ── Grade documents ───────────────────────────────────────────────────────────

class _Grade(BaseModel):
    relevant: bool = Field(description="Tài liệu có liên quan trực tiếp tới câu hỏi không")
    reason: str = Field(description="Giải thích ngắn gọn vì sao relevant/irrelevant")


def grade_documents(state: GraphState) -> dict:
    """Chấm điểm liên quan cho từng chunk bằng structured output (chính xác hơn parse free-text)."""
    span = state.get("_trace_span")
    question = state["question"]
    graded: list[GradedChunk] = []

    with trace_step(span, "grade_documents", input=f"{len(state.get('documents', []))} chunks") as t:
        for chunk in state.get("documents", []):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Đánh giá xem TÀI LIỆU có liên quan trực tiếp để trả lời CÂU HỎI không."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Câu hỏi: {question}\n\nTài liệu: {chunk.text}",
                },
            ]
            grade = completion.chat_parsed(messages, _Grade, GenerationParams(temperature=0.0))
            graded.append({"chunk": chunk, "relevant": grade.relevant, "reason": grade.reason})

        relevant_docs = [g["chunk"] for g in graded if g["relevant"]]
        t["output"] = f"{len(relevant_docs)}/{len(graded)} relevant"
    return {"documents": relevant_docs, "graded": graded}


# ── Router: có cần web search fallback không? ────────────────────────────────

def route_after_grading(state: GraphState) -> str:
    """Conditional edge: đủ chunk relevant → generate thẳng; thiếu → web_search."""
    n_relevant = len(state.get("documents", []))
    if n_relevant >= settings.agent_min_relevant_chunks:
        return "generate"
    return "web_search"


# ── Web search fallback ───────────────────────────────────────────────────────

def web_search_node(state: GraphState) -> dict:
    """CRAG fallback: không đủ chunk relevant từ vector DB → tìm trên web."""
    span = state.get("_trace_span")
    with trace_step(span, "web_search", input=state["question"]) as t:
        results = web_search(state["question"])
        t["output"] = f"{len(results)} web results"
    merged = state.get("documents", []) + results
    return {"documents": merged, "web_search_used": True}


# ── Generate ──────────────────────────────────────────────────────────────────

def generate(state: GraphState) -> dict:
    """Sinh câu trả lời cuối cùng, bám vào documents đã có (grounded)."""
    from app.prompts.templates import build_messages

    span = state.get("_trace_span")
    with trace_step(span, "generate", input=state["question"]) as t:
        messages = build_messages(state["question"], state.get("documents", []))
        answer = completion.chat(messages, GenerationParams(temperature=0.1))
        t["output"] = answer
    return {"generation": answer}
