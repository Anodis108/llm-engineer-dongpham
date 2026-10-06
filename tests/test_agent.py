"""Test Buổi 6 — Agentic RAG (CRAG + Query Decomposition).

Mock completion.chat_parsed/chat, retriever.retrieve, tools_web.web_search — LangGraph
chạy THẬT (verify control flow đúng: route sang generate hay web_search).
"""

from __future__ import annotations

from app.agent import nodes
from app.retrieval.retriever import RetrievedChunk

# ── decompose_query ───────────────────────────────────────────────────────────

def test_decompose_skips_short_question(monkeypatch):
    """Câu hỏi ngắn (< AGENT_DECOMPOSE_MIN_CHARS) không gọi LLM, trả về chính nó."""
    monkeypatch.setattr(nodes.settings, "agent_decompose_min_chars", 80)

    called = {"n": 0}
    monkeypatch.setattr(
        nodes.completion, "chat_parsed", lambda *a, **k: called.update(n=called["n"] + 1)
    )

    out = nodes.decompose_query({"question": "Điều 46 nói gì?"})
    assert out["sub_questions"] == ["Điều 46 nói gì?"]
    assert called["n"] == 0   # không gọi LLM


def test_decompose_long_question_calls_llm(monkeypatch):
    """Câu hỏi dài → gọi chat_parsed, trả sub_questions từ kết quả (giới hạn max)."""
    monkeypatch.setattr(nodes.settings, "agent_decompose_min_chars", 10)
    monkeypatch.setattr(nodes.settings, "agent_max_sub_questions", 2)

    class FakeResult:
        sub_questions = ["câu 1", "câu 2", "câu 3"]

    monkeypatch.setattr(nodes.completion, "chat_parsed", lambda *a, **k: FakeResult())

    out = nodes.decompose_query({"question": "So sánh TNHH và cổ phần theo luật 2020"})
    assert out["sub_questions"] == ["câu 1", "câu 2"]   # cắt còn max=2


# ── send_retrieve / parallel_retrieve / dedupe_documents (Send API) ──────────

def test_send_retrieve_creates_one_send_per_sub_question():
    """send_retrieve (router) sinh đúng 1 Send task cho mỗi sub-question."""
    from langgraph.types import Send

    sends = nodes.send_retrieve({"sub_questions": ["sub1", "sub2", "sub3"]})
    assert len(sends) == 3
    assert all(isinstance(s, Send) for s in sends)
    assert [s.arg["sub_question"] for s in sends] == ["sub1", "sub2", "sub3"]


def test_send_retrieve_falls_back_to_question_when_no_sub_questions():
    sends = nodes.send_retrieve({"question": "câu hỏi gốc"})
    assert len(sends) == 1
    assert sends[0].arg["sub_question"] == "câu hỏi gốc"


def test_parallel_retrieve_writes_raw_documents(monkeypatch):
    """Mỗi nhánh Send chỉ thấy `sub_question` riêng, ghi ra `raw_documents` (không phải `documents`)."""
    c = RetrievedChunk(text="Điều 46 ...", source="a.md", score=0.5)
    monkeypatch.setattr(nodes, "retrieve", lambda q, top_k=None: [c] if q == "sub1" else [])

    out = nodes.parallel_retrieve({"sub_question": "sub1"})
    assert out == {"raw_documents": [c]}


def test_dedupe_documents_merges_and_dedupes():
    """dedupe_documents đọc `raw_documents` (đã gộp qua reducer), ghi ra `documents` (không reducer)."""
    c1 = RetrievedChunk(text="Điều 46 ...", source="a.md", score=0.5)
    c2 = RetrievedChunk(text="Điều 46 ...", source="a.md", score=0.9)  # trùng text, score cao hơn
    c3 = RetrievedChunk(text="Điều 111 ...", source="b.md", score=0.7)

    # raw_documents mô phỏng kết quả ĐÃ GỘP bởi reducer operator.add từ nhiều nhánh Send.
    out = nodes.dedupe_documents({"raw_documents": [c1, c3, c2]})
    docs = out["documents"]
    assert len(docs) == 2                      # đã dedupe theo text
    assert docs[0].score == 0.9                # giữ bản score cao hơn, sắp giảm dần
    assert docs[0].text == "Điều 46 ..."


def test_dedupe_documents_does_not_accumulate_across_calls():
    """Bug đã sửa: dedupe_documents phải GHI ĐÈ `documents`, không cộng dồn qua nhiều lần gọi.

    (Trước khi sửa, `documents` có reducer operator.add nên gọi dedupe nhiều lần
    trong một fan-in sẽ nhân bản dữ liệu — xem ghi chú trong state.py.)
    """
    c1 = RetrievedChunk(text="A", source="a.md", score=0.5)
    out1 = nodes.dedupe_documents({"raw_documents": [c1]})
    out2 = nodes.dedupe_documents({"raw_documents": [c1]})
    assert out1 == out2 == {"documents": [c1]}   # không tích luỹ giữa 2 lần gọi độc lập


# ── grade_documents + router ─────────────────────────────────────────────────

def test_grade_documents_filters_irrelevant(monkeypatch):
    docs = [
        RetrievedChunk(text="liên quan", source="a.md"),
        RetrievedChunk(text="không liên quan", source="b.md"),
    ]

    class FakeGrade:
        def __init__(self, relevant):
            self.relevant = relevant
            self.reason = "..."

    grades = iter([FakeGrade(True), FakeGrade(False)])
    monkeypatch.setattr(nodes.completion, "chat_parsed", lambda *a, **k: next(grades))

    out = nodes.grade_documents({"question": "q", "documents": docs})
    assert len(out["documents"]) == 1
    assert out["documents"][0].text == "liên quan"
    assert len(out["graded"]) == 2   # graded giữ CẢ hai, kể cả bị loại


def test_router_generate_when_enough_relevant(monkeypatch):
    monkeypatch.setattr(nodes.settings, "agent_min_relevant_chunks", 1)
    state = {"documents": [RetrievedChunk(text="x", source="a")]}
    assert nodes.route_after_grading(state) == "generate"


def test_router_web_search_when_no_relevant(monkeypatch):
    monkeypatch.setattr(nodes.settings, "agent_min_relevant_chunks", 1)
    state = {"documents": []}
    assert nodes.route_after_grading(state) == "web_search"


# ── web_search_node ───────────────────────────────────────────────────────────

def test_web_search_node_appends_and_flags(monkeypatch):
    web_hit = RetrievedChunk(text="web result", source="https://x", score=0.6)
    monkeypatch.setattr(nodes, "web_search", lambda q: [web_hit])

    out = nodes.web_search_node({"question": "q", "documents": []})
    assert out["web_search_used"] is True
    assert out["documents"] == [web_hit]


# ── generate ──────────────────────────────────────────────────────────────────

def test_generate_calls_chat_with_documents(monkeypatch):
    monkeypatch.setattr(nodes.completion, "chat", lambda messages, params: "Trả lời cuối.")
    out = nodes.generate({"question": "q", "documents": []})
    assert out["generation"] == "Trả lời cuối."


# ── Full graph (LangGraph thật, node đã mock) ────────────────────────────────

def test_full_graph_routes_to_web_search_when_no_relevant_docs(monkeypatch):
    """End-to-end: không có chunk relevant nào → graph phải rẽ qua web_search rồi generate."""
    from app.agent import graph as graph_mod

    monkeypatch.setattr(graph_mod, "_build_graph", graph_mod._build_graph)  # noop, giữ cache rõ ràng
    graph_mod._build_graph.cache_clear()

    monkeypatch.setattr(nodes.settings, "agent_decompose_min_chars", 10_000)  # skip decompose LLM call
    monkeypatch.setattr(nodes, "retrieve", lambda query, top_k=None: [])       # không tìm được gì

    class FakeGrade:
        relevant = False
        reason = "irrelevant"

    monkeypatch.setattr(nodes.completion, "chat_parsed", lambda *a, **k: FakeGrade())
    monkeypatch.setattr(
        nodes, "web_search", lambda q: [RetrievedChunk(text="web info", source="https://x")]
    )
    monkeypatch.setattr(nodes.completion, "chat", lambda messages, params: "Câu trả lời từ web.")

    result = graph_mod.run_agent("Câu hỏi ngắn")

    assert result["web_search_used"] is True
    assert result["generation"] == "Câu trả lời từ web."
    graph_mod._build_graph.cache_clear()


def test_full_graph_parallel_retrieve_no_loss_no_duplication(monkeypatch):
    """End-to-end: 3 sub-questions chạy song song qua Send — mỗi chunk phải xuất hiện
    ĐÚNG MỘT LẦN trong kết quả cuối (guard chống regression bug nhân bản/mất dữ liệu)."""
    from app.agent import graph as graph_mod

    graph_mod._build_graph.cache_clear()
    monkeypatch.setattr(nodes.settings, "agent_decompose_min_chars", 5)

    by_sub_question = {
        "sub A": [RetrievedChunk(text="Chunk A", source="a.md", score=0.9)],
        "sub B": [RetrievedChunk(text="Chunk B", source="b.md", score=0.8)],
        "sub C": [RetrievedChunk(text="Chunk C", source="c.md", score=0.7)],
    }
    monkeypatch.setattr(nodes, "retrieve", lambda q, top_k=None: by_sub_question.get(q, []))

    class FakeSubQuestions:
        sub_questions = ["sub A", "sub B", "sub C"]

    class FakeGrade:
        relevant = True
        reason = "ok"

    def fake_chat_parsed(messages, schema, params=None):
        return FakeSubQuestions() if schema is nodes._SubQuestions else FakeGrade()

    monkeypatch.setattr(nodes.completion, "chat_parsed", fake_chat_parsed)
    monkeypatch.setattr(nodes.completion, "chat", lambda messages, params: "OK")

    result = graph_mod.run_agent("câu hỏi dài đủ mọi thứ")

    texts = sorted(d.text for d in result["documents"])
    assert texts == ["Chunk A", "Chunk B", "Chunk C"]   # đúng 1 lần mỗi chunk
    graph_mod._build_graph.cache_clear()
