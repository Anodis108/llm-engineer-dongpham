"""Pipeline orchestrator — nối guardrails + retrieval + prompt + LLM.

Đây là "xương sống" RAG. Ở Buổi 1, bước retrieve trả [] nên thực chất chỉ là
chatbot thuần LLM. Từ Buổi 5, chỉ cần retriever.retrieve() trả chunk thật là
toàn bộ pipeline thành RAG — KHÔNG phải sửa file này.

Luồng:
    guardrails.check_input(question)   — raise GuardrailViolation nếu injection
    → retrieve(query)
    → build_messages(q, ctx)
    → llm.chat / chat_stream / chat_parsed
    guardrails.check_output(answer, ctx) — chỉ ở answer() non-streaming; xem lý
                                            do trong docstring answer_stream()

── Module III, Bài 7, Section 4: trace chia theo BƯỚC ──────────────────────────

Trước Bài 7, mỗi request chỉ có 1 run phẳng: biết "request này mất 4.2s" nhưng
không biết 4.2s đó nằm ở đâu. Giờ mỗi bước là 1 span con:

    answer
    ├── retrieve            (bao gồm query_rewrite / search / rerank)
    ├── llm_generate        ← gần như luôn là chỗ ăn thời gian
    └── output_guard

Đây là khác biệt giữa biết "hệ thống chậm" và biết "LLM chậm, retrieval ổn" —
một bên phải đoán, một bên chỉ vào đúng chỗ.
"""

from __future__ import annotations

from collections.abc import Iterator

from app.config import settings
from app.guardrails.checks import check_input, check_output
from app.llm import completion
from app.llm.params import GenerationParams
from app.monitoring.tracing import trace_answer, trace_step, trace_stream
from app.observability.metrics import run_tags
from app.prompts.templates import build_messages
from app.retrieval.retriever import retrieve
from app.schemas.domain import LegalAnswer


def _base_tags(feature: str, **extra) -> dict:
    """Tag nền cho mọi trace — mỗi tag ứng với một trục rollback (Bài 6/Bài 7).

    `model` lấy từ settings chứ không hardcode: đổi LLM_MODEL sang model local
    là dashboard tự tách nhóm, thấy ngay chất lượng/latency khác nhau thế nào.
    """
    return run_tags(feature=feature, model=settings.llm_model, **extra)


def _retrieve_step(span, question: str):
    """Span `retrieve` + trả về (chunks, span con để retrieve() lồng tiếp vào)."""
    with trace_step(span, "retrieve", input=question) as t:
        chunks = retrieve(question, parent_span=t.get("_span"))
        t["output"] = {"n_chunks": len(chunks)}
    return chunks


def answer(question: str, params: GenerationParams | None = None) -> str:
    """Trả lời dạng text (non-streaming). Có đủ input + output guardrails."""
    check_input(question)
    with trace_answer("answer", question, metadata=_base_tags("answer")) as t:
        span = t.get("_span")

        chunks = _retrieve_step(span, question)

        prompt_meta: dict = {}
        messages = build_messages(question, chunks, meta=prompt_meta)
        # Gắn vào trace SAU khi biết prompt nào thực sự được dùng (registry có
        # thể đã rơi xuống cache/fallback — xem app/prompts/templates.py).
        t["tags"] = run_tags(**prompt_meta)

        with trace_step(span, "llm_generate", input={"n_messages": len(messages)}) as gt:
            raw_answer = completion.chat(messages, params)
            # Không ghi cả câu trả lời vào span này: nó đã nằm ở output của run
            # gốc rồi, ghi 2 lần chỉ làm trace nặng gấp đôi.
            gt["output"] = {"n_chars": len(raw_answer)}

        with trace_step(span, "output_guard", input={"n_context": len(chunks)}) as ot:
            result = check_output(raw_answer, [c.text for c in chunks])
            ot["output"] = {"valid": result.valid, "issues": result.issues}
            if not result.valid:
                # Guardrail chặn = tín hiệu an toàn → tail sampling luôn giữ
                # trace này bất kể TRACE_SAMPLE_RATE (Bài 7, Section 4).
                t["guardrail"] = True

        t["output"] = result.answer
    return result.answer


def answer_stream(
    question: str, params: GenerationParams | None = None
) -> Iterator[str]:
    """Trả lời dạng streaming (Section 6).

    Chỉ có input guardrail. Output guardrail không áp dụng được ở đây: token
    đã gửi tới client ngay khi sinh ra, nên không có cách nào "thay bằng
    fallback" sau khi phát hiện vấn đề — muốn kiểm tra output cho luồng
    streaming cần buffer toàn bộ trước (mất lợi ích của streaming) hoặc chấp
    nhận đánh đổi latency-thấp/không-guardrail-output.

    Không có span con: token được phát ngay khi sinh, nên không có ranh giới
    "bước" nào để bọc. Xem app/monitoring/tracing.py::trace_stream.
    """
    check_input(question)
    chunks = retrieve(question)
    prompt_meta: dict = {}
    messages = build_messages(question, chunks, meta=prompt_meta)
    yield from trace_stream(
        "answer_stream",
        question,
        completion.chat_stream(messages, params),
        metadata=_base_tags("answer_stream", **prompt_meta),
    )


def answer_structured(
    question: str, params: GenerationParams | None = None
) -> LegalAnswer:
    """Trả lời dạng structured output theo schema LegalAnswer (Section 4).

    Chỉ có input guardrail — LegalAnswer đã tự mang confidence/needs_lawyer,
    một hình thức "self-reported groundedness" riêng của schema này.
    """
    check_input(question)
    with trace_answer("answer_structured", question, metadata=_base_tags("answer_structured")) as t:
        span = t.get("_span")

        chunks = _retrieve_step(span, question)

        prompt_meta: dict = {}
        messages = build_messages(question, chunks, meta=prompt_meta)

        with trace_step(span, "llm_generate", input={"n_messages": len(messages)}) as gt:
            result = completion.chat_parsed(messages, LegalAnswer, params)
            gt["output"] = {"confidence": result.confidence, "needs_lawyer": result.needs_lawyer}

        # Gộp tag prompt + tag riêng của endpoint trong MỘT dict: gán `t["tags"]`
        # hai lần sẽ ghi đè lần trước và làm mất tag prompt.
        #
        # `confidence` là tự khai của model, KHÔNG phải điểm groundedness đã
        # kiểm chứng — ghi vào trace để đối chiếu về sau, đừng dùng làm nhãn.
        t["tags"] = run_tags(**prompt_meta, confidence=result.confidence)
        t["output"] = result.model_dump_json()
    return result
