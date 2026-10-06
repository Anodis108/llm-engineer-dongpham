"""Chat routes — Bài 1: endpoint /chat, /chat/stream, /chat/structured.

API-first: chatbot lộ ra qua HTTP để dễ tích hợp (web, mobile, test).
Streaming dùng StreamingResponse để đẩy token real-time (Section 6).

Module III, Bài 4 (Section 4) thêm `/chat/stream-sse`: bản streaming SSE
async, production-grade — xem docstring endpoint đó cho bảng so sánh chi tiết.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    AgentChatResponse,
    AgentSource,
    ChatRequest,
    ChatResponse,
    OptimizationStats,
)
from app.llm.params import GenerationParams
from app.optimization.caching import SemanticCache
from app.optimization.routing import rule_based_router
from app.pipeline import answer, answer_stream, answer_structured
from app.schemas.domain import LegalAnswer

router = APIRouter(prefix="/chat", tags=["chat"])


_embedder_cache = {}


def _get_st_model():
    """Lazy-load SentenceTransformer (multilingual for Vietnamese)."""
    if "model" not in _embedder_cache:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder_cache["model"] = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except ImportError:
            _embedder_cache["model"] = None
    return _embedder_cache["model"]


def _hash_embedder(text: str | list[str]):
    """Fallback hash-based embedder."""
    import hashlib

    import numpy as np

    if isinstance(text, list):
        return [_hash_embedder(t) for t in text]
    h = hashlib.sha256(text.encode()).digest()
    vec = np.array([float(byte) for byte in h[:32]])
    return vec / (np.linalg.norm(vec) + 1e-8)


def _sentence_embedder(text: str | list[str]):
    """Embed using SentenceTransformer, fallback to hash if unavailable."""
    model = _get_st_model()

    if model is None:
        # Fallback to hash embedder
        return _hash_embedder(text)

    if isinstance(text, list):
        return model.encode(text, convert_to_numpy=True)
    else:
        return model.encode([text], convert_to_numpy=True)[0]


_semantic_cache = SemanticCache(embedder=_sentence_embedder, threshold=0.8)


def _params_from(req: ChatRequest) -> GenerationParams | None:
    """Dựng GenerationParams từ override trong request (nếu có)."""
    overrides = {}
    if req.temperature is not None:
        overrides["temperature"] = req.temperature
    if req.max_completion_tokens is not None:
        overrides["max_completion_tokens"] = req.max_completion_tokens
    return GenerationParams(**overrides) if overrides else None


@router.post("", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    """Trả lời non-streaming."""
    # Buổi 8: Routing decision (determines which model should be used)
    # In production, this would be passed to answer() to select LLM
    routing_model = rule_based_router(req.question)

    # Buổi 8: Check semantic cache
    cached_answer = _semantic_cache.get(req.question)
    cache_hit = cached_answer is not None

    if cache_hit:
        text = cached_answer
    else:
        text = answer(req.question, _params_from(req))
        _semantic_cache.set(req.question, text)

    optimization = OptimizationStats(
        routing_model=routing_model,
        routing_method="rule_based",
        cache_hit=cache_hit,
    )
    return ChatResponse(answer=text, optimization=optimization)


@router.post("/stream")
def chat_stream_endpoint(req: ChatRequest) -> StreamingResponse:
    """Trả lời streaming (text/plain, từng token một)."""
    gen = answer_stream(req.question, _params_from(req))
    return StreamingResponse(gen, media_type="text/plain; charset=utf-8")


@router.post("/stream-sse")
async def chat_stream_sse_endpoint(req: ChatRequest, request: Request) -> StreamingResponse:
    """Module III, Bài 4, Section 4 — bản PRODUCTION của streaming, khác `/chat/stream`:

    | | `/chat/stream` (Bài 1) | `/chat/stream-sse` (Bài 4) |
    |---|---|---|
    | Protocol | `text/plain` thô | **SSE** (`text/event-stream`) — chuẩn web, có `[DONE]` |
    | I/O | SYNC (`def`) — block event loop | **ASYNC** (`async def`) + AsyncOpenAI |
    | Client bỏ đi | Vẫn sinh tiếp → đốt tiền | **`is_disconnected()`** → dừng ngay |
    | Quá tải | Không giới hạn → 429 hàng loạt | **Semaphore** backpressure |
    | Lỗi giữa stream | Vỡ kết nối | Gửi `{"error": ...}` rồi đóng sạch |

    Giữ CẢ HAI endpoint có chủ đích: học viên so sánh trực tiếp được 2 cách,
    và code Module I (chat.html, test cũ) không phải sửa.
    """
    from app.guardrails.checks import check_input
    from app.llm.async_client import astream_chat
    from app.prompts.templates import build_messages
    from app.retrieval.retriever import retrieve

    check_input(req.question)  # guardrail vẫn chạy TRƯỚC khi tốn token (Buổi 7)

    async def event_stream():
        try:
            chunks = retrieve(req.question)
            messages = build_messages(req.question, chunks)

            async for token in astream_chat(messages, _params_from(req)):
                # Client đóng tab giữa chừng -> dừng sinh token, khỏi tốn tiền.
                if await request.is_disconnected():
                    break
                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

            yield "data: [DONE]\n\n"
        except Exception as exc:  # noqa: BLE001 — lỗi phải đi qua SSE, không vỡ kết nối
            yield f"data: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Bài 5, Section 3: nginx mặc định buffer response -> client không
            # thấy token nào tới khi xong. Header này tắt buffering ở tầng proxy
            # (bổ sung cho `proxy_buffering off` trong deploy/nginx.conf).
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/structured", response_model=LegalAnswer)
def chat_structured_endpoint(req: ChatRequest) -> LegalAnswer:
    """Trả lời structured output theo schema LegalAnswer (Section 4)."""
    return answer_structured(req.question, _params_from(req))


@router.post("/agent", response_model=AgentChatResponse)
def chat_agent_endpoint(req: ChatRequest) -> AgentChatResponse:
    """Agentic RAG / CRAG (Buổi 6): retrieve → grade → web fallback nếu cần → generate.

    Trả thêm sources + web_search_used để thấy được "vì sao model trả lời vậy" —
    khác /chat thường vốn chỉ trả text.
    """
    from app.agent.graph import run_agent

    # Buổi 8: Check semantic cache
    cached_answer = _semantic_cache.get(req.question)
    cache_hit = cached_answer is not None

    # Buổi 8: Routing decision (determines which model should be used)
    # In production, this would be passed to run_agent() to select LLM
    routing_model = rule_based_router(req.question)

    if cache_hit:
        state = {"generation": cached_answer, "documents": []}
    else:
        state = run_agent(req.question)
        _semantic_cache.set(req.question, state.get("generation", ""))

    documents = state.get("documents", [])

    optimization = OptimizationStats(
        routing_model=routing_model,
        routing_method="rule_based",
        cache_hit=cache_hit,
    )
    return AgentChatResponse(
        answer=state.get("generation", ""),
        sources=[
            AgentSource(text=d.text, source=d.source, score=d.score) for d in documents
        ],
        web_search_used=state.get("web_search_used", False),
        sub_questions=state.get("sub_questions", []),
        optimization=optimization,
    )
