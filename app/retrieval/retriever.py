"""Retriever — Buổi 5 (RAG Pipeline). NATIVE (không LangChain).

Đây là nơi "lỗ hổng RAG" (treo từ Buổi 1) được LẤP. retrieve() giờ:
  1. (tùy chọn) Query rewriting — LLM viết lại câu hỏi thành query tìm kiếm tốt hơn.
  2. Embed query (OpenAI) → semantic search trong Qdrant, lấy top-N rộng.
  3. (tùy chọn) Re-rank bằng cross-encoder → giữ top-K hẹp, chính xác hơn.

Interface `retrieve(query, top_k) -> list[RetrievedChunk]` GIỮ NGUYÊN từ Buổi 1,
nên pipeline.py / prompts KHÔNG phải sửa — chỉ cần nó trả chunk thật thay vì [].

Bật/tắt rewriting & rerank qua config (mặc định: rewriting off, rerank off) để
demo chạy nhẹ, không kéo dep nặng khi chưa cần.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import settings


@dataclass(slots=True)
class RetrievedChunk:
    """Một đoạn văn bản lấy được từ retrieval, kèm metadata."""

    text: str
    source: str = ""
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


def _rewrite_query(query: str) -> str:
    """Dùng LLM viết lại câu hỏi thành query tìm kiếm pháp lý tối ưu (native SDK)."""
    from app.llm import completion
    from app.llm.params import GenerationParams

    messages = [
        {
            "role": "system",
            "content": (
                "Bạn là chuyên gia tìm kiếm thông tin pháp lý. Viết lại câu hỏi thành "
                "MỘT query tìm kiếm ngắn gọn, dùng thuật ngữ pháp lý chính xác. "
                "Chỉ trả về query, không giải thích."
            ),
        },
        {"role": "user", "content": query},
    ]
    rewritten = completion.chat(messages, GenerationParams(temperature=0.0))
    return rewritten.strip() or query


def _hit_summary(hits) -> list[dict]:
    """Tóm tắt hit để ghi vào trace — CHỈ source + score, không ghi cả văn bản.

    Ghi cả `text` sẽ làm mỗi trace phình lên vài chục KB, tốn tiền lưu trữ mà
    gần như không ai đọc (Bài 7, Section 4: trace ghi CÁI ĐÃ XẢY RA, không phải
    bản sao dữ liệu). Muốn xem văn bản thì mở tài liệu gốc theo `source`.
    """
    return [
        {"source": h.metadata.get("source", ""), "score": round(float(h.score), 4)}
        for h in hits
    ]


def retrieve(
    query: str, top_k: int | None = None, parent_span=None
) -> list[RetrievedChunk]:
    """Tìm các chunk liên quan nhất tới `query`.

    Args:
        query: câu hỏi người dùng.
        top_k: số chunk trả về cuối cùng (mặc định settings.rag_top_k).
        parent_span: span LangSmith để lồng các bước con vào (Bài 7). None =
            không trace, đúng hành vi cũ. Ba span con: query_rewrite / search /
            rerank — nhìn được bước nào ăn thời gian khi retrieval chậm.
    """
    # Import ở đây để tránh vòng import và giữ module nhẹ.
    from app.monitoring.tracing import trace_step
    from app.retrieval import vectorstore
    from app.retrieval.embeddings import embed_query

    top_k = top_k or settings.rag_top_k

    with trace_step(parent_span, "query_rewrite", input=query) as t:
        search_query = _rewrite_query(query) if settings.rag_query_rewriting else query
        t["output"] = search_query

    # Nếu bật rerank: lấy rộng hơn (fetch_k) rồi mới lọc xuống top_k.
    fetch_k = settings.rag_fetch_k if settings.rag_rerank_enabled else top_k
    with trace_step(parent_span, "search", input=search_query) as t:
        hits = vectorstore.search(embed_query(search_query), top_k=fetch_k)
        t["output"] = _hit_summary(hits)

    if settings.rag_rerank_enabled and hits:
        from app.retrieval.rerank import rerank

        with trace_step(parent_span, "rerank", input=len(hits)) as t:
            hits = rerank(query, hits, top_k=top_k)
            t["output"] = _hit_summary(hits)

    return [
        RetrievedChunk(
            text=h.text,
            source=h.metadata.get("source", ""),
            score=h.score,
            metadata=h.metadata,
        )
        for h in hits
    ]
