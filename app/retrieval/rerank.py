"""Re-ranking — Buổi 5 (RAG Pipeline), NATIVE.

Cross-encoder đánh giá trực tiếp cặp (query, chunk) → chính xác hơn bi-encoder
(embedding) nhưng chậm hơn. Pipeline: retrieve top-N (rộng) → rerank → giữ top-K (hẹp).

OPTIONAL: cần sentence-transformers (dep nặng). Bật/tắt qua config RERANK_ENABLED.
Model + import đều lazy → không bật thì không tải gì.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import settings
from app.retrieval.vectorstore import SearchHit


@lru_cache(maxsize=1)
def _get_cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.rerank_model)


def rerank(query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
    """Sắp xếp lại hits theo điểm cross-encoder, giữ top_k.

    Nếu không có hit thì trả nguyên. Score của hit được thay bằng điểm rerank.
    """
    if not hits:
        return hits

    model = _get_cross_encoder()
    pairs = [(query, h.text) for h in hits]
    scores = model.predict(pairs)

    # strict=True: model.predict trả đúng 1 score cho mỗi pair. Lệch số lượng
    # nghĩa là model/input hỏng — thà raise còn hơn ghép nhầm score với hit.
    ranked = sorted(zip(hits, scores, strict=True), key=lambda x: x[1], reverse=True)
    result: list[SearchHit] = []
    for hit, score in ranked[:top_k]:
        result.append(SearchHit(text=hit.text, score=float(score), metadata=hit.metadata))
    return result
