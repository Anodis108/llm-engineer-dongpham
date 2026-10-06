"""Vector store — Buổi 4 (Embeddings & Vector Databases).

Dùng Qdrant (native qdrant-client). Mặc định ":memory:" cho demo — không cần Docker.
Production: đặt QDRANT_URL=http://localhost:6333 (Qdrant chạy qua Docker).

Metric: COSINE (khớp OpenAI embedding). Interface (add/search/count/SearchHit) giữ
NGUYÊN để retriever (Buổi 5) và các buổi sau dùng lại — đổi backend chỉ sửa file này.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from functools import lru_cache

from app.config import settings


@dataclass(slots=True)
class SearchHit:
    """Một kết quả tìm kiếm từ vector store."""

    text: str
    score: float                 # càng cao càng liên quan (cosine similarity)
    metadata: dict


@lru_cache(maxsize=1)
def _get_client():
    """Qdrant client. ":memory:" → in-process; url khác → kết nối server."""
    from qdrant_client import QdrantClient

    if settings.qdrant_url == ":memory:":
        return QdrantClient(location=":memory:")
    return QdrantClient(url=settings.qdrant_url)


def _ensure_collection() -> None:
    """Tạo collection nếu chưa có (idempotent)."""
    from qdrant_client.models import Distance, VectorParams

    client = _get_client()
    name = settings.vectorstore_collection
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=settings.embedding_dim, distance=Distance.COSINE
            ),
        )


def _to_point_id(raw_id: str) -> str:
    """Qdrant point id phải là UUID hoặc int. Map string id ổn định → UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw_id))


def add(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict] | None = None,
) -> None:
    """Thêm/ghi đè documents đã embed. Text + metadata lưu trong payload."""
    from qdrant_client.models import PointStruct

    _ensure_collection()
    metadatas = metadatas or [{} for _ in ids]
    points = [
        PointStruct(
            id=_to_point_id(raw_id),
            vector=vector,
            payload={"text": doc, **meta},
        )
        # strict=True: 4 mảng song song phải khớp độ dài. Lệch = id gắn nhầm
        # vector, hỏng âm thầm cả index (retrieval trả rác mà không có lỗi nào).
        for raw_id, vector, doc, meta in zip(ids, embeddings, documents, metadatas, strict=True)
    ]
    _get_client().upsert(collection_name=settings.vectorstore_collection, points=points)


def search(
    query_vector: list[float],
    top_k: int = 5,
    where: dict | None = None,
) -> list[SearchHit]:
    """Tìm top_k documents gần query nhất.

    Args:
        query_vector: vector câu truy vấn (đã embed_query).
        top_k: số kết quả.
        where: metadata filter, ví dụ {"source": "luat-dn-2020"}.
    """
    query_filter = None
    if where:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        query_filter = Filter(
            must=[
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in where.items()
            ]
        )

    # query_points() thay cho search() đã deprecated ở qdrant-client mới.
    response = _get_client().query_points(
        collection_name=settings.vectorstore_collection,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
    )
    return [
        SearchHit(
            text=h.payload.get("text", ""),
            score=h.score,  # Qdrant COSINE trả similarity trực tiếp (cao = tốt)
            metadata={k: v for k, v in h.payload.items() if k != "text"},
        )
        for h in response.points
    ]


def count() -> int:
    """Số document trong collection (0 nếu chưa tạo)."""
    client = _get_client()
    name = settings.vectorstore_collection
    if not client.collection_exists(name):
        return 0
    return client.count(collection_name=name).count
