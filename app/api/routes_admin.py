"""Admin routes — tiện ích vận hành cho demo (KHÔNG phải nghiệp vụ Buổi nào).

/admin/ingest: chạy lại app.retrieval.ingest.ingest() trong CHÍNH process server
đang chạy. Cần thiết vì QDRANT_URL=:memory: mặc định tạo 1 Qdrant in-process
riêng cho mỗi process — chạy `python -m scripts.ingest` ở process khác (ví dụ
CLI) sẽ KHÔNG nạp được dữ liệu cho server đang chạy. Nút "Ingest dữ liệu" trong
chat.html gọi endpoint này để nạp đúng vào Qdrant mà server đang dùng.

Endpoint này cũng là bước bắt buộc trong CD: `deploy/smoke.py` gọi nó trước khi
hỏi câu kiểm tra, vì bản deploy mới luôn khởi động với Qdrant rỗng.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import settings
from app.retrieval import vectorstore

router = APIRouter(prefix="/admin", tags=["admin"])


class IngestResponse(BaseModel):
    chunks_indexed: int
    total_in_collection: int
    source_dir: str


@router.post("/ingest", response_model=IngestResponse)
def ingest_endpoint() -> IngestResponse:
    from app.retrieval.ingest import ingest

    total = ingest()
    return IngestResponse(
        chunks_indexed=total,
        total_in_collection=vectorstore.count(),
        source_dir=settings.rag_source_dir,
    )


@router.get("/collection-status")
def collection_status() -> dict:
    return {
        "collection": settings.vectorstore_collection,
        "count": vectorstore.count(),
        "qdrant_url": settings.qdrant_url,
    }
