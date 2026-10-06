"""CLI ingestion — Buổi 5 (RAG Pipeline).

Wrapper mỏng quanh `app.retrieval.ingest.ingest`. Logic thật nằm trong `app/`
(Module III, Bài 6 chuyển sang đó) vì `/admin/ingest` cần gọi nó, mà `app/`
không được import `scripts/` — và Docker image cố tình không COPY `scripts/`.

Chạy:
    python -m scripts.ingest                 # ingest từ RAG_SOURCE_DIR (mặc định data/legal_docs)
    python -m scripts.ingest ./duong/dan     # ingest từ thư mục khác

LƯU Ý: với QDRANT_URL=:memory: (mặc định) mỗi process có 1 Qdrant riêng. Chạy
lệnh này ở terminal sẽ nạp vào store của CHÍNH process này, KHÔNG phải store của
server đang chạy. Muốn nạp cho server đang chạy, gọi POST /admin/ingest.

Cần OPENAI_API_KEYS (embedding).
"""

from __future__ import annotations

import sys

# Re-export để `from scripts.ingest import ingest` (các demo cũ) vẫn chạy.
from app.retrieval.ingest import ingest

__all__ = ["ingest"]


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else None
    ingest(src)
