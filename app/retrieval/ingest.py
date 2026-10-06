"""Ingestion pipeline — Buổi 5 (RAG Pipeline). NATIVE, không LangChain.

    load_documents → chunk_documents → embed_passages → vectorstore.add

Module III, Bài 6: hàm này CHUYỂN từ `scripts/ingest.py` sang đây. Lý do không
phải cho đẹp: `app/api/routes_admin.py` (endpoint `/admin/ingest`) cần gọi nó,
mà `app/` import `scripts/` là sai tầng — và trong Docker image `scripts/`
KHÔNG được COPY vào (chỉ `app/` + `data/legal_docs`), nên endpoint đó nổ
`ModuleNotFoundError: No module named 'scripts'`.

Đúng ra `scripts/` là lớp CLI mỏng gọi vào `app/`, không phải nơi chứa logic
nghiệp vụ. `scripts/ingest.py` giờ chỉ còn là wrapper giữ lệnh cũ chạy được.

Với QDRANT_URL=:memory: dữ liệu KHÔNG persist giữa các process — muốn giữ, chạy
Qdrant server và đặt QDRANT_URL.
"""

from __future__ import annotations

from app.config import settings
from app.retrieval import vectorstore
from app.retrieval.chunking import chunk_documents
from app.retrieval.embeddings import embed_passages
from app.retrieval.loader import load_documents


def ingest(source_dir: str | None = None, batch_size: int = 64) -> int:
    source_dir = source_dir or settings.rag_source_dir

    print(f"[1/4] Load tài liệu từ {source_dir} ...")
    docs = load_documents(source_dir)
    print(f"      → {len(docs)} tài liệu/trang")

    print(f"[2/4] Chunking (size={settings.rag_chunk_size}, overlap={settings.rag_chunk_overlap}) ...")
    chunks = chunk_documents(
        docs,
        chunk_size=settings.rag_chunk_size,
        overlap=settings.rag_chunk_overlap,
        contextual=settings.rag_contextual_chunking,
    )
    print(f"      → {len(chunks)} chunks")
    if not chunks:
        print("      (không có gì để index — thư mục rỗng?)")
        return 0

    print("[3/4] Embedding + [4/4] Index vào Qdrant (theo batch) ...")
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [c.text for c in batch]
        # id ổn định theo source + chunk_index để chạy lại không tạo trùng.
        ids = [f"{c.metadata.get('source','?')}::{start + i}" for i, c in enumerate(batch)]
        metas = [c.metadata for c in batch]
        vectors = embed_passages(texts)
        vectorstore.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)

    total = vectorstore.count()
    print(f"\nHoàn tất. Collection '{settings.vectorstore_collection}' có {total} chunks.")
    return total
