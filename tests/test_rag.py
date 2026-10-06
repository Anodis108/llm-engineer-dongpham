"""Test Buổi 5 — RAG Pipeline (loader, chunking, retriever).

Chunking là logic thuần → test trực tiếp. Loader test qua tmp files.
Retriever mock embed + vectorstore (không gọi API / server).
"""

from __future__ import annotations

from app.retrieval import retriever as retriever_mod
from app.retrieval.chunking import chunk_documents
from app.retrieval.loader import LoadedDoc, load_documents

# ── Chunking ─────────────────────────────────────────────────────────────────

def test_chunk_short_doc_stays_single():
    docs = [LoadedDoc(text="Đoạn ngắn.", metadata={"source": "a.md"})]
    chunks = chunk_documents(docs, chunk_size=512, overlap=64)
    assert len(chunks) == 1
    assert chunks[0].text == "Đoạn ngắn."
    assert chunks[0].metadata["source"] == "a.md"
    assert chunks[0].metadata["chunk_index"] == 0


def test_chunk_long_doc_splits_and_respects_size():
    # Text dài gồm nhiều câu ngăn cách bằng ". "
    text = ". ".join(f"Cau so {i} co mot chut noi dung" for i in range(60))
    docs = [LoadedDoc(text=text, metadata={"source": "b.md"})]
    chunks = chunk_documents(docs, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    # Không chunk nào vượt quá size + overlap (overlap thêm vào đầu chunk sau).
    assert all(len(c.text) <= 100 + 10 for c in chunks)
    # chunk_index tăng dần
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_chunk_metadata_inherited():
    docs = [LoadedDoc(text="x" * 300, metadata={"source": "c.pdf", "page": 3})]
    chunks = chunk_documents(docs, chunk_size=100, overlap=0)
    assert all(c.metadata["source"] == "c.pdf" and c.metadata["page"] == 3 for c in chunks)


# ── Contextual Retrieval (heading-based) ─────────────────────────────────────

def test_contextual_prepends_heading_path():
    """Chunk markdown được prepend chuỗi heading H1 > H2 vào text + metadata."""
    md = "# Luật ABC\n\n## Điều 5. Vốn góp\n\nNội dung điều 5 về vốn góp công ty."
    docs = [LoadedDoc(text=md, metadata={"source": "luat.md"})]
    chunks = chunk_documents(docs, chunk_size=512, overlap=0, contextual=True)

    assert len(chunks) == 1
    c = chunks[0]
    assert c.text.startswith("Luật ABC > Điều 5. Vốn góp\n\n")
    assert c.metadata["heading"] == "Luật ABC > Điều 5. Vốn góp"
    assert "Nội dung điều 5" in c.text


def test_heading_stack_pops_same_level():
    """Hai điều cùng cấp (##) không lồng nhau — heading path không tích luỹ sai."""
    md = "# Luật\n\n## Điều 1\n\nND một.\n\n## Điều 2\n\nND hai."
    docs = [LoadedDoc(text=md, metadata={"source": "x.md"})]
    chunks = chunk_documents(docs, chunk_size=512, overlap=0, contextual=True)

    headings = [c.metadata["heading"] for c in chunks]
    assert "Luật > Điều 1" in headings
    assert "Luật > Điều 2" in headings
    # Điều 2 KHÔNG được chứa "Điều 1" trong path.
    assert all("Điều 1 > Điều 2" not in h for h in headings)


def test_contextual_off_no_prefix():
    """Tắt contextual → không prepend, kể cả markdown."""
    md = "# Luật\n\n## Điều 5\n\nNội dung."
    docs = [LoadedDoc(text=md, metadata={"source": "luat.md"})]
    chunks = chunk_documents(docs, chunk_size=512, overlap=0, contextual=False)
    assert not any("Điều 5 >" in c.text or c.text.startswith("Luật >") for c in chunks)
    assert "heading" not in chunks[0].metadata


# ── Loader ───────────────────────────────────────────────────────────────────

def test_load_txt_and_md(tmp_path):
    (tmp_path / "a.txt").write_text("Nội dung txt", encoding="utf-8")
    (tmp_path / "b.md").write_text("# Tiêu đề\nNội dung md", encoding="utf-8")
    (tmp_path / "skip.csv").write_text("bỏ,qua", encoding="utf-8")  # không hỗ trợ

    docs = load_documents(str(tmp_path))
    sources = sorted(d.metadata["source"] for d in docs)
    assert sources == ["a.txt", "b.md"]   # .csv bị bỏ qua, không raise


def test_load_missing_dir_raises():
    import pytest

    with pytest.raises(FileNotFoundError):
        load_documents("/khong/ton/tai/xyz")


# ── Retriever ────────────────────────────────────────────────────────────────

def test_retrieve_maps_hits_to_chunks(monkeypatch):
    """retrieve embed query → search → map sang RetrievedChunk (giữ source/score)."""
    from app.retrieval.vectorstore import SearchHit

    monkeypatch.setattr(retriever_mod.settings, "rag_query_rewriting", False)
    monkeypatch.setattr(retriever_mod.settings, "rag_rerank_enabled", False)
    monkeypatch.setattr(retriever_mod.settings, "rag_top_k", 2)

    import app.retrieval.embeddings as emb
    import app.retrieval.vectorstore as vs

    monkeypatch.setattr(emb, "embed_query", lambda q: [0.1, 0.2, 0.3])
    monkeypatch.setattr(
        vs,
        "search",
        lambda vector, top_k, where=None: [
            SearchHit(text="Điều 46 ...", score=0.9, metadata={"source": "luat.md"}),
            SearchHit(text="Điều 111 ...", score=0.7, metadata={"source": "luat.md"}),
        ],
    )

    chunks = retriever_mod.retrieve("Công ty TNHH?")
    assert len(chunks) == 2
    assert chunks[0].text == "Điều 46 ..."
    assert chunks[0].source == "luat.md"
    assert chunks[0].score == 0.9


def test_retrieve_empty_when_no_hits(monkeypatch):
    monkeypatch.setattr(retriever_mod.settings, "rag_query_rewriting", False)
    monkeypatch.setattr(retriever_mod.settings, "rag_rerank_enabled", False)

    import app.retrieval.embeddings as emb
    import app.retrieval.vectorstore as vs

    monkeypatch.setattr(emb, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(vs, "search", lambda vector, top_k, where=None: [])

    assert retriever_mod.retrieve("gì đó") == []
