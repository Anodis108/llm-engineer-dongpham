"""Chunking — Buổi 5 (RAG Pipeline), NATIVE (không LangChain).

Recursive character splitter: thử tách theo các separator từ "thô" đến "mịn"
(đoạn văn → câu → từ) để chunk không cắt giữa câu khi có thể. Tương đương
RecursiveCharacterTextSplitter nhưng tự viết, không phụ thuộc LangChain.

CONTEXTUAL RETRIEVAL (heading-based): với file markdown, mỗi chunk được PREPEND
chuỗi heading của section chứa nó (VD: "Luật Doanh nghiệp 2020 > Điều 47. Góp vốn").
Nhờ đó chunk tự đứng vững về ngữ cảnh → embedding + retrieval chính xác hơn, và LLM
biết chunk thuộc điều luật nào. (Biến thể nhẹ của Contextual Retrieval — Anthropic.)

chunk_size=512, overlap=64 (điểm khởi đầu tốt theo handbook).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.retrieval.loader import LoadedDoc

# Thứ tự separator: thử tách theo cái "to" trước, nhỏ dần.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Markdown ATX heading: 1-6 dấu # + khoảng trắng + tiêu đề.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(slots=True)
class Chunk:
    """Một đoạn văn bản đã chia, kèm metadata thừa hưởng từ document gốc."""

    text: str
    metadata: dict = field(default_factory=dict)


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Chia text thành các đoạn <= chunk_size, cố gắng cắt ở ranh giới tự nhiên."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    # Tìm separator "thô nhất" mà thực sự có trong text.
    sep = next((s for s in _SEPARATORS if s and s in text), "")
    if sep == "":
        # Không còn ranh giới → cắt cứng theo ký tự, có overlap.
        return _hard_split(text, chunk_size, overlap)

    parts = text.split(sep)
    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = part if not current else current + sep + part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # part đơn lẻ vẫn quá dài → đệ quy tách bằng separator mịn hơn.
            if len(part) > chunk_size:
                chunks.extend(_split_text(part, chunk_size, overlap))
                current = ""
            else:
                current = part
    if current:
        chunks.append(current)

    return _apply_overlap(chunks, overlap)


def _hard_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Cắt cứng theo ký tự khi không còn ranh giới tự nhiên."""
    step = max(1, chunk_size - overlap)
    return [text[i : i + chunk_size] for i in range(0, len(text), step)]


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Thêm phần đuôi của chunk trước vào đầu chunk sau (giữ ngữ cảnh ở ranh giới)."""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    out = [chunks[0]]
    # strict=False là CỐ Ý: `chunks[1:]` ngắn hơn `chunks` đúng 1 phần tử — đây
    # là cách ghép cặp (chunk trước, chunk sau), không phải lỗi lệch độ dài.
    for prev, cur in zip(chunks, chunks[1:], strict=False):
        tail = prev[-overlap:]
        out.append(tail + cur)
    return out


# ── Contextual Retrieval: tách markdown theo heading ─────────────────────────

@dataclass(slots=True)
class _Section:
    """Một section markdown: chuỗi heading (path) + nội dung text bên dưới."""

    heading_path: list[str]   # VD: ["Luật Doanh nghiệp 2020", "Điều 47. Góp vốn"]
    text: str


def _split_markdown_sections(text: str) -> list[_Section]:
    """Parse markdown thành các section, mỗi section biết chuỗi heading tổ tiên.

    Duy trì stack heading theo cấp (#=1, ##=2...): gặp heading cấp N thì cắt bỏ
    mọi heading cấp >= N khỏi stack rồi push heading mới → heading_path luôn là
    đường dẫn từ gốc tới section hiện tại.
    """
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []   # (level, title)
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(_Section([t for _, t in stack], body))
        buf.clear()

    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            flush()  # kết thúc section trước khi mở heading mới
            level = len(m.group(1))
            title = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buf.append(line)
    flush()
    return sections


def _prefix_from(heading_path: list[str]) -> str:
    """Chuỗi context prepend vào chunk, VD: 'Luật... > Điều 47. Góp vốn\n\n'."""
    return (" > ".join(heading_path) + "\n\n") if heading_path else ""


def _chunk_markdown(
    doc: LoadedDoc, chunk_size: int, overlap: int
) -> list[Chunk]:
    """Chunk một markdown doc, prepend heading path vào mỗi chunk (contextual retrieval)."""
    chunks: list[Chunk] = []
    idx = 0
    for section in _split_markdown_sections(doc.text):
        prefix = _prefix_from(section.heading_path)
        # Chừa chỗ cho prefix để chunk (kèm prefix) không vượt quá chunk_size.
        budget = max(64, chunk_size - len(prefix))
        for piece in _split_text(section.text, budget, overlap):
            meta = {
                **doc.metadata,
                "chunk_index": idx,
                "heading": " > ".join(section.heading_path),
            }
            text = prefix + piece
            meta["chunk_size"] = len(text)
            chunks.append(Chunk(text=text, metadata=meta))
            idx += 1
    return chunks


def _is_markdown(doc: LoadedDoc) -> bool:
    return str(doc.metadata.get("source", "")).lower().endswith(".md")


def chunk_documents(
    docs: list[LoadedDoc],
    chunk_size: int = 512,
    overlap: int = 64,
    contextual: bool = True,
) -> list[Chunk]:
    """Chia list document đã load thành list Chunk, giữ metadata gốc + chunk_index.

    Args:
        contextual: bật Contextual Retrieval cho markdown (prepend heading path).
            Doc không phải markdown luôn dùng recursive splitter thường.
    """
    chunks: list[Chunk] = []
    for doc in docs:
        if contextual and _is_markdown(doc):
            chunks.extend(_chunk_markdown(doc, chunk_size, overlap))
        else:
            for idx, piece in enumerate(_split_text(doc.text, chunk_size, overlap)):
                meta = {**doc.metadata, "chunk_index": idx, "chunk_size": len(piece)}
                chunks.append(Chunk(text=piece, metadata=meta))
    return chunks
