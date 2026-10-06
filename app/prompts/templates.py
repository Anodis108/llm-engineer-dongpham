"""Prompt templates — Bài 1, Section 3 (Prompt Patterns).

Minh hoạ role prompting (persona luật sư VN) + chỗ chèn context cho RAG (Buổi 5).
Ở Buổi 1, context luôn rỗng (retriever là stub) — system prompt được thiết kế để
xử lý cả trường hợp "không có tài liệu" một cách trung thực.

Module III, Bài 6: nguồn chân lý của prompt đã chuyển sang LangSmith Prompt Hub.
`build_messages()` giờ là lớp mỏng: dựng `context_section` rồi để registry render.
Hằng số `LEGAL_SYSTEM_PROMPT` bên dưới KHÔNG bị xoá — nó là tầng fallback cuối
cùng (bài học Section 4: "Registry hosted cần fallback"), để LangSmith có sập
thì app vẫn trả lời được thay vì 500.

Bật registry bằng PROMPT_REGISTRY=langsmith trong .env. Mặc định "local" để
test/CI chạy hermetic — không gọi mạng, không cần credentials.
"""

from __future__ import annotations

import logging

from app.config import settings
from app.prompts.registry import PromptUnavailable, get_registry
from app.retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

# Role prompting (Section 3): persona cụ thể, có quy tắc rõ ràng.
# Đây là bản v1 của prompt "rag-answer" (xem app/prompts/definitions.py) — giữ
# nguyên văn làm fallback, KHÔNG sửa ở đây nữa. Muốn đổi hành vi prompt thì
# thêm version mới trong definitions.py rồi push lên LangSmith.
LEGAL_SYSTEM_PROMPT = """Bạn là trợ lý pháp lý chuyên về luật doanh nghiệp Việt Nam.

Quy tắc:
- Trả lời chính xác, ngắn gọn, bằng tiếng Việt.
- Khi có TÀI LIỆU THAM KHẢO bên dưới, CHỈ trả lời dựa trên tài liệu đó và trích dẫn nguồn.
- Khi KHÔNG có tài liệu tham khảo, nói rõ rằng câu trả lời dựa trên hiểu biết chung
  và khuyến nghị người dùng kiểm chứng với văn bản luật chính thức.
- Luôn khuyên tham khảo luật sư cho các vụ việc cụ thể.
- Luôn tránh đưa ra tư vấn pháp lý cụ thể cho từng trường hợp, chỉ cung cấp thông tin chung.
"""

# Tên prompt trên registry (không kèm tiền tố project — registry tự ghép).
RAG_ANSWER_PROMPT = "rag-answer"


def _build_context_section(context_chunks: list[RetrievedChunk] | None) -> str:
    """Phần context chèn vào cuối system prompt — rỗng khi không có chunk nào.

    Tách riêng vì đây là BIẾN của template (`{context_section}`), không phải
    nội dung prompt: prompt trên LangSmith không đổi, chỉ giá trị truyền vào đổi.
    """
    if not context_chunks:
        return ""

    # Hiển thị source (từ Buổi 5 chunk có metadata thật) để model trích dẫn đúng nguồn.
    context_block = "\n\n".join(
        f"[Nguồn {i + 1}: {c.source or 'tài liệu'}] {c.text}"
        for i, c in enumerate(context_chunks)
    )
    return f"\n\n--- TÀI LIỆU THAM KHẢO ---\n{context_block}"


def _build_from_constant(question: str, context_section: str) -> list[dict]:
    """Tầng fallback cuối — hành vi y hệt trước Bài 6."""
    return [
        {"role": "system", "content": LEGAL_SYSTEM_PROMPT + context_section},
        {"role": "user", "content": question},
    ]


def build_messages(
    question: str,
    context_chunks: list[RetrievedChunk] | None = None,
    meta: dict | None = None,
) -> list[dict]:
    """Dựng danh sách messages cho một câu hỏi.

    Nếu có context_chunks (từ Buổi 5 trở đi) thì chèn vào trước câu hỏi —
    đây chính là bước "Augment" của RAG. Ở Buổi 1, context_chunks rỗng.

    Thứ tự 3 tầng (Bài 6, Section 4): LangSmith → cache đĩa → hằng số ở trên.
    Tầng 1-2 nằm trong registry; ở đây chỉ bắt lỗi cuối cùng.

    `meta` là tham số RA (out-param) tuỳ chọn: nếu truyền vào, hàm ghi thêm
    `prompt_source` / `prompt_version` của lần render này. Dùng để gắn tag cho
    trace (Bài 7) mà không phải đổi kiểu trả về — call site cũ không truyền
    `meta` thì không thấy khác gì.

    Vì sao cần tag `prompt_source`: nếu registry rơi xuống tầng cache/fallback,
    app VẪN trả lời bình thường — chỉ là trả lời bằng prompt cũ. Không có tag
    này thì sự cố đó hoàn toàn vô hình trên dashboard.
    """
    context_section = _build_context_section(context_chunks)

    if settings.prompt_registry == "langsmith":
        try:
            result = get_registry().render(
                RAG_ANSWER_PROMPT,
                context_section=context_section,
                question=question,
            )
            logger.debug(
                "prompt rag-answer render từ %s (%s)", result.source, result.commit_hash[:8]
            )
            if meta is not None:
                meta["prompt_source"] = result.source
                meta["prompt_version"] = result.commit_hash[:8] or "unknown"
            return result.messages
        except PromptUnavailable as exc:
            # Rơi hết 2 tầng trên. App vẫn phải trả lời được — chỉ chất lượng
            # prompt là bản cũ hơn, không phải lỗi 500.
            logger.warning("Prompt registry không dùng được (%s) — dùng hằng số", exc)

    if meta is not None:
        meta["prompt_source"] = "fallback"
        meta["prompt_version"] = f"{RAG_ANSWER_PROMPT}-v1-const"
    return _build_from_constant(question, context_section)
