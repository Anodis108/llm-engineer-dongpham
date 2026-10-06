"""Che PII trước khi trace ra ngoài — Module III, Bài 7, Section 3 + 4.

`redact_pii()` trong app/guardrails/pii.py che PII trên MỘT chuỗi, dùng cho
đường trả lời người dùng. Ở đây cần thêm một thứ mà bản đó không làm: che PII
trong cả một CẤU TRÚC (dict/list lồng nhau) — vì payload trace không phải một
chuỗi mà là `{"input": {...}, "documents": [...]}`.

Điểm mấu chốt về THỨ TỰ: che phải xảy ra TRƯỚC khi dữ liệu rời khỏi process.
Trace đã gửi lên LangSmith rồi thì không rút lại được — xoá run trên dashboard
không xoá bản đã nằm trong backup/log của họ. Nên đây là việc phải làm ở tầng
export (app/monitoring/tracing.py), không phải việc của người dùng dashboard.

Vì sao không che ở tầng guardrail input: che ở đó sẽ làm hỏng câu trả lời (model
cần thấy SĐT để trả lời "số này của ai"). Che ở tầng export thì dữ liệu tới model
vẫn đầy đủ, chỉ bản GHI LẠI bị che — đúng ranh giới.
"""

from __future__ import annotations

from typing import Any

from app.guardrails.pii import redact_pii

# Che luôn trong metadata: người dùng có thể dán PII vào trường tự do, và
# metadata là chỗ dễ quên nhất.
_MAX_DEPTH = 8


def redact_deep(value: Any, _depth: int = 0) -> Any:
    """Che PII trong chuỗi/dict/list lồng nhau. Giữ nguyên kiểu dữ liệu.

    `_depth` chặn đệ quy vô hạn — payload trace có thể chứa object tự tham
    chiếu, và treo process vì che PII thì tệ hơn là không che.
    """
    if _depth >= _MAX_DEPTH:
        return "[QUÁ SÂU — BỎ QUA]"
    if isinstance(value, str):
        return redact_pii(value)
    if isinstance(value, dict):
        return {k: redact_deep(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_deep(v, _depth + 1) for v in value]
    return value
