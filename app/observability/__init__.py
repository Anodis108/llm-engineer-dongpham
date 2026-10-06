"""Observability — Module III, Bài 7 (Observability & Guardrails).

Bốn mảnh, mỗi mảnh trả lời một câu hỏi khác nhau:

    sampling.py  — Ghi cái gì?        (tail sampling: giữ 100% ca đáng xem)
    redaction.py — Ghi như thế nào?   (che PII trước khi dữ liệu rời process)
    metrics.py   — Ghi kèm gì?        (tag theo trục rollback để lọc được)
    feedback.py  — Người dùng nghĩ gì? (explicit + implicit signal)

Chúng không độc lập: tag ở metrics.py chính là thứ khiến một trace đã sample
trở nên có ích, và feedback gắn vào run_id do tracing.py sinh ra. Xem
app/monitoring/tracing.py để thấy chỗ ráp nối.
"""
