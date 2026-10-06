"""Tag cho trace — Module III, Bài 7, Section 4.

Bài học: một trace không tag thì chỉ trả lời được "request này chậm". Có tag thì
trả lời được "TẤT CẢ request dùng prompt v2 đều chậm" — tức là từ debug một ca
sang tìm được nguyên nhân hệ thống.

Bộ tag tối thiểu, chọn theo đúng các TRỤC ROLLBACK của Bài 6 — mỗi tag tương ứng
một thứ có thể rollback độc lập, nên khi trace chỉ ra bất thường thì biết ngay
gạt công tắc nào:

    feature        — endpoint/tính năng (answer, stream, agent, multi_agent...)
    model          — model đang phục vụ (gpt-4o-mini, llama3.2:3b, Qwen2.5-3B...)
    prompt_version — version prompt đang chạy  → rollback bằng prompt_rollback.py
    prompt_source  — langsmith | cache | fallback (registry có rơi tầng không?)
    cache          — hit | miss                  → trục cache
    cascade        — bật/tắt cascade            → trục cost

`prompt_source` đáng chú ý riêng: nếu dashboard cho thấy tỉ lệ `fallback` tăng,
nghĩa là LangSmith đang có vấn đề và app đang âm thầm chạy prompt cũ. Không có
tag này thì sự cố đó vô hình — app vẫn trả lời, chỉ là trả lời bằng prompt khác.
"""

from __future__ import annotations

from typing import Any

# Nguồn chân lý của danh sách tag. Thêm tag mới thì thêm ở đây để
# scripts/observability_demo.py và test cùng thấy.
TAG_KEYS: tuple[str, ...] = (
    "feature",
    "model",
    "prompt_version",
    "prompt_source",
    "cache",
    "cascade",
)


def run_tags(**tags: Any) -> dict[str, str]:
    """Lọc tag rỗng/None rồi ép về str.

    Lọc quan trọng hơn nó trông: tag `model=""` và không có tag `model` trông
    KHÁC nhau trên dashboard (một cái là nhóm rỗng, một cái là thiếu dữ liệu),
    và nhóm rỗng luôn gây khó chịu khi lọc.
    """
    return {
        key: str(value)
        for key, value in tags.items()
        if value is not None and str(value) != ""
    }


def with_tags(metadata: dict[str, Any] | None, **tags: Any) -> dict[str, Any]:
    """Gộp tag vào metadata có sẵn, không ghi đè metadata cũ."""
    return {**(metadata or {}), **run_tags(**tags)}
