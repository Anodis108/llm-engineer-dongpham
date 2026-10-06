"""Feedback loop — Module III, Bài 7, Section 4.

Bài học: chất lượng LLM không đo được bằng unit test. Cách duy nhất biết sản
phẩm có tốt lên là thu tín hiệu từ người dùng thật — nhưng có hai loại tín hiệu
với chi phí và độ tin cậy rất khác nhau, và trộn lẫn chúng là sai lầm phổ biến:

    EXPLICIT (thumbs up/down) — người dùng CHỦ ĐỘNG bấm.
        Đáng tin: người ta chỉ bấm khi thật sự có ý kiến.
        Nhưng HIẾM: tỉ lệ phản hồi thường 1-5%. Và lệch nặng về phía tiêu cực —
        người hài lòng thì đọc xong rồi đi, người bực mới bấm dislike. Nên
        "tỉ lệ dislike 20%" KHÔNG có nghĩa 20% câu trả lời tệ.

    IMPLICIT (copy, hỏi lại, bỏ đi) — suy ra từ hành vi, không cần ai bấm.
        Có ở gần như MỌI request → đủ dữ liệu để thấy xu hướng.
        Nhưng MỜ: "hỏi lại" có thể vì câu trả lời dở, cũng có thể vì người dùng
        vừa nghĩ ra câu hỏi mới. Không được coi là nhãn vàng.

Vì vậy: ghi CẢ HAI vào cùng một thang điểm (để vẽ chung một biểu đồ xu hướng),
nhưng gắn nhãn nguồn (`feedback_source_type` + `extra["signal_kind"]`) để khi
cần kết luận chặt thì lọc riêng explicit, còn khi cần phát hiện sớm thì nhìn
implicit.

Chú ý: đây là tín hiệu NGƯỜI DÙNG, khác hẳn điểm eval trong Bài 2 (judge chấm
theo golden set). Hai thứ bổ sung cho nhau — eval biết câu trả lời có ĐÚNG
không, feedback biết người dùng có HÀI LÒNG không. Lệch nhau là thông tin:
judge 4.8/5 mà dislike cao nghĩa là tiêu chí chấm đang không đo thứ người dùng
quan tâm.
"""

from __future__ import annotations

from enum import Enum

from app.config import settings


# (str, Enum) chứ không phải StrEnum, dù pyproject khai requires-python >=3.11:
# env conda của lớp (`llm-engineer`) chạy 3.10, mà StrEnum chỉ có từ 3.11. Đây là
# chỗ DUY NHẤT trong repo cần noqa: UP042.
#
# (str, Enum) hợp lệ ở mọi phiên bản Python và hành vi dùng ở đây giống hệt
# StrEnum — giá trị enum vẫn so sánh được với chuỗi. Nếu nâng env dev lên 3.11
# thì đổi sang StrEnum và xoá dòng noqa này.
class Signal(str, Enum):  # noqa: UP042
    """Các tín hiệu thu được. Giá trị dùng luôn làm khoá trên dashboard."""

    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    COPIED = "copied"
    REGENERATED = "regenerated"
    ABANDONED = "abandoned"


EXPLICIT_SIGNALS = frozenset({Signal.THUMBS_UP, Signal.THUMBS_DOWN})
IMPLICIT_SIGNALS = frozenset({Signal.COPIED, Signal.REGENERATED, Signal.ABANDONED})

# Quy về [0, 1] để explicit và implicit nằm chung một thang.
#
# `abandoned` = 0.0 chứ không phải âm: bỏ đi là tín hiệu KHÔNG có kết luận
# (người dùng có thể chỉ hết giờ), không phải bằng chứng câu trả lời tệ. Cho nó
# điểm âm sẽ kéo điểm trung bình xuống vì lý do không liên quan tới chất lượng.
SCORES: dict[Signal, float] = {
    Signal.THUMBS_UP: 1.0,
    Signal.THUMBS_DOWN: 0.0,
    Signal.COPIED: 1.0,
    Signal.REGENERATED: 0.0,
    Signal.ABANDONED: 0.0,
}

# Khoá feedback trên LangSmith. Một khoá duy nhất cho cả hai loại để vẽ được
# một đường xu hướng; phân biệt loại bằng metadata, không bằng khoá — tách khoá
# sẽ làm dashboard phải ghép 2 biểu đồ và không ai làm thế.
FEEDBACK_KEY = "user_signal"


def score_for(signal: Signal | str) -> float:
    return SCORES[Signal(signal)]


def kind_for(signal: Signal | str) -> str:
    """`explicit` hay `implicit` — quyết định tín hiệu này nặng bao nhiêu."""
    return "explicit" if Signal(signal) in EXPLICIT_SIGNALS else "implicit"


def submit_feedback(
    run_id: str,
    signal: Signal | str,
    *,
    comment: str | None = None,
    client=None,
    extra: dict | None = None,
) -> dict:
    """Ghi 1 tín hiệu người dùng lên run tương ứng trên LangSmith.

    Trả về dict mô tả những gì đã gửi (không trả object của SDK) để test không
    cần mạng và call site không phụ thuộc kiểu dữ liệu của LangSmith.

    `client=None` + MONITORING_ENABLED=false → no-op, trả về mô tả mà không gửi.
    """
    signal = Signal(signal)
    payload = {
        "run_id": str(run_id),
        "key": FEEDBACK_KEY,
        "score": score_for(signal),
        "comment": comment,
        "kind": kind_for(signal),
        "signal": signal.value,
    }

    if client is None and not settings.monitoring_enabled:
        return payload  # no-op có chủ đích — xem docstring module

    if client is None:
        from langsmith import Client

        client = Client(
            api_key=settings.langsmith_api_key, api_url=settings.langsmith_endpoint
        )

    client.create_feedback(
        run_id=run_id,
        key=FEEDBACK_KEY,
        score=payload["score"],
        comment=comment,
        # `extra` chứ không phải `value`: `value` bị LangSmith dùng cho feedback
        # dạng categorical, nhét dict vào đó sẽ hỏng phần hiển thị.
        extra={
            "signal": signal.value,
            "signal_kind": payload["kind"],
            **(extra or {}),
        },
    )
    return payload


def from_chat_event(event: str, *, dwell_seconds: float = 0.0) -> Signal | None:
    """Suy tín hiệu implicit từ hành vi UI. None = hành vi này không nói gì.

    Ngưỡng `dwell_seconds` cho `abandoned`: đóng tab sau 1 giây thì không kết
    luận được gì (có thể bấm nhầm); ngồi 30 giây rồi bỏ đi mới đáng ghi.
    """
    mapping = {
        "copy_answer": Signal.COPIED,
        "regenerate": Signal.REGENERATED,
        "thumbs_up": Signal.THUMBS_UP,
        "thumbs_down": Signal.THUMBS_DOWN,
    }
    if event == "close":
        return Signal.ABANDONED if dwell_seconds >= 30 else None
    return mapping.get(event)
