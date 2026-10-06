"""Sampling cho trace — Module III, Bài 7, Section 4 (Observability & Guardrails).

Vấn đề: trace MỌI request thì chi phí lưu trữ tăng tuyến tính theo traffic, mà
95% request là những câu giống nhau — nhìn 1000 trace bình thường không cho thêm
thông tin gì so với nhìn 50. Ngược lại, bỏ trace của request LỖI thì mất đúng
thứ duy nhất đáng xem.

Hai cách, và bài học chọn cách thứ hai:

    HEAD sampling — quyết định NGAY khi request bắt đầu, trước khi biết kết quả.
        Rẻ (không phải buffer gì), nhưng mù: không thể nói "giữ 100% request
        lỗi" vì lúc quyết định chưa biết request nào sẽ lỗi. Lỗi cũng bị lấy
        mẫu 5% như mọi request khác.

    TAIL sampling — quyết định SAU khi request xong, khi đã biết latency/lỗi/
        guardrail. Giữ được 100% ca đáng xem trong khi vẫn lấy mẫu thưa phần
        bình thường. Đổi lại: phải giữ trace trong RAM tới lúc kết thúc, nên
        process chết giữa chừng là mất trace đó.

File này là phần QUYẾT ĐỊNH; phần buffer + flush nằm trong
app/monitoring/tracing.py (xem `_Run._flush_tree`).

Quy tắc, theo đúng thứ tự ưu tiên dưới đây (thứ tự quan trọng — "giữ vì lỗi"
phải thắng "bỏ vì cache hit", vì một request lỗi hiếm khi là cache hit, còn nếu
có thì đó chính là ca lạ nhất cần xem):

    1. Lỗi exception          → LUÔN giữ
    2. Guardrail chặn         → LUÔN giữ  (đây là tín hiệu an toàn, không phải rác)
    3. Latency vượt ngưỡng    → LUÔN giữ  (p99 là thứ ta cần điều tra)
    4. Cache hit              → KHÔNG giữ (kết quả tất định, tái hiện được)
    5. Còn lại                → lấy mẫu theo tỉ lệ (mặc định 5%)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# Tỉ lệ lấy mẫu cho request "bình thường". 5% là điểm cân bằng thường dùng:
# đủ để thấy xu hướng trên dashboard, không đủ để hoá đơn lưu trữ thành vấn đề.
DEFAULT_SAMPLE_RATE = 0.05

# Ngưỡng "chậm". Bài học: 5s là mức bắt đầu khiến người dùng rời đi với chatbot.
DEFAULT_SLOW_LATENCY_S = 5.0


@dataclass(frozen=True, slots=True)
class SamplingDecision:
    sample: bool
    reason: str


def should_sample(
    *,
    error: bool = False,
    guardrail: bool = False,
    latency_s: float = 0.0,
    cache_hit: bool = False,
    slow_latency_s: float = DEFAULT_SLOW_LATENCY_S,
    rate: float | None = None,
    rng: random.Random | None = None,
) -> SamplingDecision:
    """Quyết định giữ hay bỏ trace. Thuần tuý, không I/O → test được không cần mạng.

    `rng` tiêm được để test tất định (không phải seed global random).
    """
    if error:
        return SamplingDecision(True, "error")
    if guardrail:
        return SamplingDecision(True, "guardrail")
    if latency_s >= slow_latency_s:
        return SamplingDecision(True, "slow")
    if cache_hit:
        return SamplingDecision(False, "cache_hit")

    r = DEFAULT_SAMPLE_RATE if rate is None else rate
    draw = (rng or random).random()
    return SamplingDecision(draw < r, f"random({r:.0%})")


@dataclass
class SamplingStats:
    """Đếm số trace giữ/bỏ theo lý do — dùng cho scripts/observability_demo.py.

    Không phải metric production (không đẩy đi đâu cả): đây là công cụ để TRẢ LỜI
    CÂU HỎI "sampling có đang giữ đúng thứ đáng giữ không", thứ mà nhìn log
    từng dòng không thấy được.
    """

    kept: dict[str, int] = field(default_factory=dict)
    dropped: dict[str, int] = field(default_factory=dict)

    def record(self, decision: SamplingDecision) -> None:
        bucket = self.kept if decision.sample else self.dropped
        bucket[decision.reason] = bucket.get(decision.reason, 0) + 1

    @property
    def total(self) -> int:
        return sum(self.kept.values()) + sum(self.dropped.values())

    @property
    def keep_rate(self) -> float:
        return sum(self.kept.values()) / self.total if self.total else 0.0

    def summary(self) -> str:
        lines = [f"tổng {self.total} request — giữ {sum(self.kept.values())} ({self.keep_rate:.0%})"]
        for reason, n in sorted(self.kept.items(), key=lambda kv: -kv[1]):
            lines.append(f"  giữ    {reason:16s} {n}")
        for reason, n in sorted(self.dropped.items(), key=lambda kv: -kv[1]):
            lines.append(f"  bỏ     {reason:16s} {n}")
        return "\n".join(lines)
