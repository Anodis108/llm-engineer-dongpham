"""Test observability — Module III, Bài 7 (sampling, redaction, tags, feedback).

Không test nào gọi mạng: client LangSmith thay bằng object giả.
"""

from __future__ import annotations

import random

import pytest

from app.monitoring import tracing
from app.observability import feedback, metrics, redaction, sampling

# ─── Sampling ────────────────────────────────────────────────────────────────


def test_error_is_always_sampled():
    """Điểm mấu chốt của TAIL sampling: giữ 100% ca lỗi, bất kể tỉ lệ."""
    d = sampling.should_sample(error=True, rate=0.0)
    assert d.sample and d.reason == "error"


def test_guardrail_is_always_sampled():
    d = sampling.should_sample(guardrail=True, rate=0.0)
    assert d.sample and d.reason == "guardrail"


def test_slow_request_is_always_sampled():
    d = sampling.should_sample(latency_s=7.5, slow_latency_s=5.0, rate=0.0)
    assert d.sample and d.reason == "slow"


def test_fast_request_below_threshold_is_not_slow():
    d = sampling.should_sample(latency_s=0.3, slow_latency_s=5.0, rate=1.0)
    assert d.reason.startswith("random")


def test_cache_hit_is_dropped():
    """Kết quả tất định và tái hiện được — không đáng tốn chỗ lưu."""
    d = sampling.should_sample(cache_hit=True, rate=1.0)
    assert not d.sample and d.reason == "cache_hit"


def test_error_beats_cache_hit():
    """Thứ tự ưu tiên: request vừa lỗi vừa là cache hit là ca LẠ nhất, phải giữ."""
    d = sampling.should_sample(error=True, cache_hit=True, rate=0.0)
    assert d.sample and d.reason == "error"


def test_normal_request_follows_rate():
    rng = random.Random(0)
    decisions = [sampling.should_sample(rate=0.25, rng=rng) for _ in range(2000)]
    kept = sum(d.sample for d in decisions)
    assert 0.22 < kept / len(decisions) < 0.28


def test_rate_zero_drops_everything_normal():
    rng = random.Random(0)
    assert not any(sampling.should_sample(rate=0.0, rng=rng).sample for _ in range(50))


def test_sampling_stats_counts_by_reason():
    stats = sampling.SamplingStats()
    stats.record(sampling.should_sample(error=True))
    stats.record(sampling.should_sample(cache_hit=True))
    stats.record(sampling.should_sample(cache_hit=True))

    assert stats.total == 3
    assert stats.kept == {"error": 1}
    assert stats.dropped == {"cache_hit": 2}
    assert stats.keep_rate == pytest.approx(1 / 3)
    assert "error" in stats.summary()


def test_sampling_stats_empty_is_safe():
    assert sampling.SamplingStats().keep_rate == 0.0


# ─── Redaction ───────────────────────────────────────────────────────────────


def test_redact_deep_handles_nested_structures():
    payload = {
        "question": "Gọi 0912345678 giúp tôi",
        "docs": [{"text": "email a@b.com", "score": 0.9}],
        "meta": (1, "cccd 123456789012"),
    }
    out = redaction.redact_deep(payload)

    assert out["question"] == "Gọi [PHONE_REDACTED] giúp tôi"
    assert out["docs"][0]["text"] == "email [EMAIL_REDACTED]"
    assert out["docs"][0]["score"] == 0.9  # giữ nguyên kiểu, không ép về str
    assert out["meta"] == [1, "cccd [CCCD_REDACTED]"]


def test_redact_deep_leaves_non_strings_alone():
    assert redaction.redact_deep(42) == 42
    assert redaction.redact_deep(None) is None


def test_redact_deep_stops_at_max_depth():
    """Object tự tham chiếu không được làm treo process."""
    node: dict = {}
    node["self"] = node
    assert redaction.redact_deep(node) is not None


# ─── Tags ────────────────────────────────────────────────────────────────────


def test_run_tags_drops_empty_values():
    tags = metrics.run_tags(feature="answer", model="", cache=None, cascade="on")
    assert tags == {"feature": "answer", "cascade": "on"}


def test_run_tags_stringifies():
    assert metrics.run_tags(confidence=0.8) == {"confidence": "0.8"}


def test_with_tags_preserves_existing_metadata():
    out = metrics.with_tags({"latency_s": 1.2}, feature="answer")
    assert out == {"latency_s": 1.2, "feature": "answer"}


# ─── Feedback ────────────────────────────────────────────────────────────────


def test_explicit_and_implicit_are_classified():
    assert feedback.kind_for(feedback.Signal.THUMBS_UP) == "explicit"
    assert feedback.kind_for(feedback.Signal.COPIED) == "implicit"


def test_scores_are_normalised_to_unit_range():
    for signal in feedback.Signal:
        assert 0.0 <= feedback.score_for(signal) <= 1.0


def test_abandoned_is_neutral_not_negative():
    """Bỏ đi không kết luận được gì — cho điểm âm sẽ kéo lệch xu hướng."""
    assert feedback.score_for(feedback.Signal.ABANDONED) == 0.0


def test_submit_feedback_is_noop_when_monitoring_disabled(monkeypatch):
    monkeypatch.setattr(feedback.settings, "monitoring_enabled", False)
    payload = feedback.submit_feedback("run-1", feedback.Signal.THUMBS_UP)
    assert payload["score"] == 1.0
    assert payload["kind"] == "explicit"


def test_submit_feedback_sends_with_kind_in_extra(monkeypatch):
    sent = {}

    class _Client:
        def create_feedback(self, **kw):
            sent.update(kw)

    feedback.submit_feedback(
        "run-1", feedback.Signal.REGENERATED, comment="sai", client=_Client()
    )
    assert sent["key"] == feedback.FEEDBACK_KEY
    assert sent["score"] == 0.0
    assert sent["extra"]["signal_kind"] == "implicit"


def test_from_chat_event_maps_behaviour():
    assert feedback.from_chat_event("copy_answer") == feedback.Signal.COPIED
    assert feedback.from_chat_event("thumbs_down") == feedback.Signal.THUMBS_DOWN


def test_from_chat_event_ignores_quick_close():
    """Đóng tab sau 1s có thể chỉ là bấm nhầm — không phải tín hiệu."""
    assert feedback.from_chat_event("close", dwell_seconds=1.0) is None
    assert feedback.from_chat_event("close", dwell_seconds=45) == feedback.Signal.ABANDONED


def test_from_chat_event_ignores_unknown():
    assert feedback.from_chat_event("scroll") is None


# ─── Tích hợp vào tracing (tail sampling + redaction) ────────────────────────


class _FakeClient:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def create_run(self, **kwargs):
        self.calls.append(("create_run", kwargs))

    def update_run(self, **kwargs):
        self.calls.append(("update_run", kwargs))


@pytest.fixture
def traced(monkeypatch):
    """Bật monitoring, tiêm client giả, trả về (fake_client, bật/tắt sampling)."""
    monkeypatch.setattr(tracing.settings, "monitoring_enabled", True)
    monkeypatch.setattr(tracing.settings, "trace_redact_pii", True)
    fake = _FakeClient()
    monkeypatch.setattr(tracing, "_get_client", lambda: fake)
    return fake


def test_normal_request_is_dropped_when_rate_zero(traced, monkeypatch):
    monkeypatch.setattr(tracing.settings, "trace_sample_rate", 0.0)
    with tracing.trace_answer("answer", "câu hỏi") as t:
        t["output"] = "trả lời"
    assert traced.calls == []


def test_error_request_is_kept_even_when_rate_zero(traced, monkeypatch):
    """Tail sampling: quyết định SAU khi biết lỗi, nên rate=0 vẫn giữ ca lỗi."""
    monkeypatch.setattr(tracing.settings, "trace_sample_rate", 0.0)
    with pytest.raises(ValueError):
        with tracing.trace_answer("answer", "câu hỏi"):
            raise ValueError("boom")

    update_calls = [kw for name, kw in traced.calls if name == "update_run"]
    assert update_calls[0]["error"] == "boom"


def test_guardrail_signal_forces_keep(traced, monkeypatch):
    monkeypatch.setattr(tracing.settings, "trace_sample_rate", 0.0)
    with tracing.trace_answer("answer", "câu hỏi") as t:
        t["output"] = "chặn"
        t["guardrail"] = True

    metadata = [kw["metadata"] for name, kw in traced.calls if name == "create_run"][0]
    assert metadata["sample_reason"] == "guardrail"


def test_cache_hit_signal_drops_even_when_rate_one(traced, monkeypatch):
    monkeypatch.setattr(tracing.settings, "trace_sample_rate", 1.0)
    with tracing.trace_answer("answer", "câu hỏi") as t:
        t["output"] = "trả lời"
        t["cache_hit"] = True
    assert traced.calls == []


def test_parent_is_flushed_before_child(traced, monkeypatch):
    """LangSmith dựng cây theo parent_run_id — gửi con trước sẽ tạo cây mồ côi."""
    monkeypatch.setattr(tracing.settings, "trace_sample_rate", 1.0)
    with tracing.trace_answer("answer", "q") as t:
        with tracing.trace_step(t.get("_span"), "retrieve", input="q") as t2:
            t2["output"] = ["doc"]
        t["output"] = "trả lời"

    create_calls = [kw for name, kw in traced.calls if name == "create_run"]
    assert [c["name"] for c in create_calls] == ["answer", "retrieve"]
    assert "parent_run_id" not in create_calls[0]
    assert create_calls[1]["parent_run_id"] == create_calls[0]["id"]


def test_deeply_nested_spans_are_flushed_in_tree_order(traced, monkeypatch):
    monkeypatch.setattr(tracing.settings, "trace_sample_rate", 1.0)
    with tracing.trace_answer("answer", "q") as t:
        with tracing.trace_step(t.get("_span"), "retrieve") as t2:
            with tracing.trace_step(t2.get("_span"), "search") as t3:
                t3["output"] = "hits"
        t["output"] = "x"

    names = [kw["name"] for name, kw in traced.calls if name == "create_run"]
    assert names == ["answer", "retrieve", "search"]


def test_pii_is_redacted_before_export(traced, monkeypatch):
    """Trace đã gửi lên LangSmith thì không rút lại được — che phải ở đây."""
    monkeypatch.setattr(tracing.settings, "trace_sample_rate", 1.0)
    with tracing.trace_answer("answer", "SĐT tôi là 0912345678") as t:
        t["output"] = "Email liên hệ: a@b.com"

    create_kw = [kw for name, kw in traced.calls if name == "create_run"][0]
    update_kw = [kw for name, kw in traced.calls if name == "update_run"][0]
    assert create_kw["inputs"]["input"] == "SĐT tôi là [PHONE_REDACTED]"
    assert update_kw["outputs"]["output"] == "Email liên hệ: [EMAIL_REDACTED]"


def test_redaction_can_be_disabled(traced, monkeypatch):
    monkeypatch.setattr(tracing.settings, "trace_sample_rate", 1.0)
    monkeypatch.setattr(tracing.settings, "trace_redact_pii", False)
    with tracing.trace_answer("answer", "SĐT 0912345678") as t:
        t["output"] = "ok"

    create_kw = [kw for name, kw in traced.calls if name == "create_run"][0]
    assert create_kw["inputs"]["input"] == "SĐT 0912345678"


def test_tags_are_merged_into_metadata(traced, monkeypatch):
    monkeypatch.setattr(tracing.settings, "trace_sample_rate", 1.0)
    with tracing.trace_answer("answer", "q", metadata={"feature": "answer"}) as t:
        t["tags"] = {"prompt_source": "cache"}
        t["output"] = "x"

    metadata = [kw["metadata"] for name, kw in traced.calls if name == "create_run"][0]
    assert metadata["feature"] == "answer"
    assert metadata["prompt_source"] == "cache"
    assert "latency_s" in metadata


def test_langsmith_failure_does_not_break_request(monkeypatch, caplog):
    """Monitoring hỏng phải im lặng — trace là quan sát, không phải đường tới hạn."""
    monkeypatch.setattr(tracing.settings, "monitoring_enabled", True)
    monkeypatch.setattr(tracing.settings, "trace_sample_rate", 1.0)

    class _Broken:
        def create_run(self, **kw):
            raise ConnectionError("langsmith sập")

        def update_run(self, **kw):
            raise ConnectionError("langsmith sập")

    monkeypatch.setattr(tracing, "_get_client", lambda: _Broken())

    with tracing.trace_answer("answer", "q") as t:  # không được raise
        t["output"] = "vẫn trả lời được"

    assert "Không ghi được trace" in caplog.text


def test_stream_partial_output_is_recorded_on_disconnect(traced, monkeypatch):
    """Client ngắt giữa stream vẫn để lại trace phần đã phát — đó là tín hiệu."""
    monkeypatch.setattr(tracing.settings, "trace_sample_rate", 1.0)

    def _tokens():
        yield "a"
        yield "b"

    gen = tracing.trace_stream("answer_stream", "q", _tokens())
    assert next(gen) == "a"
    gen.close()  # mô phỏng client ngắt kết nối

    update_calls = [kw for name, kw in traced.calls if name == "update_run"]
    assert update_calls[0]["outputs"] == {"output": "a"}
