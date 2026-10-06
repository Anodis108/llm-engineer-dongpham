"""Test Buổi 7 — Monitoring (LangSmith hooks, tối thiểu).

Khi MONITORING_ENABLED=false (mặc định), trace_answer/trace_stream phải là
no-op hoàn toàn — không import package `langsmith` (chưa cài trong dev env).
Khi bật, mock `_get_client` để không cần key/network thật.
"""

from __future__ import annotations

from app.monitoring import tracing


def test_trace_answer_is_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(tracing.settings, "monitoring_enabled", False)

    with tracing.trace_answer("answer", "câu hỏi") as t:
        t["output"] = "trả lời"

    # Không raise, không import langsmith (nếu import sẽ ModuleNotFoundError
    # vì package chưa cài trong dev env).


def test_trace_stream_is_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(tracing.settings, "monitoring_enabled", False)

    tokens = iter(["a", "b", "c"])
    result = list(tracing.trace_stream("answer_stream", "câu hỏi", tokens))
    assert result == ["a", "b", "c"]


class _FakeClient:
    """Ghi lại mọi create_run/update_run — đủ để verify không cần network thật."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def create_run(self, **kwargs):
        self.calls.append(("create_run", kwargs))

    def update_run(self, **kwargs):
        self.calls.append(("update_run", kwargs))


def test_trace_answer_calls_langsmith_when_enabled(monkeypatch):
    monkeypatch.setattr(tracing.settings, "monitoring_enabled", True)
    # trace_sample_rate=1.0: các test dưới kiểm tra CƠ CHẾ GHI trace, không
    # kiểm tra sampling. Để mặc định 5% thì test sẽ đỏ ngẫu nhiên.
    monkeypatch.setattr(tracing.settings, "trace_sample_rate", 1.0)

    fake = _FakeClient()
    monkeypatch.setattr(tracing, "_get_client", lambda: fake)

    with tracing.trace_answer("answer", "câu hỏi") as t:
        t["output"] = "trả lời"

    create_calls = [kw for name, kw in fake.calls if name == "create_run"]
    update_calls = [kw for name, kw in fake.calls if name == "update_run"]

    assert create_calls[0]["name"] == "answer"
    assert create_calls[0]["inputs"] == {"input": "câu hỏi"}
    assert update_calls[0]["outputs"] == {"output": "trả lời"}
    assert update_calls[0]["run_id"] == create_calls[0]["id"]


def test_trace_step_nests_under_parent_run(monkeypatch):
    """trace_step tạo child run với parent_run_id đúng bằng run_id của span cha
    (lồng nhau trong LangSmith — giống app/agent/nodes.py xuyên _trace_span)."""
    monkeypatch.setattr(tracing.settings, "monitoring_enabled", True)
    # trace_sample_rate=1.0: các test dưới kiểm tra CƠ CHẾ GHI trace, không
    # kiểm tra sampling. Để mặc định 5% thì test sẽ đỏ ngẫu nhiên.
    monkeypatch.setattr(tracing.settings, "trace_sample_rate", 1.0)

    fake = _FakeClient()
    monkeypatch.setattr(tracing, "_get_client", lambda: fake)

    with tracing.trace_answer("answer", "câu hỏi") as t:
        with tracing.trace_step(t.get("_span"), "retrieve", input="q") as t2:
            t2["output"] = ["doc1"]
        t["output"] = "trả lời"

    create_calls = [kw for name, kw in fake.calls if name == "create_run"]
    assert len(create_calls) == 2
    parent_run_id = create_calls[0]["id"]
    assert create_calls[1]["name"] == "retrieve"
    assert create_calls[1]["parent_run_id"] == parent_run_id


def test_trace_answer_reports_error_without_losing_it(monkeypatch):
    """update() gọi 2 lần khi có lỗi (set error rồi set output trong finally) —
    lần gọi sau không được xoá mất error đã ghi (bug đã sửa trong _Run.update)."""
    monkeypatch.setattr(tracing.settings, "monitoring_enabled", True)
    # trace_sample_rate=1.0: các test dưới kiểm tra CƠ CHẾ GHI trace, không
    # kiểm tra sampling. Để mặc định 5% thì test sẽ đỏ ngẫu nhiên.
    monkeypatch.setattr(tracing.settings, "trace_sample_rate", 1.0)

    fake = _FakeClient()
    monkeypatch.setattr(tracing, "_get_client", lambda: fake)

    try:
        with tracing.trace_answer("answer", "câu hỏi"):
            raise ValueError("boom")
    except ValueError:
        pass

    update_calls = [kw for name, kw in fake.calls if name == "update_run"]
    assert update_calls[0]["error"] == "boom"


def test_trace_stream_calls_langsmith_when_enabled(monkeypatch):
    monkeypatch.setattr(tracing.settings, "monitoring_enabled", True)
    # trace_sample_rate=1.0: các test dưới kiểm tra CƠ CHẾ GHI trace, không
    # kiểm tra sampling. Để mặc định 5% thì test sẽ đỏ ngẫu nhiên.
    monkeypatch.setattr(tracing.settings, "trace_sample_rate", 1.0)

    fake = _FakeClient()
    monkeypatch.setattr(tracing, "_get_client", lambda: fake)

    tokens = iter(["a", "b"])
    result = list(tracing.trace_stream("answer_stream", "q", tokens))

    assert result == ["a", "b"]
    create_calls = [kw for name, kw in fake.calls if name == "create_run"]
    update_calls = [kw for name, kw in fake.calls if name == "update_run"]
    assert create_calls[0]["name"] == "answer_stream"
    assert update_calls[0]["outputs"] == {"output": "ab"}
