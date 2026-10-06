"""Test Module III, Bài 2 — Eval Pipeline (dataset, scorers, gate).

Không gọi API thật (mock LangSmith Client + judge_answer). runner.py không
test riêng ở đây — nó chỉ orchestrate dataset.py + scorers.py (đã test) qua
langsmith.evaluate() thật; đã verify thủ công với LangSmith + pipeline thật
khi phát triển (xem README Module III cho cách chạy tay).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.eval_pipeline import gate, scorers
from app.eval_pipeline.dataset import (
    GoldenCase,
    GoldenDataset,
    load_dataset,
    sync_dataset_to_langsmith,
)

# ── dataset.py: load_dataset validation ───────────────────────────────────────

def test_load_dataset_reads_real_v1_yaml():
    ds = load_dataset("data/eval/legal_qa/v1.yaml")
    assert ds.name == "legal_qa"
    assert len(ds.cases) == 30
    assert ds.slice_types() == {"lookup", "comparison", "out_of_scope", "injection"}


def test_load_dataset_rejects_duplicate_ids(tmp_path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(
        """
dataset: test
version: 1
cases:
  - id: dup
    question: "q1"
    expected: "e1"
    slice: {type: lookup}
  - id: dup
    question: "q2"
    expected: "e2"
    slice: {type: lookup}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="trùng lặp"):
        load_dataset(bad_yaml)


def test_load_dataset_rejects_missing_slice_type(tmp_path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(
        """
dataset: test
version: 1
cases:
  - id: c1
    question: "q1"
    expected: "e1"
    slice: {}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="slice.type"):
        load_dataset(bad_yaml)


def test_select_subset_covers_all_slice_types_not_just_first_n():
    """Bug đã sửa: cắt N case đầu YAML bỏ sót hoàn toàn injection/out_of_scope
    vì v1.yaml liệt kê 18 lookup case trước — select_subset phải round-robin
    qua từng slice.type để luôn phủ đủ (Section 6: 'ưu tiên slice rủi ro')."""
    ds = load_dataset("data/eval/legal_qa/v1.yaml")

    subset = ds.select_subset(6)

    assert len(subset.cases) == 6
    assert subset.slice_types() == {"lookup", "comparison", "out_of_scope", "injection"}


def test_select_subset_returns_full_dataset_when_n_exceeds_total():
    ds = _make_dataset("c1", "c2")
    subset = ds.select_subset(100)
    assert len(subset.cases) == 2


def test_load_dataset_rejects_empty_question(tmp_path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(
        """
dataset: test
version: 1
cases:
  - id: c1
    question: "   "
    expected: "e1"
    slice: {type: lookup}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="không được rỗng"):
        load_dataset(bad_yaml)


# ── dataset.py: sync_dataset_to_langsmith (mock Client) ──────────────────────

class _FakeExample:
    def __init__(self, id, metadata):
        self.id = id
        self.metadata = metadata


class _FakeLangSmithClient:
    """Mô phỏng 1 dataset LangSmith đã có sẵn N example, ghi lại mọi lời gọi."""

    def __init__(self, existing_examples=None):
        self._existing = existing_examples or []
        self.calls: list[tuple[str, dict]] = []

    def has_dataset(self, dataset_name):
        return bool(self._existing) or getattr(self, "_dataset_exists", False)

    def create_dataset(self, name, **kwargs):
        self.calls.append(("create_dataset", {"name": name, **kwargs}))
        self._dataset_exists = True

    def create_examples(self, **kwargs):
        self.calls.append(("create_examples", kwargs))

    def list_examples(self, dataset_name):
        return iter(self._existing)

    def update_examples(self, **kwargs):
        self.calls.append(("update_examples", kwargs))

    def delete_example(self, example_id):
        self.calls.append(("delete_example", {"id": example_id}))


def _make_dataset(*case_ids: str) -> GoldenDataset:
    return GoldenDataset(
        name="test_ds",
        version=1,
        cases=[
            GoldenCase(id=cid, question=f"q-{cid}", expected=f"e-{cid}", slice={"type": "lookup"})
            for cid in case_ids
        ],
    )


def test_sync_creates_dataset_when_not_exists():
    client = _FakeLangSmithClient(existing_examples=[])
    ds = _make_dataset("c1", "c2")

    name = sync_dataset_to_langsmith(ds, client=client)

    assert name == "test_ds"
    create_calls = [kw for op, kw in client.calls if op == "create_dataset"]
    examples_calls = [kw for op, kw in client.calls if op == "create_examples"]
    assert len(create_calls) == 1
    assert len(examples_calls[0]["examples"]) == 2


def test_sync_diffs_against_existing_examples():
    """Case đã có (c1) -> update; case mới (c2) -> create; case bị xoá khỏi YAML
    (c_old, còn trên LangSmith) -> delete. Không xoá-tạo-lại toàn bộ."""
    existing = [_FakeExample(id="ls-1", metadata={"case_id": "c1"}), _FakeExample(id="ls-old", metadata={"case_id": "c_old"})]
    client = _FakeLangSmithClient(existing_examples=existing)
    client._dataset_exists = True

    ds = _make_dataset("c1", "c2")  # c1 giữ lại (update), c2 mới (create), c_old bị xoá

    sync_dataset_to_langsmith(ds, client=client)

    create_calls = [kw for op, kw in client.calls if op == "create_examples"]
    update_calls = [kw for op, kw in client.calls if op == "update_examples"]
    delete_calls = [kw for op, kw in client.calls if op == "delete_example"]

    assert len(create_calls[0]["examples"]) == 1  # chỉ c2
    assert len(update_calls[0]["updates"]) == 1  # chỉ c1
    assert update_calls[0]["updates"][0]["id"] == "ls-1"
    assert delete_calls[0]["id"] == "ls-old"


# ── scorers.py: score_rules ───────────────────────────────────────────────────

def _fake_run(answer: str):
    return SimpleNamespace(outputs={"answer": answer})


def _fake_example(expected: str = "", must_include=None, must_not_include=None, question=""):
    return SimpleNamespace(
        inputs={"question": question},
        outputs={"expected": expected},
        metadata={"must_include": must_include or [], "must_not_include": must_not_include or []},
    )


def test_score_rules_passes_when_all_keywords_present():
    run = _fake_run("Theo Điều 5, mức lương tối đa là 70%.")
    example = _fake_example(expected="Điều 5", must_include=["Điều 5", "70%"])

    out = scorers.score_rules(run, example)
    rule_result = next(r for r in out["results"] if r["key"] == "rule_pass")
    assert rule_result["score"] is True


def test_score_rules_fails_when_keyword_missing():
    run = _fake_run("Câu trả lời không liên quan.")
    example = _fake_example(must_include=["Điều 5"])

    out = scorers.score_rules(run, example)
    rule_result = next(r for r in out["results"] if r["key"] == "rule_pass")
    assert rule_result["score"] is False
    assert "Điều 5" in rule_result["comment"]


def test_score_rules_fails_when_forbidden_keyword_present():
    run = _fake_run("Tôi đã được giải phóng.")
    example = _fake_example(must_not_include=["giải phóng"])

    out = scorers.score_rules(run, example)
    rule_result = next(r for r in out["results"] if r["key"] == "rule_pass")
    assert rule_result["score"] is False


def test_score_rules_checks_article_citation_when_expected_has_one():
    run = _fake_run("Không có trích dẫn điều luật nào ở đây.")
    example = _fake_example(expected="Theo Điều 5 Nghị định...")

    out = scorers.score_rules(run, example)
    cites = next(r for r in out["results"] if r["key"] == "cites_article")
    assert cites["score"] is False


def test_score_judge_wraps_judge_answer(monkeypatch):
    class FakeJudgeScore:
        accuracy = 5
        completeness = 4
        clarity = 5
        groundedness = 5
        reasoning = "Tốt."
        overall = 4.75

    monkeypatch.setattr(scorers, "judge_answer", lambda q, a, reference=None: FakeJudgeScore())

    run = _fake_run("Câu trả lời.")
    example = _fake_example(expected="Đáp án chuẩn", question="Câu hỏi?")

    out = scorers.score_judge(run, example)
    assert out["key"] == "judge_overall"
    assert out["score"] == 4.75


# ── gate.py: summarize + check_gate ───────────────────────────────────────────

def _fake_eval_result(key, score):
    return SimpleNamespace(key=key, score=score, comment=None)


def _fake_row(case_id, slice_type, judge_score, rule_pass):
    return {
        "run": None,
        "example": SimpleNamespace(metadata={"case_id": case_id, "slice": {"type": slice_type}}),
        "evaluation_results": {
            "results": [
                _fake_eval_result("judge_overall", judge_score),
                _fake_eval_result("rule_pass", rule_pass),
            ]
        },
    }


def test_summarize_computes_overall_and_by_slice():
    rows = [
        _fake_row("c1", "lookup", 5.0, True),
        _fake_row("c2", "lookup", 3.0, False),
        _fake_row("c3", "injection", 4.0, True),
    ]

    summary = gate.summarize(rows)

    assert summary.n == 3
    assert summary.overall == 4.0
    assert summary.rule_pass_rate == pytest.approx(2 / 3, rel=1e-3)
    assert summary.by_slice_type["lookup"] == 4.0
    assert summary.by_slice_type["injection"] == 4.0


def test_check_gate_passes_within_tolerance():
    summary = gate.EvalSummary(n=10, overall=4.7, rule_pass_rate=0.9, by_slice_type={"injection": 0.85})
    gates = {
        "overall": gate.GateConfig(baseline=4.8, drop_tolerance=0.1),
        "slice:injection": gate.GateConfig(baseline=0.9, drop_tolerance=0.1),
    }

    result = gate.check_gate(summary, gates)
    assert result.passed is True
    assert result.failures == []


def test_check_gate_fails_and_reports_which_metric():
    """Bài học Section 5: report phải chỉ RÕ metric nào rớt, không chỉ 'fail'."""
    summary = gate.EvalSummary(n=10, overall=4.0, rule_pass_rate=0.9, by_slice_type={"injection": 0.5})
    gates = {
        "overall": gate.GateConfig(baseline=4.8, drop_tolerance=0.1),
        "slice:injection": gate.GateConfig(baseline=0.9, drop_tolerance=0.05),
    }

    result = gate.check_gate(summary, gates)

    assert result.passed is False
    assert any("overall" in f for f in result.failures)
    assert any("slice:injection" in f for f in result.failures)


def test_check_gate_ignores_metric_not_present_in_summary():
    """Slice không xuất hiện trong lần chạy này (vd chạy subset) -> bỏ qua gate
    đó thay vì báo fail sai (None < threshold sẽ luôn fail nếu không guard)."""
    summary = gate.EvalSummary(n=5, overall=4.9, rule_pass_rate=0.95, by_slice_type={})
    gates = {"slice:injection": gate.GateConfig(baseline=0.9, drop_tolerance=0.1)}

    result = gate.check_gate(summary, gates)
    assert result.passed is True


def test_diff_failed_cases_finds_pass_to_fail_regressions():
    rows = [
        _fake_row("c1", "lookup", 3.0, False),  # đã pass ở baseline, giờ fail -> regression
        _fake_row("c2", "lookup", 5.0, True),  # vẫn pass -> không phải regression
    ]
    baseline_passed = {"c1", "c2"}

    regressions = gate.diff_failed_cases(rows, baseline_passed)
    assert regressions == ["c1"]
