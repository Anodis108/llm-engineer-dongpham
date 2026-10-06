"""Test Buổi 7 — Evaluation (LLM-as-Judge + RAG metrics native).

Mock completion.chat_parsed cho mọi lời gọi LLM — không tốn API, verify logic
wiring (tính overall đúng, faithfulness đếm claim đúng, summarize gộp đúng).
"""

from __future__ import annotations

from app.eval import judge as judge_mod
from app.eval import metrics as metrics_mod
from app.eval import ragas_native as rn

# ── judge.py ─────────────────────────────────────────────────────────────────

def test_judge_answer_computes_overall(monkeypatch):
    fake = judge_mod.JudgeScore(
        accuracy=4, completeness=5, clarity=3, groundedness=4, reasoning="ok"
    )
    monkeypatch.setattr(judge_mod.completion, "chat_parsed", lambda *a, **k: fake)

    score = judge_mod.judge_answer("q", "a")
    assert score.overall == (4 + 5 + 3 + 4) / 4


def test_judge_batch_calls_judge_answer_per_item(monkeypatch):
    calls = []
    fake = judge_mod.JudgeScore(
        accuracy=4, completeness=4, clarity=4, groundedness=4, reasoning="ok"
    )

    def fake_chat_parsed(messages, schema, params=None):
        calls.append(messages)
        return fake

    monkeypatch.setattr(judge_mod.completion, "chat_parsed", fake_chat_parsed)

    items = [{"question": "q1", "answer": "a1"}, {"question": "q2", "answer": "a2"}]
    results = judge_mod.judge_batch(items)
    assert len(results) == 2
    assert len(calls) == 2


# ── ragas_native.py ──────────────────────────────────────────────────────────

def test_faithfulness_all_claims_supported(monkeypatch):
    class FakeClaims:
        claims = ["claim 1", "claim 2"]

    class FakeSupport:
        supported = True

    def fake_chat_parsed(messages, schema, params=None):
        return FakeClaims() if schema is rn._Claims else FakeSupport()

    monkeypatch.setattr(rn.completion, "chat_parsed", fake_chat_parsed)
    assert rn.faithfulness("answer text", ["context"]) == 1.0


def test_faithfulness_partial_support(monkeypatch):
    class FakeClaims:
        claims = ["claim 1", "claim 2"]

    results = iter([True, False])

    class FakeSupport:
        def __init__(self, supported):
            self.supported = supported

    def fake_chat_parsed(messages, schema, params=None):
        if schema is rn._Claims:
            return FakeClaims()
        return FakeSupport(next(results))

    monkeypatch.setattr(rn.completion, "chat_parsed", fake_chat_parsed)
    assert rn.faithfulness("answer", ["context"]) == 0.5


def test_faithfulness_no_claims_defaults_to_faithful(monkeypatch):
    class FakeClaims:
        claims = []

    monkeypatch.setattr(rn.completion, "chat_parsed", lambda *a, **k: FakeClaims())
    assert rn.faithfulness("", ["context"]) == 1.0


def test_answer_relevancy_returns_score(monkeypatch):
    class FakeRelevancy:
        relevant = True
        score = 0.85

    monkeypatch.setattr(rn.completion, "chat_parsed", lambda *a, **k: FakeRelevancy())
    assert rn.answer_relevancy("q", "a") == 0.85


def test_context_precision_empty_contexts_returns_zero():
    assert rn.context_precision("q", [], "ground truth") == 0.0


def test_evaluate_rag_skips_recall_precision_without_ground_truth(monkeypatch):
    class FakeRelevancy:
        relevant = True
        score = 0.9

    class FakeClaims:
        claims = []

    def fake_chat_parsed(messages, schema, params=None):
        if schema is rn._Claims:
            return FakeClaims()
        return FakeRelevancy()

    monkeypatch.setattr(rn.completion, "chat_parsed", fake_chat_parsed)

    result = rn.evaluate_rag("q", "a", ["ctx"], ground_truth=None)
    assert "faithfulness" in result
    assert "answer_relevancy" in result
    assert "context_recall" not in result
    assert "context_precision" not in result


# ── metrics.py ───────────────────────────────────────────────────────────────

def test_evaluate_combines_judge_and_rag(monkeypatch):
    fake_score = judge_mod.JudgeScore(
        accuracy=4, completeness=4, clarity=4, groundedness=4, reasoning="ok"
    )

    class FakeClaims:
        claims = []

    class FakeRelevancy:
        relevant = True
        score = 0.9

    def fake_chat_parsed(messages, schema, params=None):
        if schema is judge_mod.JudgeScore:
            return fake_score
        if schema is rn._Claims:
            return FakeClaims()
        return FakeRelevancy()

    monkeypatch.setattr(judge_mod.completion, "chat_parsed", fake_chat_parsed)
    monkeypatch.setattr(rn.completion, "chat_parsed", fake_chat_parsed)

    result = metrics_mod.evaluate("q", "a", contexts=["ctx"])
    assert result.judge.overall == 4.0
    assert "faithfulness" in result.rag


def test_evaluate_no_contexts_skips_rag(monkeypatch):
    fake = judge_mod.JudgeScore(
        accuracy=4, completeness=4, clarity=4, groundedness=4, reasoning="ok"
    )
    monkeypatch.setattr(judge_mod.completion, "chat_parsed", lambda *a, **k: fake)

    result = metrics_mod.evaluate("q", "a")
    assert result.rag == {}


def test_summarize_averages_correctly():
    from app.eval.judge import JudgeScore

    results = [
        metrics_mod.EvalResult(
            question="q1",
            judge=JudgeScore(accuracy=4, completeness=4, clarity=4, groundedness=4, reasoning="x"),
            rag={"faithfulness": 0.8},
        ),
        metrics_mod.EvalResult(
            question="q2",
            judge=JudgeScore(accuracy=2, completeness=2, clarity=2, groundedness=2, reasoning="y"),
            rag={"faithfulness": 1.0},
        ),
    ]
    summary = metrics_mod.summarize(results)
    assert summary["n_samples"] == 2
    assert summary["judge"]["accuracy"] == 3.0
    assert summary["rag"]["faithfulness"] == 0.9


def test_summarize_empty_list_returns_empty_dict():
    assert metrics_mod.summarize([]) == {}
