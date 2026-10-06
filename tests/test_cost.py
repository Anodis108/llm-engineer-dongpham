"""Test Module III, Bài 3 — Cost Optimization & Caching.

Không gọi API thật. Kiểm tra 4 mảng:
  - tracker.py: cost_of, CostTracker (by_feature/by_user/percentile), breakdown.
  - cache_exact.py: cache key chứa prompt_version, hit/miss, peek/store_result.
  - optimization/caching.py: SemanticCache nâng cấp (TTL, volatile, stats).
  - cascade.py: escalate khi không tự tin, không escalate khi tự tin.
  - budget.py: check_budget, guard_user_budget.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.cost import budget, cascade
from app.cost.cache_exact import ExactCache, InMemoryStore, cache_key
from app.cost.tracker import CostTracker, breakdown_pct, breakdown_tokens, cost_of
from app.optimization.caching import SemanticCache, is_volatile

# ── tracker.py ─────────────────────────────────────────────────────────────

def test_cost_of_computes_from_pricing_table():
    cost = cost_of("gpt-4o-mini", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert cost == pytest.approx(0.15 + 0.60)


def test_cost_of_unknown_model_returns_zero_not_raise():
    assert cost_of("unknown-model-xyz", 1000, 1000) == 0.0


def test_cost_tracker_by_feature_and_by_user():
    t = CostTracker()
    t.record(feature="rag", user_id="u1", model="gpt-4o-mini", prompt_tokens=1000, completion_tokens=100)
    t.record(feature="rag", user_id="u2", model="gpt-4o", prompt_tokens=1000, completion_tokens=100)
    t.record(feature="summarize", user_id="u1", model="gpt-4o-mini", prompt_tokens=500, completion_tokens=50)

    by_feature = t.by_feature()
    by_user = t.by_user()

    assert set(by_feature.keys()) == {"rag", "summarize"}
    assert by_feature["rag"] > by_feature["summarize"]
    assert set(by_user.keys()) == {"u1", "u2"}


def test_cost_tracker_cache_hit_costs_zero():
    t = CostTracker()
    rec = t.record(feature="rag", user_id="u1", model="gpt-4o", prompt_tokens=5000, completion_tokens=500, cache_hit=True)
    assert rec.cost_usd == 0.0


def test_cost_tracker_cache_hit_rate():
    t = CostTracker()
    t.record(feature="rag", user_id="u1", model="gpt-4o-mini", prompt_tokens=100, completion_tokens=10, cache_hit=True)
    t.record(feature="rag", user_id="u1", model="gpt-4o-mini", prompt_tokens=100, completion_tokens=10, cache_hit=False)
    assert t.cache_hit_rate() == 0.5


def test_cost_tracker_percentile_p99_higher_than_p50_with_outlier():
    t = CostTracker()
    for _ in range(9):
        t.record(feature="rag", user_id="u1", model="gpt-4o-mini", prompt_tokens=100, completion_tokens=10)
    t.record(feature="rag", user_id="u1", model="gpt-4o", prompt_tokens=100_000, completion_tokens=10_000)  # outlier

    assert t.percentile(0.99) > t.percentile(0.5)


def test_breakdown_tokens_shows_context_dominant():
    """Section 1: context bloat chiếm phần lớn input — đúng ví dụ bài học."""
    tokens = breakdown_tokens(system="x" * 1600, context="y" * 12000, history="z" * 800, question="q" * 400)
    pct = breakdown_pct(tokens)
    assert pct["context"] > 50  # context chiếm hơn nửa, giống ví dụ ~70% bài học


def test_breakdown_pct_handles_all_empty():
    assert breakdown_pct(breakdown_tokens()) == {"system": 0.0, "context": 0.0, "history": 0.0, "question": 0.0}


# ── cache_exact.py ─────────────────────────────────────────────────────────

def test_cache_key_changes_when_prompt_version_bumps():
    """Section 3, warning chính của bài học: version PHẢI nằm trong key."""
    key_v1 = cache_key("rag", 1, "gpt-4o-mini", "câu hỏi", {})
    key_v2 = cache_key("rag", 2, "gpt-4o-mini", "câu hỏi", {})
    assert key_v1 != key_v2


def test_cache_key_deterministic_for_same_input():
    k1 = cache_key("rag", 1, "gpt-4o-mini", "câu hỏi", {"temperature": 0.2})
    k2 = cache_key("rag", 1, "gpt-4o-mini", "câu hỏi", {"temperature": 0.2})
    assert k1 == k2


def test_exact_cache_hit_on_second_identical_call():
    cache = ExactCache()
    call_count = {"n": 0}

    def call_fn():
        call_count["n"] += 1
        return f"answer-{call_count['n']}"

    a1, hit1 = cache.get_or_call(prompt_name="rag", prompt_version=1, model="gpt-4o-mini", rendered_prompt="q", params={}, call_fn=call_fn)
    a2, hit2 = cache.get_or_call(prompt_name="rag", prompt_version=1, model="gpt-4o-mini", rendered_prompt="q", params={}, call_fn=call_fn)

    assert hit1 is False
    assert hit2 is True
    assert a1 == a2 == "answer-1"
    assert call_count["n"] == 1


def test_exact_cache_miss_after_version_bump():
    """Bug đã bắt được khi verify thật: bump version phải MISS, không trả answer cũ."""
    cache = ExactCache()
    call_count = {"n": 0}

    def call_fn():
        call_count["n"] += 1
        return f"answer-{call_count['n']}"

    cache.get_or_call(prompt_name="rag", prompt_version=1, model="gpt-4o-mini", rendered_prompt="q", params={}, call_fn=call_fn)
    a2, hit2 = cache.get_or_call(prompt_name="rag", prompt_version=2, model="gpt-4o-mini", rendered_prompt="q", params={}, call_fn=call_fn)

    assert hit2 is False
    assert a2 == "answer-2"
    assert call_count["n"] == 2


def test_exact_cache_peek_and_store_result_pair():
    """peek()/store_result() — dùng khi cần chèn tầng khác giữa check và store
    (xem scripts/cost_replay_demo.py: exact -> semantic -> LLM thật)."""
    cache = ExactCache()

    assert cache.peek(prompt_name="rag", prompt_version=1, model="gpt-4o-mini", rendered_prompt="q", params={}) is None

    cache.store_result(prompt_name="rag", prompt_version=1, model="gpt-4o-mini", rendered_prompt="q", params={}, answer="cached-answer")

    assert cache.peek(prompt_name="rag", prompt_version=1, model="gpt-4o-mini", rendered_prompt="q", params={}) == "cached-answer"


def test_exact_cache_stats_hit_rate():
    cache = ExactCache()
    cache.get_or_call(prompt_name="rag", prompt_version=1, model="m", rendered_prompt="a", params={}, call_fn=lambda: "x")
    cache.get_or_call(prompt_name="rag", prompt_version=1, model="m", rendered_prompt="a", params={}, call_fn=lambda: "x")

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5


def test_in_memory_store_ttl_expires():
    store = InMemoryStore()
    store.setex("k", ttl_seconds=-1, value="v")  # đã hết hạn ngay
    assert store.get("k") is None


# ── optimization/caching.py: SemanticCache nâng cấp ──────────────────────────

def _fake_embedder(text: str) -> np.ndarray:
    """Embedder toy: câu chứa 'A' -> vector [1,0], chứa 'B' -> [0,1]."""
    v = np.array([1.0, 0.0]) if "A" in text else np.array([0.0, 1.0])
    return v / np.linalg.norm(v)


def test_semantic_cache_hit_for_similar_question():
    cache = SemanticCache(embedder=_fake_embedder, threshold=0.9)
    cache.set("Câu hỏi A", "Answer A")
    assert cache.get("Câu hỏi A") == "Answer A"


def test_semantic_cache_miss_for_dissimilar_question():
    cache = SemanticCache(embedder=_fake_embedder, threshold=0.9)
    cache.set("Câu hỏi A", "Answer A")
    assert cache.get("Câu hỏi B") is None


def test_is_volatile_detects_year():
    assert is_volatile("Mức lương tối thiểu vùng năm 2024 là bao nhiêu?") is True


def test_is_volatile_detects_percentage():
    assert is_volatile("Nếu lợi nhuận vượt kế hoạch 15% thì sao?") is True


def test_is_volatile_false_for_plain_question():
    assert is_volatile("Ai chi trả tiền lương cho người đại diện?") is False


def test_semantic_cache_skips_volatile_questions_on_set():
    """Section 4, false hit table: câu có tham số ngày/số KHÔNG được lưu vào cache."""
    cache = SemanticCache(embedder=_fake_embedder, threshold=0.9, skip_volatile=True)
    cache.set("Câu hỏi A năm 2024", "Answer 2024")
    assert cache.size() == 0


def test_semantic_cache_skips_volatile_questions_on_get():
    cache = SemanticCache(embedder=_fake_embedder, threshold=0.9, skip_volatile=True)
    cache.skip_volatile = False  # tạm tắt để set thành công cho test setup
    cache.set("Câu hỏi A năm 2024", "Answer 2024")
    cache.skip_volatile = True

    assert cache.get("Câu hỏi A năm 2020") is None  # volatile query luôn miss dù có data gần giống


def test_semantic_cache_ttl_evicts_old_entries():
    cache = SemanticCache(embedder=_fake_embedder, threshold=0.9, ttl_seconds=-1)  # hết hạn ngay
    cache.set("Câu hỏi A", "Answer A")
    assert cache.get("Câu hỏi A") is None  # evicted trước khi tìm


def test_semantic_cache_stats_tracks_hits_and_misses():
    cache = SemanticCache(embedder=_fake_embedder, threshold=0.9)
    cache.set("Câu hỏi A", "Answer A")
    cache.get("Câu hỏi A")  # hit
    cache.get("Câu hỏi B")  # miss

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5


def test_semantic_cache_backward_compatible_without_new_params():
    """Đảm bảo app/api/routes_chat.py (caller cũ) không cần đổi gì khi nâng cấp."""
    cache = SemanticCache(embedder=_fake_embedder, threshold=0.8)
    cache.set("Câu hỏi A", "Answer A")
    assert cache.get("Câu hỏi A") == "Answer A"
    assert cache.size() == 1


# ── cascade.py ────────────────────────────────────────────────────────────

def _fake_usage(prompt_tokens=100, completion_tokens=50):
    return SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


def test_cascade_does_not_escalate_when_confident(monkeypatch):
    from app.cost.tracker import tracker

    tracker.clear()
    call_log = []

    def fake_chat_with_usage(messages, params, model=None):
        call_log.append(model)
        return "Câu trả lời rõ ràng và đầy đủ.", _fake_usage()

    monkeypatch.setattr("app.llm.completion.chat_with_usage", fake_chat_with_usage)

    result = cascade.answer_cascade("Câu hỏi dễ", build_messages_fn=lambda q: [{"role": "user", "content": q}])

    assert result.escalated is False
    assert result.model_used == cascade.CHEAP_MODEL
    assert call_log == [cascade.CHEAP_MODEL]


def test_cascade_escalates_when_not_confident(monkeypatch):
    from app.cost.tracker import tracker

    tracker.clear()
    call_log = []

    def fake_chat_with_usage(messages, params, model=None):
        call_log.append(model)
        if model == cascade.CHEAP_MODEL:
            return "Tôi không tìm thấy thông tin này.", _fake_usage()
        return "Câu trả lời chi tiết từ model mạnh.", _fake_usage()

    monkeypatch.setattr("app.llm.completion.chat_with_usage", fake_chat_with_usage)

    result = cascade.answer_cascade("Câu hỏi khó", build_messages_fn=lambda q: [{"role": "user", "content": q}])

    assert result.escalated is True
    assert result.model_used == cascade.STRONG_MODEL
    assert call_log == [cascade.CHEAP_MODEL, cascade.STRONG_MODEL]


def test_escalate_rate_computed_correctly():
    results = [
        cascade.CascadeResult(answer="a", model_used="gpt-4o-mini", escalated=False, cost_usd=0.0),
        cascade.CascadeResult(answer="b", model_used="gpt-4o", escalated=True, cost_usd=0.0),
        cascade.CascadeResult(answer="c", model_used="gpt-4o-mini", escalated=False, cost_usd=0.0),
        cascade.CascadeResult(answer="d", model_used="gpt-4o", escalated=True, cost_usd=0.0),
    ]
    assert cascade.escalate_rate(results) == 0.5


def test_escalate_rate_empty_list_is_zero():
    assert cascade.escalate_rate([]) == 0.0


# ── budget.py ─────────────────────────────────────────────────────────────

def test_check_budget_projects_based_on_days_elapsed():
    t = CostTracker()
    t.record(feature="x", user_id="u1", model="gpt-4o", prompt_tokens=1_000_000, completion_tokens=100_000)
    # spent ~= 2.5 + 1.0 = 3.5 tại ngày 10/30 -> projected = spent/10*30 = 10.5
    status = budget.check_budget(days_elapsed=10, days_in_month=30, tracker_=t)
    assert status.projected == pytest.approx(status.spent / 10 * 30)


def test_check_budget_fires_alert_when_threshold_crossed():
    t = CostTracker()
    # spend đủ lớn để vượt 50% của MONTHLY_BUDGET_USD (800 -> ngưỡng 400)
    t.record(feature="x", user_id="u1", model="gpt-4o", prompt_tokens=200_000_000, completion_tokens=0)
    status = budget.check_budget(days_elapsed=1, days_in_month=30, tracker_=t)
    assert 0.5 in status.alerts_fired


def test_guard_user_budget_raises_when_exceeded():
    t = CostTracker()
    t.record(feature="x", user_id="u1", model="gpt-4o", prompt_tokens=1_000_000, completion_tokens=0)

    with pytest.raises(budget.BudgetExceeded) as exc_info:
        budget.guard_user_budget("u1", daily_cap_usd=0.01, tracker_=t)

    assert exc_info.value.user_id == "u1"


def test_guard_user_budget_passes_when_under_cap():
    t = CostTracker()
    t.record(feature="x", user_id="u1", model="gpt-4o-mini", prompt_tokens=100, completion_tokens=10)

    budget.guard_user_budget("u1", daily_cap_usd=100.0, tracker_=t)  # không raise


def test_guard_user_budget_only_counts_matching_user():
    t = CostTracker()
    t.record(feature="x", user_id="u1", model="gpt-4o", prompt_tokens=10_000_000, completion_tokens=0)  # tốn tiền
    t.record(feature="x", user_id="u2", model="gpt-4o-mini", prompt_tokens=10, completion_tokens=1)

    budget.guard_user_budget("u2", daily_cap_usd=0.01, tracker_=t)  # u2 chi rất ít -> không raise
