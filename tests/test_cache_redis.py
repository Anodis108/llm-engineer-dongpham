"""Test RedisStore — mock redis client, không cần Redis server thật.

Verify RedisStore implement ĐÚNG CacheStore protocol (get/setex) và
ExactCache hoạt động y hệt với InMemoryStore khi đổi sang RedisStore.
"""

from __future__ import annotations

from app.cost.cache_exact import ExactCache
from app.cost.cache_redis import RedisStore


class _FakeRedisClient:
    """Giả lập redis.Redis đủ cho get/setex."""

    def __init__(self):
        self._data: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def setex(self, key: str, ttl_seconds: int, value: str) -> None:
        self._data[key] = value


def test_redis_store_get_setex_roundtrip():
    store = RedisStore(client=_FakeRedisClient())
    assert store.get("missing") is None

    store.setex("k1", 60, "value1")
    assert store.get("k1") == "value1"


def test_exact_cache_works_with_redis_store():
    """ExactCache không cần biết store là InMemory hay Redis — cùng protocol."""
    store = RedisStore(client=_FakeRedisClient())
    cache = ExactCache(store=store)

    calls = {"n": 0}

    def call_fn():
        calls["n"] += 1
        return f"answer-{calls['n']}"

    a1, hit1 = cache.get_or_call(
        prompt_name="rag", prompt_version=1, model="gpt-4o-mini",
        rendered_prompt="câu hỏi", params={}, call_fn=call_fn,
    )
    a2, hit2 = cache.get_or_call(
        prompt_name="rag", prompt_version=1, model="gpt-4o-mini",
        rendered_prompt="câu hỏi", params={}, call_fn=call_fn,
    )

    assert hit1 is False
    assert hit2 is True
    assert a1 == a2 == "answer-1"
    assert calls["n"] == 1
