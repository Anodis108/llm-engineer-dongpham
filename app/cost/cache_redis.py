"""RedisStore — Module III, Bài 4, Section 1 (Containerisation).

`ExactCache` (Bài 3) tách store qua `CacheStore` protocol đúng để không phải
sửa gì ở logic cache khi đổi backend — implement 2 method (`get`/`setex`) là đủ.
`InMemoryStore` không chia sẻ giữa nhiều process/instance; RedisStore giải
quyết đúng vấn đề đó cho khi app chạy nhiều replica (nhiều container cùng
đọc/ghi chung 1 cache).

Import `redis` bị hoãn vào trong `__init__` (không phải top-level) để phần còn
lại của test suite/app không bắt buộc cài `redis` package khi không dùng.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from redis import Redis


class RedisStore:
    """CacheStore implementation dùng Redis thật — xem docker-compose.yml (service `redis`)."""

    def __init__(self, url: str = "redis://localhost:6379", client: Redis | None = None):
        if client is not None:
            self._client = client
        else:
            import redis

            self._client = redis.from_url(url, decode_responses=True)

    def get(self, key: str) -> str | None:
        return self._client.get(key)

    def setex(self, key: str, ttl_seconds: int, value: str) -> None:
        self._client.setex(key, ttl_seconds, value)
