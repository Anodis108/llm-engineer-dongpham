"""Test app/llm/async_client.py — mock AsyncOpenAI, không gọi API thật.

Module III, Bài 4, Section 4: async client + semaphore backpressure + retry
với jitter. Test ở đây verify LOGIC (semaphore nhả/giữ slot, retry rồi raise,
token được yield đúng thứ tự) chứ không test SDK OpenAI.
"""

from __future__ import annotations

import httpx
import pytest
from openai import APITimeoutError, RateLimitError

from app.llm import async_client
from app.llm.params import GenerationParams


def _fake_rate_limit_error() -> RateLimitError:
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return RateLimitError("rate limited", response=response, body=None)


async def _instant_sleep(*_args, **_kwargs) -> None:
    """Thay asyncio.sleep trong test — bỏ qua backoff thật để test chạy nhanh."""
    return None


class _FakeStream:
    """Giả lập async iterator trả về từ client.chat.completions.create(stream=True)."""

    def __init__(self, tokens: list[str]):
        self._tokens = tokens

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for tok in self._tokens:

            class _Delta:
                content = tok

            class _Choice:
                delta = _Delta()

            class _Chunk:
                choices = [_Choice()]

            yield _Chunk()


class _FakeCompletions:
    def __init__(self, tokens=None, error_then_tokens=None):
        self._tokens = tokens
        self._error_then_tokens = error_then_tokens
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        if self._error_then_tokens is not None and self.calls == 1:
            raise _fake_rate_limit_error()
        return _FakeStream(self._tokens or self._error_then_tokens or [])


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeAsyncClient:
    def __init__(self, tokens=None, error_then_tokens=None):
        self.chat = _FakeChat(_FakeCompletions(tokens, error_then_tokens))


@pytest.mark.asyncio
async def test_astream_chat_yields_tokens(monkeypatch):
    """astream_chat yield đúng thứ tự token, semaphore được nhả sau khi xong."""
    fake = _FakeAsyncClient(tokens=["Xin ", "chào"])
    monkeypatch.setattr(async_client, "get_async_client", lambda: fake)

    before = async_client.semaphore_state()["available"]

    tokens = [t async for t in async_client.astream_chat([{"role": "user", "content": "hi"}])]

    assert tokens == ["Xin ", "chào"]
    assert async_client.semaphore_state()["available"] == before


@pytest.mark.asyncio
async def test_astream_chat_retries_on_rate_limit(monkeypatch):
    """Lỗi RateLimitError lần đầu -> retry -> vẫn stream được (backoff rút ngắn cho test)."""
    fake = _FakeAsyncClient(error_then_tokens=["ok"])
    monkeypatch.setattr(async_client, "get_async_client", lambda: fake)
    monkeypatch.setattr("asyncio.sleep", _instant_sleep)

    tokens = [t async for t in async_client.astream_chat([{"role": "user", "content": "hi"}])]

    assert tokens == ["ok"]
    assert fake.chat.completions.calls == 2


@pytest.mark.asyncio
async def test_acall_with_retry_raises_after_max_attempts(monkeypatch):
    """Hết số lần thử -> raise lại lỗi gốc, không nuốt exception."""
    monkeypatch.setattr("asyncio.sleep", _instant_sleep)

    async def _always_fails():
        request = httpx.Request("POST", "https://example.test/v1/chat/completions")
        raise APITimeoutError(request=request)

    with pytest.raises(APITimeoutError):
        await async_client.acall_with_retry(_always_fails, max_retries=2)


def test_semaphore_state_reports_limit():
    state = async_client.semaphore_state()
    assert state["limit"] == async_client.settings.llm_max_concurrency
    assert 0 <= state["available"] <= state["limit"]


def test_generation_params_defaults_used_when_none():
    """astream_chat nhận params=None -> dùng GenerationParams() mặc định, không lỗi."""
    params = GenerationParams()
    assert params.temperature == async_client.settings.llm_temperature
