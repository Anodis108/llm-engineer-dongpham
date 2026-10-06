"""Async LLM client — Module III, Bài 4, Section 4 (FastAPI Patterns cho LLM).

Vì sao cần file này khi đã có client.py (sync)?
  Bài học Section 4 cảnh báo rõ: "đừng gọi SDK sync trong endpoint async (block
  event loop)". `client.py` trả `OpenAI` (sync) và `resilience.retry_with_backoff`
  dùng `time.sleep()` — cả hai đều CHẶN event loop nếu gọi từ `async def`.
  File này là bản async tương ứng: `AsyncOpenAI` + `asyncio.sleep()`.

3 pattern bài học yêu cầu, gom ở đây:
  1. Async client       — AsyncOpenAI, không block event loop.
  2. Backpressure       — Semaphore giới hạn số request LLM đồng thời; request
     thứ N+1 xếp hàng thay vì làm sập upstream/vượt rate limit.
  3. Retry có jitter    — exponential backoff + jitter cho 429/timeout, dùng
     `asyncio.sleep` (KHÔNG phải time.sleep).

Tái dùng key rotation của Bài 1 (`RotatingKeyPool` qua `client._get_pool()`) —
không viết lại logic chọn key.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator

from openai import APITimeoutError, AsyncOpenAI, RateLimitError

from app.config import settings
from app.llm.backends import get_backend
from app.llm.client import _get_pool
from app.llm.params import GenerationParams

# Section 4: giới hạn số lời gọi LLM ĐỒNG THỜI. Request thứ (N+1) sẽ `await`
# ở đây cho tới khi có slot — đây chính là "backpressure": thay vì đẩy hết tải
# lên provider (gây 429 hàng loạt), ta xếp hàng có kiểm soát ở tầng app.
# Đặt ở module level -> dùng CHUNG cho mọi request trong 1 process uvicorn worker.
_LLM_SEMAPHORE = asyncio.Semaphore(settings.llm_max_concurrency)


def get_async_client() -> AsyncOpenAI:
    """AsyncOpenAI trỏ tới backend đang cấu hình — song song với client.get_client().

    `max_retries=0`: TẮT retry tự động của SDK để tự quản backoff + jitter
    (bài học Section 4: SDK retry không có jitter, dễ gây thundering herd khi
    nhiều client cùng retry một lúc).
    """
    backend = get_backend(settings.llm_backend)
    base_url = settings.llm_base_url or backend.default_base_url or None
    api_key = _get_pool().get_key() if backend.requires_real_key else backend.dummy_key

    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=settings.llm_timeout_seconds,
        max_retries=0,
    )


async def acall_with_retry(fn, *, max_retries: int | None = None):
    """Bản ASYNC của resilience.retry_with_backoff — `asyncio.sleep` thay `time.sleep`.

    Dùng `await acall_with_retry(lambda: client.chat.completions.create(...))`.
    Bắt cả RateLimitError (429) lẫn APITimeoutError, đúng bảng "Xử lý rate limit
    từ provider" trong bài học.
    """
    attempts = max_retries if max_retries is not None else settings.llm_max_retries

    for attempt in range(attempts):
        try:
            return await fn()
        except (RateLimitError, APITimeoutError):
            if attempt == attempts - 1:
                raise
            # exp backoff + jitter, cap ở 16s (bài học: min(2**attempt, 16) + jitter)
            backoff = min(2**attempt, 16) + random.uniform(0, 1)
            await asyncio.sleep(backoff)

    raise RuntimeError("acall_with_retry: hết số lần thử")


async def astream_chat(
    messages: list[dict], params: GenerationParams | None = None
) -> AsyncIterator[str]:
    """Stream token bất đồng bộ, có backpressure qua semaphore.

    Semaphore giữ slot TRONG SUỐT quá trình stream (không chỉ lúc mở kết nối) —
    vì 1 stream đang chạy vẫn chiếm tài nguyên phía provider. Nhả slot khi
    generator kết thúc hoặc bị đóng (caller `break` do client disconnect).
    """
    params = params or GenerationParams()

    async with _LLM_SEMAPHORE:
        client = get_async_client()

        async def _open():
            return await client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                stream=True,
                **params.to_openai_kwargs(),
            )

        stream = await acall_with_retry(_open)

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


def semaphore_state() -> dict[str, int]:
    """Số slot còn trống / tổng — để `/health` và benchmark quan sát backpressure."""
    return {
        "limit": settings.llm_max_concurrency,
        "available": _LLM_SEMAPHORE._value,  # noqa: SLF001 — chỉ để quan sát, không điều khiển
    }
