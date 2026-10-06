"""Benchmark backpressure — Module III, Bài 4, Section 4 (Hands-on, Bước 3).

Bắn N request đồng thời vào `/chat/stream-sse`, đo p50/p95 latency + throughput,
để THẤY được hiệu ứng của `LLM_MAX_CONCURRENCY` (semaphore) thay vì chỉ đọc lý
thuyết. So sánh 2 lần chạy với giá trị semaphore khác nhau:

    # Terminal 1 — chạy server với semaphore NHỎ
    LLM_MAX_CONCURRENCY=5 uvicorn app.main:app --port 8000

    # Terminal 2 — bắn 50 request đồng thời
    python -m scripts.concurrency_benchmark --requests 50 --concurrency 50

    # Đổi lại LLM_MAX_CONCURRENCY=50, chạy lại, so sánh p95/throughput.

Kỳ vọng bài học: semaphore nhỏ -> p95 cao hơn (request xếp hàng) nhưng KHÔNG có
429 hàng loạt từ provider; semaphore lớn hơn giới hạn thật của provider -> ăn
429, retry, p95 còn TỆ hơn. Không có "số đúng tuyệt đối" — phải đo mới biết.

Dùng endpoint /chat/stream-sse thật (không mock) — cần OPENAI_API_KEYS hợp lệ
hoặc backend local (ollama/vllm) đang chạy, vì mục đích là đo latency THẬT.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx

DEFAULT_URL = "http://localhost:8000/chat/stream-sse"
DEFAULT_QUESTION = "Mức lương tối thiểu vùng I hiện nay là bao nhiêu?"


async def _one_request(client: httpx.AsyncClient, url: str, question: str) -> tuple[float, bool]:
    """Gửi 1 request, đọc hết stream. Trả (latency_seconds, ok)."""
    start = time.perf_counter()
    try:
        async with client.stream("POST", url, json={"question": question}, timeout=60.0) as resp:
            async for _ in resp.aiter_bytes():
                pass
            ok = resp.status_code == 200
    except httpx.HTTPError:
        ok = False
    return time.perf_counter() - start, ok


async def run_benchmark(url: str, question: str, total_requests: int, concurrency: int) -> dict:
    """Bắn `total_requests` request, tối đa `concurrency` cái cùng lúc (client-side)."""
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    failures = 0

    async with httpx.AsyncClient() as client:

        async def _bounded():
            nonlocal failures
            async with sem:
                latency, ok = await _one_request(client, url, question)
                latencies.append(latency)
                if not ok:
                    failures += 1

        wall_start = time.perf_counter()
        await asyncio.gather(*[_bounded() for _ in range(total_requests)])
        wall_time = time.perf_counter() - wall_start

    latencies.sort()
    return {
        "total_requests": total_requests,
        "failures": failures,
        "wall_time_s": round(wall_time, 2),
        "throughput_rps": round(total_requests / wall_time, 2) if wall_time > 0 else 0,
        "p50_s": round(statistics.median(latencies), 3) if latencies else None,
        "p95_s": round(latencies[int(len(latencies) * 0.95) - 1], 3) if latencies else None,
        "max_s": round(max(latencies), 3) if latencies else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--requests", type=int, default=50, help="Tổng số request bắn")
    parser.add_argument(
        "--concurrency", type=int, default=50, help="Số request đồng thời phía CLIENT"
    )
    args = parser.parse_args()

    print(f"Bắn {args.requests} request (client concurrency={args.concurrency}) tới {args.url}")
    result = asyncio.run(
        run_benchmark(args.url, args.question, args.requests, args.concurrency)
    )

    print()
    for key, value in result.items():
        print(f"  {key:16s}: {value}")


if __name__ == "__main__":
    main()
