"""Smoke test sau deploy — Module III, Bài 6, Section 4 (Deployment Strategies).

Vì sao KHÔNG dùng "container đã start" làm tiêu chí deploy thành công:
    Một bản deploy hỏng rất nhiều kiểu mà `/health` vẫn trả 200 — model name sai
    (chỉ lộ khi gọi thật), hết quota API key, Qdrant rỗng nên retrieval trả 0
    chunk, prompt registry trả sai version. Container "healthy" chỉ nghĩa là
    process còn sống.

Nên smoke test phải đi hết một đường NGƯỜI DÙNG THẬT: health → ingest dữ liệu →
hỏi 1 câu đã biết đáp án → kiểm tra câu trả lời có chứa từ khoá bắt buộc.

Đây là bước chặn cuối trong cd.yml: smoke fail → rollback về bản last-known-good
ngay, trước khi người dùng kịp thấy.

Chạy:
    python deploy/smoke.py --url http://localhost:8000
    python deploy/smoke.py --url https://llm.example.com --skip-ingest
"""

from __future__ import annotations

import argparse
import sys
import time

import httpx

# Câu hỏi + từ khoá lấy từ golden set (data/eval/legal_qa/v1.yaml, case
# legal_qa_001) — câu trả lời ĐÚNG phải chứa đủ các từ này. Dùng lại case có
# sẵn thay vì bịa câu mới để smoke test và eval đo cùng một thứ.
KNOWN_QUESTION = "Nghị định này quy định về chế độ gì?"
MUST_INCLUDE = ["tiền lương", "thù lao", "tiền thưởng"]


class SmokeFailure(Exception):
    """Một bước smoke test thất bại — cd.yml bắt lỗi này để rollback."""


def _fail(step: str, detail: str) -> None:
    # Format GitHub Actions annotation: hiện thẳng trên UI, không phải lục log.
    print(f"::error::smoke/{step}: {detail}")
    raise SmokeFailure(f"{step}: {detail}")


def check_health(client: httpx.Client) -> None:
    t0 = time.perf_counter()
    r = client.get("/health")
    ms = (time.perf_counter() - t0) * 1000

    if r.status_code != 200:
        _fail("health", f"HTTP {r.status_code} (mong đợi 200)")
    print(f"  ✓ /health            200  ({ms:.0f}ms)")


def check_ingest(client: httpx.Client) -> None:
    """Nạp dữ liệu vào Qdrant CỦA CHÍNH process server.

    QDRANT_URL=:memory: tạo 1 Qdrant in-process riêng cho mỗi process — chạy
    scripts.ingest ở process khác sẽ nạp vào một store khác, server vẫn rỗng.
    Nên phải gọi qua /admin/ingest để nạp đúng vào store server đang dùng.
    """
    r = client.post("/admin/ingest", timeout=300.0)
    if r.status_code != 200:
        _fail("ingest", f"HTTP {r.status_code}: {r.text[:200]}")

    body = r.json()
    total = body.get("total_in_collection", 0)
    if total <= 0:
        _fail("ingest", f"collection rỗng sau ingest (total={total})")
    print(f"  ✓ /admin/ingest      {total} chunk trong collection")


def check_answer(client: httpx.Client, question: str, must_include: list[str]) -> None:
    t0 = time.perf_counter()
    r = client.post("/chat", json={"question": question}, timeout=120.0)
    ms = (time.perf_counter() - t0) * 1000

    if r.status_code != 200:
        _fail("answer", f"HTTP {r.status_code}: {r.text[:200]}")

    answer = (r.json().get("answer") or "").strip()
    if not answer:
        _fail("answer", "câu trả lời rỗng")

    missing = [kw for kw in must_include if kw.lower() not in answer.lower()]
    if missing:
        _fail(
            "answer",
            f"câu trả lời thiếu {missing} — RAG có thể không lấy được tài liệu. "
            f"Nhận được: {answer[:200]!r}",
        )
    print(f"  ✓ /chat              đủ từ khoá {must_include}  ({ms:.0f}ms)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke test sau deploy")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--timeout", type=float, default=15.0, help="timeout cho /health")
    ap.add_argument("--skip-ingest", action="store_true", help="bỏ bước nạp dữ liệu")
    ap.add_argument("--skip-rag", action="store_true", help="chỉ kiểm tra /health")
    ap.add_argument("--question", default=KNOWN_QUESTION)
    ap.add_argument(
        "--must-include", default=",".join(MUST_INCLUDE),
        help="từ khoá bắt buộc, ngăn cách bằng dấu phẩy",
    )
    args = ap.parse_args()

    print(f"Smoke test: {args.url}")
    must_include = [k.strip() for k in args.must_include.split(",") if k.strip()]

    try:
        with httpx.Client(base_url=args.url, timeout=args.timeout) as client:
            check_health(client)
            if args.skip_rag:
                print("\nSMOKE PASS (/health, bỏ qua RAG).")
                return
            if not args.skip_ingest:
                check_ingest(client)
            check_answer(client, args.question, must_include)
    except SmokeFailure:
        print("\nSMOKE FAIL — không promote bản này.")
        sys.exit(1)
    except httpx.HTTPError as exc:
        print(f"::error::smoke/connect: không gọi được {args.url}: {exc}")
        print("\nSMOKE FAIL — không promote bản này.")
        sys.exit(1)

    print("\nSMOKE PASS.")


if __name__ == "__main__":
    main()
