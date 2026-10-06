"""Demo Module III, Bài 3 — Cache 2 tầng cho Vietnamese chatbot (Section 8, Hands-on).

Chạy tập replay (data/cost/replay_questions.yaml, 46 câu ~43% trùng/gần giống)
qua RAG pipeline (app.pipeline) dưới 4 cấu hình, đo cost + hit rate mỗi tầng —
đúng 5 bước bài học: baseline -> +tầng 1 (exact) -> +tầng 2 (semantic) -> báo cáo
-> (mở rộng) cascade.

Chạy:
    python -m scripts.cost_replay_demo --config no-cache
    python -m scripts.cost_replay_demo --config exact
    python -m scripts.cost_replay_demo --config exact+semantic
    python -m scripts.cost_replay_demo --config cascade
    python -m scripts.cost_replay_demo --compare      # chạy cả 4, in bảng so sánh

Cần OPENAI_API_KEYS + Qdrant đã ingest (script tự ingest nếu chưa có, giống
scripts/eval_pipeline_demo.py — dùng QDRANT_URL=:memory: nên phải ingest lại
mỗi lần chạy process mới).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import yaml

from app.cost.cache_exact import ExactCache
from app.cost.cascade import answer_cascade, escalate_rate
from app.cost.tracker import CostTracker
from app.optimization.caching import SemanticCache

REPLAY_PATH = "data/cost/replay_questions.yaml"
PROMPT_NAME = "rag_answer"
PROMPT_VERSION = 1  # bump số này để demo Section 3: cache tự "reset" theo version


def load_replay(path: str = REPLAY_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["questions"]


def _get_embedder():
    """Tái dùng đúng embedder app/api/routes_chat.py dùng cho semantic cache —
    SentenceTransformer nếu có, fallback hash-based nếu không (offline demo)."""
    from app.api.routes_chat import _sentence_embedder

    return _sentence_embedder


@dataclass(slots=True)
class ReplayResult:
    config: str
    n_questions: int
    total_cost: float
    exact_hit_rate: float
    semantic_hit_rate: float
    escalate_rate: float
    elapsed_s: float


def run_config(config: str, questions: list[dict]) -> ReplayResult:
    """1 lần chạy toàn bộ replay dưới 1 cấu hình cache. Mỗi lần chạy dùng
    CostTracker RIÊNG (không dùng chung tracker.py:tracker toàn cục) để 4 lần
    chạy không cộng dồn chi phí lẫn nhau khi --compare."""
    from app.guardrails.checks import check_input, check_output
    from app.llm import completion
    from app.llm.params import GenerationParams
    from app.prompts.templates import build_messages
    from app.retrieval.retriever import retrieve

    local_tracker = CostTracker()
    exact_cache = ExactCache() if config in ("exact", "exact+semantic") else None
    semantic_cache = (
        SemanticCache(embedder=_get_embedder(), threshold=0.92)
        if config == "exact+semantic"
        else None
    )
    cascade_results = []

    start = time.perf_counter()
    for item in questions:
        question = item["question"]

        if config == "cascade":
            result = answer_cascade(
                question,
                build_messages_fn=lambda q: build_messages(q, retrieve(q)),
                feature="cost_replay",
                user_id="replay",
            )
            cascade_results.append(result)
            continue

        # Kiến trúc tóm tắt bài học (Section 4 mermaid, "T1 -> T2 -> Routing"):
        # TẦNG 1 (exact) LUÔN CHECK TRƯỚC tầng 2 — rẻ hơn (hash lookup, không
        # cần embed) và chính xác 100% (không có false hit như semantic). Nếu
        # đảo thứ tự, semantic cache "nuốt" luôn cả case trùng y hệt (repeat_*)
        # vì similarity=1.0 với chính câu đã lưu, khiến exact_hit_rate luôn về 0
        # — chính là lỗi ban đầu đã bắt được khi verify script này.
        if exact_cache is not None:
            exact_hit_answer = exact_cache.peek(
                prompt_name=PROMPT_NAME, prompt_version=PROMPT_VERSION, model="gpt-4o-mini",
                rendered_prompt=question, params={},
            )
            if exact_hit_answer is not None:
                local_tracker.record(
                    feature="cost_replay", user_id="replay", model="gpt-4o-mini",
                    prompt_tokens=0, completion_tokens=0, cache_hit=True,
                )
                continue

        if semantic_cache is not None:
            cached = semantic_cache.get(question)
            if cached is not None:
                local_tracker.record(
                    feature="cost_replay", user_id="replay", model="gpt-4o-mini",
                    prompt_tokens=0, completion_tokens=0, cache_hit=True,
                )
                continue

        check_input(question)
        chunks = retrieve(question)
        messages = build_messages(question, chunks)
        text, usage = completion.chat_with_usage(messages, GenerationParams())
        local_tracker.record(
            feature="cost_replay", user_id="replay", model="gpt-4o-mini",
            prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
        )
        checked = check_output(text, [c.text for c in chunks])
        answer_text = checked.answer

        if exact_cache is not None:
            exact_cache.store_result(
                prompt_name=PROMPT_NAME, prompt_version=PROMPT_VERSION, model="gpt-4o-mini",
                rendered_prompt=question, params={}, answer=answer_text,
            )
        if semantic_cache is not None:
            semantic_cache.set(question, answer_text)

    elapsed = time.perf_counter() - start

    return ReplayResult(
        config=config,
        n_questions=len(questions),
        total_cost=local_tracker.total_cost() + sum(r.cost_usd for r in cascade_results),
        exact_hit_rate=exact_cache.stats()["hit_rate"] if exact_cache else 0.0,
        semantic_hit_rate=semantic_cache.stats()["hit_rate"] if semantic_cache else 0.0,
        escalate_rate=escalate_rate(cascade_results) if cascade_results else 0.0,
        elapsed_s=elapsed,
    )


def print_result(result: ReplayResult) -> None:
    print(f"\n─── Config: {result.config} ───")
    print(f"n_questions        = {result.n_questions}")
    print(f"total_cost         = ${result.total_cost:.4f}")
    print(f"cost_per_request   = ${result.total_cost / result.n_questions:.5f}")
    print(f"exact_hit_rate     = {result.exact_hit_rate:.1%}")
    print(f"semantic_hit_rate  = {result.semantic_hit_rate:.1%}")
    if result.config == "cascade":
        print(f"escalate_rate      = {result.escalate_rate:.1%}")
    print(f"elapsed_s          = {result.elapsed_s:.1f}s")


def print_comparison(results: list[ReplayResult]) -> None:
    print("\n" + "═" * 70)
    print("BẢNG SO SÁNH (cùng tập replay)")
    print("═" * 70)
    header = f"{'Config':<20}{'Total Cost':>14}{'Cost/req':>12}{'Exact Hit':>12}{'Semantic Hit':>14}"
    print(header)
    print("-" * len(header))
    baseline_cost = results[0].total_cost if results else 0.0
    for r in results:
        note = ""
        if baseline_cost > 0 and r is not results[0]:
            pct_change = (r.total_cost / baseline_cost - 1) * 100  # âm = tiết kiệm, dương = tốn hơn
            note = f"(-{abs(pct_change):.0f}%)" if pct_change < 0 else f"(+{pct_change:.0f}%)"
        print(
            f"{r.config:<20}${r.total_cost:<13.4f}${r.total_cost / r.n_questions:<11.5f}"
            f"{r.exact_hit_rate:>11.1%}{r.semantic_hit_rate:>14.1%}  {note}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", choices=["no-cache", "exact", "exact+semantic", "cascade"], default="no-cache",
    )
    parser.add_argument("--compare", action="store_true", help="Chạy cả 4 cấu hình, in bảng so sánh")
    parser.add_argument("--skip-ingest", action="store_true")
    args = parser.parse_args()

    if not args.skip_ingest:
        print("[ingest] nạp lại RAG data trước khi replay (bỏ qua bằng --skip-ingest)...")
        from scripts.ingest import ingest

        ingest()

    questions = load_replay()
    print(f"\n[replay] {len(questions)} câu hỏi từ {REPLAY_PATH}")

    if args.compare:
        results = [run_config(c, questions) for c in ["no-cache", "exact", "exact+semantic", "cascade"]]
        for r in results:
            print_result(r)
        print_comparison(results)
    else:
        result = run_config(args.config, questions)
        print_result(result)
        # Section 7: đối chiếu 1 lần chạy replay với ngân sách THÁNG — minh hoạ
        # "budget" là khái niệm tích luỹ theo thời gian, không phải per-run;
        # dùng MONTHLY_BUDGET_USD trực tiếp (không gọi check_budget() — hàm đó
        # đọc từ tracker.py:tracker toàn cục, còn run_config() dùng tracker
        # RIÊNG mỗi lần chạy để 4 config trong --compare không cộng dồn cost).
        from app.cost.budget import MONTHLY_BUDGET_USD

        pct = result.total_cost / MONTHLY_BUDGET_USD * 100
        print(f"\n[budget] chi phí lần chạy này = ${result.total_cost:.4f} "
              f"({pct:.3f}% ngân sách tháng ${MONTHLY_BUDGET_USD})")


if __name__ == "__main__":
    main()
