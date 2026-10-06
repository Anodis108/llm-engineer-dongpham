"""Demo Buổi 7 — chạy evaluation (LLM-as-Judge + RAG metrics) trên một test set.

Minh hoạ "Evaluation Checklist trước khi deploy" cuối bài học: chạy dataset qua
evaluate_dataset(), in bảng tổng hợp, và áp threshold như một eval gate CI thật
sẽ làm (faithfulness > 0.85, answer_relevancy > 0.85...).

Chạy:
    python -m scripts.eval_demo

Cần OPENAI_API_KEYS (judge + RAG metrics đều gọi LLM).
"""

from __future__ import annotations

import json

from app.eval.metrics import evaluate_dataset, summarize

DATASET_PATH = "data/eval_dataset.jsonl"

# Ngưỡng tối thiểu để "pass" — khớp Evaluation Checklist trong bài học.
THRESHOLDS = {
    "judge.overall": 3.5,       # thang 1-5
    "rag.faithfulness": 0.85,
    "rag.answer_relevancy": 0.85,
}


def load_dataset(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    samples = load_dataset(DATASET_PATH)
    print(f"Đánh giá {len(samples)} sample từ {DATASET_PATH}...\n")

    results = evaluate_dataset(samples)

    # strict=True: bỏ sót sample nào là điểm eval sai — phải raise, không im lặng.
    for s, r in zip(samples, results, strict=True):
        d = r.as_dict()
        print(f"Q: {s['question']}")
        print(f"A: {s['answer']}")
        print(f"Context: {s['contexts']}")
        print(f"Ground truth: {s['ground_truth']}")
        print(
            f"  judge  : accuracy={d['judge']['accuracy']} completeness={d['judge']['completeness']} "
            f"clarity={d['judge']['clarity']} groundedness={d['judge']['groundedness']} "
            f"(overall={d['judge_overall']:.2f})"
        )
        if r.rag:
            rag_str = " ".join(f"{k}={v:.2f}" for k, v in r.rag.items())
            print(f"  rag    : {rag_str}")
        print()

    summary = summarize(results)
    print("─── Tổng hợp (eval gate) ───")
    print(f"n_samples: {summary['n_samples']}")
    print(f"judge.overall           = {summary['judge']['overall']:.2f}  "
          f"(ngưỡng >= {THRESHOLDS['judge.overall']})")
    if "faithfulness" in summary["rag"]:
        print(f"rag.faithfulness        = {summary['rag']['faithfulness']:.2f}  "
              f"(ngưỡng >= {THRESHOLDS['rag.faithfulness']})")
    if "answer_relevancy" in summary["rag"]:
        print(f"rag.answer_relevancy    = {summary['rag']['answer_relevancy']:.2f}  "
              f"(ngưỡng >= {THRESHOLDS['rag.answer_relevancy']})")

    passed = (
        summary["judge"]["overall"] >= THRESHOLDS["judge.overall"]
        and summary["rag"].get("faithfulness", 1.0) >= THRESHOLDS["rag.faithfulness"]
        and summary["rag"].get("answer_relevancy", 1.0) >= THRESHOLDS["rag.answer_relevancy"]
    )
    print(f"\nEval gate: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
