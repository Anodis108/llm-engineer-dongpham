"""Evaluation — Buổi 7 (LLM Evaluation & Guardrails).

Điểm vào thống nhất cho evaluation: gộp LLM-as-Judge (judge.py) và RAG metrics
kiểu RAGAS (ragas_native.py). Dùng trong CI/eval gate: chặn deploy nếu điểm
dưới ngưỡng (xem "Evaluation Checklist" cuối bài học).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.eval.judge import JudgeScore, judge_answer
from app.eval.ragas_native import evaluate_rag


@dataclass(slots=True)
class EvalResult:
    """Kết quả evaluation đầy đủ cho một sample — judge + RAG metrics."""

    question: str
    judge: JudgeScore
    rag: dict

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "judge": self.judge.model_dump(),
            "judge_overall": self.judge.overall,
            "rag": self.rag,
        }


def evaluate(
    question: str,
    answer: str,
    contexts: list[str] | None = None,
    reference: str | None = None,
    ground_truth: str | None = None,
) -> EvalResult:
    """Đánh giá đầy đủ một câu trả lời: LLM-as-Judge + RAG metrics (nếu có context).

    Args:
        question: câu hỏi gốc.
        answer: câu trả lời cần đánh giá.
        contexts: các chunk đã dùng để trả lời (để tính faithfulness/relevancy).
        reference: câu trả lời chuẩn cho judge (tùy chọn).
        ground_truth: câu trả lời chuẩn cho context_recall/precision (tùy chọn).
    """
    contexts = contexts or []
    context_text = "\n\n".join(contexts) if contexts else None

    judge_score = judge_answer(question, answer, reference=reference, context=context_text)
    rag_scores = evaluate_rag(question, answer, contexts, ground_truth) if contexts else {}

    return EvalResult(question=question, judge=judge_score, rag=rag_scores)


def evaluate_dataset(samples: list[dict]) -> list[EvalResult]:
    """Đánh giá một tập test dataset. Mỗi sample: {question, answer, contexts?,
    reference?, ground_truth?} — khớp format test_data trong bài học."""
    return [
        evaluate(
            question=s["question"],
            answer=s["answer"],
            contexts=s.get("contexts"),
            reference=s.get("reference"),
            ground_truth=s.get("ground_truth"),
        )
        for s in samples
    ]


def summarize(results: list[EvalResult]) -> dict:
    """Trung bình các metric trên toàn dataset — dùng cho eval gate trong CI."""
    if not results:
        return {}

    n = len(results)
    judge_avg = {
        "accuracy": sum(r.judge.accuracy for r in results) / n,
        "completeness": sum(r.judge.completeness for r in results) / n,
        "clarity": sum(r.judge.clarity for r in results) / n,
        "groundedness": sum(r.judge.groundedness for r in results) / n,
        "overall": sum(r.judge.overall for r in results) / n,
    }

    rag_keys = set().union(*(r.rag.keys() for r in results)) if any(r.rag for r in results) else set()
    rag_avg = {
        k: sum(r.rag.get(k, 0.0) for r in results if k in r.rag)
        / max(1, sum(1 for r in results if k in r.rag))
        for k in rag_keys
    }

    return {"n_samples": n, "judge": judge_avg, "rag": rag_avg}
