"""Eval Runner — Module III, Bài 2, Section 4 (Từ eval thủ công sang eval pipeline).

Dùng thẳng `langsmith.evaluate()` làm runner thay vì tự viết ThreadPoolExecutor
+ lưu lịch sử JSON (như code mẫu trong bài học) — LangSmith đã làm sẵn: chạy
song song trên toàn bộ Dataset, gọi evaluators, LƯU LỊCH SỬ run tự động (mỗi
lần chạy = 1 "experiment" gắn với dataset version, xem trên LangSmith UI).
Runner ở đây chỉ còn việc bọc `pipeline.answer()` thành `target` đúng chữ ký
LangSmith cần (dict in -> dict out) và xử lý riêng case injection/out_of_scope
(kỳ vọng bị guardrails CHẶN, không phải lỗi).
"""

from __future__ import annotations

from app.eval_pipeline.dataset import (
    GoldenDataset,
    default_langsmith_client,
    sync_dataset_to_langsmith,
)
from app.eval_pipeline.scorers import DEFAULT_EVALUATORS
from app.guardrails.checks import GuardrailViolation


def _target(inputs: dict) -> dict:
    """`target` cho langsmith.evaluate() — nhận example.inputs, trả dict output.

    Case injection kỳ vọng bị `check_input` CHẶN (guardrails, Module I Bài 7)
    — đây là THÀNH CÔNG của guardrail, không phải lỗi hệ thống, nên bắt riêng
    và trả answer mô tả việc bị chặn thay vì để GuardrailViolation văng lên
    làm cả run trong LangSmith bị đánh dấu "error".
    """
    from app.pipeline import answer

    try:
        result = answer(inputs["question"])
    except GuardrailViolation as exc:
        result = f"[Bị chặn bởi guardrails: {exc.reason}]"
    return {"answer": result}


def run_eval(
    dataset: GoldenDataset,
    *,
    experiment_prefix: str | None = None,
    max_concurrency: int = 8,
    client=None,
):
    """Section 4: sync dataset lên LangSmith rồi chạy toàn bộ qua evaluate().

    Trả về `ExperimentResults` — object của LangSmith, đọc được `.to_pandas()`
    hoặc duyệt trực tiếp để build report (xem report.py:build_report()).
    """
    from langsmith.evaluation import evaluate

    # client=None -> Client() mặc định của LangSmith đọc LANGCHAIN_API_KEY (biến
    # môi trường KHÁC với LANGSMITH_API_KEY mà settings.py/tracing.py dùng) —
    # luôn tự tạo client qua settings nếu caller không truyền, để nhất quán
    # nguồn credential DUY NHẤT của repo (app/config.py).
    if client is None:
        client = default_langsmith_client()

    dataset_name = sync_dataset_to_langsmith(dataset, client=client)

    return evaluate(
        _target,
        data=dataset_name,
        evaluators=DEFAULT_EVALUATORS,
        experiment_prefix=experiment_prefix or f"{dataset.name}-v{dataset.version}",
        max_concurrency=max_concurrency,
        client=client,
    )
