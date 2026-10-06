"""Demo Module III, Bài 2 — Eval Pipeline chạy trên LangSmith.

Khác scripts/eval_demo.py (Module I, Bài 7 — chạy 1 lần, không version dataset,
không gate theo slice): script này minh hoạ TOÀN BỘ luồng "Từ eval thủ công
sang eval pipeline" (Section 4) + Regression Gate (Section 5):

    golden set YAML (version hoá) → sync LangSmith Dataset → evaluate() chạy
    song song → aggregate (tổng + THEO SLICE) → so baseline → exit 0/1

Chạy:
    python -m scripts.eval_pipeline_demo                      # full dataset
    python -m scripts.eval_pipeline_demo --subset 6            # 6 case đầu (nhanh, rẻ — dùng cho PR)
    python -m scripts.eval_pipeline_demo --skip-ingest         # bỏ qua bước ingest lại RAG data
    python -m scripts.eval_pipeline_demo --subset 6 --report pr.md --json pr.json

Cần OPENAI_API_KEYS (judge + RAG) và LANGSMITH_API_KEY (MONITORING_ENABLED
không bắt buộc — script tự tạo LangSmith Client riêng cho phần eval, độc lập
với tracing production).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.eval_pipeline.dataset import load_dataset
from app.eval_pipeline.gate import GateConfig, check_gate, summarize
from app.eval_pipeline.runner import run_eval

DATASET_PATH = "data/eval/legal_qa/v1.yaml"

# Section 5: baseline + drop_tolerance — khớp GATES trong bài học. Baseline ở
# đây là ước lượng ban đầu; production nên tính từ vài lần chạy thật (Section 5:
# "Đo variance nền" trước khi chọn tolerance).
GATES = {
    "overall": GateConfig(baseline=4.0, drop_tolerance=0.5),
    "rule_pass_rate": GateConfig(baseline=0.7, drop_tolerance=0.15),
    "slice:injection": GateConfig(baseline=0.9, drop_tolerance=0.3),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_PATH, help="Đường dẫn golden set YAML")
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Chỉ chạy N case đầu (nhanh/rẻ — dùng cho eval gate trên PR, Section 6)",
    )
    parser.add_argument(
        "--skip-ingest", action="store_true",
        help="Bỏ qua ingest lại RAG data (dùng khi Qdrant server đã có dữ liệu, không phải :memory:)",
    )
    parser.add_argument(
        "--json", default=None, metavar="PATH",
        help="Ghi kết quả dạng JSON (Bài 6: CI đọc để dựng PR comment)",
    )
    parser.add_argument(
        "--report", default=None, metavar="PATH",
        help="Ghi báo cáo Markdown (dùng làm nội dung PR comment / job summary)",
    )
    args = parser.parse_args()

    if not args.skip_ingest:
        print("[ingest] nạp lại RAG data trước khi eval (bỏ qua bằng --skip-ingest)...")
        from scripts.ingest import ingest

        ingest()

    dataset = load_dataset(args.dataset)
    if args.subset:
        # select_subset ưu tiên phủ ĐỦ slice (round-robin), không cắt N case
        # đầu YAML — tránh bỏ sót hoàn toàn slice rủi ro (injection/out_of_scope)
        # nếu chúng nằm cuối file (Section 6: "ưu tiên slice rủi ro").
        dataset = dataset.select_subset(args.subset)

    print(f"\n[eval] chạy {len(dataset.cases)} case từ '{dataset.name}' v{dataset.version}...")
    results = run_eval(dataset, experiment_prefix=f"{dataset.name}-v{dataset.version}")
    rows = list(results)

    print(f"\nXem chi tiết trên LangSmith: {results.url}\n")

    summary = summarize(rows)
    print("─── Tổng hợp ───")
    print(f"n_cases           = {summary.n}")
    print(f"overall (judge)   = {summary.overall}")
    print(f"rule_pass_rate    = {summary.rule_pass_rate}")
    print("by_slice.type:")
    for slice_type, score in sorted(summary.by_slice_type.items()):
        print(f"  {slice_type:15s} = {score}")

    gate_result = check_gate(summary, GATES)

    print("\n─── Eval Gate ───")
    if gate_result.passed:
        print("PASS — mọi metric trong tolerance so với baseline.")
    else:
        print("FAIL — các metric sau vượt tolerance:")
        for failure in gate_result.failures:
            print(f"  ✗ {failure}")

    # ── Đầu ra máy đọc được (Bài 6, Section 4: "CI cần output có cấu trúc") ──
    # Eval gate chỉ hữu ích nếu kết quả tới được chỗ người quyết định: comment
    # trên PR + job summary. Log của CI thì không ai đọc.
    payload = _payload(dataset, results, summary, gate_result)

    if args.json:
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n[report] JSON → {args.json}")

    if args.report:
        Path(args.report).write_text(_render_markdown(payload), encoding="utf-8")
        print(f"[report] Markdown → {args.report}")

    sys.exit(0 if gate_result.passed else 1)


def _payload(dataset, results, summary, gate_result) -> dict:
    """Kết quả eval dạng có cấu trúc — nguồn duy nhất cho JSON + Markdown."""
    return {
        "dataset": dataset.name,
        "dataset_version": dataset.version,
        "n_cases": summary.n,
        "overall": summary.overall,
        "rule_pass_rate": summary.rule_pass_rate,
        "by_slice_type": dict(sorted(summary.by_slice_type.items())),
        "gate": {
            "passed": gate_result.passed,
            "failures": list(gate_result.failures),
        },
        "langsmith_url": results.url,
        "experiment": getattr(results, "experiment_name", ""),
    }


def _render_markdown(p: dict) -> str:
    """Báo cáo cho PR comment — FAIL lên đầu để người review thấy ngay."""
    status = "✅ PASS" if p["gate"]["passed"] else "❌ FAIL"
    lines = [
        f"## Eval gate: {status}",
        "",
        f"`{p['dataset']}` v{p['dataset_version']} — {p['n_cases']} case"
        f" · [chi tiết trên LangSmith]({p['langsmith_url']})",
        "",
        "| Metric | Giá trị |",
        "| --- | --- |",
        f"| overall (judge) | {p['overall']} |",
        f"| rule_pass_rate | {p['rule_pass_rate']} |",
    ]
    for slice_type, score in p["by_slice_type"].items():
        lines.append(f"| slice:{slice_type} | {score} |")

    if p["gate"]["failures"]:
        lines += ["", "**Vượt tolerance so với baseline:**", ""]
        lines += [f"- {f}" for f in p["gate"]["failures"]]

    lines += [
        "",
        "<sub>Baseline và tolerance đặt trong `scripts/eval_pipeline_demo.py::GATES`.</sub>",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
