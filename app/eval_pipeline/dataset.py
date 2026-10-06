"""Golden Dataset — Module III, Bài 2, Section 2 (Golden Dataset).

Golden set sống ở 2 nơi có chủ đích khác nhau:
  - YAML (`data/eval/legal_qa/v1.yaml`) — NGUỒN CHÂN LÝ, version-controlled
    trong git, review qua pull request như code (bài học: "Review khi thêm
    case; mỗi case là một 'định nghĩa đúng'").
  - LangSmith Dataset — bản ĐỒNG BỘ để `evaluate()` (runner.py) chạy trên đó
    và LangSmith tự lưu lịch sử run/so sánh giữa các lần chạy.

`sync_dataset_to_langsmith()` đồng bộ 1 CHIỀU: YAML → LangSmith. Sync THEO
DIFF (so `case_id` trong metadata, không xoá-tạo-lại toàn bộ mỗi lần) để giữ
được lịch sử run gắn với example cũ trên LangSmith — xoá/tạo lại toàn bộ sẽ
mất liên kết đó.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from app.config import settings


def default_langsmith_client():
    """Client LangSmith đọc key từ settings (app/config.py — điểm DUY NHẤT
    chạm secrets trong repo), KHÔNG dựa vào Client() tự đọc biến môi trường
    mặc định (LANGCHAIN_API_KEY) — settings đọc alias LANGSMITH_API_KEY riêng
    của repo này (xem monitoring/tracing.py, cùng convention)."""
    from langsmith import Client

    return Client(api_key=settings.langsmith_api_key, api_url=settings.langsmith_endpoint)


@dataclass(slots=True)
class GoldenCase:
    """1 case trong golden set — khớp field trong file YAML (Section 2)."""

    id: str
    question: str
    expected: str
    slice: dict = field(default_factory=dict)
    must_include: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GoldenDataset:
    name: str
    version: int
    cases: list[GoldenCase]

    def slice_types(self) -> set[str]:
        return {c.slice.get("type", "unknown") for c in self.cases}

    def select_subset(self, n: int) -> GoldenDataset:
        """Section 6: subset cho eval gate trên PR — "ưu tiên slice rủi ro",
        KHÔNG cắt cơ học N case đầu (nguy hiểm: nếu YAML liệt kê hết lookup
        trước rồi mới tới injection/out_of_scope như v1.yaml hiện có, cắt đầu
        sẽ bỏ sót HOÀN TOÀN 2 slice rủi ro nhất).

        Lấy round-robin qua từng slice.type để phân bổ đều, không phải random
        — đảm bảo n case luôn có mặt đủ các slice miễn n >= số slice.
        """
        if n >= len(self.cases):
            return self

        by_type: dict[str, list[GoldenCase]] = {}
        for case in self.cases:
            by_type.setdefault(case.slice.get("type", "unknown"), []).append(case)

        selected: list[GoldenCase] = []
        round_idx = 0
        while len(selected) < n:
            added_this_round = False
            for cases in by_type.values():
                if round_idx < len(cases):
                    selected.append(cases[round_idx])
                    added_this_round = True
                    if len(selected) == n:
                        break
            if not added_this_round:
                break
            round_idx += 1

        return GoldenDataset(name=self.name, version=self.version, cases=selected)


def load_dataset(path: str | Path) -> GoldenDataset:
    """Đọc + validate golden set YAML (Section 2: Maintain — version hoá dataset).

    Validate tối thiểu: mỗi case có `id` duy nhất, `question`/`expected` không
    rỗng, `slice.type` có mặt (bắt buộc để tính `by_slice` ở runner.py —
    Section 4, "Aggregate tổng VÀ theo slice" chính là cách bắt regression vô hình).
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cases = []
    seen_ids: set[str] = set()
    for raw_case in raw["cases"]:
        case = GoldenCase(
            id=raw_case["id"],
            question=raw_case["question"],
            expected=raw_case["expected"],
            slice=raw_case.get("slice", {}),
            must_include=raw_case.get("must_include", []),
            must_not_include=raw_case.get("must_not_include", []),
        )
        if not case.question.strip() or not case.expected.strip():
            raise ValueError(f"Case '{case.id}': question/expected không được rỗng.")
        if case.id in seen_ids:
            raise ValueError(f"Case id trùng lặp: '{case.id}' — mỗi case phải có id duy nhất.")
        if "type" not in case.slice:
            raise ValueError(f"Case '{case.id}': thiếu slice.type — cần cho aggregate theo slice.")
        seen_ids.add(case.id)
        cases.append(case)

    return GoldenDataset(name=raw["dataset"], version=raw["version"], cases=cases)


def _to_langsmith_examples(dataset: GoldenDataset) -> dict[str, dict]:
    """Map GoldenCase -> dict examples LangSmith, key theo case.id.

    `split` = slice.type — cho phép filter subset ngay trên LangSmith UI/API
    theo đúng khái niệm "slice" của bài học mà không cần thêm field custom.
    """
    return {
        case.id: {
            "inputs": {"question": case.question},
            "outputs": {"expected": case.expected},
            "metadata": {
                "case_id": case.id,
                "slice": case.slice,
                "must_include": case.must_include,
                "must_not_include": case.must_not_include,
                "dataset_version": dataset.version,
            },
            "split": case.slice.get("type", "unknown"),
        }
        for case in dataset.cases
    }


def sync_dataset_to_langsmith(dataset: GoldenDataset, *, client=None) -> str:
    """Đồng bộ YAML -> LangSmith Dataset theo DIFF (Section 2: Maintain).

    - Dataset chưa tồn tại: tạo mới + push toàn bộ case.
    - Dataset đã tồn tại: so `case_id` hiện có vs YAML — update case đã đổi
      nội dung, create case mới, DELETE case không còn trong YAML (đã bị xoá/
      đổi id). Không xoá-tạo-lại toàn bộ dataset để giữ lịch sử run cũ.

    Returns: tên dataset đã sync (dùng làm `data=` cho evaluate() ở runner.py).
    """
    if client is None:
        client = default_langsmith_client()

    wanted = _to_langsmith_examples(dataset)

    if not client.has_dataset(dataset_name=dataset.name):
        client.create_dataset(dataset.name, description=f"Golden set '{dataset.name}' v{dataset.version}")
        client.create_examples(
            dataset_name=dataset.name,
            examples=[{"metadata": ex["metadata"], **ex} for ex in wanted.values()],
        )
        return dataset.name

    existing = list(client.list_examples(dataset_name=dataset.name))
    existing_by_case_id = {
        ex.metadata.get("case_id"): ex for ex in existing if ex.metadata and ex.metadata.get("case_id")
    }

    to_create = [ex for case_id, ex in wanted.items() if case_id not in existing_by_case_id]
    to_update = [
        {"id": existing_by_case_id[case_id].id, **ex}
        for case_id, ex in wanted.items()
        if case_id in existing_by_case_id
    ]
    to_delete_ids = [
        ex.id for case_id, ex in existing_by_case_id.items() if case_id not in wanted
    ]

    if to_create:
        client.create_examples(dataset_name=dataset.name, examples=to_create)
    if to_update:
        client.update_examples(dataset_name=dataset.name, updates=to_update)
    for example_id in to_delete_ids:
        client.delete_example(example_id)

    return dataset.name
