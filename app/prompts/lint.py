"""Prompt lint — Module III, Bài 6, Section 4 (CI Patterns cho LLM).

Chặn ở CI những lỗi prompt rẻ mà đắt: thiếu metadata, version sai kiểu, template
quá dài, và — lỗi phổ biến nhất theo bài học — **biến dùng trong template nhưng
chưa khai báo** (hoặc ngược lại: khai báo rồi đổi tên trong template mà quên
cập nhật danh sách).

Khác code mẫu trong bài (lint file YAML): ở đây prompt sống trên LangSmith, nên
lint chạy trên `PromptDefinition` trong app/prompts/definitions.py TRƯỚC khi
push. Không cần credentials LangSmith → chạy được trong CI, không tốn gì.

Chạy:
    python -m app.prompts.lint          # exit 1 nếu có lỗi
"""

from __future__ import annotations

import re
import sys

from app.prompts.definitions import ALL_PROMPTS, PromptDefinition

# Bắt buộc phải có, khớp REQUIRED_META trong bài học.
REQUIRED_META = ("name", "version", "model", "owner", "changelog", "description")
MAX_TEMPLATE_CHARS = 12000
_KEBAB_CASE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def _template_variables(p: PromptDefinition) -> set[str]:
    """Biến template THỰC SỰ dùng — hỏi thẳng ChatPromptTemplate, không tự parse
    regex (parse tay sẽ bỏ sót biến trong message thứ 2 trở đi)."""
    return set(p.template.input_variables)


def lint_definition(p: PromptDefinition) -> list[str]:
    """Trả về danh sách lỗi của 1 definition. Rỗng = hợp lệ."""
    errs: list[str] = []
    label = f"{p.name} v{p.version}"

    for field in REQUIRED_META:
        value = getattr(p, field, None)
        if value is None or (isinstance(value, str) and not value.strip()):
            errs.append(f"{label}: thiếu metadata '{field}'")

    if not isinstance(p.version, int) or isinstance(p.version, bool) or p.version < 1:
        errs.append(f"{label}: version phải là số nguyên >= 1 (đang là {p.version!r})")

    if not _KEBAB_CASE.match(p.name):
        errs.append(f"{label}: name phải là kebab-case (vd 'rag-answer'), đang là {p.name!r}")

    # Độ dài tính trên template ĐÃ GHÉP mọi message — prompt phình thường do
    # nhồi few-shot/context vào 1 message, không phải do nhiều message.
    total_chars = sum(
        len(m.prompt.template) for m in p.template.messages
    )
    if total_chars > MAX_TEMPLATE_CHARS:
        errs.append(
            f"{label}: template quá dài ({total_chars} > {MAX_TEMPLATE_CHARS} ký tự)"
        )

    used = _template_variables(p)
    declared = set(p.variables)

    missing = used - declared
    if missing:
        errs.append(f"{label}: biến dùng nhưng chưa khai báo: {sorted(missing)}")

    unused = declared - used
    if unused:
        errs.append(
            f"{label}: biến khai báo nhưng không dùng trong template: {sorted(unused)} "
            "(dấu hiệu đổi tên biến ở template mà quên cập nhật `variables`)"
        )

    return errs


def lint_all(prompts: list[PromptDefinition] | None = None) -> list[str]:
    """Lint toàn bộ definitions + phát hiện trùng (name, version)."""
    prompts = ALL_PROMPTS if prompts is None else prompts
    errs: list[str] = []

    seen: dict[tuple[str, int], int] = {}
    for p in prompts:
        key = (p.name, p.version)
        seen[key] = seen.get(key, 0) + 1
        errs.extend(lint_definition(p))

    for (name, version), count in seen.items():
        if count > 1:
            errs.append(f"{name} v{version}: định nghĩa trùng ({count} lần)")

    return errs


def main() -> None:
    errs = lint_all()
    for e in errs:
        # Format GitHub Actions annotation — hiện thẳng trên PR diff.
        print(f"::error::{e}")
    if errs:
        print(f"\n{len(errs)} lỗi prompt lint.")
        sys.exit(1)
    print(f"prompt lint OK ({len(ALL_PROMPTS)} definitions).")


if __name__ == "__main__":
    main()
