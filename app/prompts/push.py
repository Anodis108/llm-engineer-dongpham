"""Push prompt definitions lên LangSmith Prompt Hub — Module III, Bài 6.

Đây là bước "prompt as code" (Bài 1, Section 3 + Bài 6, Section 4): prompt nằm
trong git dưới dạng `PromptDefinition`, được lint ở CI (app/prompts/lint.py),
rồi push lên LangSmith như một bước deploy RIÊNG với deploy code.

Vì sao tách khỏi deploy code:
    Sửa 1 câu trong prompt không cần build lại image. Push version mới + trỏ
    alias `production` sang đó = deploy prompt trong vài giây; rollback = trỏ
    alias về commit cũ (scripts/prompt_rollback.py). Đây là "3 trục rollback"
    của Bài 6: image tag / prompt alias / model config — độc lập nhau.

Quy ước version (Bài 1, Section 3):
    `name` ổn định (vd "rag-answer"), `version` là số nguyên tăng dần. Mỗi
    version push lên = 1 COMMIT của cùng repo đó. Chỉ version CAO NHẤT của mỗi
    name được gắn alias — các version cũ vẫn nằm đó làm lịch sử để rollback về.

Chạy:
    python -m app.prompts.push                  # push tất cả, gắn alias production
    python -m app.prompts.push --dry-run        # in ra sẽ push gì, không gọi API
    python -m app.prompts.push --alias staging  # đẩy vào nhánh thử nghiệm
"""

from __future__ import annotations

import argparse
import sys

from app.config import settings
from app.prompts.definitions import ALL_PROMPTS, PromptDefinition, latest_per_name
from app.prompts.lint import lint_all

# Prompt là repo CÔNG KHAI trên LangSmith theo mặc định của `push_prompt`.
# Với prompt chứa thông tin nội bộ, đặt `is_public=False` — ở đây prompt chỉ là
# persona luật sư chung nên để mặc định, nhưng nêu rõ để người đọc biết đây là
# lựa chọn chứ không phải sót.


def _client():
    from langsmith import Client

    return Client(
        api_key=settings.langsmith_api_key, api_url=settings.langsmith_endpoint
    )


def push_all(
    prompts: list[PromptDefinition] | None = None,
    alias: str | None = None,
    client=None,
    dry_run: bool = False,
) -> list[dict]:
    """Push mọi definition theo thứ tự version tăng dần. Trả về kết quả từng cái.

    KHÔNG push nếu lint còn lỗi — prompt hỏng mà lên registry còn tệ hơn code
    hỏng, vì nó ảnh hưởng chất lượng câu trả lời mà không có exception nào cả.
    """
    prompts = ALL_PROMPTS if prompts is None else prompts
    alias = alias or settings.prompt_alias

    errs = lint_all(prompts)
    if errs:
        for e in errs:
            print(f"::error::{e}")
        raise SystemExit(f"{len(errs)} lỗi lint — không push. Sửa trước.")

    newest = latest_per_name()
    results: list[dict] = []

    for p in sorted(prompts, key=lambda x: (x.name, x.version)):
        # Chỉ version mới nhất giữ alias; version cũ push lên làm commit lịch sử.
        gets_alias = newest[p.name] is p
        tags = [alias] if gets_alias else None

        row = {
            "identifier": p.identifier,
            "version": p.version,
            "alias": alias if gets_alias else None,
        }

        if dry_run:
            row["commit_hash"] = "(dry-run)"
            results.append(row)
            print(
                f"[dry-run] {p.identifier} v{p.version} "
                f"→ {'alias=' + alias if gets_alias else 'commit (không alias)'}"
            )
            continue

        commit_hash = _client_or(client).push_prompt(
            p.identifier,
            object=p.template,  # BẮT BUỘC LangChain object — xem docstring definitions.py
            commit_tags=tags,
            commit_description=p.commit_message,
            description=p.description,
        )
        row["commit_hash"] = commit_hash
        results.append(row)
        print(
            f"✓ {p.identifier} v{p.version} → commit {commit_hash[:8]}"
            f"{' [alias ' + alias + ']' if gets_alias else ''}"
        )

    return results


def _client_or(client):
    return client if client is not None else _client()


def main() -> None:
    ap = argparse.ArgumentParser(description="Push prompt lên LangSmith Prompt Hub")
    ap.add_argument("--alias", default=None, help="alias gắn cho version mới nhất")
    ap.add_argument("--dry-run", action="store_true", help="không gọi API, chỉ in kế hoạch")
    args = ap.parse_args()

    if not args.dry_run and not settings.langsmith_api_key:
        sys.exit("Thiếu LANGSMITH_API_KEY — không push được.")

    if not args.dry_run:
        print(f"Workspace project: {settings.langsmith_project}")

    results = push_all(alias=args.alias, dry_run=args.dry_run)
    print(f"\nĐã push {len(results)} commit.")


if __name__ == "__main__":
    main()
