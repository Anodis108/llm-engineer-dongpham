"""Rollback prompt KHÔNG cần rebuild image — Module III, Bài 6, Section 4.

Bài học "3 trục rollback độc lập": image tag / prompt alias / model config.
Script này là trục thứ hai. Khi prompt v2 làm chất lượng câu trả lời tụt, cách
sửa KHÔNG phải build lại image rồi deploy lại (mất 5-10 phút) mà là trỏ alias
`production` về commit v1 (mất vài giây, không downtime).

Cách hoạt động:
    LangSmith coi mỗi version prompt là 1 COMMIT bất biến của repo đó; alias
    (`production`) là con trỏ tới 1 commit. Rollback = lấy nội dung của commit
    cũ, push lại thành commit mới và gắn alias vào đó.

    Vì sao push commit MỚI thay vì trỏ con trỏ ngược lại (kiểu `git reset`)?
    Để lịch sử chỉ tiến — luôn thấy được "đã rollback lúc nào, về bản nào" ngay
    trong danh sách commit. Tương đương `git revert` chứ không phải `git reset`.

    Đổi lại: PHẢI xoá cache đĩa của app (data/prompt_cache/) sau khi rollback,
    nếu không app còn phục vụ bản cache cũ tới hết TTL. Script tự xoá.

Chạy:
    python scripts/prompt_rollback.py --list
    python scripts/prompt_rollback.py --previous          # về commit liền trước
    python scripts/prompt_rollback.py --to <commit_hash>
    python scripts/prompt_rollback.py --previous --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Cho phép chạy trực tiếp `python scripts/...` từ gốc repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402


def _client():
    from langsmith import Client

    return Client(
        api_key=settings.langsmith_api_key, api_url=settings.langsmith_endpoint
    )


def _identifier(name: str) -> str:
    return f"{settings.langsmith_project}-{name}"


def _clear_cache(name: str, alias: str) -> Path | None:
    """Xoá cache đĩa để app không phục vụ bản cũ tới hết TTL."""
    path = Path(settings.prompt_cache_dir) / f"{name}__{alias}.json"
    if path.exists():
        path.unlink()
        return path
    return None


def _commits(client, identifier: str, limit: int = 20):
    """Danh sách commit, mới nhất trước."""
    rows = list(client.list_prompt_commits(identifier, limit=limit))
    return sorted(rows, key=lambda c: c.created_at, reverse=True)


def list_commits(client, name: str, alias: str, limit: int = 20) -> None:
    identifier = _identifier(name)
    print(f"Prompt: {identifier}   (alias đang xét: {alias})\n")

    try:
        current = client.pull_prompt_commit(f"{identifier}:{alias}")
        current_hash = current.commit_hash
    except Exception as exc:  # noqa: BLE001
        current_hash = None
        print(f"  ! Không resolve được alias {alias!r}: {exc}\n")

    for c in _commits(client, identifier, limit):
        mark = "← " + alias if c.commit_hash == current_hash else "  "
        when = c.created_at.strftime("%Y-%m-%d %H:%M")
        print(f"  {c.commit_hash[:12]}  {when}  {mark}  {c.description or ''}")


def rollback(
    client, name: str, alias: str, target: str | None, dry_run: bool = False
) -> int:
    identifier = _identifier(name)

    try:
        current = client.pull_prompt_commit(f"{identifier}:{alias}")
    except Exception as exc:  # noqa: BLE001
        print(f"::error::Không đọc được alias {alias!r} của {identifier}: {exc}")
        return 1

    if target is None:  # --previous: commit liền trước commit đang giữ alias
        commits = _commits(client, identifier)
        hashes = [c.commit_hash for c in commits]
        if current.commit_hash not in hashes:
            print(f"::error::Commit hiện tại không nằm trong {len(hashes)} commit gần nhất.")
            return 1
        idx = hashes.index(current.commit_hash)
        if idx + 1 >= len(hashes):
            print("::error::Không có commit nào cũ hơn để rollback về.")
            return 1
        target = hashes[idx + 1]

    if target == current.commit_hash:
        print("Alias đã trỏ đúng commit đó — không làm gì.")
        return 0

    # Lấy nội dung commit đích rồi push lại thành commit mới, gắn alias.
    old = client.pull_prompt(f"{identifier}:{target}")

    print(f"Rollback {identifier}")
    print(f"  từ  {current.commit_hash[:12]}  {current.description or ''}")
    print(f"  về  {target[:12]}")

    if dry_run:
        print("\n[dry-run] Không gọi API, không xoá cache.")
        return 0

    new_hash = client.push_prompt(
        identifier,
        object=old,
        commit_tags=[alias],
        commit_description=f"rollback: {alias} → {target[:12]}",
        description=current.description,
    )
    print(f"  → commit mới {new_hash[:8]} đang giữ alias {alias!r}")

    removed = _clear_cache(name, alias)
    if removed:
        print(f"  → đã xoá cache {removed}")
    else:
        print("  → không có cache đĩa cần xoá")

    print("\nApp dùng bản mới ở request kế tiếp (không cần restart, không rebuild image).")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Rollback prompt qua alias LangSmith")
    ap.add_argument("--prompt", default="rag-answer", help="tên prompt (không tiền tố project)")
    ap.add_argument("--alias", default=None, help="alias cần trỏ (mặc định: PROMPT_ALIAS)")
    ap.add_argument("--list", action="store_true", help="chỉ liệt kê commit rồi thoát")
    ap.add_argument("--previous", action="store_true", help="rollback về commit liền trước")
    ap.add_argument("--to", default=None, help="rollback về commit hash cụ thể")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    alias = args.alias or settings.prompt_alias
    if not settings.langsmith_api_key:
        sys.exit("Thiếu LANGSMITH_API_KEY — không gọi được LangSmith.")

    client = _client()

    if args.list or (not args.previous and not args.to):
        list_commits(client, args.prompt, alias)
        return

    target = args.to if args.to else None
    sys.exit(rollback(client, args.prompt, alias, target, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
