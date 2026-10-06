"""Diễn tập rollback CD chạy hoàn toàn ở local — Module III, Bài 6, Section 4.

Mục đích: chứng minh bằng máy, không phải bằng lời, hai điều mà bài học khẳng định:

  1. **Container healthy KHÔNG có nghĩa là deploy thành công.** Bản deploy hỏng
     trong bài này vẫn trả `/health` 200 — nó chỉ chết ở `/chat`. Nếu tiêu chí
     promote là "container đã start", bản hỏng này sẽ lên production.

  2. **Rollback là thao tác vài giây, không phải build lại.** Image bất biến
     theo tag; rollback chỉ là trỏ tag `latest` về image cũ rồi restart. Không
     compile, không build, không chờ CI.

Kịch bản (6 bước, tự động):
    [1] build image A (code hiện tại)                     → tag :good
    [2] "deploy" A → ghi .current=good
    [3] smoke A                                           → PASS
    [4] build image B (pipeline hỏng giả lập)             → tag :bad
        "deploy" B → ghi .lkg=good, .current=bad
    [5] smoke B                                           → FAIL (/health 200, /chat 500)
    [6] rollback → .current=good → smoke A                → PASS

Bước [4] dựng image B bằng 1 Dockerfile 2 dòng `FROM :good` + COPY đè
app/pipeline.py bằng bản hỏng — đúng hình dạng của một lần deploy lỗi thật
(đổi code, không đổi hạ tầng), và build chỉ mất ~1 giây vì tái dùng layer.

Chạy:
    python scripts/cicd_demo.py                  # đầy đủ, gọi API thật cho smoke
    python scripts/cicd_demo.py --health-only    # chỉ /health (nhanh, miễn phí)
    python scripts/cicd_demo.py --keep           # giữ container lại để xem

Cần: Docker đang chạy. Với smoke đầy đủ cần thêm OPENAI_API_KEYS (đọc từ .env).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STATE_DIR = REPO / "data" / "cicd_demo"
CONTAINER = "llm-cicd-demo"
IMAGE_GOOD = "llm-engineer-demo:good"
IMAGE_BAD = "llm-engineer-demo:bad"

# Pipeline hỏng: /health vẫn 200 (nó không đụng tới module này), nhưng /chat
# nổ 500. Đây chính là kiểu hỏng mà chỉ smoke test mới bắt được.
BROKEN_PIPELINE = '''"""Bản pipeline HỎNG giả lập cho diễn tập rollback (scripts/cicd_demo.py).

Không phải code thật — chỉ tồn tại để bước [5] của drill có một bản deploy
"trông thì sống" (container up, /health 200) nhưng hỏng ở đường người dùng thật.
"""

from __future__ import annotations

from collections.abc import Iterator

_BOOM = "demo: regression giả lập — bản deploy này hỏng ở /chat, /health vẫn 200"


def answer(question: str, params=None) -> str:
    raise RuntimeError(_BOOM)


def answer_stream(question: str, params=None) -> Iterator[str]:
    raise RuntimeError(_BOOM)


def answer_structured(question: str, params=None):
    raise RuntimeError(_BOOM)
'''


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO, text=True, **kw)


def step(n: int, total: int, text: str) -> None:
    print(f"\n{'─' * 68}\n[{n}/{total}] {text}\n{'─' * 68}")


def docker(*args: str, check: bool = True, capture: bool = False):
    return run(
        ["docker", *args],
        check=check,
        capture_output=capture,
    )


def read_tag(name: str) -> str:
    path = STATE_DIR / name
    return path.read_text().strip() if path.exists() else ""


def write_tag(name: str, value: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / name).write_text(value + "\n")


def stop_container() -> None:
    docker("rm", "-f", CONTAINER, check=False, capture=True)


def start_container(image: str, port: int) -> None:
    """Chạy container với ĐÚNG cách release.sh chạy: image bất biến + tag latest."""
    docker("tag", image, "llm-engineer-demo:latest")
    stop_container()

    cmd = [
        "run", "-d", "--name", CONTAINER,
        "-p", f"127.0.0.1:{port}:8000",
    ]
    # Nạp .env nếu có — smoke đầy đủ cần OPENAI_API_KEYS để trả lời thật.
    env_file = REPO / ".env"
    if env_file.exists():
        cmd += ["--env-file", str(env_file)]
    cmd += [
        "-e", "PROMPT_REGISTRY=local",  # drill không phụ thuộc LangSmith sống/chết
        "llm-engineer-demo:latest",
    ]
    docker(*cmd, capture=True)

    # Chờ /health thật, không chỉ "docker run trả về 0".
    deadline = time.time() + 60
    while time.time() < deadline:
        r = run(
            ["curl", "-fsS", "--max-time", "2", f"http://127.0.0.1:{port}/health"],
            capture_output=True,
        )
        if r.returncode == 0:
            return
        time.sleep(1)
    raise SystemExit("::error::container không healthy sau 60s")


def smoke(port: int, health_only: bool) -> bool:
    cmd = [sys.executable, "deploy/smoke.py", "--url", f"http://127.0.0.1:{port}"]
    if health_only:
        cmd.append("--skip-rag")
    r = run(cmd)
    return r.returncode == 0


def build_good() -> None:
    docker("build", "-t", IMAGE_GOOD, ".", capture=True)


def build_bad() -> None:
    """Dựng image hỏng bằng overlay 1 file — nhanh, không đụng code trong repo."""
    broken = STATE_DIR / "broken_pipeline.py"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text(BROKEN_PIPELINE, encoding="utf-8")

    dockerfile = STATE_DIR / "Dockerfile.broken"
    dockerfile.write_text(
        f"FROM {IMAGE_GOOD}\nCOPY {broken.name} /app/app/pipeline.py\n",
        encoding="utf-8",
    )

    docker("build", "-f", str(dockerfile), "-t", IMAGE_BAD, str(STATE_DIR), capture=True)


def deploy(tag_image: str, tag_name: str, port: int, record_lkg: bool) -> None:
    """Bản local của `deploy/release.sh deploy` — cùng ngữ nghĩa .current/.lkg."""
    current = read_tag("current")
    if record_lkg and current and current != tag_name:
        write_tag("lkg", current)
        print(f"  → .lkg = {current}")

    print(f"  → .current = {tag_name}")
    start_container(tag_image, port)
    write_tag("current", tag_name)


def rollback(port: int) -> None:
    """Bản local của `deploy/release.sh rollback`."""
    lkg = read_tag("lkg")
    if not lkg:
        raise SystemExit("::error::chưa có .lkg — không rollback được")
    image = IMAGE_GOOD if lkg == "good" else IMAGE_BAD

    print(f"  → rollback: .current {read_tag('current')} → {lkg}")
    start_container(image, port)
    write_tag("current", lkg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Diễn tập rollback CD ở local")
    ap.add_argument("--port", type=int, default=8099, help="cổng host cho container demo")
    ap.add_argument("--health-only", action="store_true", help="smoke chỉ kiểm tra /health (miễn phí)")
    ap.add_argument("--keep", action="store_true", help="giữ container lại sau khi xong")
    args = ap.parse_args()

    if shutil.which("docker") is None:
        raise SystemExit("::error::không tìm thấy docker")

    if run(["docker", "info"], capture_output=True).returncode != 0:
        raise SystemExit("::error::Docker daemon chưa chạy")

    if not args.health_only and not (REPO / ".env").exists():
        print("! Không có .env → smoke sẽ không gọi được /chat. Dùng --health-only.")

    try:
        step(1, 6, f"Build image A (code hiện tại) → {IMAGE_GOOD}")
        build_good()
        print("  ✓ build xong")

        step(2, 6, "Deploy A")
        deploy(IMAGE_GOOD, "good", args.port, record_lkg=False)

        step(3, 6, "Smoke A (kỳ vọng PASS)")
        if not smoke(args.port, args.health_only):
            raise SystemExit("::error::smoke A fail — code hiện tại đã hỏng, dừng drill")
        print("  ✓ PASS")

        step(4, 6, f"Build image B (pipeline hỏng giả lập) → {IMAGE_BAD}")
        build_bad()
        print("  ✓ build xong")
        deploy(IMAGE_BAD, "bad", args.port, record_lkg=True)

        step(5, 6, "Smoke B (kỳ vọng FAIL — /health vẫn 200)")
        if smoke(args.port, args.health_only):
            if args.health_only:
                print(
                    "  ! --health-only: /health 200 nên smoke PASS. Đây CHÍNH LÀ bài học:\n"
                    "    health check một mình không bắt được bản deploy hỏng."
                )
            else:
                raise SystemExit("::error::smoke B pass — kịch bản hỏng không hoạt động")
        else:
            print("  ✓ FAIL đúng như mong đợi — không promote bản này")

        step(6, 6, "Rollback về last-known-good rồi smoke lại")
        rollback(args.port)
        if not smoke(args.port, args.health_only):
            raise SystemExit("::error::smoke sau rollback vẫn fail")
        print("  ✓ PASS — đã về bản tốt")

    finally:
        if args.keep:
            print(f"\n(giữ container {CONTAINER} ở cổng {args.port})")
        else:
            stop_container()
            print(f"\n(đã dọn container {CONTAINER})")

    print(
        "\nKẾT LUẬN\n"
        "  • Bản B: container up, /health 200 — nhưng /chat 500. Nếu tiêu chí promote\n"
        "    là 'container đã start' thì bản này đã lên production.\n"
        "  • Rollback chỉ đổi tag + restart: vài giây, không build lại, không chờ CI.\n"
        "  • Đây là lý do cd.yml đặt smoke test TRƯỚC bước promote, và rollback\n"
        "    tự động khi smoke fail.\n"
    )


if __name__ == "__main__":
    main()
