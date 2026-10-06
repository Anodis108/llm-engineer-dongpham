# Module III, Bài 4, Section 1 — Multi-stage build.
#
# Stage "builder": có compiler + toàn bộ dev tooling để cài dependency (một số
# package Python cần biên dịch native extension). Stage "runtime": KHÔNG có gì
# ngoài Python + package đã cài sẵn — image cuối nhỏ hơn nhiều và không lộ
# compiler/toolchain ra production (giảm bề mặt tấn công).
#
# QUAN TRỌNG (bài học nhấn mạnh): model weights KHÔNG nằm trong image này.
# App chỉ gọi API (OpenAI Cloud / Ollama / vLLM qua base_url) — không có
# weight nào để bake vào image. Nếu sau này serve model local trong container,
# mount weights qua volume hoặc tải lúc khởi động, KHÔNG COPY vào layer build
# (mỗi lần đổi code lại phải build lại weight nặng hàng chục GB).

# ─── Stage 1: builder ─────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# requirements.txt đổi RẤT ít so với code — copy riêng để tận dụng layer cache
# của Docker: sửa app/*.py không phải cài lại dependency mỗi lần build.
COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ─── Stage 2: runtime ──────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user — container bị compromise thì không có quyền root trên host.
RUN groupadd --system app && useradd --system --gid app --home /app app

WORKDIR /app

# Chỉ copy package đã cài từ builder, không copy compiler/cache pip.
COPY --from=builder /install /usr/local

COPY app ./app
COPY data/legal_docs ./data/legal_docs

RUN chown -R app:app /app
USER app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# HEALTHCHECK dùng chính /health đã có — không cần thêm dependency (curl).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
