#!/usr/bin/env bash
# Module III, Bài 5, Section 2 — cloud-init / startup script cho Compute Engine VM.
#
# Chạy MỘT LẦN khi VM khởi tạo lần đầu (deploy.sh truyền file này qua
# `--metadata-from-file startup-script=deploy/startup.sh` lúc `gcloud compute
# instances create`). Cài Docker + nginx + certbot, tạo thư mục cho app.
#
# KHÔNG bake OPENAI_API_KEYS/secrets vào script này hay vào image — app đọc
# secret lúc runtime qua GCP Secret Manager (USE_SECRET_MANAGER=true trong
# /opt/llm-app/.env, xem app/config.py::_load_secret_overrides) hoặc qua
# .env được deploy.sh copy riêng (KHÔNG commit .env vào git).
#
# THỨ TỰ QUAN TRỌNG: script này chạy ngay khi VM boot, TRƯỚC khi deploy.sh kịp
# copy nginx.conf / systemd unit lên. Nên mọi bước phụ thuộc vào file đó phải
# có kiểm tra tồn tại — nếu không, `set -e` sẽ giết script giữa chừng và VM
# boot xong trong trạng thái thiếu nửa cấu hình.
set -euo pipefail

echo "[startup] Cài Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sh
    systemctl enable --now docker
fi

echo "[startup] Cài nginx + certbot..."
apt-get update -y
apt-get install -y nginx certbot python3-certbot-nginx

mkdir -p /opt/llm-app
mkdir -p /var/www/certbot

echo "[startup] Kiểm tra systemd unit..."
# deploy.sh copy deploy/llm-app.service vào /etc/systemd/system/ SAU bước này,
# nên ở lần boot đầu tiên unit chưa tồn tại — `enable` sẽ fail. Bỏ qua là đúng:
# deploy.sh sẽ enable lại sau khi copy unit vào.
systemctl daemon-reload
if [ -f /etc/systemd/system/llm-app.service ]; then
    systemctl enable llm-app.service || true
else
    echo "[startup]   chưa có llm-app.service (deploy.sh copy sau) — bỏ qua"
fi

echo "[startup] Kiểm tra nginx..."
# Cùng lý do: nginx.conf được deploy.sh copy sau. Ở lần boot đầu chưa có file,
# mà symlink vào một file không tồn tại sẽ làm `nginx -t` fail.
if [ -f /etc/nginx/sites-available/llm-app ]; then
    ln -sf /etc/nginx/sites-available/llm-app /etc/nginx/sites-enabled/llm-app
    rm -f /etc/nginx/sites-enabled/default
    nginx -t && systemctl reload nginx
else
    echo "[startup]   chưa có nginx config (deploy.sh copy sau) — bỏ qua"
fi

echo "[startup] Xong. deploy.sh sẽ cấu hình .env, nginx và HTTPS ở bước sau."
