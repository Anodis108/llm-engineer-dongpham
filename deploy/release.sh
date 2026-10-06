#!/usr/bin/env bash
# Release/rollback trên VM — Module III, Bài 6, Section 4 (Deployment Strategies).
#
# Script này chạy TRÊN VM (cd.yml gọi qua `gcloud compute ssh`). Tách khỏi
# cd.yml vì 2 lý do:
#   1. Logic rollback phải chạy được cả khi GitHub Actions sập — thao tác viên
#      gõ tay `release.sh rollback` trên VM là xong.
#   2. Bash trong YAML không test được; bash trong file thì `bash -n` được.
#
# Mô hình: image bất biến theo git SHA. "Deploy" = đổi tag `latest` trỏ sang
# SHA mới rồi restart. "Rollback" = trỏ `latest` về SHA của bản last-known-good.
# KHÔNG build lại gì cả — nên rollback mất vài giây.
#
#   ./release.sh deploy <image_uri_theo_sha>
#   ./release.sh smoke [đường_dẫn_smoke.py]
#   ./release.sh rollback
#   ./release.sh status
set -euo pipefail

APP_DIR=/opt/llm-app
LOCAL_IMAGE=llm-engineer-demo
CURRENT_FILE="$APP_DIR/.current"
LKG_FILE="$APP_DIR/.lkg"
SERVICE=llm-app
CONTAINER=llm-app

log() { echo "==> $*"; }
die() { echo "::error::$*" >&2; exit 1; }

# Chờ service trả /health thật, không chỉ "systemctl start trả về 0".
wait_healthy() {
    local tries=${1:-30}
    for _ in $(seq "$tries"); do
        if curl -fsS --max-time 3 http://127.0.0.1:8000/health >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done
    return 1
}

write_tag() {  # write_tag <file> <tag>
    echo "$2" | sudo tee "$1" >/dev/null
}

do_deploy() {
    local image_uri="$1"
    [ -n "$image_uri" ] || die "thiếu image_uri"

    local new_tag="${image_uri##*:}"
    local old_tag=""
    [ -f "$CURRENT_FILE" ] && old_tag="$(sudo cat "$CURRENT_FILE")"

    log "Deploy $new_tag (bản đang chạy: ${old_tag:-chưa có})"

    # Ghi LKG TRƯỚC khi đổi tag: nếu bước sau hỏng, rollback vẫn có đích.
    if [ -n "$old_tag" ] && [ "$old_tag" != "$new_tag" ]; then
        write_tag "$LKG_FILE" "$old_tag"
        log "Đã ghi last-known-good = $old_tag"
    fi

    sudo docker pull "$image_uri"
    sudo docker tag "$image_uri" "$LOCAL_IMAGE:latest"
    write_tag "$CURRENT_FILE" "$new_tag"
    # Ghi lại đường dẫn registry (bỏ tag) để `rollback` biết pull image cũ từ đâu
    # khi nó đã bị prune khỏi máy.
    write_tag "$APP_DIR/.registry_image" "${image_uri%:*}"

    sudo systemctl restart "$SERVICE"

    if wait_healthy 30; then
        log "Service healthy với $new_tag"
    else
        # Không tự rollback ở đây: cd.yml còn phải chạy smoke test (kiểm tra
        # chất lượng câu trả lời) rồi mới quyết định. Rollback sớm sẽ che mất
        # lỗi mà smoke test cần báo.
        die "Service KHÔNG healthy sau 60s với $new_tag"
    fi
}

do_rollback() {
    [ -f "$LKG_FILE" ] || die "chưa có last-known-good — không rollback được"
    local lkg_tag
    lkg_tag="$(sudo cat "$LKG_FILE")"
    [ -n "$lkg_tag" ] || die "file .lkg rỗng"

    log "Rollback về last-known-good: $lkg_tag"

    # Image có thể đã bị prune khỏi máy → thử pull lại theo tag đã ghi.
    local image_uri
    image_uri="$(sudo cat "$APP_DIR/.registry_image" 2>/dev/null || true)"
    [ -n "$image_uri" ] || die "thiếu $APP_DIR/.registry_image (đường dẫn image trên registry)"

    if ! sudo docker image inspect "${image_uri%:*}:$lkg_tag" >/dev/null 2>&1; then
        log "Image $lkg_tag không còn trên máy — pull lại"
        sudo docker pull "${image_uri%:*}:$lkg_tag"
    fi

    sudo docker tag "${image_uri%:*}:$lkg_tag" "$LOCAL_IMAGE:latest"
    write_tag "$CURRENT_FILE" "$lkg_tag"
    sudo systemctl restart "$SERVICE"

    wait_healthy 30 || die "Rollback xong nhưng service vẫn không healthy"
    log "Đã rollback về $lkg_tag và service healthy"
}

do_smoke() {
    # Chạy smoke test BẰNG python trong chính container vừa deploy. Lý do không
    # cài Python + httpx lên VM: container đã có sẵn cả hai, và như vậy ta test
    # đúng artifact sẽ phục vụ người dùng — không phải một môi trường song song.
    local src="${1:-/opt/llm-app/smoke.py}"
    [ -f "$src" ] || die "không thấy $src"

    sudo docker cp "$src" "$CONTAINER:/tmp/smoke.py"
    # --skip-ingest=false: bản deploy mới luôn khởi động với Qdrant RỖNG
    # (QDRANT_URL=:memory:), nên phải nạp dữ liệu trước khi hỏi câu kiểm tra.
    sudo docker exec "$CONTAINER" python /tmp/smoke.py --url http://127.0.0.1:8000
}

do_status() {
    echo "current = $(sudo cat "$CURRENT_FILE" 2>/dev/null || echo '(chưa có)')"
    echo "lkg     = $(sudo cat "$LKG_FILE" 2>/dev/null || echo '(chưa có)')"
    sudo systemctl is-active "$SERVICE" || true
    sudo docker images "$LOCAL_IMAGE" --format '{{.Repository}}:{{.Tag}}  {{.CreatedSince}}'
}

case "${1:-}" in
    deploy)   shift; do_deploy "${1:-}" ;;
    smoke)    shift; do_smoke "${1:-}" ;;
    rollback) do_rollback ;;
    status)   do_status ;;
    *) die "dùng: $0 {deploy <image_uri>|smoke [path]|rollback|status}" ;;
esac
