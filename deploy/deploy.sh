#!/usr/bin/env bash
# Module III, Bài 5, Section 5-6 — dựng hạ tầng lên Compute Engine (chạy TỪ MÁY LOCAL).
#
# Đây là script BOOTSTRAP: chạy MỘT LẦN trước buổi học để dựng VM, secret, IP
# tĩnh, nginx và HTTPS. Nó KHÔNG deploy code — việc đó do GitHub Actions làm
# (.github/workflows/deploy.yml) mỗi khi có commit mới lên main.
#
# Tách như vậy là có chủ đích: hạ tầng đổi rất ít, code đổi liên tục. Trộn hai
# thứ vào một script nghĩa là mỗi lần sửa một dòng code lại phải chạy lại toàn
# bộ phần provision — chậm, và có nguy cơ đụng vào thứ đang chạy tốt.
#
# Luồng tổng thể:
#
#   [máy bạn]  deploy.sh          → VM + secret + nginx + cert + SSH key   (1 lần)
#   [GitHub]   deploy.yml         → build → GHCR → ssh VM → smoke         (mỗi push)
#
# Yêu cầu: `gcloud` CLI đã login (`gcloud auth login`) + set project
# (`gcloud config set project YOUR_PROJECT_ID`).
#
# Dùng:
#   PROJECT_ID=my-gcp-project ./deploy/deploy.sh
#
# Biến tuỳ chọn (mặc định trong ngoặc):
#   ZONE (asia-southeast1-b)   REGION (asia-southeast1)   VM_NAME (llm-app-vm)
#   DOMAIN (tự suy từ IP qua sslip.io — xem bước 4)
#   CERTBOT_EMAIL (không có → đăng ký cert không kèm email)
#   MONITORING_ENABLED (false — bật true chỉ khi đã nạp LANGSMITH_API_KEY thật)
#
# Idempotent theo từng bước — chạy lại an toàn.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Thiếu PROJECT_ID — export PROJECT_ID=your-gcp-project}"
ZONE="${ZONE:-asia-southeast1-b}"
REGION="${REGION:-asia-southeast1}"
VM_NAME="${VM_NAME:-llm-app-vm}"
ADDRESS_NAME="${ADDRESS_NAME:-llm-app-ip}"
SERVICE_ACCOUNT_NAME="${SERVICE_ACCOUNT_NAME:-llm-app-sa}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/llm-app-deploy}"
MONITORING_ENABLED="${MONITORING_ENABLED:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> [1/8] Firewall — chỉ mở 80/443 ra Internet (Section 5)"
gcloud compute firewall-rules create allow-http-https \
    --project="$PROJECT_ID" \
    --allow=tcp:80,tcp:443 \
    --target-tags=llm-app \
    --direction=INGRESS \
    --description="HTTP/HTTPS cho llm-app — port 8000 KHÔNG expose, chỉ bind 127.0.0.1" \
    || echo "  (rule đã tồn tại, bỏ qua)"

echo "==> [2/8] Service Account — thay key file bằng IAM (Section 4)"
gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
    --project="$PROJECT_ID" \
    --display-name="LLM App Compute Engine SA" \
    || echo "  (service account đã tồn tại, bỏ qua)"

SA_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# Chỉ quyền ĐỌC secret — nguyên tắc least privilege, SA này KHÔNG được tạo/xoá secret.
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/secretmanager.secretAccessor" \
    --condition=None >/dev/null

echo "==> [3/8] Secret Manager — tạo secret rỗng nếu chưa có (Section 4)"
for secret_id in llm-engineer-openai-api-keys llm-engineer-tavily-api-key llm-engineer-langsmith-api-key; do
    gcloud secrets describe "$secret_id" --project="$PROJECT_ID" &> /dev/null && continue
    printf "PLACEHOLDER" | gcloud secrets create "$secret_id" \
        --project="$PROJECT_ID" --data-file=- --replication-policy=automatic
    echo "  Đã tạo secret rỗng: $secret_id"
done

echo "==> [4/8] IP tĩnh + domain"
# IP tĩnh là BẮT BUỘC ở đây, không phải cho đẹp: hostname sslip.io mã hoá IP
# ngay trong tên miền (34.87.12.5 -> 34-87-12-5.sslip.io). IP đổi sau một lần
# stop/start VM là hostname trỏ sai chỗ và cert HTTPS hết hiệu lực.
if ! gcloud compute addresses describe "$ADDRESS_NAME" \
        --region="$REGION" --project="$PROJECT_ID" &> /dev/null; then
    gcloud compute addresses create "$ADDRESS_NAME" \
        --region="$REGION" --project="$PROJECT_ID"
fi

VM_IP="$(gcloud compute addresses describe "$ADDRESS_NAME" \
    --region="$REGION" --project="$PROJECT_ID" --format='get(address)')"

# sslip.io là DNS wildcard công khai: tên miền chứa IP thì tự resolve về IP đó,
# và nằm trong Public Suffix List nên Let's Encrypt cấp cert bình thường. Nghĩa
# là có HTTPS đầy đủ mà không cần mua hay cấu hình tên miền nào.
if [ -z "${DOMAIN:-}" ]; then
    DOMAIN="${VM_IP//./-}.sslip.io"
    echo "  DOMAIN không được đặt — suy từ IP tĩnh: $DOMAIN"
fi
echo "  IP tĩnh: $VM_IP   Domain: $DOMAIN"

echo "==> [5/8] Tạo VM (nếu chưa có) — gắn Service Account + IP tĩnh"
if ! gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --project="$PROJECT_ID" &> /dev/null; then
    gcloud compute instances create "$VM_NAME" \
        --project="$PROJECT_ID" --zone="$ZONE" \
        --machine-type="${MACHINE_TYPE:-e2-small}" \
        --tags=llm-app \
        --address="$ADDRESS_NAME" \
        --service-account="$SA_EMAIL" \
        --scopes=cloud-platform \
        --image-family=debian-12 --image-project=debian-cloud \
        --metadata-from-file=startup-script="${SCRIPT_DIR}/startup.sh"
    echo "  VM tạo mới — đợi ~60s để startup-script cài Docker/nginx xong."
    sleep 60
else
    echo "  VM đã tồn tại, bỏ qua tạo mới."
    echo "  LƯU Ý: nếu VM cũ chưa gắn IP tĩnh $ADDRESS_NAME, phải xoá và tạo lại"
    echo "         (GCE không cho đổi IP của VM đang tồn tại)."
fi

echo "==> [6/8] Copy config lên VM (systemd unit, nginx, .env production)"
sed "s/YOUR_DOMAIN_HERE/${DOMAIN}/g" "${SCRIPT_DIR}/nginx.conf" > /tmp/nginx-llm-app.conf

gcloud compute scp "${SCRIPT_DIR}/llm-app.service" "${VM_NAME}:/tmp/llm-app.service" \
    --zone="$ZONE" --project="$PROJECT_ID"
gcloud compute scp /tmp/nginx-llm-app.conf "${VM_NAME}:/tmp/nginx-llm-app.conf" \
    --zone="$ZONE" --project="$PROJECT_ID"

# .env production KHÔNG chứa secret thật (USE_SECRET_MANAGER=true -> app tự đọc
# Secret Manager) — chỉ cấu hình non-sensitive + trỏ project id.
#
# MONITORING_ENABLED mặc định false: secret LangSmith vừa tạo còn là placeholder
# "PLACEHOLDER", bật lên sẽ khiến mỗi request chờ một lần gọi mạng thất bại.
# Nạp key thật rồi hãy bật (xem cuối file này).
cat > /tmp/llm-app.env <<EOF
USE_SECRET_MANAGER=true
GCP_PROJECT_ID=${PROJECT_ID}
LLM_BACKEND=openai
LLM_MODEL=gpt-4o-mini
LLM_MAX_CONCURRENCY=20
QDRANT_URL=:memory:
MONITORING_ENABLED=${MONITORING_ENABLED}
LANGSMITH_PROJECT=llm-engineer-demo
PROMPT_REGISTRY=local
APP_NAME=Vietnamese Legal Assistant
EOF
gcloud compute scp /tmp/llm-app.env "${VM_NAME}:/tmp/llm-app.env" \
    --zone="$ZONE" --project="$PROJECT_ID"

gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT_ID" --command="
    sudo mkdir -p /opt/llm-app &&
    sudo mv /tmp/llm-app.env /opt/llm-app/.env &&
    sudo chmod 600 /opt/llm-app/.env &&
    sudo chown root:root /opt/llm-app/.env &&
    sudo mv /tmp/llm-app.service /etc/systemd/system/llm-app.service &&
    sudo mkdir -p /etc/nginx/sites-available &&
    sudo mv /tmp/nginx-llm-app.conf /etc/nginx/sites-available/llm-app &&
    sudo systemctl daemon-reload &&
    sudo systemctl enable llm-app &&
    sudo ln -sf /etc/nginx/sites-available/llm-app /etc/nginx/sites-enabled/llm-app &&
    sudo rm -f /etc/nginx/sites-enabled/default &&
    sudo nginx -t && sudo systemctl reload nginx
"
# KHÔNG dùng `systemctl enable --now`: chưa có image nào trên máy (bản deploy đầu
# tiên do GitHub Actions đẩy lên), nên start ngay sẽ fail. Chỉ enable — service
# sẽ được `release.sh deploy` restart ở lần deploy đầu tiên.

echo "==> [7/8] HTTPS qua certbot (Let's Encrypt)"
# Chạy được vì bước 6 đã để nginx phục vụ HTTP với đúng server_name, còn DNS thì
# sslip.io đã resolve sẵn về IP tĩnh. certbot sẽ tự nhân bản block `listen 80`
# trong nginx.conf sang một block `listen 443 ssl` kèm chứng chỉ.
CERTBOT_ARGS=(-d "$DOMAIN" --non-interactive --agree-tos --redirect)
if [ -n "${CERTBOT_EMAIL:-}" ]; then
    CERTBOT_ARGS+=(-m "$CERTBOT_EMAIL")
else
    # Không có email thật thì đừng bịa `admin@<domain>` — địa chỉ đó không tồn
    # tại, và Let's Encrypt sẽ gửi cảnh báo hết hạn vào hư không.
    CERTBOT_ARGS+=(--register-unsafely-without-email)
fi

gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT_ID" \
    --command="sudo certbot --nginx ${CERTBOT_ARGS[*]}"

echo "==> [8/8] SSH key cho GitHub Actions"
# Key RIÊNG cho CI, không dùng lại key cá nhân của bạn: thu hồi được bằng cách
# xoá key khỏi VM metadata mà không ảnh hưởng tới quyền truy cập của bạn.
if [ ! -f "$SSH_KEY_PATH" ]; then
    ssh-keygen -t ed25519 -f "$SSH_KEY_PATH" -N "" -C "github-actions-deploy" >/dev/null
    echo "  Đã tạo key mới: $SSH_KEY_PATH"
else
    echo "  Dùng key đã có: $SSH_KEY_PATH"
fi

PUBKEY="$(cat "${SSH_KEY_PATH}.pub")"
# GCE nhận key qua metadata `ssh-keys`, định dạng mỗi dòng `user:ssh-ed25519 AAAA...`
# Đọc key cũ ra trước rồi mới ghi lại — `add-metadata` ghi ĐÈ cả khoá ssh-keys,
# nên ghi thẳng sẽ xoá mất key của chính bạn và bạn tự khoá mình khỏi VM.
EXISTING_KEYS="$(gcloud compute instances describe "$VM_NAME" \
    --zone="$ZONE" --project="$PROJECT_ID" --format=json \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print(next((i["value"] for i in d.get("metadata",{}).get("items",[]) if i["key"]=="ssh-keys"), ""))' \
    2>/dev/null || true)"

if grep -qF "$PUBKEY" <<<"$EXISTING_KEYS"; then
    echo "  Key đã có trên VM, bỏ qua."
else
    printf '%s\ndeploy:%s\n' "$EXISTING_KEYS" "$PUBKEY" | sed '/^$/d' > /tmp/llm-app-ssh-keys.txt
    gcloud compute instances add-metadata "$VM_NAME" \
        --zone="$ZONE" --project="$PROJECT_ID" \
        --metadata-from-file=ssh-keys=/tmp/llm-app-ssh-keys.txt >/dev/null
    rm -f /tmp/llm-app-ssh-keys.txt
    echo "  Đã thêm public key vào VM (user: deploy)"
fi

cat <<EOF

════════════════════════════════════════════════════════════════════════
 Xong phần hạ tầng. Còn 3 bước tay trước khi pipeline chạy được.
════════════════════════════════════════════════════════════════════════

[1] Nạp secret THẬT (secret đang là chuỗi "PLACEHOLDER"):
      printf '%s' "sk-..." | gcloud secrets versions add llm-engineer-openai-api-keys --data-file=-
      printf '%s' "tvly-..." | gcloud secrets versions add llm-engineer-tavily-api-key --data-file=-

[2] Khai báo trên GitHub (Settings → Secrets and variables → Actions):
      Secrets:  VM_SSH_KEY  = nội dung file dưới đây (copy cả 2 dòng BEGIN/END)
                  pbcopy < ${SSH_KEY_PATH}
      Variables: VM_HOST = ${VM_IP}
                 VM_USER = deploy
                 APP_URL = https://${DOMAIN}

[3] Kiểm tra HẠ TẦNG (app chưa chạy — image do pipeline đẩy lên ở lần push đầu):
      ssh -i ${SSH_KEY_PATH} deploy@${VM_IP} \\
          'sudo docker --version; sudo ls /opt/llm-app/; systemctl is-active nginx'
      curl -sI https://${DOMAIN}/ | head -1

    Lệnh curl trả 502 là ĐÚNG ở bước này, không phải lỗi: nginx đã sống và đang
    proxy tới 127.0.0.1:8000, nhưng chưa có container nào listen ở đó.
    /opt/llm-app/release.sh cũng CHƯA có — pipeline scp nó lên mỗi lần deploy.

 Bước cuối: push một commit lên main → tab Actions → xem pipeline tự deploy.
 Khi đó /health mới thật sự trả 200.

 ⚠ Lần deploy đầu sẽ ĐỎ ở bước docker pull với `denied`: package trên ghcr.io
   mặc định là PRIVATE kể cả khi repo public, mà VM pull ẩn danh. Vào
   repo → Packages → Package settings → Danger Zone → Change visibility → Public,
   rồi Re-run workflow. Chi tiết: deploy/README.md.
EOF
