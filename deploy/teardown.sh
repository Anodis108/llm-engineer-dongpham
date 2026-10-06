#!/usr/bin/env bash
# Module III, Bài 6 — xoá sạch mọi thứ deploy.sh đã dựng, để chạy lại từ đầu.
#
#   PROJECT_ID=my-gcp-project ./deploy/teardown.sh
#
# Xoá theo ĐÚNG THỨ TỰ NGƯỢC với lúc dựng: VM trước (nó đang giữ IP tĩnh), rồi
# IP, rồi secret, rồi service account, rồi firewall. Đảo thứ tự sẽ làm gcloud
# báo lỗi phụ thuộc (`resourceInUseByAnotherResource`) — vẫn xoá được nhưng phải
# chạy lại vài lần.
#
# KHÔNG đụng tới: project GCP, billing account, repo GitHub, package trên
# ghcr.io, GitHub Secrets/Variables, trace trên LangSmith. Phần cuối file liệt
# kê những thứ phải xoá tay.
#
# Biến (mặc định khớp deploy.sh):
#   ZONE (asia-southeast1-b)   REGION (asia-southeast1)
#   VM_NAME (llm-app-vm)       ADDRESS_NAME (llm-app-ip)
#   SERVICE_ACCOUNT_NAME (llm-app-sa)   FIREWALL_RULE (allow-http-https)
#   SSH_KEY_PATH (~/.ssh/llm-app-deploy)
#   KEEP_SSH_KEY (false — true thì giữ lại key local, dùng khi chỉ muốn dựng lại VM)
#   FORCE (false — true thì bỏ qua bước xác nhận, dùng cho CI)
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Thiếu PROJECT_ID — export PROJECT_ID=your-gcp-project}"
ZONE="${ZONE:-asia-southeast1-b}"
REGION="${REGION:-asia-southeast1}"
VM_NAME="${VM_NAME:-llm-app-vm}"
ADDRESS_NAME="${ADDRESS_NAME:-llm-app-ip}"
SERVICE_ACCOUNT_NAME="${SERVICE_ACCOUNT_NAME:-llm-app-sa}"
FIREWALL_RULE="${FIREWALL_RULE:-allow-http-https}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/llm-app-deploy}"
KEEP_SSH_KEY="${KEEP_SSH_KEY:-false}"

SA_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

say()  { echo "==> $*"; }
skip() { echo "    (không có sẵn, bỏ qua)"; }

# ── Xác nhận — script này không hoàn tác được ─────────────────────────────────
if [ "${FORCE:-false}" != "true" ]; then
    cat <<EOF
Sẽ XOÁ VĨNH VIỄN trong project "$PROJECT_ID":
    VM            $VM_NAME  (zone $ZONE, xoá cả boot disk)
    IP tĩnh       $ADDRESS_NAME  (region $REGION)
    3 secret      llm-engineer-{openai-api-keys,tavily-api-key,langsmith-api-key}
    Service acct  $SA_EMAIL  (+ IAM binding)
    Firewall      $FIREWALL_RULE
    SSH key       $SSH_KEY_PATH  (trừ khi KEEP_SSH_KEY=true)

EOF
    printf "Gõ 'xoa' để xác nhận: "
    read -r answer
    [ "$answer" = "xoa" ] || { echo "Đã huỷ, không xoá gì."; exit 1; }
fi

say "[1/6] Xoá VM $VM_NAME"
# --delete-disks=boot: boot disk không tự xoá theo VM ở một số cấu hình, để mặc
# định cũng là 'boot' nhưng ghi rõ ra cho chắc.
gcloud compute instances delete "$VM_NAME" \
    --zone="$ZONE" --project="$PROJECT_ID" \
    --delete-disks=boot --quiet || skip

say "[2/6] Giải phóng IP tĩnh $ADDRESS_NAME"
# Phải sau bước 1: IP đang gắn vào VM thì không xoá được.
gcloud compute addresses delete "$ADDRESS_NAME" \
    --region="$REGION" --project="$PROJECT_ID" --quiet || skip

say "[3/6] Xoá 3 secret (xoá secret là xoá luôn mọi version)"
for secret_id in llm-engineer-openai-api-keys llm-engineer-tavily-api-key llm-engineer-langsmith-api-key; do
    printf '    %-38s ' "$secret_id"
    if gcloud secrets delete "$secret_id" --project="$PROJECT_ID" --quiet 2>/dev/null; then
        echo "đã xoá"
    else
        echo "không có, bỏ qua"
    fi
done

say "[4/6] Gỡ IAM binding rồi xoá service account"
# Gỡ binding trước: xoá SA mà còn binding sẽ để lại principal chết
# (deleted:serviceAccount:...) nằm trong policy của project.
gcloud projects remove-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/secretmanager.secretAccessor" \
    --condition=None --quiet >/dev/null 2>&1 || skip
gcloud iam service-accounts delete "$SA_EMAIL" \
    --project="$PROJECT_ID" --quiet || skip

say "[5/6] Xoá firewall rule $FIREWALL_RULE"
gcloud compute firewall-rules delete "$FIREWALL_RULE" \
    --project="$PROJECT_ID" --quiet || skip

say "[6/6] Key SSH cho CI"
if [ "$KEEP_SSH_KEY" = "true" ]; then
    echo "    KEEP_SSH_KEY=true — giữ lại $SSH_KEY_PATH"
elif [ -f "$SSH_KEY_PATH" ]; then
    rm -f "$SSH_KEY_PATH" "$SSH_KEY_PATH.pub"
    echo "    đã xoá $SSH_KEY_PATH{,.pub}"
else
    skip
fi

# File tạm deploy.sh ghi ra trong lúc chạy.
rm -f /tmp/llm-app.env /tmp/nginx-llm-app.conf /tmp/llm-app.service /tmp/llm-app-ssh-keys.txt

cat <<EOF

════════════════════════════════════════════════════════════════════════
 Xong phần GCP. Còn 3 thứ phải xoá tay.
════════════════════════════════════════════════════════════════════════

[1] GitHub → Settings → Secrets and variables → Actions
    Xoá secret  VM_SSH_KEY
    Xoá variable VM_HOST, VM_USER, APP_URL
    (Không xoá thì lần dựng lại sẽ dùng giá trị cũ — VM_HOST trỏ vào IP đã
     giải phóng, pipeline đỏ ở bước ssh với lỗi timeout khó hiểu.)

[2] Package trên ghcr.io — không bắt buộc
    Repo → Packages → llm-engineer-demo → Package settings → Delete.
    Để lại cũng không sao, chỉ là image cũ nằm đó. NHƯNG nếu bạn đã set
    package sang Public thì nó vẫn public sau khi xoá project.

[3] Trace trên LangSmith — không bắt buộc
    Vào project llm-engineer-demo trên smith.langchain.com xoá nếu muốn
    buổi demo sau bắt đầu từ trang sạch.

⚠  LET'S ENCRYPT CÓ RATE LIMIT — đọc trước khi dựng lại liên tục
   Certbot xin cert cho <IP>.sslip.io. Let's Encrypt giới hạn 5 chứng chỉ
   TRÙNG NHAU (cùng tập tên miền) mỗi tuần. Teardown rồi dựng lại giữ nguyên
   IP tĩnh nghĩa là xin lại đúng tên miền đó — lần thứ 6 trong tuần sẽ bị từ
   chối và bước 7 của deploy.sh đỏ.

   Cách né: xoá luôn IP tĩnh (script này làm rồi) để lần sau lấy IP MỚI →
   domain mới → không dính giới hạn. Muốn giữ IP cũ thì thêm --staging vào
   lệnh certbot khi thử, chỉ bỏ --staging ở lần chạy thật.
EOF
