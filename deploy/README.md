# Deploy — Module III, Bài 5 & 6 (Cloud Deployment + CI/CD)

Runbook triển khai `llm-engineer-demo` lên **Google Compute Engine** (VM chạy
liên tục — đúng lựa chọn bài học cho app có state cục bộ + cần kiểm soát
nginx/TLS trực tiếp, khác Cloud Run scale-to-zero hay GKE nhiều service).

> Bối cảnh lựa chọn hạ tầng (Section 1 — Cloud Landscape):
>
> | | Compute Engine | Cloud Run | GKE | Vertex AI |
> |---|---|---|---|---|
> | Dùng khi | 1 service, cần GPU/chạy liên tục | Stateless, traffic thất thường (scale-to-zero) | Nhiều service, cần orchestration | Serve model managed, không tự quản hạ tầng |
> | Ở đây | ✅ VM đơn, nginx + TLS tự quản | ❌ app dùng Qdrant `:memory:` — mỗi instance một DB rỗng | quá mức cần cho demo 1 service | không áp dụng (app tự host logic, không chỉ serve model) |

## Hai đường ống — chọn một

Repo có **hai** workflow deploy. Chúng dạy cùng một bài học, khác nhau ở phần
xác thực — chọn theo mục đích, không phải theo mức độ "xịn":

| | `deploy.yml` (demo) | `cd.yml` (production) |
|---|---|---|
| Xác thực | SSH key (1 secret) | Workload Identity Federation (0 key file) |
| Registry | `ghcr.io` (dùng `GITHUB_TOKEN`) | Artifact Registry (cần auth GCP) |
| Truy cập VM | `ssh` thường | `gcloud compute ssh` |
| Setup | ~5 phút | ~30 phút (pool + provider + IAM binding + Environment) |
| Mặc định | **bật** | **tắt** |
| Dùng khi | Lớp học, demo, thử nghiệm | Thật, chạy production |

**Chỉ một trong hai được bật tại một thời điểm.** Cả hai đều treo trên
`workflow_run` của `ci.yml`, nên bật cả hai thì mỗi push sẽ deploy hai lần lên
cùng một VM — hai tiến trình giành nhau đổi tag `latest` và restart service.
`cd.yml` vì vậy chỉ chạy khi biến `ENABLE_WIF_DEPLOY` = `'true'`
(Settings → Secrets and variables → Actions → Variables). Muốn chuyển sang
đường WIF: đặt biến đó, **và** xoá hoặc vô hiệu hoá `deploy.yml`.

Phần dưới mô tả **`deploy.yml`** — đường demo. `cd.yml` dùng chung
`release.sh` / `smoke.py` / `deploy.sh` nên hạ tầng y hệt, chỉ khác bước auth.

## Cấu trúc thư mục

```
deploy/
├── startup.sh       # cloud-init: cài Docker + nginx + certbot trên VM lần đầu
├── llm-app.service  # systemd unit: chạy container app, Restart=always
├── nginx.conf       # reverse proxy — CHỈ block :80, certbot tự nhân bản sang :443
├── deploy.sh        # BOOTSTRAP (chạy 1 lần từ máy bạn): VM, IAM, secret, nginx, TLS, SSH key
├── release.sh       # chạy TRÊN VM: deploy / smoke / rollback / status
├── smoke.py         # smoke test: /health → /admin/ingest → /chat, kiểm tra từ khoá
└── DEMO-CICD.md     # kịch bản buổi học (3 commit)
```

Phân chia trách nhiệm: **`deploy.sh` dựng hạ tầng một lần, `deploy.yml` đưa code
lên mỗi lần push.** Hạ tầng đổi rất ít, code đổi liên tục — trộn hai thứ vào một
script nghĩa là mỗi lần sửa một dòng code lại phải chạy lại toàn bộ phần
provision.

## Phần A — Bootstrap (chạy một lần, trước buổi học)

### Trước khi chạy

1. Cài `gcloud` CLI, login: `gcloud auth login && gcloud config set project YOUR_PROJECT_ID`.
2. Bật API cần thiết:
   ```bash
   gcloud services enable compute.googleapis.com secretmanager.googleapis.com \
       iam.googleapis.com
   ```
3. **Không cần domain.** Script tự suy tên miền từ IP tĩnh qua `sslip.io` (xem bước 4).

### Chạy

```bash
export PROJECT_ID=my-gcp-project
./deploy/deploy.sh
```

`deploy.sh` làm tuần tự (idempotent — chạy lại an toàn):

1. **Firewall** — chỉ mở `80`/`443` ra Internet, gắn network tag `llm-app`. Port
   app (8000) **không** có rule riêng — nó bind `127.0.0.1` trong container
   (xem `llm-app.service`), nginx là điểm vào public duy nhất.
2. **Service Account + IAM** — tạo SA riêng cho VM, cấp đúng 1 role
   `roles/secretmanager.secretAccessor` (least privilege — SA này ĐỌC được
   secret nhưng không tạo/xoá). VM dùng Application Default Credentials qua SA
   này, **không có key file JSON nào trên VM hay trong image**.
3. **Secret Manager** — tạo 3 secret rỗng (nếu chưa có). Script KHÔNG tự điền
   giá trị thật — bạn chạy tay, để secret không lọt vào lịch sử shell của CI/log.
4. **IP tĩnh + domain** — cấp IP tĩnh rồi suy `DOMAIN` dạng `<ip>.sslip.io`
   (ví dụ `34-87-12-5.sslip.io`). IP tĩnh là bắt buộc vì hostname mã hoá IP
   trong tên miền — IP đổi là cert HTTPS hết hiệu lực. Truyền `DOMAIN=...` nếu
   bạn có tên miền thật.
5. **Tạo VM** — `e2-small`, Debian 12, gắn IP tĩnh + SA, chạy `startup.sh` qua
   metadata (cài Docker/nginx/certbot).
6. **Copy config lên VM** — `nginx.conf` (đã thay `YOUR_DOMAIN_HERE`), `llm-app.service`,
   và `.env` production. `.env` này **KHÔNG chứa secret thật**, chỉ có
   `USE_SECRET_MANAGER=true` + `GCP_PROJECT_ID` để app tự đọc Secret Manager lúc
   khởi động (xem `app/config.py`). Service chỉ được `enable`, **không** `--now`:
   chưa có image nào trên máy, bản deploy đầu tiên do pipeline đẩy lên.
7. **HTTPS qua certbot** — chạy được vì bước 6 đã để nginx phục vụ HTTP với đúng
   `server_name`, còn DNS thì `sslip.io` resolve sẵn. certbot tự nhân bản block
   `listen 80` sang `listen 443 ssl` kèm chứng chỉ. Gói `certbot` của Debian cài
   kèm `certbot.timer` nên gia hạn tự động — không cần cron tay. Kiểm tra bằng
   `systemctl list-timers certbot`.
8. **SSH key cho GitHub Actions** — sinh keypair riêng cho CI (không dùng lại key
   cá nhân của bạn), thêm public key vào VM metadata dưới user `deploy`. Đọc key
   cũ ra trước khi ghi để không tự khoá mình khỏi VM.

### Ba bước tay còn lại

Script in ra ở cuối, tóm tắt:

```bash
# [1] Nạp secret thật (đang là chuỗi "PLACEHOLDER")
printf '%s' "sk-..." | gcloud secrets versions add llm-engineer-openai-api-keys --data-file=-

# [2] GitHub → Settings → Secrets and variables → Actions
#     Secrets:   VM_SSH_KEY = nội dung ~/.ssh/llm-app-deploy (copy cả 2 dòng)
#     Variables: VM_HOST = <IP tĩnh>   VM_USER = deploy   APP_URL = https://<domain>

# [3] Kiểm tra HẠ TẦNG — app chưa chạy, image do pipeline đẩy lên ở lần push đầu
ssh -i ~/.ssh/llm-app-deploy deploy@<VM_HOST> \
    'sudo docker --version; sudo ls /opt/llm-app/; systemctl is-active nginx'
curl -sI https://<domain>/ | head -1
```

### Đọc kết quả kiểm tra cho đúng

Sau bootstrap, **hai thứ trông như lỗi nhưng thực ra là đúng**:

| Bạn thấy | Nghĩa là |
|---|---|
| `curl https://<domain>/` → **502** | nginx sống và đang proxy tới `127.0.0.1:8000`, nhưng chưa có container nào listen. Sẽ hết sau lần deploy đầu. |
| `sudo /opt/llm-app/release.sh` → **command not found** | `release.sh` và `smoke.py` do pipeline `scp` lên **mỗi lần deploy** (để rollback chạy được cả khi workflow đổi). Bootstrap cố ý không copy. |

Cái **phải** đúng ở giai đoạn này: SSH vào được, `sudo` không hỏi mật khẩu,
`docker --version` chạy, `/opt/llm-app/.env` tồn tại, `nginx` active.

`systemctl is-active llm-app` sẽ trả `inactive` — cũng đúng: service đã `enable`
nhưng chưa `start` vì chưa có image.

## Phần B — Vòng lặp hằng ngày (mỗi push)

```
git push main
  └─ ci.yml        prompt lint → ruff → pytest        (chạy trước, rẻ)
       └─ deploy.yml  (chỉ khi ci xanh)
            ├─ build → ghcr.io/<owner>/<repo>:<git-sha>
            ├─ scp release.sh + smoke.py lên VM
            ├─ release.sh deploy <image_uri>   ← pull, đổi tag, restart
            ├─ release.sh smoke                ← hỏi 1 câu đã biết đáp án
            └─ if: failure() → release.sh rollback về last-known-good
```

Ba nguyên tắc thể hiện trong đó:

- **Chỉ deploy khi CI xanh.** `workflow_run` trên `ci.yml`, không phải `on: push`
  — nếu không sẽ deploy cả commit chưa qua test.
- **Image bất biến, tag = git SHA.** Deploy = trỏ tag, rollback = trỏ ngược lại.
  Không bao giờ deploy tag `latest` từ registry.
- **Promote theo smoke test, không theo trạng thái container.** `/health` trả 200
  kể cả khi Qdrant rỗng và `/chat` trả 500. Container "healthy" ≠ deploy thành công.

## Bắt buộc: đặt package trên ghcr.io thành Public

**Package trên GHCR mặc định là PRIVATE, kể cả khi repo là public.** Đây là bẫy
hay gặp nhất khi dựng đường demo này, và triệu chứng rất dễ chẩn đoán nhầm: CI
xanh, build + push xanh, rồi bước deploy chết với `denied` khi VM `docker pull`.

VM pull ẩn danh, nên package phải public. Làm **một lần**, sau lần build đầu tiên:

> GitHub → repo → **Packages** (cột phải) → chọn package `llm-engineer-demo`
> → **Package settings** → kéo xuống **Danger Zone** → **Change visibility**
> → **Public** → gõ tên package để xác nhận.

Lần deploy đầu tiên sẽ đỏ ở bước này — đó là dự kiến, không phải lỗi cấu hình.
Đặt public xong thì **Re-run** workflow, nó xanh.

Cảnh báo của GitHub: đã chuyển public thì **không chuyển lại private được**.

### Vì sao không dùng token thay vì public?

Vì như vậy phải tạo một PAT dài hạn và nhét vào Secret Manager — đúng cái
anti-pattern mà Bài 5 dạy tránh. Image public + không credential nào là lựa chọn
vừa đơn giản hơn vừa đúng bài học hơn.

Nếu buộc phải để image private: tạo PAT (classic) scope `read:packages`, lưu vào
Secret Manager, và thêm bước `docker login ghcr.io` vào `release.sh` trước khi
pull. Repo private cũng cần thêm bước này cho chính `deploy.yml`.

## Rollback

Ba cách, từ tự động tới tay:

```bash
# 1. Tự động — deploy.yml tự làm khi smoke fail. Không cần thao tác gì.

# 2. Tay trên VM (chạy được cả khi GitHub Actions sập)
gcloud compute ssh llm-app-vm --zone=asia-southeast1-b --command="sudo /opt/llm-app/release.sh rollback"

# 3. Xem trạng thái hiện tại
gcloud compute ssh llm-app-vm --zone=asia-southeast1-b --command="sudo /opt/llm-app/release.sh status"
```

`release.sh` giữ hai file trạng thái trên VM: `.current` (tag đang chạy) và
`.lkg` (last-known-good). `.lkg` được ghi **trước** khi đổi tag, nên luôn có đích
để quay về kể cả khi bước ngay sau đó hỏng.

**Rollback không cần build lại** — image cũ vẫn nằm trên registry theo tag SHA,
chỉ đổi con trỏ. Mất vài giây thay vì vài phút.

## Xem log

```bash
gcloud compute ssh llm-app-vm --zone=asia-southeast1-b --command="sudo journalctl -u llm-app -f"
```

Restart sau khi đổi secret (không cần deploy lại image):

```bash
gcloud compute ssh llm-app-vm --zone=asia-southeast1-b --command="sudo systemctl restart llm-app"
```

## Secret Manager — vì sao không dùng key file?

Bài học Section 4 nhấn mạnh 2 lỗi phổ biến: (1) commit `.env` chứa key thật vào
git, (2) bake secret vào Docker image lúc build (`ARG`/`ENV` trong Dockerfile —
lộ qua `docker history`). Cách ở đây né cả hai:

- Secret **không nằm trong image** — image chỉ có code, không có giá trị bí mật nào.
- Secret **không nằm trong `.env` deploy lên VM** — `.env` chỉ có
  `USE_SECRET_MANAGER=true` + `GCP_PROJECT_ID`, giá trị thật đọc từ Secret
  Manager lúc `app/config.py::get_settings()` chạy (runtime, mỗi lần container
  khởi động).
- Auth vào Secret Manager qua **Service Account gắn sẵn vào VM** (Application
  Default Credentials) — không cần biến `GOOGLE_APPLICATION_CREDENTIALS` trỏ
  tới key file nào.
- Local dev **không cần Secret Manager** — `USE_SECRET_MANAGER=false` (mặc
  định trong `.env.example`), app đọc thẳng `.env` như trước giờ. Xem test
  `tests/test_secret_manager.py` — verify cả 2 luồng bằng mock, không gọi GCP thật.

## Sự cố thường gặp

| Triệu chứng | Nguyên nhân | Xử lý |
|---|---|---|
| `nginx -t` báo `no "ssl_certificate" is defined` | `nginx.conf` có sẵn `listen 443 ssl` mà chưa có cert | Đừng viết block 443 vào `nginx.conf` — để certbot tự thêm |
| `deploy.sh` chết ở bước 6 | Lỗi trên, hoặc `systemctl enable --now` khi chưa có image | Kiểm tra `nginx.conf` chỉ có block `:80` |
| certbot fail `Could not find a virtual host` | nginx chưa reload với `server_name` đúng | Chạy lại bước 6 rồi bước 7 |
| Smoke fail ở `check_answer` | Qdrant rỗng (app `QDRANT_URL=:memory:`) | `release.sh smoke` đã tự ingest — nếu vẫn fail, xem log container |
| VM pull image `denied` | Package trên ghcr.io còn private (mặc định) | Đặt Public — xem mục "Bắt buộc: đặt package trên ghcr.io thành Public" |
| Deploy `ssh: Permission denied` | `VM_SSH_KEY` sai hoặc key chưa vào VM metadata | `ssh -i ~/.ssh/llm-app-deploy deploy@<VM_HOST> echo OK` để tự kiểm |
| Mỗi request chậm bất thường | `MONITORING_ENABLED=true` mà `LANGSMITH_API_KEY` còn là `PLACEHOLDER` | Nạp key thật, hoặc để `MONITORING_ENABLED=false` |

## Chi phí

`e2-small` ở `asia-southeast1` ≈ $0.025/giờ. IP tĩnh tốn thêm ~$0.004/giờ khi
VM **đang dừng** (GCE tính phí IP nhàn rỗi). Nhớ dừng VM sau buổi học:

```bash
gcloud compute instances stop llm-app-vm --zone=asia-southeast1-b
```

Muốn free tier thật: `e2-micro` miễn phí nhưng chỉ ở `us-central1`/`us-west1`/`us-east1`.
Truyền `ZONE=us-central1-a MACHINE_TYPE=e2-micro ./deploy/deploy.sh`. Đổi khu vực
nghĩa là latency tới OpenAI tăng — với lớp học thì không đáng kể.
