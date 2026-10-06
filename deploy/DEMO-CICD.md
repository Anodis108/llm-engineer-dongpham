# CI/CD trên Google Compute Engine

---

## Chuẩn bị

### 1. Bootstrap hạ tầng

```bash
export PROJECT_ID=llm-engineer-demo
./deploy/deploy.sh
```

### 2. Nạp secret thật

Ba secret vừa tạo còn là chuỗi `PLACEHOLDER`. Phải nạp giá trị thật — nếu không,
`/health` vẫn trả 200 nhưng mọi request gọi OpenAI sẽ hỏng.

```bash
read -rs "OPENAI_KEY?OpenAI API key (sk-...): "; echo
printf '%s' "$OPENAI_KEY" | gcloud secrets versions add llm-engineer-openai-api-keys \
    --project="$PROJECT_ID" --data-file=-
unset OPENAI_KEY

read -rs "TAVILY_KEY?Tavily API key (tvly-...): "; echo
printf '%s' "$TAVILY_KEY" | gcloud secrets versions add llm-engineer-tavily-api-key \
    --project="$PROJECT_ID" --data-file=-
unset TAVILY_KEY

read -rs "LANGSMITH_KEY?LangSmith API key (lsv2_...): "; echo
printf '%s' "$LANGSMITH_KEY" | gcloud secrets versions add llm-engineer-langsmith-api-key \
    --project="$PROJECT_ID" --data-file=-
unset LANGSMITH_KEY
```

`read -rs` để key không lọt vào lịch sử shell. Kiểm tra lại:

```bash
for s in openai-api-keys tavily-api-key langsmith-api-key; do
    printf '%-18s ' "$s"
    gcloud secrets versions access latest --secret="llm-engineer-$s" --project="$PROJECT_ID" \
        | grep -qx PLACEHOLDER && echo "CHƯA NẠP" || echo "đã nạp"
done
```

### 3. Bật monitoring

`.env` trên VM do `deploy.sh` ghi, mặc định `MONITORING_ENABLED=false` — trace
LangSmith là no-op. Bật lên:

```bash
export ZONE=asia-southeast1-b
gcloud compute ssh llm-app-vm --zone="$ZONE" --project="$PROJECT_ID" --command='
    set -e
    sudo sed -i "s/^MONITORING_ENABLED=.*/MONITORING_ENABLED=true/" /opt/llm-app/.env
    sudo grep "^MONITORING_ENABLED=" /opt/llm-app/.env
    sudo systemctl restart llm-app
'
```

Phải in ra `MONITORING_ENABLED=true`. Chạy lại `deploy.sh` sẽ ghi đè `.env` về
`false` — lần sau dùng `MONITORING_ENABLED=true ./deploy/deploy.sh`.

### 4. Khai báo trên GitHub

Settings → Secrets and variables → Actions:

| Loại | Tên | Giá trị |
|---|---|---|
| Secret | `VM_SSH_KEY` | `pbcopy < ~/.ssh/llm-app-deploy` |
| Variable | `VM_HOST` | IP tĩnh — `gcloud compute addresses describe llm-app-ip --region=asia-southeast1 --format='get(address)'` |
| Variable | `VM_USER` | `deploy` |
| Variable | `APP_URL` | `https://<IP-đổi-dấu-chấm-thành-gạch>.sslip.io` |

**Không** đặt `ENABLE_WIF_DEPLOY`. Biến đó bật `cd.yml`, mà cả hai workflow đều
treo trên `workflow_run` của `ci` — đặt vào là mỗi push deploy hai lần lên cùng
một VM.

### 5. Push lần đầu

```bash
git push origin main
```

Lần này sẽ đỏ ở bước `docker pull` với `denied`: package trên ghcr.io mặc định
private kể cả khi repo public, mà VM pull ẩn danh.

Repo → **Packages** → `llm-engineer-demo` → **Package settings** → **Danger Zone**
→ **Change visibility** → **Public** → **Re-run** workflow.

### Checklist

- [ ] `curl -s https://<domain>/health` trả `{"status":"ok",...}`
- [ ] `ci` và `deploy` đều đã xanh ít nhất một lần trong tab Actions
- [ ] Ba secret đã nạp (lệnh kiểm tra ở bước 2)
- [ ] Package trên ghcr.io đã Public
- [ ] `MONITORING_ENABLED=true` trên VM, và một request thật đã sinh trace trong
      project `llm-engineer-demo` trên LangSmith
- [ ] `ssh -i ~/.ssh/llm-app-deploy deploy@<VM_HOST> 'echo OK; sudo /opt/llm-app/release.sh status'`
      chạy được — dùng đúng key và đúng đường GitHub sẽ dùng. **Chỉ chạy được SAU
      lần deploy đầu**: `release.sh` do pipeline scp lên, trước đó chưa tồn tại.
- [ ] `git checkout .` — không còn thay đổi dở dang từ lần thử trước

---

## Màn 1 — Commit tốt: pipeline xanh (5 phút)

Thêm một trường vào `health()` trong `app/main.py`:

```python
"version": "v2",
```

```bash
git add app/main.py && git commit -m "health: thêm trường version" && git push
```

**Xem tab Actions:** workflow `ci` chạy trước (lint → ruff → pytest), xanh xong
mới tới `deploy` (build → push ghcr → ssh → smoke).

```bash
curl -s https://<domain>/health     # → có "version":"v2"
```

---

## Màn 2 — Commit phá test: CI đỏ, không deploy (5 phút)

Trong `app/observability/sampling.py`, hàm `should_sample()`: chuyển khối
`if cache_hit:` lên **trước** khối `if error:`.

```bash
git commit -am "sampling: gộp nhánh cache_hit lên trước" && git push
```

**Kết quả mong đợi:** `test_error_beats_cache_hit` đỏ — `1 failed, 34 passed`
trong `tests/test_observability.py`.

**Xem tab Actions:** workflow `ci` đỏ. Workflow `deploy` **không hề chạy**.

```bash
curl -s https://<domain>/health     # → vẫn là v2, bản cũ nguyên vẹn
```

Hoàn tác trước khi sang màn 3:

```bash
git revert --no-edit HEAD && git push
```

---

## Màn 3 — Qua CI nhưng hỏng lúc chạy: rollback tự động (10 phút)

Trong `app/config.py`, sửa một ký tự ở `rag_source_dir`:

```
"./data/legal_docs"  →  "./data/legal_doc"
```

```bash
git commit -am "config: sửa đường dẫn tài liệu" && git push
```

**Kết quả mong đợi:** `279 passed` — toàn bộ test xanh, ruff xanh, CI xanh. Không
test nào phủ đường dẫn đó.

**Xem tab Actions theo thứ tự:**

| Bước | Kết quả mong đợi |
|---|---|
| `ci` | ✅ xanh |
| build + push ghcr | ✅ xanh |
| `release.sh deploy` | ✅ xanh — `/health` trả 200, systemd báo service active |
| `release.sh smoke` | ❌ **đỏ** — `/admin/ingest` trả 200 nhưng `total_in_collection = 0` |
| Rollback tự động | ✅ chạy `if: failure()` → về last-known-good |
| Smoke lại | ✅ xanh — app đã hồi |

```bash
curl -s https://<domain>/health     # → vẫn sống, vẫn là bản tốt
```

---

## Câu hỏi thảo luận

1. Màn 3 lẽ ra có thể bắt được ở tầng test không? Nếu có, test đó trông thế nào —
   và vì sao viết nó **không** thay thế được smoke test?
2. Nếu `smoke.py` cũng mock hết I/O như unit test, màn 3 sẽ diễn ra thế nào?
3. `release.sh` ghi `.lkg` **trước** khi đổi tag. Nếu ghi sau thì hỏng ở đâu?
4. Vì sao `deploy.yml` dùng `concurrency: cancel-in-progress: false`, còn `ci.yml`
   dùng `true`?
5. Repo này còn một workflow `cd.yml` dùng Workload Identity Federation thay vì
   SSH key. Đánh đổi giữa hai cách là gì — và với production thật thì chọn cách nào?

---

## Dọn dẹp sau buổi

**Chỉ dừng, giữ hạ tầng cho buổi sau** — IP tĩnh vẫn tính phí nhàn rỗi
~$0.004/h:

```bash
gcloud compute instances stop llm-app-vm --zone=asia-southeast1-b
```

**Xoá sạch để dựng lại từ đầu:**

```bash
PROJECT_ID=llm-engineer-demo ./deploy/teardown.sh
```

Script xoá VM, IP tĩnh, 3 secret, service account, firewall và SSH key local —
theo đúng thứ tự ngược lúc dựng, nên chạy lại lần hai cũng không lỗi. Ba thứ
phải xoá tay, script in ra ở cuối:

1. GitHub → Settings → Secrets and variables → Actions: `VM_SSH_KEY`, `VM_HOST`,
   `VM_USER`, `APP_URL`. **Đừng bỏ qua bước này** — `VM_HOST` còn trỏ vào IP đã
   giải phóng thì lần dựng sau pipeline đỏ ở bước `ssh` với lỗi timeout.
2. Package `llm-engineer-demo` trên ghcr.io (tuỳ chọn).
3. Trace trong project `llm-engineer-demo` trên LangSmith (tuỳ chọn).

Giữ lại key local để đỡ phải cập nhật `VM_SSH_KEY` trên GitHub:

```bash
KEEP_SSH_KEY=true PROJECT_ID=llm-engineer-demo ./deploy/teardown.sh
```

> ⚠ **Let's Encrypt giới hạn 5 chứng chỉ trùng nhau mỗi tuần.** Domain là
> `<IP>.sslip.io`, nên teardown rồi dựng lại giữ nguyên IP tĩnh là xin lại đúng
> tên miền đó — lần thứ 6 trong tuần, bước 7 của `deploy.sh` đỏ. `teardown.sh`
> xoá luôn IP tĩnh nên lần sau sẽ lấy IP mới và không dính giới hạn.

Nếu giữ VM mà muốn thu hồi quyền SSH của CI khi khoá học kết thúc:

```bash
gcloud compute instances remove-metadata llm-app-vm --keys=ssh-keys
```
