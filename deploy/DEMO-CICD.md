# Kịch bản buổi học — CI/CD trên Google Compute Engine

Runbook để dạy Module III, Bài 6 trong một buổi. Toàn bộ lệnh trong đây đã được
chạy thử và kiểm chứng.

**Mục tiêu:** sau buổi này sinh viên trả lời được ba câu —

1. Vì sao **chỉ deploy khi CI xanh**, thay vì deploy thẳng từ `push`?
2. Vì sao **container healthy không có nghĩa là deploy thành công**?
3. Vì sao **rollback phải là một lệnh**, không phải một quy trình?

## Chuẩn bị trước buổi (làm trước, không làm trước mặt sinh viên)

```bash
export PROJECT_ID=my-gcp-project
./deploy/deploy.sh
```

Rồi làm 3 bước tay mà script in ra ở cuối (nạp secret thật, khai báo
`VM_SSH_KEY` + `VM_HOST` + `VM_USER` + `APP_URL` trên GitHub, kiểm tra
`/health`). Chi tiết ở [README.md](README.md).

**Checklist trước khi vào lớp:**

- [ ] `curl https://<domain>/health` trả `{"status":"ok",...}`
- [ ] Tab Actions có ít nhất một lần `ci` + `deploy` đã xanh
- [ ] `ssh deploy@<VM_HOST> sudo /opt/llm-app/release.sh status` chạy được
- [ ] Repo **public** (để VM pull được image từ ghcr.io)
- [ ] Đã thử `python -m scripts.cicd_demo` ở máy bạn — màn 0 cần nó chạy được
- [ ] Đã `git checkout .` — không còn thay đổi dở dang từ lần thử trước

> **Đừng cho sinh viên SSH vào VM.** `/opt/llm-app/.env` và Secret Manager đều
> dẫn tới key thật. Và về mặt sư phạm thì cho họ SSH vào deploy là dạy ngược bài
> học — CI/CD tồn tại chính vì không ai phải SSH vào.

---

## Màn 0 — Diễn tập tại chỗ (5 phút, không cần cloud)

Chạy trước để sinh viên thấy toàn bộ vòng đời mà không phải chờ mạng:

```bash
python -m scripts.cicd_demo
```

Script dựng 2 image bằng Docker thật, deploy bản tốt, rồi deploy bản hỏng và cho
nó tự rollback. **Nói với sinh viên:** "Đây là toàn bộ bài học hôm nay, chạy trên
máy tôi. Phần còn lại chỉ là đưa nó lên cloud."

---

## Màn 1 — Commit tốt: pipeline xanh (5 phút)

Sinh viên sửa một thứ **nhìn thấy được** để chứng minh bản mới thật sự lên sóng.
Thêm một trường vào `/health`:

```bash
# sửa app/main.py, trong hàm health(), thêm một dòng:
#     "version": "v2",
```

```bash
git add app/main.py && git commit -m "health: thêm trường version" && git push
```

**Chiếu màn hình:** tab Actions → workflow `ci` chạy trước (lint → ruff → pytest),
xanh xong mới tới `deploy` (build → push ghcr → ssh → smoke).

```bash
curl https://<domain>/health     # → thấy "version":"v2"
```

**Chốt:** không ai SSH vào, không ai gõ `docker pull`. Commit lên là app lên.

---

## Màn 2 — Commit phá test: CI đỏ, không deploy (5 phút)

Sửa thứ tự ưu tiên trong `app/observability/sampling.py` — đổi chỗ `error` và
`cache_hit`. Đây là lỗi rất dễ mắc khi refactor, và tinh tế: nó chỉ sai ở **một
tổ hợp hiếm** (request vừa lỗi vừa là cache hit).

```bash
# trong should_sample(), chuyển khối `if cache_hit:` lên TRƯỚC khối `if error:`
git commit -am "sampling: gộp nhánh cache_hit lên trước" && git push
```

Kết quả đã kiểm chứng: `test_error_beats_cache_hit` đỏ, `1 failed, 34 passed`.

**Chiếu màn hình:** workflow `ci` đỏ. Workflow `deploy` **không hề chạy** — vì nó
treo trên `workflow_run` của `ci`, không phải trên `push`.

```bash
curl https://<domain>/health     # → vẫn là v2, bản cũ nguyên vẹn
```

**Chốt:** đây là lý do `on: workflow_run` chứ không phải `on: push`. Nếu deploy
treo trực tiếp trên `push`, commit này đã lên production rồi.

Nhớ hoàn tác trước khi sang màn 3:

```bash
git revert --no-edit HEAD && git push
```

---

## Màn 3 — Commit qua CI nhưng hỏng lúc chạy: rollback tự động (10 phút)

Đây là màn quan trọng nhất. Sửa **một ký tự** trong đường dẫn tài liệu:

```bash
# app/config.py, dòng rag_source_dir:
#   "./data/legal_docs"  →  "./data/legal_doc"
git commit -am "config: sửa đường dẫn tài liệu" && git push
```

Kết quả đã kiểm chứng: **279 passed** — toàn bộ test xanh, ruff xanh, CI xanh.
Không test nào phủ đường dẫn đó, vì test mock hết phần I/O.

**Chiếu màn hình theo thứ tự:**

| Bước | Sinh viên thấy |
|---|---|
| `ci` | ✅ xanh — test không bắt được lỗi này |
| build + push ghcr | ✅ xanh |
| `release.sh deploy` | ✅ xanh — `/health` trả 200, systemd báo service active |
| `release.sh smoke` | ❌ **đỏ** — `/admin/ingest` trả 200 nhưng `total_in_collection = 0` |
| Rollback tự động | ✅ chạy `if: failure()` → về last-known-good |
| Smoke lại | ✅ xanh — app đã hồi |

```bash
curl https://<domain>/health     # → vẫn sống, vẫn là bản tốt
```

**Chốt — đây là câu trả lời cho câu hỏi số 2:**

> Container "healthy" chỉ có nghĩa là **process còn sống**. `/health` trả 200 kể
> cả khi Qdrant rỗng và `/chat` trả lời sai. Muốn biết deploy có thành công
> không, phải **hỏi app một câu đã biết đáp án** — đó là việc của `smoke.py`.

Và câu hỏi số 3:

> Rollback ở đây mất vài giây vì **không build lại gì cả** — image cũ vẫn nằm
> trên registry theo tag SHA, chỉ đổi con trỏ `.lkg` → `.current`. Rollback nhanh
> hay chậm là quyết định ở khâu **đóng gói image**, không phải ở khâu rollback.

---

## Tổng kết ba màn

| Màn | Lỗi ở đâu | Ai bắt | Kết cục |
|---|---|---|---|
| 1 | không có | — | app lên sóng |
| 2 | logic, có test phủ | `pytest` | **không deploy** — production chưa từng thấy bản lỗi |
| 3 | cấu hình, không test phủ | `smoke.py` | deploy rồi **tự rollback** |

Ba tầng chặn, ba mức chi phí: test rẻ nhất chặn sớm nhất; smoke đắt hơn nhưng bắt
được thứ test không thấy; rollback là lưới cuối.

## Câu hỏi thảo luận

1. Màn 3 lẽ ra có thể bắt được ở tầng test không? Nếu có, test đó trông thế nào —
   và vì sao viết nó **không** thay thế được smoke test?
2. Nếu `smoke.py` cũng mock hết I/O như unit test, màn 3 sẽ diễn ra thế nào?
3. `release.sh` ghi `.lkg` **trước** khi đổi tag. Nếu ghi sau thì hỏng ở đâu?
4. Vì sao `deploy.yml` dùng `concurrency: cancel-in-progress: false`, còn `ci.yml`
   dùng `true`?
5. Repo này còn một workflow `cd.yml` dùng Workload Identity Federation thay vì
   SSH key. Đánh đổi giữa hai cách là gì — và với production thật thì chọn cách nào?

## Dọn dẹp sau buổi

```bash
# Dừng VM (IP tĩnh vẫn tính phí nhàn rỗi ~$0.004/h)
gcloud compute instances stop llm-app-vm --zone=asia-southeast1-b

# Xoá hẳn nếu không dùng nữa
gcloud compute instances delete llm-app-vm --zone=asia-southeast1-b
gcloud compute addresses delete llm-app-ip --region=asia-southeast1
```

Nhớ xoá `VM_SSH_KEY` khỏi GitHub Secrets nếu không dùng lại, và thu hồi key trên
VM (`gcloud compute instances remove-metadata llm-app-vm --keys=ssh-keys`) khi
khoá học kết thúc.
