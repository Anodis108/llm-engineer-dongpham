# Module III — LLM Ops

Module III chuyển từ "xây agent" (Module II) sang "vận hành production": biến các
kỹ thuật eval thủ công (Module I Bài 7, Module II Bài 5) thành **pipeline tự động**
— golden dataset version hoá, chạy song song, chấm điểm nhiều tầng, so baseline,
và chặn merge trên CI khi có regression.

> Cài đặt / chạy / test chung: xem [README.md](README.md).
> Module I (RAG pháp lý): xem [README.module1.md](README.module1.md).
> Module II (AI Agent): xem [README.module2.md](README.module2.md).

---

## Lộ trình xây dựng

| Buổi | Chủ đề | Lắp vào codebase |
|------|--------|------------------|
| **2** | **LLM Evaluation Pipelines** | `eval_pipeline/` — **golden dataset trên LangSmith, rule + judge scorer, regression gate** |
| **3** | **Cost Optimization & Caching** | `cost/` — **token economics, exact/semantic cache 2 tầng, cascading, budget governance** |
| **4** | **Containerisation & LLM Serving** | `Dockerfile`, `docker-compose.yml`, `app/llm/async_client.py` — **multi-stage build, RedisStore, async serving (semaphore/backpressure/SSE)** |
| **5** | **Cloud Deployment** | `deploy/` — **Compute Engine + nginx + systemd + Secret Manager runbook** |

### Buổi 2 — LLM Evaluation Pipelines

RAGAS/LLM-as-judge (Module I, Bài 7) và Task Success/Trajectory eval (Module II,
Bài 5) chấm **1 output** tại 1 thời điểm. Buổi này biến chúng thành pipeline chạy
tự động trên toàn bộ golden set, aggregate theo slice (bắt **"regression vô
hình"** — điểm tổng ổn nhưng 1 nhóm case tụt hẳn), và gate chặn merge.

**Golden Dataset (Section 2)** — [`data/eval/legal_qa/v1.yaml`](data/eval/legal_qa/v1.yaml):
30 case theo đúng nội dung THẬT của [`data/legal_docs/luat-doanh-nghiep-2026-trich.md`](data/legal_docs/luat-doanh-nghiep-2026-trich.md)
(Nghị định tiền lương/thù lao — không phải "Luật Doanh nghiệp" như tên file gợi ý),
chia 4 slice:

| Slice | Số case | Kỳ vọng |
|---|---|---|
| `lookup` | 18 | Tra đúng 1 Điều/khoản cụ thể |
| `comparison` (multihop) | 6 | Tổng hợp ≥ 2 phần nội dung (vd so 2 trường hợp lương tối đa) |
| `out_of_scope` | 3 | Hỏi ngoài tài liệu (vd Luật Doanh nghiệp thật) → kỳ vọng từ chối/không suy diễn |
| `injection` | 3 | Prompt injection → kỳ vọng bỏ qua chỉ dẫn độc hại, không lộ system prompt |

Mỗi case: `id`, `question`, `expected`, `slice` (đủ `type`/`difficulty`/`multihop`/`out_of_scope`),
`must_include`/`must_not_include` (rule-based assertion).

[`app/eval_pipeline/dataset.py`](app/eval_pipeline/dataset.py) — `load_dataset()`
đọc + validate YAML (id trùng, question/expected rỗng, thiếu `slice.type` đều
raise ngay). `sync_dataset_to_langsmith()` đồng bộ **YAML → LangSmith Dataset**
theo DIFF (so `case_id`, chỉ update/create/delete phần thay đổi — KHÔNG xoá-tạo-lại
toàn bộ, giữ lịch sử run gắn với example cũ). `split` trên LangSmith = `slice.type`,
filter subset được ngay trên UI.

```bash
python -c "
from app.eval_pipeline.dataset import load_dataset, sync_dataset_to_langsmith
ds = load_dataset('data/eval/legal_qa/v1.yaml')
sync_dataset_to_langsmith(ds)
"
```

> **Vì sao golden set sống ở 2 nơi (YAML + LangSmith)?** YAML là NGUỒN CHÂN LÝ,
> version-controlled trong git, review qua pull request như code (bài học:
> "mỗi case là một 'định nghĩa đúng'; sai một case = sai thước đo mãi mãi").
> LangSmith là bản đồng bộ để `evaluate()` chạy trên đó và tự lưu lịch sử run.

**Eval Methods (Section 3)** — [`app/eval_pipeline/scorers.py`](app/eval_pipeline/scorers.py),
xếp tầng từ rẻ đến đắt:

- `score_rules` — must_include/must_not_include + regex `Điều \d+` (~0đ, tức thì, không cần LLM).
- `score_judge` — **tái dùng nguyên xi** [`app/eval/judge.py`](app/eval/judge.py)
  (Module I, Bài 7: rubric tuyệt đối, 6 loại bias đã giảm thiểu), chỉ bọc lại đúng
  chữ ký evaluator LangSmith cần: `(run, example) -> dict`.

**Runner (Section 4)** — [`app/eval_pipeline/runner.py`](app/eval_pipeline/runner.py)
dùng thẳng `langsmith.evaluate()` thay vì tự viết `ThreadPoolExecutor` + lưu lịch
sử JSON (như code mẫu trong bài học) — LangSmith đã chạy song song trên toàn
Dataset, gọi evaluators, và lưu mỗi lần chạy thành 1 "experiment" gắn với dataset
version. `target` là `pipeline.answer()` (Module I) bọc lại: case `injection` bị
`GuardrailViolation` chặn được coi là **thành công của guardrail**, không phải lỗi.

**Regression Gate (Section 5)** — [`app/eval_pipeline/gate.py`](app/eval_pipeline/gate.py):
`summarize()` đọc `ExperimentResults` trực tiếp (không cần `pandas`), tính tổng +
**theo từng `slice.type`**. `check_gate()` so với baseline + `drop_tolerance` cho
từng metric — kể cả gate riêng cho 1 slice cụ thể (vd `slice:injection`), đúng ví
dụ `GATES` trong bài học. Report chỉ RÕ metric nào rớt (`overall: 4.0 < 4.8 - 0.1`),
không chỉ số tổng.

```bash
# Chạy full 30 case + ingest lại RAG data + áp gate — exit 1 nếu FAIL (dùng cho CI)
python -m scripts.eval_pipeline_demo

# Subset 6 case cho vòng lặp nhanh khi dev (round-robin qua slice — LUÔN phủ đủ
# injection/out_of_scope dù chúng nằm cuối YAML, không cắt N case đầu)
python -m scripts.eval_pipeline_demo --subset 6

# Đã ingest rồi (Qdrant server, không phải :memory:) — bỏ qua ingest lại
python -m scripts.eval_pipeline_demo --skip-ingest
```

> **Vì sao `--subset` không đơn giản là "N case đầu YAML"?** `data/eval/legal_qa/v1.yaml`
> liệt kê 18 case `lookup` trước rồi mới tới `injection`/`out_of_scope` — cắt cơ
> học sẽ bỏ sót HOÀN TOÀN 2 slice rủi ro nhất. `GoldenDataset.select_subset()`
> round-robin qua từng `slice.type` để bất kỳ N nào (kể cả N nhỏ) vẫn phủ đủ 4 slice.

**CI Integration (Section 6)** — [`.github/workflows/eval-gate.yml`](.github/workflows/eval-gate.yml):
2 tầng đúng bài học — PR chạm `prompts/`/`retrieval/`/`eval_pipeline/`/`data/eval/`
chạy **subset 20 case, chặn merge nếu FAIL**; cron nightly chạy **full dataset,
không chặn** (`continue-on-error: true`), chỉ báo cáo xu hướng.

```bash
# .env: cần cả 2 (embedding/chat + đăng ký trace/dataset)
OPENAI_API_KEYS=sk-...
LANGSMITH_API_KEY=lsv2_pt_...
```

### Buổi 3 — Cost Optimization & Caching

Module I, Bài 8 đã giới thiệu 3 đòn bẩy (prompt caching, semantic caching, model
routing). Buổi này **mở rộng sang cấp hệ thống**: đo cost trước khi tối ưu, cache
2 tầng HOÀN CHỈNH với invalidation đúng, cascading (khác routing), và cost
governance. Package mới `app/cost/` — không viết lại semantic cache/routing đã
có, chỉ nâng cấp + bổ sung phần bài học yêu cầu mà Module I chưa có.

**Token Economics + đo trước khi tối ưu (Section 1-2)** — [`app/cost/tracker.py`](app/cost/tracker.py):
`CostTracker` ghi mỗi request (feature/user/model/cache_hit), trả lời đúng 3 lát
cắt bài học nêu — `by_feature()` (tính năng nào ngốn ngân sách), `by_user()` (ai
lạm dụng), `percentile()` (p50 vs p99). `breakdown_tokens()` ước lượng % token
theo nguồn (system/context/history/question) — minh hoạ trực quan **context
bloat là kẻ giết ngân sách**. `app/llm/completion.py` có thêm `chat_with_usage()`
(trả kèm `usage` thật từ API, khác `chat()` chỉ trả text) để tính tiền chính xác
thay vì ước lượng.

```bash
python -c "
from app.cost.tracker import breakdown_tokens, breakdown_pct
pct = breakdown_pct(breakdown_tokens(system='x'*1600, context='y'*12000, history='z'*800, question='q'*400))
print(pct)  # context chiếm phần lớn — giống ví dụ ~70% trong bài học
"
```

**Caching tầng 1: Exact Response Cache (Section 3)** — [`app/cost/cache_exact.py`](app/cost/cache_exact.py):
`ExactCache.get_or_call()` cache TOÀN BỘ câu trả lời cho input giống hệt. Điểm
quan trọng nhất bài học nhấn mạnh: **`prompt_version` PHẢI nằm trong `cache_key()`**
— bump version tự động tạo "namespace" mới, cache cũ hết hạn tự nhiên (đã verify
bằng test: cùng câu hỏi, version cũ → hit; version mới → miss lại).

```bash
python -c "
from app.cost.cache_exact import ExactCache
cache = ExactCache()
n = {'x': 0}
def call(): n['x'] += 1; return f'answer-{n[\"x\"]}'
a1, hit1 = cache.get_or_call(prompt_name='rag', prompt_version=1, model='gpt-4o-mini', rendered_prompt='câu hỏi', params={}, call_fn=call)
a2, hit2 = cache.get_or_call(prompt_name='rag', prompt_version=1, model='gpt-4o-mini', rendered_prompt='câu hỏi', params={}, call_fn=call)
a3, hit3 = cache.get_or_call(prompt_name='rag', prompt_version=2, model='gpt-4o-mini', rendered_prompt='câu hỏi', params={}, call_fn=call)  # bump version
print(hit1, hit2, hit3)  # False True False — v2 KHÔNG trả answer cũ của v1
"
```

> **Vì sao chưa dùng Redis thật?** `CacheStore` là 1 protocol (`get`/`setex`) —
> `InMemoryStore` (ở đây) implement cho demo, không cần Docker. Module III, Bài 4
> (Containerisation) sẽ thêm `RedisStore` implement ĐÚNG 2 method đó vào
> `docker-compose.yml`, KHÔNG đổi gì trong `ExactCache`/logic cache — minh hoạ giá
> trị của việc tách interface store khỏi logic ngay từ đầu.

**Caching tầng 2: Semantic Cache (Section 4)** — nâng cấp [`app/optimization/caching.py`](app/optimization/caching.py)
(giữ tương thích ngược hoàn toàn với `app/api/routes_chat.py` đang dùng `get()`/`set()`):

- `ttl_seconds` — cache stale khi tài liệu nguồn cập nhật.
- `is_volatile()` — loại câu có **năm/tỷ lệ %** cụ thể khỏi semantic cache (false
  hit nguy hiểm nhất theo bài học: "... năm 2020" vs "... năm 2024" similarity
  cao nhưng câu trả lời khác hẳn).
- `stats()` → `hit_rate` — quyết định semantic cache có đáng dùng (bài học: FAQ
  20-40% mới đáng, câu hỏi đa dạng <5% thì không).

**Model Cascading (Section 5)** — [`app/cost/cascade.py`](app/cost/cascade.py):
khác **routing** (`app/optimization/routing.py`, Module I — quyết định model
TRƯỚC dựa trên query), cascading gọi model rẻ TRƯỚC rồi ESCALATE lên model mạnh
dựa trên OUTPUT (không tự tin → escalate). `escalate_rate()` đo tỉ lệ — > 50%
nghĩa là cascade đang LỖ (trả tiền cả 2 model cho phần lớn traffic).

**Cost Governance (Section 7)** — [`app/cost/budget.py`](app/cost/budget.py):
`check_budget()` (spent + dự phóng cuối tháng theo tốc độ chi tiêu, bắn alert ở
50%/80%/100%), `guard_user_budget()` (hard cap/ngày, raise `BudgetExceeded` →
nối vào `app/main.py` exception handler → HTTP 429, giống cách `GuardrailViolation`
→ HTTP 400 ở Module I).

**Hands-on: Cache 2 tầng cho Vietnamese chatbot (Section 8)** — [`scripts/cost_replay_demo.py`](scripts/cost_replay_demo.py)
chạy tập replay [`data/cost/replay_questions.yaml`](data/cost/replay_questions.yaml)
(46 câu ~43% trùng/gần giống, tái dùng 18 câu `lookup` từ golden set Bài 2 làm
nền) qua 4 cấu hình, đúng 5 bước bài học:

```bash
python -m scripts.cost_replay_demo --config no-cache       # baseline
python -m scripts.cost_replay_demo --config exact           # + tầng 1
python -m scripts.cost_replay_demo --config exact+semantic  # + tầng 2
python -m scripts.cost_replay_demo --config cascade         # mở rộng
python -m scripts.cost_replay_demo --compare                # chạy cả 4, in bảng so sánh
```

> **Vì sao exact cache PHẢI check trước semantic cache?** Thứ tự sai (semantic
> trước) khiến semantic "nuốt" luôn cả case trùng y hệt (similarity=1.0 với
> chính câu đã lưu) — `exact_hit_rate` luôn về 0%, dù exact cache có tồn tại.
> Đây là bug thật bắt được khi verify script này lần đầu — kiến trúc đúng
> (Section 4, mermaid tóm tắt bài học) là **tầng 1 → tầng 2 → gọi LLM**, tầng 1
> rẻ hơn (hash lookup, không cần embed) và chính xác 100%.

### Buổi 4 — Containerisation & LLM Serving

Đưa app đã có (Module I/II + Bài 2-3) vào Docker đúng chuẩn production, và
nâng cấp serving từ sync (`/chat/stream`, Bài 1) lên async có backpressure —
2 việc tách biệt, không đổi logic RAG/agent đã xây.

**Multi-stage Docker build (Section 1)** — [`Dockerfile`](../Dockerfile):
stage `builder` cài dependency (cần compiler cho một số package native), stage
`runtime` chỉ copy package đã cài + code, chạy bằng **non-root user**, có
`HEALTHCHECK` gọi thẳng `/health` (không cần thêm `curl`). Layer xếp theo tần
suất đổi — `requirements.txt` copy TRƯỚC `app/`, sửa code không phải cài lại
dependency. [`.dockerignore`](../.dockerignore) loại `.git`, `tests/`,
`training/`, notebook.

> **Model weights KHÔNG nằm trong image.** App chỉ gọi API (OpenAI Cloud/Ollama/
> vLLM qua `base_url`) — không có gì để bake vào image. Nếu sau này serve
> model local trong container, mount weight qua volume (xem service `vllm`
> dưới) chứ không `COPY` vào layer build.

```bash
docker build -t llm-engineer-demo:latest .
docker compose up -d              # Qdrant + Redis + app
docker compose --profile vllm up  # + vLLM local (tuỳ chọn, cần GPU)
```

**LLM Serving nâng cao (Section 2-3)** — [`docker-compose.yml`](../docker-compose.yml)
thêm service `vllm` (profile riêng, không chạy mặc định vì cần GPU + image
nặng) minh hoạ 2 khái niệm bài học: **PagedAttention** (KV cache quản lý như
phân trang bộ nhớ ảo OS, ~96% tận dụng bộ nhớ so với ~20-40% cách naive) và
**continuous batching** (lập lịch ở mức iteration, request xong nhường slot
ngay — khác static batching phải đợi cả batch xong).

**Redis cache tầng 1 (Section 1, nối Bài 3)** — [`app/cost/cache_redis.py`](../app/cost/cache_redis.py):
`RedisStore` implement đúng `CacheStore` protocol (`get`/`setex`) đã tách sẵn
từ Bài 3 — `ExactCache` KHÔNG đổi gì khi chuyển từ `InMemoryStore` (demo, 1
process) sang Redis thật (chia sẻ giữa nhiều container/replica). Đây chính là
giá trị của việc tách interface từ đầu thay vì viết cứng vào 1 backend.

**FastAPI Patterns cho LLM (Section 4)** — [`app/llm/async_client.py`](../app/llm/async_client.py)
+ endpoint mới [`POST /chat/stream-sse`](../app/api/routes_chat.py):

| | `/chat/stream` (Bài 1) | `/chat/stream-sse` (Bài 4, mới) |
|---|---|---|
| Protocol | `text/plain` thô | **SSE** (`text/event-stream`), kết thúc bằng `data: [DONE]` |
| I/O | SYNC (`def`) — block event loop nếu server có việc khác | **ASYNC** (`async def`) + `AsyncOpenAI` |
| Client bỏ đi giữa chừng | Vẫn sinh tiếp → đốt tiền API | **`request.is_disconnected()`** → dừng ngay |
| Quá tải đồng thời | Không giới hạn → 429 hàng loạt từ provider | **Semaphore backpressure** (`LLM_MAX_CONCURRENCY`) — request thừa xếp hàng |
| Lỗi giữa stream | Vỡ kết nối, client không biết vì sao | Gửi `{"error": ...}` qua SSE rồi đóng sạch |

Giữ **CẢ HAI** endpoint có chủ đích: học viên so sánh trực tiếp 2 cách, và
`chat.html`/test Module I không phải sửa gì. `async_client.py` tái dùng
`RotatingKeyPool` của [`app/llm/client.py`](../app/llm/client.py) (không viết
lại logic key rotation) — chỉ khác `AsyncOpenAI` thay `OpenAI`, và
`acall_with_retry()` dùng `asyncio.sleep()` thay `time.sleep()` (bản sync,
`resilience.retry_with_backoff`, sẽ BLOCK event loop nếu gọi từ `async def`).

```bash
curl -N -X POST http://localhost:8000/chat/stream-sse \
    -H "Content-Type: application/json" \
    -d '{"question": "Mức lương tối thiểu vùng I là bao nhiêu?"}'
```

**Hands-on: đo backpressure (Section 4, Bước 3)** — [`scripts/concurrency_benchmark.py`](../scripts/concurrency_benchmark.py)
bắn N request đồng thời vào `/chat/stream-sse`, đo p50/p95/throughput:

```bash
LLM_MAX_CONCURRENCY=5  uvicorn app.main:app --port 8000   # terminal 1
python -m scripts.concurrency_benchmark --requests 50 --concurrency 50  # terminal 2

# Đổi LLM_MAX_CONCURRENCY=50, chạy lại — so sánh p95/throughput để THẤY
# hiệu ứng semaphore, không chỉ đọc lý thuyết.
```

> **Test** (mock `AsyncOpenAI`, không gọi API thật): [`tests/test_async_client.py`](../tests/test_async_client.py)
> (semaphore nhả/giữ slot, retry+jitter, hết lượt thử thì raise) và
> [`tests/test_chat.py`](../tests/test_chat.py) (`test_chat_stream_sse_endpoint_wiring`,
> verify format SSE + `[DONE]`; `test_chat_stream_endpoint_still_works` xác nhận
> endpoint cũ không bị phá).

### Buổi 5 — Cloud Deployment

Đưa image từ Bài 4 lên **Google Compute Engine** (VM chạy liên tục — Section 1
so sánh Compute Engine/Cloud Run/GKE/Vertex AI, VM phù hợp nhất cho demo 1
service tự quản nginx/TLS). Toàn bộ script + runbook ở [`deploy/`](../deploy/README.md).

**Compute + hạ tầng (Section 2-3)** — [`deploy/startup.sh`](../deploy/startup.sh)
(cloud-init cài Docker/nginx/certbot), [`deploy/llm-app.service`](../deploy/llm-app.service)
(systemd unit, `Restart=always` tự hồi phục sau crash), [`deploy/nginx.conf`](../deploy/nginx.conf)
(reverse proxy + HTTPS qua Let's Encrypt).

> **`proxy_buffering off` là bắt buộc cho route streaming.** nginx mặc định
> buffer toàn bộ response trước khi trả — client không thấy token nào tới cho
> đến khi model sinh XONG, phá hoàn toàn SSE. `nginx.conf` tắt buffering riêng
> cho `/chat/stream-sse` VÀ `/chat/stream` (2 route streaming), khớp header
> `X-Accel-Buffering: no` mà endpoint SSE đã set sẵn (Bài 4, Section 4).

**Secret Management (Section 4)** — [`app/config.py`](../app/config.py)
đọc `OPENAI_API_KEYS`/`TAVILY_API_KEY`/`LANGSMITH_API_KEY` từ **GCP Secret
Manager lúc RUNTIME** khi `USE_SECRET_MANAGER=true` (production), fallback
`.env` khi tắt (local dev/test — mặc định). Auth qua Service Account gắn vào
VM (Application Default Credentials), **không có key file JSON nào** trên VM
hay trong image. `get_settings()` gọi `_load_secret_overrides()`: lỗi GCP bất
kỳ (mất quyền, secret không tồn tại...) chỉ log warning và fallback về `.env`
của field đó — không sập cả app.

```bash
python -c "
from app.config import Settings, _load_secret_overrides
base = Settings(USE_SECRET_MANAGER=True, GCP_PROJECT_ID='my-project')
print(_load_secret_overrides(base))  # {} nếu chưa auth GCP local — bài học: an toàn, không raise
"
```

**Firewall + IAM (Section 5)** — [`deploy/deploy.sh`](../deploy/deploy.sh)
chỉ mở `80`/`443` ra Internet; port app (8000) bind `127.0.0.1` trong container
(`llm-app.service`), không có rule firewall riêng — nginx là điểm vào public
duy nhất. Service Account chỉ có role `secretmanager.secretAccessor` (least
privilege).

**Hands-on: deploy full stack (Section 6)** — runbook từng bước đầy đủ ở
[`deploy/README.md`](../deploy/README.md):

```bash
export PROJECT_ID=my-gcp-project DOMAIN=chat.example.com
./deploy/deploy.sh   # firewall -> IAM -> secrets -> build+push -> VM -> nginx -> certbot
curl https://$DOMAIN/health
```

> **Test** (mock GCP client, không gọi Secret Manager thật): [`tests/test_secret_manager.py`](../tests/test_secret_manager.py)
> — verify override đúng field khi bật, no-op khi tắt/thiếu project_id, và lỗi
> GCP fallback về `.env` thay vì raise.

---

## Cấu trúc (Module III)

```
app/
├── eval_pipeline/         # ✓ Module III, Bài 2 — golden dataset + eval runner + regression gate
│   ├── dataset.py         #   load_dataset (YAML), sync_dataset_to_langsmith, select_subset
│   ├── scorers.py         #   score_rules, score_judge (bọc app/eval/judge.py)
│   ├── runner.py          #   run_eval — wraps langsmith.evaluate() + pipeline.answer()
│   └── gate.py            #   summarize (tổng + by_slice), check_gate, diff_failed_cases
├── cost/                  # ✓ Module III, Bài 3 (+ Bài 4) — token economics + cache 2 tầng + cascade + budget
│   ├── tracker.py         #   CostTracker (by_feature/by_user/percentile), breakdown_tokens
│   ├── cache_exact.py     #   ExactCache (tầng 1), CacheStore protocol, InMemoryStore
│   ├── cache_redis.py     #   ✓ Bài 4 — RedisStore (CacheStore thật, dùng chung nhiều instance)
│   ├── cascade.py         #   answer_cascade, escalate_rate
│   └── budget.py          #   check_budget, guard_user_budget, BudgetExceeded (→ HTTP 429)
├── llm/async_client.py    # ✓ Bài 4, Section 4 — AsyncOpenAI, semaphore backpressure, retry+jitter async
└── optimization/caching.py # ✓ nâng cấp Bài 3: TTL, is_volatile(), stats() (tầng 2, Module I gốc)
data/
├── eval/legal_qa/v1.yaml       # ✓ Bài 2 — golden set 30 case
└── cost/replay_questions.yaml  # ✓ Bài 3 — 46 câu replay (~43% trùng/gần giống)
scripts/
├── eval_pipeline_demo.py       # ✓ Bài 2 — CLI: sync dataset → evaluate() → report → gate → exit 0/1
├── cost_replay_demo.py         # ✓ Bài 3 — CLI: replay 4 cấu hình cache → bảng so sánh cost
└── concurrency_benchmark.py    # ✓ Bài 4 — CLI: bắn N request đồng thời, đo p50/p95/throughput
Dockerfile                  # ✓ Bài 4, Section 1 — multi-stage build, non-root, HEALTHCHECK
.dockerignore                # ✓ Bài 4 — loại .git/tests/training/notebook khỏi build context
docker-compose.yml            # ✓ Bài 4 — Qdrant + Redis + app + vllm (profile GPU tuỳ chọn)
deploy/                      # ✓ Module III, Bài 5 — Compute Engine + nginx + systemd + Secret Manager
├── startup.sh              #   cloud-init: cài Docker/nginx/certbot lần đầu
├── llm-app.service          #   systemd unit — Restart=always
├── nginx.conf                #   reverse proxy, proxy_buffering off cho route streaming
├── deploy.sh                 #   orchestration: firewall → IAM → secrets → build → VM → TLS
└── README.md                 #   runbook đầy đủ từng bước
.github/workflows/
└── eval-gate.yml          # ✓ CI: subset trên PR (chặn merge), full nightly (không chặn)
```

> Module I/II tái dùng LangSmith **cùng project** cho tracing (`app/monitoring/tracing.py`)
> — golden dataset + eval run của Module III xuất hiện trên cùng dashboard, dùng
> chung 1 `LANGSMITH_API_KEY`.
