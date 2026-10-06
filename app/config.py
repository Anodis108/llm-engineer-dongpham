"""Cấu hình ứng dụng — đọc từ biến môi trường / file .env.

Đây là điểm DUY NHẤT đọc secrets. Phần còn lại của code import `settings`
chứ không gọi os.environ trực tiếp → dễ test, dễ kiểm soát.

Module III, Bài 5, Section 4 (Secret Management) — production đọc secret từ
GCP Secret Manager LÚC RUNTIME, không nướng vào Docker image/build. Local dev
vẫn dùng `.env` như cũ (không ai bắt buộc cài credentials GCP để chạy test/dev).
Xem `_load_secret_overrides()` phía dưới.
"""

import logging
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Field nào lấy từ Secret Manager, map sang secret_id trên GCP. Chỉ 3 secret
# thật sự nhạy cảm — phần còn lại (model name, chunk size...) không cần secret,
# nằm trong .env/biến môi trường bình thường.
_SECRET_MANAGER_FIELDS = {
    "openai_api_keys": "llm-engineer-openai-api-keys",
    "tavily_api_key": "llm-engineer-tavily-api-key",
    "langsmith_api_key": "llm-engineer-langsmith-api-key",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ── Backend (Buổi 2: Local Serving) ──────────────────────────────────────
    # "openai" = OpenAI Cloud | "ollama" = Ollama local | "vllm" = vLLM server.
    # Cả 3 đều nói chuyện qua OpenAI-compatible API → phần còn lại của code không đổi.
    llm_backend: str = Field(default="openai", alias="LLM_BACKEND")
    # base_url override. Để trống → dùng mặc định của backend (xem llm/backends.py).
    llm_base_url: str = Field(default="", alias="LLM_BASE_URL")

    # ── OpenAI ──────────────────────────────────────────────────────────────
    # Một hoặc nhiều API key, ngăn cách bằng dấu phẩy (bật key rotation).
    # Với backend local (ollama/vllm) không cần key thật.
    openai_api_keys: str = Field(default="", alias="OPENAI_API_KEYS")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")

    # ── Generation params (mặc định) ────────────────────────────────────────
    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")
    llm_max_completion_tokens: int = Field(default=800, alias="LLM_MAX_COMPLETION_TOKENS")
    llm_top_p: float = Field(default=1.0, alias="LLM_TOP_P")

    # ── Resilience ──────────────────────────────────────────────────────────
    llm_max_retries: int = Field(default=5, alias="LLM_MAX_RETRIES")

    # ── Async serving (Module III, Bài 4, Section 4: FastAPI Patterns) ───────
    # Backpressure: số lời gọi LLM ĐỒNG THỜI tối đa trong 1 worker. Request
    # thứ (N+1) xếp hàng ở semaphore thay vì đẩy hết tải lên provider (429).
    llm_max_concurrency: int = Field(default=20, alias="LLM_MAX_CONCURRENCY")
    # Timeout rõ ràng cho mỗi lời gọi (bài học: 30-60s), tránh treo vô hạn.
    llm_timeout_seconds: float = Field(default=30.0, alias="LLM_TIMEOUT_SECONDS")

    # ── Embeddings & Vector Store (Buổi 4) ───────────────────────────────────
    # OpenAI text-embedding-3-small: 1536 dims, gọi API (dùng chung key với LLM).
    embedding_model: str = Field(
        default="text-embedding-3-small", alias="EMBEDDING_MODEL"
    )
    embedding_dim: int = Field(default=1536, alias="EMBEDDING_DIM")

    # Qdrant. Mặc định ":memory:" cho demo (không cần Docker). Production: đặt
    #   QDRANT_URL=http://localhost:6333  và chạy Qdrant qua Docker.
    qdrant_url: str = Field(default=":memory:", alias="QDRANT_URL")
    vectorstore_collection: str = Field(
        default="legal_docs", alias="VECTORSTORE_COLLECTION"
    )

    # ── RAG Pipeline (Buổi 5) ────────────────────────────────────────────────
    rag_source_dir: str = Field(default="./data/legal_docs", alias="RAG_SOURCE_DIR")
    rag_chunk_size: int = Field(default=512, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=64, alias="RAG_CHUNK_OVERLAP")
    # Contextual Retrieval: prepend heading path vào chunk markdown (cải thiện retrieval).
    rag_contextual_chunking: bool = Field(default=True, alias="RAG_CONTEXTUAL_CHUNKING")
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K")

    # Query rewriting: LLM viết lại câu hỏi trước khi search (cải thiện recall).
    rag_query_rewriting: bool = Field(default=False, alias="RAG_QUERY_REWRITING")

    # Re-ranking: cross-encoder lọc lại sau retrieve (cần sentence-transformers).
    rag_rerank_enabled: bool = Field(default=False, alias="RAG_RERANK_ENABLED")
    rag_fetch_k: int = Field(default=20, alias="RAG_FETCH_K")  # lấy rộng trước khi rerank
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANK_MODEL"
    )

    # ── Agentic RAG / CRAG (Buổi 6) ──────────────────────────────────────────
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
    # Query decomposition: chỉ decompose câu hỏi đủ dài/phức tạp (heuristic đơn giản).
    agent_decompose_min_chars: int = Field(default=80, alias="AGENT_DECOMPOSE_MIN_CHARS")
    agent_max_sub_questions: int = Field(default=4, alias="AGENT_MAX_SUB_QUESTIONS")
    # Số chunk "relevant" tối thiểu để KHÔNG cần web search fallback.
    agent_min_relevant_chunks: int = Field(default=1, alias="AGENT_MIN_RELEVANT_CHUNKS")
    agent_web_search_results: int = Field(default=3, alias="AGENT_WEB_SEARCH_RESULTS")

    # ── Module II, Bài 3: Agent Memory & Context Engineering ─────────────────
    # Long-term memory: collection Qdrant RIÊNG (tách khỏi legal_docs của Module I).
    agent_memory_collection: str = Field(
        default="user_memory", alias="AGENT_MEMORY_COLLECTION"
    )
    # Context management (Section 2-4).
    agent_max_messages: int = Field(default=20, alias="AGENT_MAX_MESSAGES")  # sliding window
    agent_keep_recent_messages: int = Field(
        default=6, alias="AGENT_KEEP_RECENT_MESSAGES"
    )  # số message giữ nguyên khi summarize
    # Ngưỡng token của "context window" để áp nguyên tắc 40-60% (ước lượng thô).
    # Đặt nhỏ có chủ đích cho demo — dễ thấy summarization kích hoạt.
    agent_context_window_tokens: int = Field(
        default=2000, alias="AGENT_CONTEXT_WINDOW_TOKENS"
    )

    # ── Module II, Bài 4: Agentic Tool Design & Integration ──────────────────
    # Số tool tối đa model được THẤY mỗi lượt (Section 2: Tool Retrieval).
    # 15 tool tổng (tools.py) > ngưỡng chính xác ổn định (5-7) của bài học.
    agent_tool_retrieval_k: int = Field(default=5, alias="AGENT_TOOL_RETRIEVAL_K")

    # ── Guardrails & Evaluation (Buổi 7) ─────────────────────────────────────
    # Prompt injection: bật thêm LLM-based check ngoài regex (tốn 1 lời gọi LLM).
    guardrails_llm_injection_check: bool = Field(
        default=False, alias="GUARDRAILS_LLM_INJECTION_CHECK"
    )
    guardrails_min_answer_len: int = Field(default=10, alias="GUARDRAILS_MIN_ANSWER_LEN")

    # ── Monitoring — LangSmith (Buổi 7, Section 4; Module III Bài 2) ─────────
    # Tắt mặc định: khi chưa điền LANGSMITH_API_KEY thật, tracing là no-op,
    # không bắt buộc cài/kích hoạt LangSmith để chạy phần còn lại của codebase.
    monitoring_enabled: bool = Field(default=False, alias="MONITORING_ENABLED")
    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="llm-engineer-demo", alias="LANGSMITH_PROJECT")
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com", alias="LANGSMITH_ENDPOINT"
    )

    # ── Sampling & redaction cho trace (Module III, Bài 7, Section 3-4) ──────
    # Tỉ lệ giữ trace cho request BÌNH THƯỜNG. Ca lỗi/chậm/guardrail luôn được
    # giữ bất kể tỉ lệ này (tail sampling — xem app/observability/sampling.py).
    # Đặt 1.0 khi cần soi kỹ, 0.0 khi muốn tắt hẳn mà không tắt monitoring.
    trace_sample_rate: float = Field(default=0.05, alias="TRACE_SAMPLE_RATE")
    # Ngưỡng "chậm" — vượt là luôn giữ trace để điều tra p99.
    trace_slow_latency_s: float = Field(default=5.0, alias="TRACE_SLOW_LATENCY_S")
    # Che PII trước khi gửi trace ra ngoài. Chỉ tắt khi debug trên dữ liệu giả.
    trace_redact_pii: bool = Field(default=True, alias="TRACE_REDACT_PII")

    # ── Prompt Registry (Module III, Bài 6 — nối nội dung Bài 1) ─────────────
    # Nguồn chân lý của prompt là LangSmith Prompt Hub. Tên prompt lấy tiền tố
    # từ `langsmith_project` (vd "llm-engineer-demo-rag-answer").
    #
    # "local" (mặc định) = dùng hằng số trong app/prompts/templates.py.
    # "langsmith" = pull prompt lúc runtime, có cache đĩa + fallback (bài học:
    # "Registry hosted cần fallback" — registry sập không được làm chết app).
    # Mặc định "local" để test/CI chạy hermetic, không gọi mạng.
    prompt_registry: str = Field(default="local", alias="PROMPT_REGISTRY")
    prompt_cache_dir: str = Field(default="data/prompt_cache", alias="PROMPT_CACHE_DIR")
    # Alias trỏ tới commit đang dùng. Đổi sang "staging" để thử version mới mà
    # không đụng production; rollback = trỏ alias này về commit cũ (xem
    # scripts/prompt_rollback.py) — KHÔNG cần rebuild image.
    prompt_alias: str = Field(default="production", alias="PROMPT_ALIAS")

    # ── App ─────────────────────────────────────────────────────────────────
    app_name: str = Field(default="Vietnamese Legal Assistant", alias="APP_NAME")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # ── Secret Management (Module III, Bài 5, Section 4) ────────────────────
    # Tắt mặc định: local dev/test dùng .env, không bắt buộc cài GCP credentials.
    # Production (deploy/startup.sh) đặt USE_SECRET_MANAGER=true.
    use_secret_manager: bool = Field(default=False, alias="USE_SECRET_MANAGER")
    gcp_project_id: str = Field(default="", alias="GCP_PROJECT_ID")

    @property
    def api_keys(self) -> list[str]:
        """Tách chuỗi key thành list, bỏ khoảng trắng và phần tử rỗng."""
        return [k.strip() for k in self.openai_api_keys.split(",") if k.strip()]

    @field_validator("openai_api_keys")
    @classmethod
    def _warn_if_empty(cls, v: str) -> str:
        # Không raise ở đây để test/import không cần key thật;
        # llm/client.py sẽ báo lỗi rõ ràng khi thực sự cần gọi API.
        return v


def _fetch_secret(project_id: str, secret_id: str) -> str | None:
    """Đọc phiên bản "latest" của 1 secret trên GCP Secret Manager.

    Trả None (không raise) khi secret không tồn tại/không có quyền — để một
    secret thiếu không sập cả app, chỉ field đó giữ giá trị từ .env/default.
    Auth qua Application Default Credentials (service account của VM/Cloud
    Run) — KHÔNG có key file nào trên máy/trong image (Bài 5, Section 4).
    """
    try:
        from google.cloud import secretmanager
    except ImportError:
        logger.warning(
            "USE_SECRET_MANAGER=true nhưng chưa cài google-cloud-secret-manager"
        )
        return None

    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("utf-8")
    except Exception:  # noqa: BLE001 — bất kỳ lỗi GCP nào cũng fallback, không raise
        logger.warning("Không đọc được secret %r từ Secret Manager, dùng .env", secret_id)
        return None


def _load_secret_overrides(base: Settings) -> dict[str, str]:
    """Đọc các field nhạy cảm từ Secret Manager, ghi đè giá trị .env.

    Chỉ chạy khi `use_secret_manager=True` VÀ có `gcp_project_id` — nếu không
    trả dict rỗng (no-op), giữ nguyên hành vi cũ cho local dev/test.
    """
    if not base.use_secret_manager or not base.gcp_project_id:
        return {}

    overrides: dict[str, str] = {}
    for field_name, secret_id in _SECRET_MANAGER_FIELDS.items():
        value = _fetch_secret(base.gcp_project_id, secret_id)
        if value:
            overrides[field_name] = value
    return overrides


@lru_cache
def get_settings() -> Settings:
    """Singleton settings (cache để không đọc .env/Secret Manager nhiều lần).

    Thứ tự ưu tiên: Secret Manager (nếu bật) > .env > default field.
    """
    base = Settings()
    overrides = _load_secret_overrides(base)
    return base.model_copy(update=overrides) if overrides else base


settings = get_settings()
