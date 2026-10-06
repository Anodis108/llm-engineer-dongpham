"""Prompt registry trên LangSmith — Module III, Bài 6 (nối Bài 1, Section 4).

App KHÔNG hardcode prompt và KHÔNG đọc file YAML trong repo: prompt sống trên
LangSmith Prompt Hub, pull về lúc runtime theo alias (`production`).

Vì sao có cache đĩa (bài học Section 4 nhấn mạnh):
    "Registry hosted cần fallback — nếu load prompt qua API lúc runtime, một sự
     cố của registry = toàn bộ app chết."
Nên thứ tự luôn là: **LangSmith → cache đĩa → hằng số hardcode**, mỗi lần rơi
xuống tầng dưới đều log warning để thấy trên dashboard/log.

Ranh giới LangChain: chỉ dùng `pull_prompt` để LẤY object về (SDK trả
ChatPromptTemplate). Ngay sau đó ta TRÍCH template thô + biến ra dataclass thuần
và render bằng `str.format` — hot path không phụ thuộc LangChain, và cache đĩa
đọc lại được mà không cần cài langchain-core.

Rollback prompt KHÔNG cần rebuild image: trỏ alias `production` về commit cũ
bằng scripts/prompt_rollback.py, rồi xoá cache (hoặc chờ TTL) là app dùng bản cũ.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

# ChatPromptTemplate bọc mỗi message trong 1 lớp riêng; tên lớp cho biết role.
# (Thuộc tính `.role` của các lớp này là None nên không dùng được.)
_ROLE_BY_CLASS = {
    "SystemMessagePromptTemplate": "system",
    "HumanMessagePromptTemplate": "user",
    "AIMessagePromptTemplate": "assistant",
}

# Cache coi là cũ sau 24h — prompt không đổi thường xuyên, nhưng cũ quá thì
# nên thử lại LangSmith thay vì mãi chạy bản cache.
CACHE_TTL_SECONDS = 86400


class PromptUnavailable(RuntimeError):
    """Cả LangSmith lẫn cache đĩa đều không dùng được → caller dùng hằng số."""


@dataclass(slots=True)
class ExtractedMessage:
    role: str
    template: str
    variables: tuple[str, ...]


@dataclass(slots=True)
class ExtractedPrompt:
    """Bản "thuần" của 1 prompt — đủ để render, không cần LangChain."""

    name: str
    messages: list[ExtractedMessage]
    commit_hash: str = ""
    fetched_at: float = 0.0
    # "langsmith" | "cache". Ghi tường minh thay vì suy từ commit_hash — bản
    # cache CŨNG có commit_hash, suy kiểu đó sẽ báo nhầm bản cache là langsmith.
    source: str = "langsmith"


@dataclass(slots=True)
class RenderResult:
    messages: list[dict]
    source: str  # "langsmith" | "cache" — "fallback" do templates.py quyết định
    commit_hash: str = ""


def _extract(prompt, name: str, commit_hash: str = "") -> ExtractedPrompt:
    """Trích ChatPromptTemplate (LangChain) → ExtractedPrompt (thuần)."""
    messages: list[ExtractedMessage] = []
    for m in prompt.messages:
        role = _ROLE_BY_CLASS.get(type(m).__name__)
        if role is None:
            raise PromptUnavailable(
                f"{name}: không map được role cho message {type(m).__name__}"
            )
        messages.append(
            ExtractedMessage(
                role=role,
                template=m.prompt.template,
                variables=tuple(m.prompt.input_variables),
            )
        )
    return ExtractedPrompt(
        name=name, messages=messages, commit_hash=commit_hash, fetched_at=time.time()
    )


def _cache_path(name: str, alias: str) -> Path:
    return Path(settings.prompt_cache_dir) / f"{name}__{alias}.json"


def _write_cache(prompt: ExtractedPrompt, alias: str) -> None:
    try:
        path = _cache_path(prompt.name, alias)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(prompt), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:  # cache ghi hỏng không được làm chết request
        logger.warning("Không ghi được prompt cache cho %r: %s", prompt.name, exc)


def _read_cache(name: str, alias: str) -> ExtractedPrompt | None:
    path = _cache_path(name, alias)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Prompt cache hỏng cho %r: %s", name, exc)
        return None

    if time.time() - raw.get("fetched_at", 0) > CACHE_TTL_SECONDS:
        logger.warning("Prompt cache hết hạn cho %r (alias=%s)", name, alias)
        return None

    return ExtractedPrompt(
        name=raw["name"],
        messages=[ExtractedMessage(**m) for m in raw["messages"]],
        commit_hash=raw.get("commit_hash", ""),
        fetched_at=raw.get("fetched_at", 0.0),
        source="cache",
    )


def _render_messages(prompt: ExtractedPrompt, values: dict) -> list[dict]:
    """Render từng message. Raise nếu thiếu biến — thà lỗi rõ còn hơn gửi
    prompt thiếu biến lên model (đúng thứ lint ở CI cố chặn trước)."""
    out: list[dict] = []
    for m in prompt.messages:
        missing = [v for v in m.variables if v not in values]
        if missing:
            raise PromptUnavailable(
                f"{prompt.name}: thiếu biến khi render: {missing}"
            )
        out.append({"role": m.role, "content": m.template.format(**values)})
    return out


class PromptRegistry:
    """Pull prompt từ LangSmith, cache đĩa, render ra dict OpenAI-style."""

    def __init__(self, client=None):
        self._client = client  # tiêm được trong test, tránh gọi mạng

    def _get_client(self):
        if self._client is None:
            from langsmith import Client

            self._client = Client(
                api_key=settings.langsmith_api_key, api_url=settings.langsmith_endpoint
            )
        return self._client

    def _pull(self, name: str, identifier: str) -> ExtractedPrompt:
        """Pull `name:alias` từ LangSmith. Raise PromptUnavailable nếu không được.

        `name` là tên TRẦN ("rag-answer"), KHÔNG lấy từ `parse_prompt_identifier`
        — hàm đó trả về tên đã kèm tiền tố project ("llm-engineer-demo-rag-answer"),
        dùng nó làm khoá cache thì ghi một đằng đọc một nẻo.
        """
        prompt = self._get_client().pull_prompt(identifier)

        # Lấy commit hash thật để biết đang chạy đúng version nào (alias có thể
        # đã bị trỏ đi chỗ khác — cần thấy được khi debug/rollback).
        commit_hash = ""
        try:
            commit = self._get_client().pull_prompt_commit(identifier)
            commit_hash = getattr(commit, "commit_hash", "") or ""
        except Exception:  # noqa: BLE001 — chỉ là metadata phụ, không chặn render
            logger.debug("Không lấy được commit hash cho %s", identifier)

        return _extract(prompt, name, commit_hash)

    def get(self, name: str, alias: str | None = None) -> ExtractedPrompt:
        """Lấy prompt: LangSmith trước, rơi xuống cache khi lỗi."""
        alias = alias or settings.prompt_alias
        identifier = f"{settings.langsmith_project}-{name}:{alias}"

        try:
            prompt = self._pull(name, identifier)
            _write_cache(prompt, alias)
            logger.info("Prompt %s lấy từ LangSmith (%s)", identifier, prompt.commit_hash[:8])
            return prompt
        except Exception as exc:  # noqa: BLE001 — mọi lỗi đều phải rơi xuống cache
            logger.warning(
                "Không pull được prompt %r từ LangSmith (%s) — thử cache đĩa",
                identifier,
                exc,
            )

        cached = _read_cache(name, alias)
        if cached is not None:
            logger.warning("Prompt %s dùng bản CACHE đĩa", identifier)
            return cached

        raise PromptUnavailable(f"{identifier}: không có LangSmith lẫn cache đĩa")

    def render(
        self, name: str, alias: str | None = None, **values
    ) -> RenderResult:
        prompt = self.get(name, alias)
        messages = _render_messages(prompt, values)
        return RenderResult(
            messages=messages, source=prompt.source, commit_hash=prompt.commit_hash
        )


_registry: PromptRegistry | None = None


def get_registry() -> PromptRegistry:
    global _registry
    if _registry is None:
        _registry = PromptRegistry()
    return _registry


def reset_registry() -> None:
    """Xoá singleton — dùng trong test để tiêm client giả."""
    global _registry
    _registry = None
