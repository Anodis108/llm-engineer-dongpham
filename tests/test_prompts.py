"""Test prompt registry + lint — Module III, Bài 6.

Không test nào gọi mạng: client LangSmith được thay bằng object giả, cache đĩa
trỏ vào tmp_path. Đây chính là lý do PROMPT_REGISTRY mặc định là "local".
"""

from __future__ import annotations

import dataclasses
import json
import time
from types import SimpleNamespace

import pytest
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.prompts import lint, registry, templates
from app.prompts.definitions import (
    ALL_PROMPTS,
    QUERY_REWRITE_V1,
    RAG_ANSWER_V1,
    RAG_ANSWER_V2,
    latest_per_name,
)
from app.prompts.registry import (
    CACHE_TTL_SECONDS,
    ExtractedMessage,
    ExtractedPrompt,
    PromptRegistry,
    PromptUnavailable,
    _extract,
    _read_cache,
    _render_messages,
    _write_cache,
)

# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "prompt_cache_dir", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def _reset_singleton():
    registry.reset_registry()
    yield
    registry.reset_registry()


def _fake_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [("system", "Luật: {context_section}"), ("human", "{question}")]
    )


class _FakeLangSmithClient:
    """Thay Client thật. `fail=True` mô phỏng LangSmith sập/mất mạng."""

    def __init__(self, prompt=None, fail: bool = False, commit_hash: str = "deadbeefcafe"):
        self._prompt = prompt if prompt is not None else _fake_prompt()
        self._fail = fail
        self._commit_hash = commit_hash

    def pull_prompt(self, identifier):
        if self._fail:
            raise ConnectionError("langsmith down")
        return self._prompt

    def pull_prompt_commit(self, identifier):
        if self._fail:
            raise ConnectionError("langsmith down")
        return SimpleNamespace(commit_hash=self._commit_hash)


# ─── Lint ────────────────────────────────────────────────────────────────────


def test_lint_passes_for_shipped_definitions():
    assert lint.lint_all() == []


def test_lint_detects_missing_metadata():
    bad = dataclasses.replace(RAG_ANSWER_V1, owner="")
    assert "thiếu metadata 'owner'" in " ".join(lint.lint_definition(bad))


@pytest.mark.parametrize("version", [0, -1, "1", True])
def test_lint_rejects_bad_version(version):
    bad = dataclasses.replace(RAG_ANSWER_V1, version=version)
    assert "version phải là số nguyên >= 1" in " ".join(lint.lint_definition(bad))


def test_lint_rejects_non_kebab_case_name():
    bad = dataclasses.replace(RAG_ANSWER_V1, name="RagAnswer")
    assert "kebab-case" in " ".join(lint.lint_definition(bad))


def test_lint_detects_variable_used_but_not_declared():
    """Lỗi phổ biến nhất theo bài học: template dùng biến chưa khai báo."""
    bad = dataclasses.replace(RAG_ANSWER_V1, variables=("question",))
    errs = " ".join(lint.lint_definition(bad))
    assert "dùng nhưng chưa khai báo" in errs and "context_section" in errs


def test_lint_detects_declared_but_unused_variable():
    """Dấu hiệu đổi tên biến ở template mà quên cập nhật `variables`."""
    bad = dataclasses.replace(
        RAG_ANSWER_V1, variables=("context_section", "question", "typo_var")
    )
    assert "khai báo nhưng không dùng" in " ".join(lint.lint_definition(bad))


def test_lint_detects_template_too_long(monkeypatch):
    monkeypatch.setattr(lint, "MAX_TEMPLATE_CHARS", 10)
    assert "quá dài" in " ".join(lint.lint_definition(RAG_ANSWER_V1))


def test_lint_detects_duplicate_name_version():
    errs = lint.lint_all([RAG_ANSWER_V1, RAG_ANSWER_V1])
    assert "định nghĩa trùng" in " ".join(errs)


def test_lint_checks_every_message_not_just_first():
    """Biến nằm ở message thứ 2 vẫn phải được kiểm — parse regex tay hay bỏ sót."""
    tpl = ChatPromptTemplate.from_messages([("system", "cố định"), ("human", "{question}")])
    p = dataclasses.replace(QUERY_REWRITE_V1, template=tpl, variables=())
    assert "dùng nhưng chưa khai báo" in " ".join(lint.lint_definition(p))


# ─── Invariant: v1 phải trùng khít hành vi cũ ────────────────────────────────


def test_v1_template_reproduces_legacy_system_prompt():
    """RAG_ANSWER_V1 render với context rỗng phải ra ĐÚNG hằng số cũ.

    Đây là lưới an toàn cho việc chuyển prompt lên registry: bản v1 không được
    đổi một ký tự nào so với app/prompts/templates.py trước Bài 6.
    """
    msgs = RAG_ANSWER_V1.template.format_messages(context_section="", question="X")
    assert msgs[0].content == templates.LEGAL_SYSTEM_PROMPT
    assert msgs[1].content == "X"


# ─── Extract / render ────────────────────────────────────────────────────────


def test_extract_maps_roles_and_variables():
    got = _extract(_fake_prompt(), "rag-answer", "abc123")
    assert [(m.role, m.template) for m in got.messages] == [
        ("system", "Luật: {context_section}"),
        ("user", "{question}"),
    ]
    assert got.messages[0].variables == ("context_section",)
    assert got.commit_hash == "abc123"
    assert got.source == "langsmith"


def test_extract_handles_round_tripped_langchain_object():
    """`pull_prompt` trả object đã qua serialize/deserialize của LangChain, không
    phải object dựng tay. Tên lớp message phải giữ nguyên qua vòng đó — nếu
    LangChain đổi tên lớp, `_extract` sẽ hỏng và test này bắt được."""
    from langchain_core.load import dumps, loads

    back = loads(dumps(RAG_ANSWER_V1.template))
    got = _extract(back, "rag-answer")
    assert [m.role for m in got.messages] == ["system", "user"]
    assert got.messages[0].template == RAG_ANSWER_V1.template.messages[0].prompt.template


def test_extract_rejects_unknown_message_class():
    class Weird:
        messages = [object()]

    with pytest.raises(PromptUnavailable, match="không map được role"):
        _extract(Weird(), "x")


def test_render_messages_substitutes_variables():
    prompt = ExtractedPrompt(
        name="p",
        messages=[
            ExtractedMessage("system", "Luật: {context_section}", ("context_section",)),
            ExtractedMessage("user", "{question}", ("question",)),
        ],
    )
    assert _render_messages(prompt, {"context_section": "Điều 1", "question": "Hỏi"}) == [
        {"role": "system", "content": "Luật: Điều 1"},
        {"role": "user", "content": "Hỏi"},
    ]


def test_render_messages_raises_on_missing_variable():
    """Thà lỗi rõ còn hơn gửi prompt thiếu biến lên model."""
    prompt = ExtractedPrompt(
        name="p", messages=[ExtractedMessage("user", "{question}", ("question",))]
    )
    with pytest.raises(PromptUnavailable, match="thiếu biến"):
        _render_messages(prompt, {})


# ─── Registry: 3 tầng ────────────────────────────────────────────────────────


def test_registry_pulls_from_langsmith_and_writes_cache(cache_dir):
    reg = PromptRegistry(client=_FakeLangSmithClient())
    result = reg.render("rag-answer", context_section="ctx", question="Hỏi?")

    assert result.source == "langsmith"
    assert result.commit_hash == "deadbeefcafe"
    assert result.messages[0]["content"] == "Luật: ctx"
    assert (cache_dir / f"rag-answer__{settings.prompt_alias}.json").exists()


def test_registry_uses_project_prefix_for_identifier(cache_dir):
    seen = []

    class _Spy(_FakeLangSmithClient):
        def pull_prompt(self, identifier):
            seen.append(identifier)
            return super().pull_prompt(identifier)

    PromptRegistry(client=_Spy()).get("rag-answer")
    assert seen == [f"{settings.langsmith_project}-rag-answer:{settings.prompt_alias}"]


def test_registry_falls_back_to_cache_when_langsmith_down(cache_dir, caplog):
    PromptRegistry(client=_FakeLangSmithClient()).get("rag-answer")  # làm nóng cache

    result = PromptRegistry(client=_FakeLangSmithClient(fail=True)).render(
        "rag-answer", context_section="", question="Hỏi?"
    )

    assert result.source == "cache"
    assert "dùng bản CACHE đĩa" in caplog.text


def test_registry_raises_when_no_cache_and_langsmith_down(cache_dir):
    with pytest.raises(PromptUnavailable, match="không có LangSmith lẫn cache đĩa"):
        PromptRegistry(client=_FakeLangSmithClient(fail=True)).get("rag-answer")


def test_expired_cache_is_ignored(cache_dir):
    _write_cache(
        ExtractedPrompt(
            name="rag-answer",
            messages=[ExtractedMessage("user", "{question}", ("question",))],
            commit_hash="old",
            fetched_at=time.time() - CACHE_TTL_SECONDS - 1,
        ),
        settings.prompt_alias,
    )
    assert _read_cache("rag-answer", settings.prompt_alias) is None


def test_corrupt_cache_is_ignored_not_raised(cache_dir):
    path = cache_dir / f"rag-answer__{settings.prompt_alias}.json"
    path.write_text("{ không phải json", encoding="utf-8")
    assert _read_cache("rag-answer", settings.prompt_alias) is None


def test_unwritable_cache_does_not_break_render(tmp_path, monkeypatch, caplog):
    """Cache ghi hỏng (đĩa đầy/quyền) không được làm chết request."""
    monkeypatch.setattr(settings, "prompt_cache_dir", str(tmp_path / "a.txt" / "b"))
    (tmp_path / "a.txt").write_text("tệp thường chặn đường", encoding="utf-8")

    result = PromptRegistry(client=_FakeLangSmithClient()).render(
        "rag-answer", context_section="", question="Hỏi?"
    )
    assert result.source == "langsmith"
    assert "Không ghi được prompt cache" in caplog.text


# ─── templates.build_messages ────────────────────────────────────────────────


def test_build_messages_local_mode_matches_constant():
    got = templates.build_messages("Hỏi gì đó")
    assert got == [
        {"role": "system", "content": templates.LEGAL_SYSTEM_PROMPT},
        {"role": "user", "content": "Hỏi gì đó"},
    ]


def test_build_messages_local_mode_appends_context():
    chunk = SimpleNamespace(text="Điều 1: ...", source="luat.md")
    got = templates.build_messages("Hỏi", [chunk])
    assert got[0]["content"] == (
        templates.LEGAL_SYSTEM_PROMPT
        + "\n\n--- TÀI LIỆU THAM KHẢO ---\n[Nguồn 1: luat.md] Điều 1: ..."
    )


def test_build_messages_langsmith_mode_uses_registry(monkeypatch, cache_dir):
    monkeypatch.setattr(settings, "prompt_registry", "langsmith")
    monkeypatch.setattr(templates, "get_registry", lambda: PromptRegistry(
        client=_FakeLangSmithClient()
    ))

    got = templates.build_messages("Hỏi?", [SimpleNamespace(text="Điều 1", source="s.md")])
    assert got[0]["content"] == "Luật: \n\n--- TÀI LIỆU THAM KHẢO ---\n[Nguồn 1: s.md] Điều 1"
    assert got[1] == {"role": "user", "content": "Hỏi?"}


def test_build_messages_falls_back_to_constant_when_registry_dead(monkeypatch, cache_dir, caplog):
    """Bài học Section 4: registry sập KHÔNG được làm app chết."""
    monkeypatch.setattr(settings, "prompt_registry", "langsmith")
    monkeypatch.setattr(templates, "get_registry", lambda: PromptRegistry(
        client=_FakeLangSmithClient(fail=True)
    ))

    got = templates.build_messages("Hỏi?")
    assert got == [
        {"role": "system", "content": templates.LEGAL_SYSTEM_PROMPT},
        {"role": "user", "content": "Hỏi?"},
    ]
    assert "Prompt registry không dùng được" in caplog.text


# ─── Definitions ─────────────────────────────────────────────────────────────


def test_latest_per_name_picks_highest_version():
    assert latest_per_name()["rag-answer"] is RAG_ANSWER_V2


def test_all_prompts_are_unique_and_identifiers_use_project_prefix():
    assert len({(p.name, p.version) for p in ALL_PROMPTS}) == len(ALL_PROMPTS)
    assert RAG_ANSWER_V1.identifier == f"{settings.langsmith_project}-rag-answer"


def test_commit_message_carries_version_and_changelog():
    assert RAG_ANSWER_V2.commit_message.startswith("v2: ")


def test_cache_file_is_plain_json_without_langchain(cache_dir):
    """Cache đọc lại được mà không cần langchain-core — hot path không phụ thuộc LC."""
    _write_cache(_extract(_fake_prompt(), "rag-answer", "h1"), settings.prompt_alias)
    raw = json.loads((cache_dir / f"rag-answer__{settings.prompt_alias}.json").read_text())
    assert raw["messages"][0]["template"] == "Luật: {context_section}"
    assert raw["source"] == "langsmith"
