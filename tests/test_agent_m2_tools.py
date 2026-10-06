"""Test Module II, Bài 4 — Agentic Tool Design & Integration.

Không gọi API thật (mock embedding/LLM). Kiểm tra 3 mảng:
  - tools.py: error handling (không raise), idempotency, composite tool.
  - tool_selection.py: hierarchical grouping (classify_intent) + tool
    retrieval (embedding similarity, mock embed_query/embed_passages).
"""

from __future__ import annotations

from app.agent_m2 import tool_selection, tools

# ── tools.py: error handling (Section 1) — không raise, trả lỗi dạng text ────

def test_check_calendar_returns_error_text_on_bad_date():
    out = tools.check_calendar.func("not-a-date")
    assert out.startswith("Lỗi:")


def test_check_calendar_happy_path():
    out = tools.check_calendar.func("2026-08-20")
    assert "2026-08-20" in out
    assert not out.startswith("Lỗi:")


def test_create_event_returns_error_dict_on_bad_date():
    out = tools.create_event.func("Họp", "not-a-date", "15:00")
    assert "error" in out


def test_search_restaurant_returns_error_on_empty_location():
    out = tools.search_restaurant.func("")
    assert out.startswith("Lỗi:")


def test_draft_email_validates_address():
    out = tools.draft_email.func("khong-hop-le", "Chào", "nội dung")
    assert out.startswith("Lỗi:")


def test_create_task_validates_priority():
    out = tools.create_task.func("Viết báo cáo", priority="urgent")
    assert out.startswith("Lỗi:")


def test_log_workout_rejects_non_positive_duration():
    out = tools.log_workout.func("chạy bộ", 0)
    assert out.startswith("Lỗi:")


# ── tools.py: idempotency (Section 1) ─────────────────────────────────────────

def test_send_reminder_same_args_returns_already_done(monkeypatch):
    monkeypatch.setattr(tools, "_IDEMPOTENCY_STORE", {})

    first = tools.send_reminder.func("Họp team", "15:00")
    second = tools.send_reminder.func("Họp team", "15:00")

    assert first["status"] == "created"
    assert second["status"] == "already_done"
    assert first["idempotency_key"] == second["idempotency_key"]


def test_send_reminder_different_args_different_key(monkeypatch):
    monkeypatch.setattr(tools, "_IDEMPOTENCY_STORE", {})

    first = tools.send_reminder.func("Họp team", "15:00")
    second = tools.send_reminder.func("Gọi khách", "16:00")

    assert first["idempotency_key"] != second["idempotency_key"]
    assert second["status"] == "created"


def test_explicit_idempotency_key_overrides_auto_generated(monkeypatch):
    monkeypatch.setattr(tools, "_IDEMPOTENCY_STORE", {})

    out = tools.book_table.func("Quán A", 4, "19:00", idempotency_key="custom-key-1")
    assert out["idempotency_key"] == "custom-key-1"

    # Cùng key custom, tham số khác → vẫn coi là đã làm (idempotency theo KEY, không theo args).
    out2 = tools.book_table.func("Quán B", 2, "20:00", idempotency_key="custom-key-1")
    assert out2["status"] == "already_done"


def test_add_to_cart_rejects_non_positive_quantity(monkeypatch):
    monkeypatch.setattr(tools, "_IDEMPOTENCY_STORE", {})
    out = tools.add_to_cart.func("Áo thun", 0)
    assert "error" in out


# ── tools.py: composite tool (Section 1: Atomic vs Composite) ───────────────

def test_book_dinner_plan_combines_calendar_and_booking(monkeypatch):
    monkeypatch.setattr(tools, "_IDEMPOTENCY_STORE", {})

    out = tools.book_dinner_plan.func("2026-08-20", "19:00", "Quán A", 4)
    assert "calendar" in out and "booking" in out
    assert "2026-08-20" in out["calendar"]
    assert out["booking"]["restaurant"] == "Quán A"
    assert out["booking"]["party_size"] == 4


def test_book_dinner_plan_returns_error_on_bad_date():
    out = tools.book_dinner_plan.func("not-a-date", "19:00", "Quán A", 4)
    assert "error" in out


# ── tools.py: TOOLS registry ──────────────────────────────────────────────────

def test_tools_registry_has_15_tools_5_groups():
    assert len(tools.TOOL_GROUPS) == 5
    assert sum(len(v) for v in tools.TOOL_GROUPS.values()) == 15
    assert len(tools.TOOLS) == 16  # 15 domain tools + 1 composite


def test_sensitive_tools_are_all_side_effect_tools():
    tool_names = {t.name for t in tools.TOOLS}
    assert tools.SENSITIVE_TOOLS.issubset(tool_names)
    # Tool chỉ đọc (không side-effect) không nên nằm trong SENSITIVE_TOOLS.
    assert "check_calendar" not in tools.SENSITIVE_TOOLS
    assert "search_restaurant" not in tools.SENSITIVE_TOOLS


# ── tool_selection.py: Hierarchical Grouping ─────────────────────────────────

def test_classify_intent_returns_valid_domain(monkeypatch):
    from app.llm import completion
    monkeypatch.setattr(completion, "chat", lambda messages, params: "dining")

    assert tool_selection.classify_intent("tìm nhà hàng gần đây") == "dining"


def test_classify_intent_returns_empty_on_unknown_domain(monkeypatch):
    from app.llm import completion
    monkeypatch.setattr(completion, "chat", lambda messages, params: "unknown_domain")

    assert tool_selection.classify_intent("câu hỏi lạ") == ""


def test_get_tools_by_group_returns_correct_tools():
    dining_tools = tool_selection.get_tools_by_group("dining")
    assert {t.name for t in dining_tools} == {"search_restaurant", "book_table", "order_food"}


def test_get_tools_by_group_unknown_group_returns_empty():
    assert tool_selection.get_tools_by_group("nonexistent") == []


# ── tool_selection.py: Tool Retrieval (embedding similarity, mock embed) ────

def _fake_vector_for(text: str) -> list[float]:
    """Vector giả — 1 chiều 'điểm domain' để test similarity mà không gọi API thật.

    Mỗi domain có 1 "trục" riêng trong vector để cosine similarity phân biệt
    rõ ràng: mô tả tool càng khớp domain câu hỏi, similarity càng cao.
    """
    domains = ["calendar", "dining", "productivity", "shopping", "wellness"]
    vec = [0.0] * len(domains)
    lowered = text.lower()
    for i, d in enumerate(domains):
        if d in lowered:
            vec[i] = 1.0
    if not any(vec):
        vec[0] = 0.01  # tránh vector 0 tuyệt đối
    return vec


async def _fake_get_mcp_tools() -> list:
    """MCP tool rỗng cho test — không spawn subprocess server thật."""
    return []


async def test_retrieve_relevant_tools_picks_matching_domain(monkeypatch):
    tool_selection.reset_cache()

    def fake_embed_passages(texts):
        # Gắn domain vào description giả lập để _fake_vector_for nhận diện đúng.
        tagged = []
        for t in tools.TOOLS:
            group = next((g for g, members in tools.TOOL_GROUPS.items() if t in members), "composite")
            tagged.append(_fake_vector_for(group))
        return tagged

    monkeypatch.setattr("app.retrieval.embeddings.embed_passages", fake_embed_passages)
    monkeypatch.setattr("app.retrieval.embeddings.embed_query", lambda q: _fake_vector_for(q))
    monkeypatch.setattr("app.agent_m2.mcp.client.get_mcp_tools", _fake_get_mcp_tools)

    results = await tool_selection.retrieve_relevant_tools("tôi muốn đặt lịch calendar", k=3)
    result_names = {t.name for t in results}
    calendar_names = {t.name for t in tool_selection.get_tools_by_group("calendar")}

    assert result_names.issubset(calendar_names)
    tool_selection.reset_cache()


async def test_retrieve_relevant_tools_respects_k(monkeypatch):
    tool_selection.reset_cache()

    monkeypatch.setattr(
        "app.retrieval.embeddings.embed_passages",
        lambda texts: [[1.0, 0.0] for _ in texts],
    )
    monkeypatch.setattr("app.retrieval.embeddings.embed_query", lambda q: [1.0, 0.0])
    monkeypatch.setattr("app.agent_m2.mcp.client.get_mcp_tools", _fake_get_mcp_tools)

    results = await tool_selection.retrieve_relevant_tools("bất kỳ câu hỏi nào", k=4)
    assert len(results) == 4
    tool_selection.reset_cache()


async def test_tool_index_is_cached_across_calls(monkeypatch):
    tool_selection.reset_cache()
    call_count = {"n": 0}

    def fake_embed_passages(texts):
        call_count["n"] += 1
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr("app.retrieval.embeddings.embed_passages", fake_embed_passages)
    monkeypatch.setattr("app.retrieval.embeddings.embed_query", lambda q: [1.0, 0.0])
    monkeypatch.setattr("app.agent_m2.mcp.client.get_mcp_tools", _fake_get_mcp_tools)

    await tool_selection.retrieve_relevant_tools("câu hỏi 1", k=3)
    await tool_selection.retrieve_relevant_tools("câu hỏi 2", k=3)

    assert call_count["n"] == 1  # index tool chỉ embed 1 LẦN, không mỗi lượt gọi
    tool_selection.reset_cache()


# ── tool_selection.py: MCP tool gộp vào index (Section 4) ────────────────────

async def test_retrieve_relevant_tools_includes_mcp_tools(monkeypatch):
    """Tool từ MCP server phải nằm TRONG index chung — retrieval không phân biệt nguồn gốc.

    Dùng vector RIÊNG (không đụng domain nào của tools.py) cho MCP tool để phép
    so sánh similarity không mập mờ với 3 tool wellness nội bộ.
    """
    tool_selection.reset_cache()

    class _FakeMCPTool:
        def __init__(self, name: str, description: str):
            self.name = name
            self.description = description

    fake_mcp_tool = _FakeMCPTool("log_workout_tool", "Ghi lại buổi tập thể dục qua MCP")
    mcp_vector = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # chiều riêng, không trùng domain nào khác

    async def fake_get_mcp_tools() -> list:
        return [fake_mcp_tool]

    def fake_embed_passages(texts):
        # Tool cuối cùng trong danh sách LUÔN là MCP tool (xem _all_tools: TOOLS + mcp_tools).
        return [[0.0] * 6 for _ in texts[:-1]] + [mcp_vector]

    monkeypatch.setattr("app.agent_m2.mcp.client.get_mcp_tools", fake_get_mcp_tools)
    monkeypatch.setattr("app.retrieval.embeddings.embed_passages", fake_embed_passages)
    monkeypatch.setattr("app.retrieval.embeddings.embed_query", lambda q: mcp_vector)

    results = await tool_selection.retrieve_relevant_tools("bất kỳ câu gì", k=1)
    assert results[0].name == "log_workout_tool"
    tool_selection.reset_cache()
