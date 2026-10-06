"""Test Module II, Bài 3 — Memory & Context Engineering.

Không gọi API thật: mock embedding/LLM. Kiểm tra 3 mảng:
  - context.py: sliding window, usage estimate, summarization, compression, re-inject.
  - memory.py: fallback store recall/save theo user_id (không cần embedding).
  - nodes/graph: recall_node chèn context, extract_and_store_node lưu fact,
    luồng recall → agent → store chạy đúng.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from app.agent_m2 import context, memory, nodes
from app.agent_m2 import graph as graph_mod

# ── context.py: sliding window ────────────────────────────────────────────────

def test_sliding_window_keeps_recent_and_system():
    msgs = [{"role": "system", "content": "sys"}] + [
        {"role": "user", "content": f"m{i}"} for i in range(30)
    ]
    out = context.sliding_window(msgs, max_messages=5)
    assert len(out) == 6  # 1 system + 5 recent
    assert out[0]["content"] == "sys"
    assert out[-1]["content"] == "m29"


def test_sliding_window_noop_when_short():
    msgs = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    assert context.sliding_window(msgs, max_messages=20) == msgs


def test_sliding_window_no_duplicate_system_when_in_recent():
    """System message nằm trong N recent không bị thêm lần 2."""
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}]
    out = context.sliding_window(msgs, max_messages=5)
    assert out.count({"role": "system", "content": "sys"}) == 1


# ── context.py: usage estimate + 40-60% rule ─────────────────────────────────

def test_estimate_tokens_and_usage():
    msgs = [{"role": "user", "content": "x" * 400}]  # ~100 tokens (4 chars/token)
    assert context.estimate_tokens(msgs) == 100
    assert context.context_usage(msgs, window_tokens=1000) == 0.1


def test_should_compact_triggers_above_threshold():
    msgs = [{"role": "user", "content": "x" * 4000}]  # ~1000 tokens
    assert context.should_compact(msgs, window_tokens=2000, threshold=0.40) is True
    assert context.should_compact(msgs, window_tokens=10000, threshold=0.40) is False


# ── context.py: summarization (mock LLM) ─────────────────────────────────────

def test_summarize_old_messages_replaces_old_keeps_recent(monkeypatch):
    monkeypatch.setattr(context.completion, "chat", lambda messages, params: "TÓM TẮT")
    msgs = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    out = context.summarize_old_messages(msgs, keep_recent=6)
    assert out[0]["role"] == "system"
    assert "TÓM TẮT" in out[0]["content"]
    assert len(out) == 7  # 1 summary + 6 recent
    assert out[-1]["content"] == "m19"


def test_summarize_noop_when_short(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(context.completion, "chat", lambda *a, **k: called.update(n=1))
    msgs = [{"role": "user", "content": "a"}]
    assert context.summarize_old_messages(msgs, keep_recent=6) == msgs
    assert called["n"] == 0  # không gọi LLM


# ── context.py: tool-output compression (mock LLM) ───────────────────────────

def test_compress_tool_result_skips_short_output(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(context.completion, "chat", lambda *a, **k: called.update(n=1))
    short = "kết quả ngắn"
    assert context.compress_tool_result(short, "query", max_tokens=300) == short
    assert called["n"] == 0  # ngắn → không nén, không tốn LLM


def test_compress_tool_result_compresses_long_output(monkeypatch):
    monkeypatch.setattr(context.completion, "chat", lambda messages, params: "ĐÃ NÉN")
    long = "x" * 2000  # > 300*4
    assert context.compress_tool_result(long, "query", max_tokens=300) == "ĐÃ NÉN"


# ── context.py: re-injection ─────────────────────────────────────────────────

def test_reinject_appends_instruction_at_end():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}]
    out = context.reinject_instructions(msgs, "QUY TẮC")
    assert out[-1]["role"] == "system"
    assert "QUY TẮC" in out[-1]["content"]
    assert len(out) == 3


# ── memory.py: fallback store (không embedding) ──────────────────────────────

def test_fallback_memory_save_and_recall_by_user(monkeypatch):
    monkeypatch.setattr(memory, "_use_qdrant", lambda: False)
    memory._FALLBACK.clear()

    memory.save_to_long_term("minh", "Minh bị dị ứng hải sản")
    memory.save_to_long_term("lan", "Lan thích món chay")

    minh_mem = memory.recall_long_term("minh", "gợi ý món ăn dị ứng", k=3)
    assert any("dị ứng" in m for m in minh_mem)
    # Không lẫn memory của user khác.
    assert all("Lan" not in m for m in minh_mem)


def test_fallback_recall_empty_when_no_memory(monkeypatch):
    monkeypatch.setattr(memory, "_use_qdrant", lambda: False)
    memory._FALLBACK.clear()
    assert memory.recall_long_term("ai", "bất kỳ", k=3) == []


def test_save_ignores_empty_fact(monkeypatch):
    monkeypatch.setattr(memory, "_use_qdrant", lambda: False)
    memory._FALLBACK.clear()
    memory.save_to_long_term("minh", "   ")
    assert memory._FALLBACK == []


# ── nodes.py: recall_node + extract_and_store_node ───────────────────────────

def test_recall_node_injects_memory_as_context(monkeypatch):
    monkeypatch.setattr(nodes.memory, "recall_long_term", lambda uid, q, k=3: ["Minh dị ứng hải sản"])

    out = nodes.recall_node({"messages": [HumanMessage(content="gợi ý món tối")], "user_id": "minh"})
    assert out["messages"][0]["role"] == "system"
    assert "dị ứng hải sản" in out["messages"][0]["content"]


def test_recall_node_noop_without_user_id():
    assert nodes.recall_node({"messages": [HumanMessage(content="hi")]}) == {}


def test_recall_node_noop_when_no_memory(monkeypatch):
    monkeypatch.setattr(nodes.memory, "recall_long_term", lambda uid, q, k=3: [])
    out = nodes.recall_node({"messages": [HumanMessage(content="hi")], "user_id": "minh"})
    assert out == {}


def test_extract_and_store_saves_fact(monkeypatch):
    saved = []
    monkeypatch.setattr(nodes.memory, "save_to_long_term", lambda uid, fact: saved.append((uid, fact)))
    from app.llm import completion
    monkeypatch.setattr(completion, "chat", lambda messages, params: "Minh làm ở công ty ABC")

    out = nodes.extract_and_store_node({
        "messages": [HumanMessage(content="tôi làm ở ABC"), AIMessage(content="ok")],
        "user_id": "minh",
    })
    assert out == {}
    assert saved == [("minh", "Minh làm ở công ty ABC")]


def test_extract_and_store_skips_none(monkeypatch):
    saved = []
    monkeypatch.setattr(nodes.memory, "save_to_long_term", lambda uid, fact: saved.append(fact))
    from app.llm import completion
    monkeypatch.setattr(completion, "chat", lambda messages, params: "NONE")

    nodes.extract_and_store_node({
        "messages": [HumanMessage(content="hôm nay trời đẹp"), AIMessage(content="vâng")],
        "user_id": "minh",
    })
    assert saved == []  # "NONE" → không lưu


def test_extract_and_store_noop_without_user_id(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(nodes.memory, "save_to_long_term", lambda *a: called.update(n=1))
    nodes.extract_and_store_node({"messages": [HumanMessage(content="hi")]})
    assert called["n"] == 0


# ── Full graph: recall → agent → store (LangGraph thật, LLM mock) ────────────

async def test_full_graph_recalls_then_stores(monkeypatch):
    """End-to-end: agent trả lời (không tool call) → graph phải qua recall + store."""
    graph_mod._build_graph.cache_clear()

    recalled = {"n": 0}
    stored = []
    monkeypatch.setattr(nodes.memory, "recall_long_term", lambda uid, q, k=3: (recalled.update(n=recalled["n"] + 1) or ["Minh dị ứng hải sản"]))
    monkeypatch.setattr(nodes.memory, "save_to_long_term", lambda uid, fact: stored.append((uid, fact)))

    class FakeLLM:
        async def ainvoke(self, messages):
            # Kiểm tra recall context đã được chèn vào messages gửi cho LLM.
            assert any("dị ứng hải sản" in str(m) for m in messages)
            return AIMessage(content="Gợi ý món chay cho bạn.")

    async def fake_llm_with_tools(query=""):
        return FakeLLM()

    monkeypatch.setattr(nodes, "_llm_with_tools", fake_llm_with_tools)
    # store_node gọi native completion.chat để trích fact.
    from app.llm import completion
    monkeypatch.setattr(completion, "chat", lambda messages, params: "Minh thích ăn chay")

    result = await graph_mod.start_conversation("thread-mem-1", "Gợi ý món tối nay", user_id="minh")

    assert result["status"] == "done"
    assert result["answer"] == "Gợi ý món chay cho bạn."
    assert recalled["n"] == 1              # đã recall
    assert stored == [("minh", "Minh thích ăn chay")]  # đã store
    graph_mod._build_graph.cache_clear()


# ── nodes.py: should_compact_route / compact_node (persist compaction) ───────

def test_should_compact_route_to_compact_when_over_threshold(monkeypatch):
    monkeypatch.setattr(nodes.settings, "agent_context_window_tokens", 100)
    big_history = [HumanMessage(content="x" * 400) for _ in range(5)]  # ~500 tokens > 40% of 100
    assert nodes.should_compact_route({"messages": big_history}) == "compact"


def test_should_compact_route_to_agent_when_under_threshold(monkeypatch):
    monkeypatch.setattr(nodes.settings, "agent_context_window_tokens", 100_000)
    small_history = [HumanMessage(content="hi")]
    assert nodes.should_compact_route({"messages": small_history}) == "agent"


def test_compact_node_removes_old_and_persists_summary(monkeypatch):
    """compact_node phải trả RemoveMessage cho phần cũ + 1 SystemMessage tóm tắt.

    Đây là điểm khác biệt cốt lõi so với summarize_old_messages (context.py):
    kết quả GHI ĐÈ state qua reducer add_messages, không phải list tạm bị vứt đi.
    """
    from langchain_core.messages import RemoveMessage, SystemMessage

    monkeypatch.setattr(nodes.settings, "agent_keep_recent_messages", 2)
    monkeypatch.setattr(nodes.context, "summarize_text", lambda old: "TÓM TẮT")

    old_msgs = [HumanMessage(content=f"m{i}", id=f"id{i}") for i in range(5)]
    recent_msgs = [HumanMessage(content="giữ lại 1"), HumanMessage(content="giữ lại 2")]
    out = nodes.compact_node({"messages": old_msgs + recent_msgs})

    removals = [m for m in out["messages"] if isinstance(m, RemoveMessage)]
    summaries = [m for m in out["messages"] if isinstance(m, SystemMessage)]

    assert {r.id for r in removals} == {f"id{i}" for i in range(5)}
    assert len(summaries) == 1
    assert "TÓM TẮT" in summaries[0].content
    # recent_msgs KHÔNG bị đưa vào RemoveMessage — vẫn còn nguyên trong state.
    assert not any(getattr(m, "id", None) in (None,) for m in removals)


def test_compact_node_noop_guard_when_not_enough_history():
    """Guard an toàn: nếu gọi trực tiếp (không qua route) mà history ngắn → không làm gì."""
    monkeypatch_history = [HumanMessage(content="chỉ 1 tin nhắn")]
    out = nodes.compact_node({"messages": monkeypatch_history})
    assert out == {}


async def test_full_graph_compacts_once_and_reuses_across_tool_loop(monkeypatch):
    """Điểm mấu chốt kiến trúc: compact CHỈ chạy 1 lần/lượt (ngay sau recall, TRƯỚC
    khi vào vòng agent⇄tools) — dù vòng lặp gọi agent_node 2 lần (trước và sau khi
    tool chạy), summarize_text chỉ được gọi ĐÚNG 1 LẦN cho cả lượt."""
    graph_mod._build_graph.cache_clear()

    monkeypatch.setattr(nodes.settings, "agent_context_window_tokens", 200)
    monkeypatch.setattr(nodes.settings, "agent_keep_recent_messages", 1)

    summarize_calls = {"n": 0}

    def fake_summarize_text(old_messages):
        summarize_calls["n"] += 1
        return "TÓM TẮT"

    monkeypatch.setattr(nodes.context, "summarize_text", fake_summarize_text)

    call_1 = AIMessage(
        content="",
        tool_calls=[{"name": "check_calendar", "args": {"start_date": "2026-08-13"}, "id": "c1"}],
    )
    call_2 = AIMessage(content="Xong rồi.")
    responses = iter([call_1, call_2])

    class FakeLLM:
        async def ainvoke(self, messages):
            return next(responses)

    async def fake_llm_with_tools(query=""):
        return FakeLLM()

    monkeypatch.setattr(nodes, "_llm_with_tools", fake_llm_with_tools)

    # Nạp sẵn nhiều message cũ vào checkpointer trước lượt này, đủ dài để vượt
    # ngưỡng compact NGAY khi recall xong (cần > 1 message để có phần "cũ" nén được).
    config = {"configurable": {"thread_id": "thread-compact-1"}}
    graph_app = graph_mod._build_graph()
    await graph_app.aupdate_state(
        config,
        {"messages": [{"role": "user", "content": "x" * 400} for _ in range(3)]},
    )

    pending = await graph_mod.start_conversation("thread-compact-1", "Chiều nay tôi có lịch gì?")
    assert pending["status"] == "pending_approval"  # dừng trước check_calendar (HITL)

    result = await graph_mod.resume_conversation("thread-compact-1", approve=True)

    assert result["status"] == "done"
    assert result["answer"] == "Xong rồi."
    # 2 lần gọi agent_node (trước & sau tool) trong CÙNG 1 lượt, nhưng compact chỉ
    # chạy ở "recall → compact" MỘT LẦN — không nén lại khi quay lại agent sau tool.
    assert summarize_calls["n"] == 1
    graph_mod._build_graph.cache_clear()
