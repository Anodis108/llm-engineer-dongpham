"""Test Module II, Bài 5 — Agent Evaluation & Observability.

Không gọi API thật (mock chat_parsed/LLM). Kiểm tra 2 mảng:
  - eval.py: evaluate_task_success, evaluate_trajectory, evaluate_run.
  - graph.py: _extract_trajectory/_extract_task (dựng lại trajectory từ
    checkpointer state) và evaluate_thread (đọc state, gọi eval, xử lý lỗi).

Dùng ĐÚNG model Pydantic thật (TaskSuccessResult/TrajectoryResult) làm fake
return value — không phải plain class — vì TrajectoryResult.overall là
@property tính từ 4 field, plain mock sẽ thiếu nó.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent_m2 import eval as agent_eval
from app.agent_m2 import graph as graph_mod

# ── eval.py: evaluate_task_success / evaluate_trajectory ────────────────────

def test_evaluate_task_success_calls_chat_parsed(monkeypatch):
    fake = agent_eval.TaskSuccessResult(success=True, score=0.9, reasoning="Đã trả lời đúng yêu cầu.")
    monkeypatch.setattr(agent_eval.completion, "chat_parsed", lambda messages, schema, params: fake)

    out = agent_eval.evaluate_task_success("Tìm nhà hàng gần đây", "3 nhà hàng: A, B, C")
    assert out.success is True
    assert out.score == 0.9


def test_evaluate_trajectory_formats_steps_and_calls_chat_parsed(monkeypatch):
    seen = {}
    fake = agent_eval.TrajectoryResult(
        efficiency=4, logical_order=5, tool_correctness=5, recovery=5, issues=[]
    )

    def fake_chat_parsed(messages, schema, params):
        seen["messages"] = messages
        return fake

    monkeypatch.setattr(agent_eval.completion, "chat_parsed", fake_chat_parsed)

    trajectory = [
        {"tool": "check_calendar", "args": {"start_date": "2026-08-20"}, "observation": "Rảnh cả ngày."},
    ]
    out = agent_eval.evaluate_trajectory("Chiều nay tôi rảnh không?", trajectory)

    assert out.overall == 4.75
    # Prompt phải chứa tên tool đã gọi.
    assert any("check_calendar" in str(m) for m in seen["messages"])


def test_evaluate_trajectory_handles_empty_trajectory(monkeypatch):
    fake = agent_eval.TrajectoryResult(
        efficiency=5, logical_order=5, tool_correctness=5, recovery=5, issues=[]
    )
    monkeypatch.setattr(agent_eval.completion, "chat_parsed", lambda messages, schema, params: fake)

    out = agent_eval.evaluate_trajectory("Xin chào", [])
    assert out.overall == 5.0


def test_evaluate_run_combines_both_dimensions(monkeypatch):
    fake_success = agent_eval.TaskSuccessResult(success=True, score=1.0, reasoning="ok")
    fake_trajectory = agent_eval.TrajectoryResult(
        efficiency=5, logical_order=5, tool_correctness=5, recovery=5, issues=[]
    )

    def fake_chat_parsed(messages, schema, params):
        return fake_success if schema is agent_eval.TaskSuccessResult else fake_trajectory

    monkeypatch.setattr(agent_eval.completion, "chat_parsed", fake_chat_parsed)

    result = agent_eval.evaluate_run("task", "output", [{"tool": "x", "args": {}, "observation": "y"}])
    assert result.task_success.success is True
    assert result.trajectory.overall == 5.0
    assert result.step_count == 1


# ── graph.py: _extract_trajectory / _extract_task ────────────────────────────

def test_extract_trajectory_matches_tool_calls_to_observations():
    messages = [
        HumanMessage(content="Chiều nay tôi rảnh không?"),
        AIMessage(
            content="",
            tool_calls=[{"name": "check_calendar", "args": {"start_date": "2026-08-20"}, "id": "call_1"}],
        ),
        ToolMessage(content="Rảnh cả ngày.", tool_call_id="call_1"),
        AIMessage(content="Chiều nay bạn rảnh."),
    ]
    trajectory = graph_mod._extract_trajectory(messages)

    assert trajectory == [
        {"tool": "check_calendar", "args": {"start_date": "2026-08-20"}, "observation": "Rảnh cả ngày."}
    ]


def test_extract_trajectory_empty_when_no_tool_calls():
    messages = [HumanMessage(content="Xin chào"), AIMessage(content="Chào bạn!")]
    assert graph_mod._extract_trajectory(messages) == []


def test_extract_trajectory_handles_multiple_steps_in_order():
    messages = [
        HumanMessage(content="task"),
        AIMessage(content="", tool_calls=[{"name": "tool_a", "args": {"x": 1}, "id": "c1"}]),
        ToolMessage(content="obs_a", tool_call_id="c1"),
        AIMessage(content="", tool_calls=[{"name": "tool_b", "args": {"y": 2}, "id": "c2"}]),
        ToolMessage(content="obs_b", tool_call_id="c2"),
        AIMessage(content="Xong."),
    ]
    trajectory = graph_mod._extract_trajectory(messages)
    assert [s["tool"] for s in trajectory] == ["tool_a", "tool_b"]
    assert [s["observation"] for s in trajectory] == ["obs_a", "obs_b"]


def test_extract_task_returns_first_human_message():
    messages = [
        HumanMessage(content="Yêu cầu gốc"),
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "c1"}]),
        ToolMessage(content="obs", tool_call_id="c1"),
        AIMessage(content="trả lời"),
    ]
    assert graph_mod._extract_task(messages) == "Yêu cầu gốc"


def test_extract_task_empty_when_no_human_message():
    assert graph_mod._extract_task([AIMessage(content="hi")]) == ""


# ── graph.py: evaluate_thread ─────────────────────────────────────────────────
# _build_graph là @lru_cache — không monkeypatch cả hàm (mất .cache_clear cho
# test khác), thay vào đó patch app/agent_m2/graph.evaluate_run và tự nạp sẵn
# state vào checkpointer thật qua graph đã build (giữ nguyên _build_graph()).

async def test_evaluate_thread_returns_error_when_no_conversation():
    out = await graph_mod.evaluate_thread("thread-eval-empty")
    assert "error" in out


async def test_evaluate_thread_returns_error_when_pending_approval(monkeypatch):
    app = graph_mod._build_graph()
    config = {"configurable": {"thread_id": "thread-eval-pending"}}
    await app.aupdate_state(
        config,
        {
            "messages": [
                HumanMessage(content="Nhắc tôi họp"),
                AIMessage(content="", tool_calls=[{"name": "send_reminder", "args": {}, "id": "c1"}]),
            ]
        },
    )

    out = await graph_mod.evaluate_thread("thread-eval-pending")
    assert "error" in out


async def test_evaluate_thread_evaluates_completed_conversation(monkeypatch):
    app = graph_mod._build_graph()
    config = {"configurable": {"thread_id": "thread-eval-done"}}
    await app.aupdate_state(
        config,
        {
            "messages": [
                HumanMessage(content="Chiều nay tôi rảnh không?"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "check_calendar", "args": {"start_date": "2026-08-20"}, "id": "c1"}],
                ),
                ToolMessage(content="Rảnh cả ngày.", tool_call_id="c1"),
                AIMessage(content="Chiều nay bạn rảnh."),
            ]
        },
    )

    fake_success = agent_eval.TaskSuccessResult(success=True, score=1.0, reasoning="ok")
    fake_trajectory = agent_eval.TrajectoryResult(
        efficiency=5, logical_order=5, tool_correctness=5, recovery=5, issues=[]
    )

    def fake_evaluate_run(task, final_output, trajectory):
        return agent_eval.AgentEvalResult(
            task_success=fake_success, trajectory=fake_trajectory, step_count=len(trajectory)
        )

    monkeypatch.setattr(graph_mod, "evaluate_run", fake_evaluate_run)

    out = await graph_mod.evaluate_thread("thread-eval-done")
    assert "error" not in out
    assert out["result"].task_success.success is True
    assert out["trajectory"] == [
        {"tool": "check_calendar", "args": {"start_date": "2026-08-20"}, "observation": "Rảnh cả ngày."}
    ]
