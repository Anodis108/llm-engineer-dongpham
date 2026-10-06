"""Graph — Module II, Bài 2-4 (LangGraph + Memory/Context + Tool Design).

Luồng Bài 2 (ReAct loop):  START → agent → (tool call?) → tools → agent → ... → END

Luồng Bài 3 (thêm memory + compaction):

    START → recall → [vượt ngưỡng 40%?] ─(có)→ compact ─┐
                                        └─(không)────────┴→ agent → (tool call?)
                                                                       ├─(có)→ tools → agent → ...
                                                                       └─(không)→ store → END

  - recall (đầu mỗi lượt):  đọc long-term memory liên quan (memory.py) → chèn vào context.
  - compact (đầu mỗi lượt, CÓ ĐIỀU KIỆN): nén history cũ, GHI ĐÈ state.messages
    (RemoveMessage + summary) — bản nén PERSIST nên các vòng agent⇄tools sau đó
    trong CÙNG lượt kế thừa, không nén lại. Tách khỏi agent_node có chủ đích, xem
    docstring compact_node/agent_node trong nodes.py (kỷ luật "one node one
    function" + lý do hiệu năng: tránh nén lặp lại mỗi vòng lặp mà không tái dùng).
  - store (cuối mỗi lượt): trích 1 sự thật đáng nhớ → ghi long-term memory.

short-term (messages) vẫn sống trong state; long-term ở vector store ngoài.

Bài 2, Section 5 (Memory kỹ thuật): compile với MemorySaver — state lưu theo
`thread_id`, agent "nhớ" hội thoại qua nhiều lượt invoke() (SHORT-TERM). Khác
với long-term memory Bài 3 (xuyên nhiều thread/session, ở vector store).

Bài 2, Section 6 (HITL): compile với `interrupt_before=["tools"]` — graph luôn
dừng TRƯỚC khi chạy tool, chờ con người phê duyệt qua resume_conversation().

Bài 4 (Tool Design, Section 2): tools.py giờ có 15 tool (5 domain + 1 composite)
— "tool sprawl" thật theo ngưỡng bài học. `ToolNode(TOOLS)` ở đây vẫn nhận ĐẦY
ĐỦ TOOLS vì nó chỉ chịu trách nhiệm THỰC THI tool khi được gọi (không quan tâm
model "thấy" bao nhiêu tool). Phần giới hạn số tool model THẤY (bind_tools) nằm
ở agent_node (nodes.py) — bind theo top-k tool liên quan tới câu hỏi hiện tại
qua tool_selection.retrieve_relevant_tools (embedding similarity).

Bài 4, Section 4 (MCP): agent_node giờ là async (cần await MCP client khi load
tool lần đầu) — start_conversation/resume_conversation ở đây chuyển thành
`async def`, dùng `ainvoke`/`aget_state`/`aupdate_state` thay vì bản sync
(Bài 2-3). routes_assistant.py (FastAPI) gọi `await` các hàm này — FastAPI hỗ
trợ route handler async natively, không cần thay đổi gì thêm ở tầng HTTP.

Bài 5 (Observability, Section 3): start_conversation/resume_conversation bọc
qua trace_answer (monitoring/tracing.py, tái dùng nguyên xi từ Module I —
KHÔNG viết lại LangSmith integration). Khác app/agent/nodes.py (CRAG) vốn tạo
NESTED span cho từng node graph: ở đây trace 1 span PHẲNG cho cả lượt (giống
cách trace_stream trace streaming) — đơn giản hơn, đủ để thấy latency/input/
output mỗi lượt trên LangSmith dashboard mà không phải xuyên `_trace_span` qua
5 node đã ổn định từ Bài 2-4. evaluate_run (eval.py, Section 1-2) chấm CHẤT
LƯỢNG (offline, on-demand qua nút "Đánh giá" ở UI) — khác trace (observability,
luôn bật, không cần bấm nút) dù cả 2 cùng nhìn vào 1 lượt chạy.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.agent_m2.eval import evaluate_run
from app.agent_m2.nodes import (
    agent_node,
    compact_node,
    extract_and_store_node,
    recall_node,
    should_compact_route,
    should_continue,
)
from app.agent_m2.state import AssistantState
from app.agent_m2.tools import TOOLS
from app.monitoring.tracing import trace_answer

# MemorySaver lưu trong RAM — đủ cho demo/dev. Production dùng SqliteSaver/
# PostgresSaver (bền vững qua restart), xem ghi chú trong bài học Section 5.
_checkpointer = MemorySaver()


@lru_cache(maxsize=1)
def _build_graph():
    graph = StateGraph(AssistantState)
    graph.add_node("recall", recall_node)      # Bài 3: đọc long-term memory
    graph.add_node("compact", compact_node)    # Bài 3: nén history (persist state)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("store", extract_and_store_node)  # Bài 3: ghi long-term memory

    graph.add_edge(START, "recall")
    # Sau recall: history vượt ngưỡng 40% → compact (nén + ghi đè state) rồi mới agent;
    # chưa vượt → thẳng tới agent. Bản nén được persist nên vòng sau không nén lại.
    graph.add_conditional_edges(
        "recall", should_compact_route, {"compact": "compact", "agent": "agent"}
    )
    graph.add_edge("compact", "agent")
    # Xong vòng lặp (không còn tool call) → store thay vì END trực tiếp.
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: "store"})
    graph.add_edge("tools", "agent")
    graph.add_edge("store", END)

    return graph.compile(checkpointer=_checkpointer, interrupt_before=["tools"])


def _pending_tool_call(result: dict) -> dict | None:
    """Trích tool call agent đang chờ duyệt từ state, None nếu graph đã kết thúc bình thường."""
    last_message = result["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None)
    return tool_calls[0] if tool_calls else None


def _extract_trajectory(messages: list) -> list[dict]:
    """Bài 5, Section 2 (Trajectory Evaluation): dựng lại chuỗi hành động từ
    checkpointer state — [{"tool", "args", "observation"}, ...] theo đúng thứ
    tự thực thi, để evaluate_trajectory() chấm được cả CON ĐƯỜNG, không chỉ
    câu trả lời cuối.

    Map tool_call_id -> {name, args} (từ AIMessage.tool_calls) rồi khớp với
    ToolMessage tương ứng (observation) — vì 2 thông tin này nằm ở 2 message
    khác nhau trong lịch sử.
    """
    calls_by_id: dict[str, dict] = {}
    for m in messages:
        for call in getattr(m, "tool_calls", None) or []:
            calls_by_id[call["id"]] = {"tool": call["name"], "args": call["args"]}

    trajectory = []
    for m in messages:
        tool_call_id = getattr(m, "tool_call_id", None)
        if tool_call_id and tool_call_id in calls_by_id:
            trajectory.append({**calls_by_id[tool_call_id], "observation": m.content})
    return trajectory


def _extract_task(messages: list) -> str:
    """Yêu cầu gốc của user = HumanMessage ĐẦU TIÊN trong lịch sử lượt hiện tại."""
    for m in messages:
        if getattr(m, "type", None) == "human":
            return m.content
    return ""


async def start_conversation(thread_id: str, message: str, user_id: str = "") -> dict:
    """Gửi lượt đầu/tiếp theo của hội thoại. Có thể dừng giữa chừng chờ duyệt tool.

    `user_id` (Bài 3): định danh user để recall/store long-term memory. Bỏ trống
    → agent chạy như Bài 2 (không dùng long-term memory).

    async (Bài 4): agent_node cần await MCP client — dùng ainvoke() thay vì
    invoke() (khác Bài 2-3 vốn sync hoàn toàn).

    Bài 5 (Observability): bọc qua trace_answer — 1 span LangSmith/lượt, gắn
    input (message)/output (answer hoặc trạng thái chờ duyệt)/latency. No-op
    hoàn toàn nếu MONITORING_ENABLED=false (xem monitoring/tracing.py).

    Returns:
        {"status": "done", "answer": str} — agent trả lời xong, không cần duyệt gì.
        {"status": "pending_approval", "tool_call": {...}} — graph dừng trước tool node.
    """
    app = _build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial: AssistantState = {"messages": [{"role": "user", "content": message}]}
    if user_id:
        initial["user_id"] = user_id

    with trace_answer("assistant_message", message, metadata={"thread_id": thread_id}) as t:
        result = await app.ainvoke(initial, config=config)
        response = _to_response(result)
        t["output"] = response.get("answer") or response
    return response


async def resume_conversation(thread_id: str, approve: bool, rejection_note: str = "") -> dict:
    """Con người phê duyệt/từ chối tool call đang chờ, rồi tiếp tục graph (Section 6).

    Từ chối: thay vì để agent gọi tool, chèn 1 tool message báo lỗi/từ chối —
    agent đọc được lý do và tự điều chỉnh (giống error recovery ở Section 3),
    thay vì graph crash hoặc treo.

    Bài 5: cũng bọc qua trace_answer — span riêng cho "phần resume" của lượt,
    tách khỏi span của start_conversation vì đây là 2 lời gọi HTTP khác nhau.
    """
    app = _build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    if not approve:
        state = await app.aget_state(config)
        last_message = state.values["messages"][-1]
        call = last_message.tool_calls[0]
        await app.aupdate_state(
            config,
            {
                "messages": [
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": f"Người dùng TỪ CHỐI hành động này. Lý do: {rejection_note or 'không nêu rõ'}.",
                    }
                ]
            },
            as_node="tools",
        )

    with trace_answer(
        "assistant_resume", f"approve={approve}", metadata={"thread_id": thread_id}
    ) as t:
        result = await app.ainvoke(None, config=config)
        response = _to_response(result)
        t["output"] = response.get("answer") or response
    return response


def _to_response(result: dict) -> dict:
    pending = _pending_tool_call(result)
    if pending is not None:
        return {"status": "pending_approval", "tool_call": {"name": pending["name"], "args": pending["args"]}}
    return {"status": "done", "answer": result["messages"][-1].content}


async def evaluate_thread(thread_id: str) -> dict:
    """Bài 5 — đọc lại checkpointer theo thread_id, chấm task success + trajectory.

    KHÔNG nhận trajectory từ caller (xem thảo luận thiết kế) — đọc thẳng từ
    state để frontend chỉ cần gửi thread_id, đúng triết lý "state sống trong
    checkpointer" (Bài 2, Section 5). Trả lỗi rõ ràng nếu thread rỗng/agent
    còn đang chờ duyệt tool (chưa có final answer để chấm).
    """
    app = _build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    state = await app.aget_state(config)
    messages = state.values.get("messages", [])

    if not messages:
        return {"error": "Chưa có hội thoại nào cho thread_id này."}
    if _pending_tool_call({"messages": messages}) is not None:
        return {"error": "Agent đang chờ phê duyệt tool — chưa có kết quả cuối để đánh giá."}

    task = _extract_task(messages)
    final_output = messages[-1].content
    trajectory = _extract_trajectory(messages)

    result = evaluate_run(task, final_output, trajectory)
    return {"result": result, "trajectory": trajectory}
