"""Context engineering — Module II, Bài 3, Section 2-4.

Các chiến lược quản lý context window cho agent chạy dài:
  - Sliding window (Section 2): giữ N message gần nhất — rẻ, mất thông tin cũ.
  - Summarization (Section 2): tóm tắt phần cũ thành 1 đoạn — giữ ý chính.
  - Tool-output compression (Section 3): nén kết quả tool trước khi vào context.
  - Context-usage estimate + nguyên tắc 40-60% (Section 4): compact CHỦ ĐỘNG.

Mọi lời gọi LLM ở đây dùng native completion.chat (app/llm, Buổi 1) — KHÔNG
qua LangChain, khác agent_node (bắt buộc dùng ChatOpenAI vì ToolNode). Summarize/
compress là tác vụ text→text thuần, không cần tool binding, nên giữ native được.
"""

from __future__ import annotations

from app.llm import completion
from app.llm.params import GenerationParams

# Ước lượng thô 1 token ≈ 4 ký tự (đủ để áp nguyên tắc 40-60%, không cần tokenizer thật).
_CHARS_PER_TOKEN = 4


def _text_of(m) -> str:
    """Lấy content dạng text từ message (dict hoặc BaseMessage)."""
    if isinstance(m, dict):
        return str(m.get("content", ""))
    return str(getattr(m, "content", ""))


def _role_of(m) -> str:
    if isinstance(m, dict):
        return m.get("role", "")
    # BaseMessage.type: "human"/"ai"/"system"/"tool"
    return getattr(m, "type", "")


def format_messages(messages: list) -> str:
    """Chuyển list message (dict hoặc BaseMessage) thành text để đưa vào prompt."""
    return "\n".join(f"{_role_of(m)}: {_text_of(m)}" for m in messages)


# ── Section 2: Sliding window ─────────────────────────────────────────────────

def sliding_window(messages: list, max_messages: int = 20) -> list:
    """Giữ N message gần nhất (system message luôn được giữ nếu có)."""
    if len(messages) <= max_messages:
        return messages
    system_msgs = [m for m in messages if _role_of(m) in ("system",)]
    recent = messages[-max_messages:]
    # Tránh nhân đôi system message đã lọt vào `recent`.
    recent_ids = {id(m) for m in recent}
    return [m for m in system_msgs if id(m) not in recent_ids] + recent


# ── Section 4: Context-usage estimate + nguyên tắc 40-60% ─────────────────────

def estimate_tokens(messages: list) -> int:
    """Ước lượng số token của toàn bộ messages (thô, theo ký tự)."""
    return sum(len(_text_of(m)) for m in messages) // _CHARS_PER_TOKEN


def context_usage(messages: list, window_tokens: int) -> float:
    """Tỷ lệ sử dụng context 0-1 so với window (để áp nguyên tắc 40-60%)."""
    if window_tokens <= 0:
        return 0.0
    return estimate_tokens(messages) / window_tokens


def should_compact(messages: list, window_tokens: int, threshold: float = 0.40) -> bool:
    """True nếu vượt ngưỡng compact (mặc định 40% — compact CHỦ ĐỘNG, Section 4).

    Quyết liệt hơn chuẩn ngành (80-95%) nhưng giảm mạnh context rot: chất lượng
    suy giảm từ ~25-40% dung lượng dù window CHƯA đầy.
    """
    return context_usage(messages, window_tokens) > threshold


# ── Section 2: Summarization ──────────────────────────────────────────────────

SUMMARY_PREFIX = "[Tóm tắt hội thoại trước]:"


def summarize_text(old_messages: list) -> str:
    """Gọi LLM tóm tắt một đoạn message thành 3-5 câu (giữ tên, số liệu, quyết định).

    Tách riêng phần gọi LLM (thuần text→text) để test được mà không đụng tới cơ
    chế state của LangGraph — dùng bởi cả summarize_old_messages (bản thuần, cho
    demo/tài liệu) lẫn compact_node (bản persist state, xem nodes.py).
    """
    prompt = (
        "Tóm tắt cuộc hội thoại sau thành 3-5 câu, giữ lại thông tin quan trọng "
        "(tên, số liệu, quyết định đã chốt, task đang làm dở):\n\n"
        + format_messages(old_messages)
    )
    return completion.chat(
        [{"role": "user", "content": prompt}], GenerationParams(temperature=0.0)
    )


def summarize_old_messages(messages: list, keep_recent: int = 6) -> list:
    """Tóm tắt phần cũ thành 1 system message, giữ nguyên keep_recent message gần nhất.

    Bản THUẦN (trả về list mới, không đụng state) — tiện minh hoạ/tài liệu và test.
    Trong graph thật, compact_node (nodes.py) mới là bản PERSIST: ghi bản nén trở
    lại state để lượt sau tái dùng, không nén lại (xem giải thích ở nodes.py).

    Tốn 1 lời gọi LLM — caller chỉ nên gọi khi should_compact() True, không mỗi lượt.
    """
    if len(messages) <= keep_recent + 1:
        return messages

    old_messages = messages[:-keep_recent]
    recent_messages = messages[-keep_recent:]
    summary = summarize_text(old_messages)
    summary_msg = {"role": "system", "content": f"{SUMMARY_PREFIX} {summary}"}
    return [summary_msg] + recent_messages


# ── Section 3: Tool-output compression ────────────────────────────────────────

def compress_tool_result(raw_result: str, query: str, max_tokens: int = 300) -> str:
    """Nén kết quả tool trước khi đưa vào context, giữ phần liên quan tới query.

    Chỉ nén khi kết quả đủ dài (> ~max_tokens) — "đừng nén quá sớm" (Section 3),
    nén mất thông tin không phục hồi được.

    LƯU Ý: hàm này CHƯA được nối vào graph flow (khác các hàm còn lại trong file,
    vốn được agent_node gọi mỗi vòng lặp). Lý do: `ToolNode` prebuilt của LangGraph
    tự thực thi tool và ghi thẳng output vào state, không có hook để chèn bước nén.
    Muốn dùng thật, thay ToolNode bằng một tool node tự viết rồi gọi hàm này lên
    output trước khi trả về. Ở đây giữ như tiện ích tham khảo + minh hoạ kỹ thuật;
    các tool demo (tools.py) trả output ngắn nên nén cũng không kích hoạt.
    """
    if len(raw_result) < max_tokens * _CHARS_PER_TOKEN:
        return raw_result

    prompt = (
        "Trích xuất thông tin liên quan đến câu hỏi từ kết quả sau. "
        "Giữ số liệu, tên riêng chính xác. Bỏ phần không liên quan.\n\n"
        f"Câu hỏi: {query}\n"
        f"Kết quả thô: {raw_result[:4000]}\n\n"
        "Thông tin liên quan (ngắn gọn):"
    )
    return completion.chat(
        [{"role": "user", "content": prompt}], GenerationParams(temperature=0.0)
    )


# ── Section 4: Re-injection chống instruction fade-out ────────────────────────

def reinject_instructions(messages: list, instructions: str) -> list:
    """Chèn lại chỉ dẫn quan trọng ở CUỐI context (vị trí attention cao nhất).

    Chống instruction fade-out: system prompt ở ĐẦU mất dần ảnh hưởng khi hội
    thoại dài (lost-in-the-middle). Đặt lại ở cuối là cách rẻ & hiệu quả nhất —
    đây chính là cách Claude Code re-inject CLAUDE.md sau mỗi lần compaction.
    """
    return messages + [{"role": "system", "content": f"[Nhắc lại chỉ dẫn]: {instructions}"}]
