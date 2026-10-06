"""Tool selection — Module II, Bài 4, Section 2 (Tool Selection).

Với 15 tool (tools.py) đã ở vùng "cần chọn lọc" theo bài học (ngưỡng chính xác
ổn định là 5-7 tool; > 20 tool độ chính xác chọn tool giảm mạnh). Ở đây minh
hoạ 2 chiến lược, dùng CHUNG cho agent thật (nodes.py bind theo kết quả retrieval):

  - classify_intent / TOOL_GROUPS   : Hierarchical Grouping (phân loại rồi nạp
    đúng 1 nhóm domain).
  - retrieve_relevant_tools         : Tool Retrieval bằng embedding similarity
    (giống RAG nhưng cho tool description thay vì tài liệu) — linh hoạt hơn
    grouping vì không cần câu hỏi khớp đúng 1 domain, và không cần bước
    classify LLM riêng.

agent_node (nodes.py) dùng retrieve_relevant_tools() — linh hoạt hơn, không bị
"khoá cứng" vào 1 domain như grouping (vd "đặt bàn ăn tối lúc rảnh" cần cả
calendar lẫn dining, grouping theo 1 nhãn duy nhất sẽ bỏ sót).

Embedding: tái dùng app/retrieval/embeddings.py (OpenAI text-embedding-3-small,
Module I) — nhất quán với memory.py (Bài 3), giữ triết lý "native SDK", không
thêm dependency mới (Chroma/LangChain vectorstore như handbook mẫu).

Section 4 (MCP): index còn gộp thêm tool từ mcp/wellness_server.py qua
mcp/client.py — model KHÔNG phân biệt được tool nội bộ (tools.py) hay tool
qua MCP, cả 2 đều vào chung 1 danh sách để retrieval chọn. `get_mcp_tools()`
là async nên `_tool_index()`/`retrieve_relevant_tools()` ở đây cũng phải async
(lru_cache không cache đúng coroutine — tự cache bằng biến module-level).
"""

from __future__ import annotations

import math

from app.agent_m2.tools import TOOL_GROUPS, TOOLS

# ── Hierarchical Grouping (Section 2) ──────────────────────────────────────────

def classify_intent(user_message: str) -> str:
    """Phân loại câu hỏi vào 1 trong các domain của TOOL_GROUPS bằng LLM.

    Heuristic rẻ hơn embedding similarity (1 lời gọi LLM ngắn, không cần
    tính toán vector) nhưng CỨNG NHẮC — chỉ trả về đúng 1 domain, không thấy
    được câu hỏi cần tool từ NHIỀU domain cùng lúc (xem so sánh ở docstring
    module, retrieve_relevant_tools linh hoạt hơn cho agent thật).
    """
    from app.llm import completion
    from app.llm.params import GenerationParams

    domains = list(TOOL_GROUPS.keys())
    prompt = (
        f"Câu hỏi sau thuộc nhóm nào: {', '.join(domains)}? "
        "Chỉ trả về đúng 1 từ trong danh sách trên.\n\n"
        f"Câu hỏi: {user_message}"
    )
    answer = completion.chat(
        [{"role": "user", "content": prompt}], GenerationParams(temperature=0.0)
    ).strip().lower()
    return answer if answer in TOOL_GROUPS else ""


def get_tools_by_group(group: str) -> list:
    """Trả danh sách tool của 1 domain; domain lạ → list rỗng."""
    return TOOL_GROUPS.get(group, [])


# ── Tool Retrieval bằng embedding (Section 2) ──────────────────────────────────

def _tool_description(t) -> str:
    # description viết theo văn phong NGƯỜI DÙNG sẽ hỏi, không theo tên hàm kỹ
    # thuật — chất lượng retrieval phụ thuộc trực tiếp vào việc này (Bài 4 tip).
    return f"{t.name}: {t.description}"


_cached_tool_index: list[tuple[str, list[float]]] | None = None
_cached_all_tools: list | None = None


async def _all_tools() -> list:
    """TOOLS nội bộ (tools.py) + tool MCP (mcp/wellness_server.py qua client.py).

    Cache module-level: chỉ gọi MCP client (spawn subprocess server) 1 lần.
    """
    global _cached_all_tools
    if _cached_all_tools is not None:
        return _cached_all_tools

    from app.agent_m2.mcp.client import get_mcp_tools

    mcp_tools = await get_mcp_tools()
    _cached_all_tools = TOOLS + mcp_tools
    return _cached_all_tools


async def _tool_index() -> list[tuple[str, list[float]]]:
    """Embed description của TOÀN BỘ tool (nội bộ + MCP) MỘT LẦN, cache lại.

    Đây là phần "index toàn bộ tool descriptions" của bài học — nặng (1 batch
    API call + 1 lần spawn MCP server) nhưng chỉ chạy 1 lần cho suốt vòng đời
    process, không mỗi lượt chat.
    """
    global _cached_tool_index
    if _cached_tool_index is not None:
        return _cached_tool_index

    from app.retrieval.embeddings import embed_passages

    all_tools = await _all_tools()
    vectors = embed_passages([_tool_description(t) for t in all_tools])
    # strict=True: embed_passages trả 1 vector cho mỗi tool. Lệch nghĩa là tool
    # bị gắn nhầm vector → chọn sai tool mà không có lỗi nào lộ ra.
    _cached_tool_index = list(zip([t.name for t in all_tools], vectors, strict=True))
    return _cached_tool_index


def _cosine(a: list[float], b: list[float]) -> float:
    # strict=True: 2 embedding khác số chiều là lỗi cấu hình (đổi EMBEDDING_MODEL
    # mà không reindex). Không có strict thì cosine vẫn "chạy" và trả điểm rác.
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def retrieve_relevant_tools(query: str, k: int = 5) -> list:
    """Top-k tool liên quan nhất tới câu hỏi (embedding similarity, Section 2).

    Index gồm cả tool nội bộ (tools.py) lẫn tool MCP (Section 4) — model không
    phân biệt được nguồn gốc, chỉ chọn theo description liên quan nhất.
    """
    from app.retrieval.embeddings import embed_query

    query_vec = embed_query(query)
    index = await _tool_index()
    scored = [(name, _cosine(query_vec, vec)) for name, vec in index]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    top_names = {name for name, _ in scored[:k]}

    all_tools = await _all_tools()
    by_name = {t.name: t for t in all_tools}
    return [by_name[name] for name in top_names if name in by_name]


def reset_cache() -> None:
    """Dùng trong test — xoá cache index để mock lại embedding/MCP giữa các test case."""
    global _cached_tool_index, _cached_all_tools
    _cached_tool_index = None
    _cached_all_tools = None
