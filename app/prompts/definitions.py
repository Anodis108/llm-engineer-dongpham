"""Prompt definitions — Module III, Bài 6 (CI/CD), nối nội dung Bài 1 (Prompt Management).

Đây là ĐỊNH NGHĨA prompt để PUSH lên LangSmith Prompt Hub. Nguồn chân lý của
prompt là LangSmith — app pull về lúc runtime (xem app/prompts/registry.py),
KHÔNG đọc file YAML trong repo.

Mỗi definition là 1 (name, version). Nhiều version cùng `name` = nhiều COMMIT
của cùng 1 prompt trên LangSmith; alias (`production`) trỏ tới 1 commit trong
đó — đổi alias = đổi version đang chạy, không cần rebuild image.

NGOẠI LỆ LangChain có chủ đích (giống app/agent_m2/nodes.py): `push_prompt`
của LangSmith SDK BẮT BUỘC object là LangChain object — `create_commit` gọi
`langchain_core.load.dumps`, không nhận string/dict thuần. Nên template ở đây
dựng bằng `ChatPromptTemplate`. Đường RENDER của app thì vẫn thuần: registry
trích template thô ra rồi render bằng `string.Template` (xem registry.py),
không đụng LangChain ở hot path.

`context_section` là biến ĐÃ ĐƯỢC DỰNG SẴN (không phải câu hỏi người dùng):
rỗng khi không có chunk nào, hoặc block "--- TÀI LIỆU THAM KHẢO ---" khi có.
Giữ đúng hành vi cũ của build_messages() — system prompt không đổi khi retrieval
rỗng.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate

from app.config import settings

# Phần CỐ ĐỊNH của prompt (instruction + ràng buộc) — tách khỏi phần điền động.
# Port nguyên văn từ app/prompts/templates.py để v1 không đổi hành vi.
#
# CHÚ Ý: khối này kết thúc bằng 1 dấu xuống dòng, và điều đó là BẮT BUỘC.
# build_messages() cũ ghép `LEGAL_SYSTEM_PROMPT + context_section` với
# LEGAL_SYSTEM_PROMPT kết thúc bằng "\n" → template ở đây cũng phải vậy. Thiếu
# dấu đó là v1 lệch hành vi so với trước Bài 6; test_prompts.py
# ::test_v1_template_reproduces_legacy_system_prompt canh đúng chỗ này.
_LEGAL_RULES = """Bạn là trợ lý pháp lý chuyên về luật doanh nghiệp Việt Nam.

Quy tắc:
- Trả lời chính xác, ngắn gọn, bằng tiếng Việt.
- Khi có TÀI LIỆU THAM KHẢO bên dưới, CHỈ trả lời dựa trên tài liệu đó và trích dẫn nguồn.
- Khi KHÔNG có tài liệu tham khảo, nói rõ rằng câu trả lời dựa trên hiểu biết chung
  và khuyến nghị người dùng kiểm chứng với văn bản luật chính thức.
- Luôn khuyên tham khảo luật sư cho các vụ việc cụ thể.
- Luôn tránh đưa ra tư vấn pháp lý cụ thể cho từng trường hợp, chỉ cung cấp thông tin chung.
"""

# v2 siết thêm ràng buộc trích dẫn — thay đổi HÀNH VI (contract) nên bump major.
_LEGAL_RULES_V2 = """Bạn là trợ lý pháp lý chuyên về luật doanh nghiệp Việt Nam.

Quy tắc:
- Trả lời chính xác, ngắn gọn, bằng tiếng Việt.
- CHỈ trả lời dựa trên TÀI LIỆU THAM KHẢO bên dưới. KHÔNG suy diễn ngoài tài liệu.
- Mọi khẳng định về quy định pháp luật PHẢI kèm số điều/khoản trích dẫn
  (ví dụ: "Điều 12, khoản 2"). Thiếu trích dẫn = câu trả lời không hợp lệ.
- Nếu tài liệu không đủ để trả lời, trả lời ĐÚNG câu sau và không thêm gì khác:
  "Tôi không tìm thấy thông tin này trong tài liệu được cung cấp."
- Luôn khuyên tham khảo luật sư cho các vụ việc cụ thể."""


@dataclass(slots=True)
class PromptDefinition:
    """1 version của 1 prompt, kèm metadata bắt buộc (Bài 1, Section 3).

    `version` là số nguyên tăng dần (quy ước lớp học, bài học Section 3). Trên
    LangSmith nó nằm trong commit_description + metadata, KHÔNG nằm trong tên
    prompt (tên ổn định, version là commit).
    """

    name: str
    version: int
    model: str
    owner: str
    changelog: str
    description: str
    template: ChatPromptTemplate
    variables: tuple[str, ...]

    @property
    def identifier(self) -> str:
        """Tên prompt trên LangSmith — tiền tố từ LANGSMITH_PROJECT.

        vd: llm-engineer-demo-rag-answer
        """
        return f"{settings.langsmith_project}-{self.name}"

    @property
    def commit_message(self) -> str:
        return f"v{self.version}: {self.changelog}"


RAG_ANSWER_V1 = PromptDefinition(
    name="rag-answer",
    version=1,
    model=settings.llm_model,
    owner="hungluu@lingolab.vn",
    changelog="Port nguyên hành vi từ app/prompts/templates.py (Module I, Bài 1).",
    description="Trợ lý pháp lý RAG — persona luật sư VN, trả lời dựa trên tài liệu tham khảo.",
    template=ChatPromptTemplate.from_messages(
        [
            ("system", _LEGAL_RULES + "{context_section}"),
            ("human", "{question}"),
        ]
    ),
    variables=("context_section", "question"),
)

RAG_ANSWER_V2 = PromptDefinition(
    name="rag-answer",
    version=2,
    model=settings.llm_model,
    owner="hungluu@lingolab.vn",
    changelog=(
        "v1 -> v2: bắt buộc trích số điều/khoản cho mọi khẳng định; "
        "thêm câu từ chối cố định khi thiếu tài liệu; bỏ suy diễn ngoài tài liệu."
    ),
    description="Trợ lý pháp lý RAG — siết ràng buộc trích dẫn điều luật, từ chối khi thiếu context.",
    template=ChatPromptTemplate.from_messages(
        [
            ("system", _LEGAL_RULES_V2 + "{context_section}"),
            ("human", "{question}"),
        ]
    ),
    variables=("context_section", "question"),
)

QUERY_REWRITE_V1 = PromptDefinition(
    name="query-rewrite",
    version=1,
    model=settings.llm_model,
    owner="hungluu@lingolab.vn",
    changelog="Tạo mới — viết lại câu hỏi trước khi search (Module I, Bài 5).",
    description="Viết lại câu hỏi người dùng thành truy vấn tìm kiếm ngắn gọn cho RAG.",
    template=ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Viết lại câu hỏi của người dùng thành một truy vấn tìm kiếm ngắn gọn, "
                "giữ nguyên thuật ngữ pháp lý. CHỈ trả về truy vấn, không giải thích.",
            ),
            ("human", "{question}"),
        ]
    ),
    variables=("question",),
)

# Thứ tự trong list = thứ tự push. Push theo đúng thứ tự version tăng dần để
# commit cuối cùng (được gắn alias) là version mới nhất.
ALL_PROMPTS: list[PromptDefinition] = [
    RAG_ANSWER_V1,
    RAG_ANSWER_V2,
    QUERY_REWRITE_V1,
]


def latest_per_name() -> dict[str, PromptDefinition]:
    """Version cao nhất của mỗi prompt — dùng để biết commit nào gắn alias."""
    out: dict[str, PromptDefinition] = {}
    for p in ALL_PROMPTS:
        current = out.get(p.name)
        if current is None or p.version > current.version:
            out[p.name] = p
    return out
