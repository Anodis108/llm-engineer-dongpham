"""RAG Evaluation — Buổi 7, Section 2. NATIVE (không import thư viện ragas).

Implement lại 4 metric cốt lõi của RAGAS bằng chính LLM client đã có (Buổi 1),
theo đúng công thức mô tả trong bài học:

  - Faithfulness      : % claims trong answer được support bởi context (không cần ground truth)
  - Answer Relevancy   : answer có liên quan đến question không (không cần ground truth)
  - Context Recall     : context có đủ info để trả lời không (cần ground_truth)
  - Context Precision  : retrieved chunks có relevant không (cần ground_truth)

Thư viện `ragas` thật kéo theo langchain + nhiều dependency nặng; bản native này
đủ để dạy đúng NGUYÊN LÝ đằng sau từng metric mà không phá nguyên tắc "native SDK"
của project.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.llm import completion
from app.llm.params import GenerationParams

# ── Faithfulness ──────────────────────────────────────────────────────────────

class _Claims(BaseModel):
    claims: list[str] = Field(description="Các claim/khẳng định độc lập trong câu trả lời")


class _ClaimSupport(BaseModel):
    supported: bool = Field(description="Claim này có được context xác nhận không")


def faithfulness(answer: str, contexts: list[str]) -> float:
    """% claims trong answer được context hỗ trợ. 0-1, không cần ground truth.

    Quy trình 2 bước (đúng thiết kế RAGAS gốc):
      1. Tách answer thành các claim độc lập.
      2. Với mỗi claim, hỏi LLM: context có support claim này không?
    """
    context_text = "\n\n".join(contexts)

    extract_messages = [
        {
            "role": "system",
            "content": "Tách câu trả lời sau thành các claim/khẳng định độc lập, "
            "mỗi claim là một sự thật có thể kiểm chứng riêng biệt.",
        },
        {"role": "user", "content": answer},
    ]
    claims = completion.chat_parsed(
        extract_messages, _Claims, GenerationParams(temperature=0.0)
    ).claims

    if not claims:
        return 1.0  # không có claim nào để kiểm chứng → coi như faithful

    n_supported = 0
    for claim in claims:
        check_messages = [
            {
                "role": "system",
                "content": "Xác định xem CLAIM có được CONTEXT xác nhận trực tiếp không. "
                "Chỉ true nếu context thực sự chứa thông tin đó.",
            },
            {"role": "user", "content": f"Context: {context_text}\n\nClaim: {claim}"},
        ]
        result = completion.chat_parsed(
            check_messages, _ClaimSupport, GenerationParams(temperature=0.0)
        )
        if result.supported:
            n_supported += 1

    return n_supported / len(claims)


# ── Answer Relevancy ──────────────────────────────────────────────────────────

class _RelevancyScore(BaseModel):
    relevant: bool = Field(description="Câu trả lời có thực sự trả lời câu hỏi không")
    score: float = Field(ge=0.0, le=1.0, description="Mức độ liên quan 0-1")


def answer_relevancy(question: str, answer: str) -> float:
    """Answer có liên quan/trả lời đúng trọng tâm question không. 0-1."""
    messages = [
        {
            "role": "system",
            "content": "Đánh giá câu trả lời có thực sự trả lời đúng trọng tâm câu hỏi "
            "không (không lạc đề, không né tránh).",
        },
        {"role": "user", "content": f"Câu hỏi: {question}\n\nCâu trả lời: {answer}"},
    ]
    result = completion.chat_parsed(
        messages, _RelevancyScore, GenerationParams(temperature=0.0)
    )
    return result.score


# ── Context Recall & Precision (cần ground_truth) ────────────────────────────

class _SentenceAttribution(BaseModel):
    can_be_attributed: bool = Field(
        description="Câu này trong ground truth có thể được suy ra từ context không"
    )


class _GroundTruthSentences(BaseModel):
    sentences: list[str] = Field(description="Các câu độc lập trong ground truth")


def context_recall(contexts: list[str], ground_truth: str) -> float:
    """% câu trong ground_truth có thể suy ra được từ context đã retrieve. 0-1.

    Đo "có retrieve đủ thông tin không" — cần ground_truth để biết "đủ" là gì.
    """
    context_text = "\n\n".join(contexts)

    split_messages = [
        {"role": "system", "content": "Tách văn bản sau thành các câu độc lập."},
        {"role": "user", "content": ground_truth},
    ]
    sentences = completion.chat_parsed(
        split_messages, _GroundTruthSentences, GenerationParams(temperature=0.0)
    ).sentences

    if not sentences:
        return 1.0

    n_attributed = 0
    for sentence in sentences:
        check_messages = [
            {
                "role": "system",
                "content": "Xác định xem CÂU sau có thể được suy ra/xác nhận từ CONTEXT không.",
            },
            {"role": "user", "content": f"Context: {context_text}\n\nCâu: {sentence}"},
        ]
        result = completion.chat_parsed(
            check_messages, _SentenceAttribution, GenerationParams(temperature=0.0)
        )
        if result.can_be_attributed:
            n_attributed += 1

    return n_attributed / len(sentences)


class _ChunkRelevance(BaseModel):
    relevant: bool = Field(description="Chunk này có liên quan để trả lời câu hỏi không")


def context_precision(question: str, contexts: list[str], ground_truth: str) -> float:
    """% retrieved chunks thực sự relevant (dùng ground_truth làm chuẩn). 0-1.

    Đo "retrieve có đúng không" — chunk không liên quan làm giảm precision dù
    recall vẫn cao (vì thông tin cần thiết vẫn nằm đâu đó trong context).
    """
    if not contexts:
        return 0.0

    n_relevant = 0
    for chunk in contexts:
        check_messages = [
            {
                "role": "system",
                "content": "Xác định xem CHUNK sau có thực sự cần thiết để trả lời "
                "CÂU HỎI, dựa trên CÂU TRẢ LỜI CHUẨN, hay không.",
            },
            {
                "role": "user",
                "content": (
                    f"Câu hỏi: {question}\n\n"
                    f"Câu trả lời chuẩn: {ground_truth}\n\n"
                    f"Chunk: {chunk}"
                ),
            },
        ]
        result = completion.chat_parsed(
            check_messages, _ChunkRelevance, GenerationParams(temperature=0.0)
        )
        if result.relevant:
            n_relevant += 1

    return n_relevant / len(contexts)


def evaluate_rag(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str | None = None,
) -> dict:
    """Chạy tất cả metric khả dụng cho một sample. Bỏ qua recall/precision nếu
    không có ground_truth (đúng đặc điểm RAGAS: 2 metric đó cần ground truth)."""
    result = {
        "faithfulness": faithfulness(answer, contexts),
        "answer_relevancy": answer_relevancy(question, answer),
    }
    if ground_truth:
        result["context_recall"] = context_recall(contexts, ground_truth)
        result["context_precision"] = context_precision(question, contexts, ground_truth)
    return result
