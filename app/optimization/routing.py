"""Model Routing — Buổi 8 (Production Optimization).

Phân loại request (dễ hay khó) rồi gửi tới model phù hợp:
  - Dễ → model nhỏ/rẻ (gpt-4o-mini)
  - Khó → model lớn/mạnh (gpt-4o)

Ba phương pháp xếp theo độ phức tạp:
  1. Rule-based: dùng heuristic (latency ~0, cost 0)
  2. Embedding similarity: so với câu mẫu (latency ~ms, cost ~0)
  3. Fine-tuned classifier: train trên data (latency ~ms, cost ~0)
"""

from __future__ import annotations

import numpy as np

# ── 1. Rule-based Routing ──────────────────────────────────────────────────────


def rule_based_router(query: str) -> str:
    """Dùng heuristic cứng để quyết định model.

    Returns: "gpt-4o-mini" (rẻ) hoặc "gpt-4o" (mạnh).
    """
    n_tokens = len(query.split())

    # Luật 1: query rất ngắn -> thường là task đơn giản
    if n_tokens < 10:
        return "gpt-4o-mini"

    # Luật 2: keyword chỉ dấu task khó
    hard_keywords = [
        "chứng minh",
        "phân tích",
        "viết code",
        "debug",
        "so sánh chi tiết",
        "step by step",
    ]
    if any(kw in query.lower() for kw in hard_keywords):
        return "gpt-4o"

    # Luật 3: có code block -> cần model mạnh
    if "```" in query:
        return "gpt-4o"

    return "gpt-4o-mini"  # mặc định dùng model rẻ


# ── 2. Embedding Similarity Routing ────────────────────────────────────────────


class EmbeddingRouter:
    """Routing dựa trên embedding similarity với câu mẫu.

    Các câu mẫu nhóm theo độ khó, embedding được tính trước.
    Khi query đến, embed câu đó rồi so với centroid mỗi nhóm.
    """

    def __init__(self, embedder, route_examples: dict[str, list[str]]):
        """embedder: function(text_or_list) -> np.ndarray.
        route_examples: {model_name: [example_queries, ...]}.
        """
        self.embedder = embedder
        self.route_examples = route_examples

        # Tính centroid embedding cho mỗi route
        self.route_centroids = {}
        for model, examples in route_examples.items():
            embs = embedder(examples)  # shape (n_examples, dim)
            if isinstance(embs, list):
                embs = np.array(embs)
            centroid = np.mean(embs, axis=0)
            self.route_centroids[model] = centroid

    def route(self, query: str) -> str:
        """Chọn model dựa trên similarity với centroid của mỗi route."""
        q_emb = self.embedder(query)
        if isinstance(q_emb, list):
            q_emb = np.array(q_emb)

        scores = {
            model: float(np.dot(centroid, q_emb))
            for model, centroid in self.route_centroids.items()
        }
        return max(scores, key=scores.get)


# ── 3. Fine-tuned Classifier Routing (stub) ───────────────────────────────────


class ClassifierRouter:
    """Routing dựa trên classifier train sẵn (ví dụ LogisticRegression trên embedding).

    Đây là stub — trong thực tế cần training data có nhãn (easy/hard) và pipeline train/load model.
    """

    def __init__(self, embedder, classifier_model_path: str | None = None):
        """embedder: function(text) -> np.ndarray.
        classifier_model_path: đường dẫn tới classifier (joblib, pickle, v.v.).

        Nếu path là None, degrade xuống rule-based router.
        """
        self.embedder = embedder
        self.classifier = None
        if classifier_model_path:
            try:
                import joblib

                self.classifier = joblib.load(classifier_model_path)
            except Exception:
                # Trong thực tế log warning; ở đây chỉ fallback
                pass

    def route(self, query: str) -> str:
        """Dự đoán model dựa trên classifier."""
        if self.classifier is None:
            # Fallback: dùng rule-based nếu classifier chưa load
            return rule_based_router(query)

        q_emb = self.embedder(query)
        if isinstance(q_emb, list):
            q_emb = np.array([q_emb])
        else:
            q_emb = np.array([q_emb])

        # Giả sử classifier.predict(X) trả về "easy" hoặc "hard"
        label = self.classifier.predict(q_emb)[0]
        return "gpt-4o-mini" if label == "easy" else "gpt-4o"
