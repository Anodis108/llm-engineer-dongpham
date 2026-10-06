"""Test chat endpoint + pipeline wiring — mock LLM & retrieval, không gọi API thật.

Lưu ý: từ Buổi 5, pipeline.answer() gọi retrieve() (RAG thật) trước khi chat, nên
phải mock CẢ retrieve lẫn chat để test không chạm OpenAI/Qdrant.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok():
    """Health check chạy không cần API key."""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model" in body


def test_chat_endpoint_wiring(monkeypatch):
    """Endpoint /chat gọi đúng pipeline. Mock retrieve + LLM để không tốn API."""
    from app import pipeline

    monkeypatch.setattr(pipeline, "retrieve", lambda question, **_kw: [])
    monkeypatch.setattr(pipeline.completion, "chat", lambda messages, params: "Trả lời demo.")

    r = client.post("/chat", json={"question": "Xin chào"})
    assert r.status_code == 200
    assert r.json()["answer"] == "Trả lời demo."


def test_chat_validation_rejects_empty_question():
    """Pydantic chặn câu hỏi rỗng (min_length=1)."""
    r = client.post("/chat", json={"question": ""})
    assert r.status_code == 422


def test_chat_stream_sse_endpoint_wiring(monkeypatch):
    """Module III, Bài 4 — /chat/stream-sse trả SSE đúng format, kết thúc bằng [DONE].

    Mock retrieve + astream_chat để không tốn API, verify format `data: ...\n\n`
    và cả 2 endpoint streaming (cũ /chat/stream, mới /chat/stream-sse) cùng tồn tại.
    """
    async def _fake_astream_chat(messages, params=None):
        for tok in ["Xin ", "chào"]:
            yield tok

    monkeypatch.setattr("app.retrieval.retriever.retrieve", lambda question, **_kw: [])
    monkeypatch.setattr("app.llm.async_client.astream_chat", _fake_astream_chat)

    r = client.post("/chat/stream-sse", json={"question": "Xin chào"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    body = r.text
    assert 'data: {"token": "Xin "}' in body
    assert 'data: {"token": "chào"}' in body
    assert body.strip().endswith("data: [DONE]")


def test_chat_stream_endpoint_still_works(monkeypatch):
    """Endpoint cũ /chat/stream (Bài 1, text/plain) không bị phá bởi việc thêm SSE mới."""
    from app import pipeline

    monkeypatch.setattr(pipeline, "retrieve", lambda question, **_kw: [])

    def _fake_chat_stream(messages, params):
        yield "Xin "
        yield "chào"

    monkeypatch.setattr(pipeline.completion, "chat_stream", _fake_chat_stream)

    r = client.post("/chat/stream", json={"question": "Xin chào"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert r.text == "Xin chào"
