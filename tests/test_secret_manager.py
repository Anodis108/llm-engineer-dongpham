"""Test app/config.py Secret Manager integration — mock GCP client, không gọi API thật.

Module III, Bài 5, Section 4: secret đọc lúc RUNTIME, có fallback .env. Test ở
đây verify LOGIC override (bật/tắt, lỗi GCP không sập app) chứ không test
Secret Manager SDK.
"""

from __future__ import annotations

from app.config import Settings, _fetch_secret, _load_secret_overrides


def test_load_secret_overrides_noop_when_disabled():
    """use_secret_manager=False (mặc định) -> không override gì, giữ nguyên .env."""
    base = Settings(USE_SECRET_MANAGER=False, OPENAI_API_KEYS="sk-from-env")
    overrides = _load_secret_overrides(base)
    assert overrides == {}


def test_load_secret_overrides_noop_without_project_id():
    """Bật use_secret_manager nhưng thiếu gcp_project_id -> vẫn no-op (an toàn)."""
    base = Settings(USE_SECRET_MANAGER=True, GCP_PROJECT_ID="", OPENAI_API_KEYS="sk-from-env")
    overrides = _load_secret_overrides(base)
    assert overrides == {}


def test_load_secret_overrides_reads_from_secret_manager(monkeypatch):
    """Bật + có project_id -> gọi Secret Manager, override đúng field."""
    base = Settings(
        USE_SECRET_MANAGER=True, GCP_PROJECT_ID="my-project", OPENAI_API_KEYS="sk-from-env"
    )

    def _fake_fetch(project_id, secret_id):
        assert project_id == "my-project"
        return {
            "llm-engineer-openai-api-keys": "sk-from-secret-manager",
        }.get(secret_id)

    monkeypatch.setattr("app.config._fetch_secret", _fake_fetch)

    overrides = _load_secret_overrides(base)
    assert overrides == {"openai_api_keys": "sk-from-secret-manager"}


def test_load_secret_overrides_falls_back_when_secret_missing(monkeypatch):
    """1 secret không tồn tại trên Secret Manager -> field đó KHÔNG bị override (giữ .env)."""
    base = Settings(USE_SECRET_MANAGER=True, GCP_PROJECT_ID="my-project")

    monkeypatch.setattr("app.config._fetch_secret", lambda project_id, secret_id: None)

    overrides = _load_secret_overrides(base)
    assert overrides == {}


def test_fetch_secret_returns_none_on_gcp_error(monkeypatch):
    """Lỗi bất kỳ khi gọi GCP (network, permission...) -> trả None, không raise."""

    class _FakeClient:
        def access_secret_version(self, request):
            raise RuntimeError("permission denied")

    class _FakeSecretManagerModule:
        @staticmethod
        def SecretManagerServiceClient():
            return _FakeClient()

    monkeypatch.setitem(
        __import__("sys").modules, "google.cloud.secretmanager", _FakeSecretManagerModule
    )

    result = _fetch_secret("my-project", "some-secret")
    assert result is None


def test_fetch_secret_returns_value_on_success(monkeypatch):
    """Gọi thành công -> trả đúng payload đã decode."""

    class _FakePayload:
        data = b"sk-secret-value"

    class _FakeResponse:
        payload = _FakePayload()

    class _FakeClient:
        def access_secret_version(self, request):
            assert request["name"] == "projects/my-project/secrets/some-secret/versions/latest"
            return _FakeResponse()

    class _FakeSecretManagerModule:
        @staticmethod
        def SecretManagerServiceClient():
            return _FakeClient()

    monkeypatch.setitem(
        __import__("sys").modules, "google.cloud.secretmanager", _FakeSecretManagerModule
    )

    result = _fetch_secret("my-project", "some-secret")
    assert result == "sk-secret-value"
