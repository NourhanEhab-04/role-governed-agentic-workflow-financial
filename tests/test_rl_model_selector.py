"""
tests/test_rl_model_selector.py
================================
Unit tests for config/model_selector.py (Step 8).

All tests are pure: no network calls, no Ollama, no Groq.
"""

import importlib
import pytest


class TestActiveBackendName:

    def test_default_is_groq(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "groq")
        import config.settings as s
        importlib.reload(s)
        import config.model_selector as ms
        importlib.reload(ms)
        assert ms.active_backend_name() == "groq"
        importlib.reload(s)

    def test_local_backend_name(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "local")
        import config.settings as s
        importlib.reload(s)
        import config.model_selector as ms
        importlib.reload(ms)
        assert ms.active_backend_name() == "local"
        importlib.reload(s)

    def test_returns_string(self):
        import config.model_selector as ms
        assert isinstance(ms.active_backend_name(), str)


class TestIsLocalBackend:

    def test_groq_backend_not_local(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "groq")
        import config.settings as s
        importlib.reload(s)
        import config.model_selector as ms
        importlib.reload(ms)
        assert ms.is_local_backend() is False
        importlib.reload(s)

    def test_local_backend_is_local(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "local")
        import config.settings as s
        importlib.reload(s)
        import config.model_selector as ms
        importlib.reload(ms)
        assert ms.is_local_backend() is True
        importlib.reload(s)

    def test_returns_bool(self):
        import config.model_selector as ms
        assert isinstance(ms.is_local_backend(), bool)


class TestGetActiveModelClientGroq:
    """Tests for groq backend — only tests that the client is returned
    (construction works), not that it can actually reach the network."""

    def test_groq_backend_returns_autogen_client(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "test-key-for-testing")
        import config.settings as s
        importlib.reload(s)
        import config.model_selector as ms
        importlib.reload(ms)
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        client = ms.get_active_model_client()
        assert isinstance(client, OpenAIChatCompletionClient)
        importlib.reload(s)

    def test_invalid_backend_raises_runtime_error(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "openai")
        import config.settings as s
        importlib.reload(s)
        import config.model_selector as ms
        importlib.reload(ms)
        with pytest.raises(RuntimeError, match="Unknown LLM_BACKEND"):
            ms.get_active_model_client()
        importlib.reload(s)

    def test_empty_backend_raises_runtime_error(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "")
        import config.settings as s
        importlib.reload(s)
        import config.model_selector as ms
        importlib.reload(ms)
        with pytest.raises(RuntimeError, match="Unknown LLM_BACKEND"):
            ms.get_active_model_client()
        importlib.reload(s)

    def test_error_message_lists_valid_backends(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "invalid")
        import config.settings as s
        importlib.reload(s)
        import config.model_selector as ms
        importlib.reload(ms)
        with pytest.raises(RuntimeError) as exc_info:
            ms.get_active_model_client()
        assert "groq" in str(exc_info.value)
        assert "local" in str(exc_info.value)
        importlib.reload(s)


class TestGetActiveModelClientLocal:

    def test_local_backend_returns_autogen_client(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "local")
        monkeypatch.setenv("LOCAL_MODEL_NAME", "llama3.1:8b")
        monkeypatch.setenv("LOCAL_MODEL_BASE_URL", "http://localhost:11434/v1")
        import config.settings as s
        importlib.reload(s)
        import config.model_selector as ms
        importlib.reload(ms)
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        client = ms.get_active_model_client()
        assert isinstance(client, OpenAIChatCompletionClient)
        importlib.reload(s)

    def test_valid_backends_constant(self):
        import config.model_selector as ms
        assert "groq" in ms._VALID_BACKENDS
        assert "local" in ms._VALID_BACKENDS

    def test_module_imports_cleanly(self):
        import config.model_selector  # noqa: F401
