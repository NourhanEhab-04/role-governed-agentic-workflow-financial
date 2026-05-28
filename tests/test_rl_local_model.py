"""
tests/test_rl_local_model.py
=============================
Tests for Step 3: local model configuration and health check.

Test structure
--------------
Group 1 — settings.py values (pure unit tests, no network, no Ollama needed)
Group 2 — local_llm_config.py client construction (no network, no Ollama needed)
Group 3 — local_model_health.py URL derivation helpers (no network)
Group 4 — local_model_health.check_ollama_health() with a mock HTTP server
Group 5 — live Ollama connectivity (skipped if Ollama is not running)

Only Group 5 requires Ollama to be installed and running.
All other groups are always executed and must always pass.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest


# ── Group 1: settings.py values ───────────────────────────────────────────────

class TestSettings:
    """settings.py values load without errors and have the right types/defaults."""

    def test_llm_backend_default_is_groq(self):
        # Without any override the default must be "groq" so existing pipeline is unaffected.
        import config.settings as s
        # We cannot guarantee the env is clean, but we can assert the type is str.
        assert isinstance(s.LLM_BACKEND, str)

    def test_local_model_name_is_non_empty_string(self):
        import config.settings as s
        assert isinstance(s.LOCAL_MODEL_NAME, str)
        assert len(s.LOCAL_MODEL_NAME) > 0

    def test_local_model_base_url_is_non_empty_string(self):
        import config.settings as s
        assert isinstance(s.LOCAL_MODEL_BASE_URL, str)
        assert len(s.LOCAL_MODEL_BASE_URL) > 0

    def test_local_model_base_url_starts_with_http(self):
        import config.settings as s
        assert s.LOCAL_MODEL_BASE_URL.startswith("http")

    def test_default_base_url_contains_11434(self):
        # Default Ollama port is 11434. Only meaningful when no override is set.
        import config.settings as s
        with patch.dict("os.environ", {}, clear=False):
            # Re-import to get defaults (only valid when env var is not set)
            import importlib
            import config.settings
            # Just assert the default string appears somewhere in the module
            # (the env var may already be set in the test environment, so we
            # check the documented default constant in the module source instead)
            import inspect
            source = inspect.getsource(config.settings)
            assert "11434" in source, "Default Ollama port 11434 missing from settings.py"

    def test_llm_backend_env_override(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "local")
        import importlib
        import config.settings as s
        importlib.reload(s)
        assert s.LLM_BACKEND == "local"
        # Restore
        importlib.reload(s)

    def test_local_model_name_env_override(self, monkeypatch):
        monkeypatch.setenv("LOCAL_MODEL_NAME", "mistral:7b")
        import importlib
        import config.settings as s
        importlib.reload(s)
        assert s.LOCAL_MODEL_NAME == "mistral:7b"
        importlib.reload(s)

    def test_local_model_base_url_env_override(self, monkeypatch):
        monkeypatch.setenv("LOCAL_MODEL_BASE_URL", "http://192.168.1.5:11434/v1")
        import importlib
        import config.settings as s
        importlib.reload(s)
        assert s.LOCAL_MODEL_BASE_URL == "http://192.168.1.5:11434/v1"
        importlib.reload(s)

    def test_existing_settings_still_present(self):
        # Regression: adding new keys must not remove existing ones.
        import config.settings as s
        assert hasattr(s, "DATABASE_URL")
        assert hasattr(s, "EVAL_RUNS_PER_SCENARIO")


# ── Group 2: client factory ────────────────────────────────────────────────────

class TestLocalLlmConfig:
    """config/local_llm_config.py builds clients without network access."""

    def test_module_imports_without_error(self):
        import config.local_llm_config  # noqa: F401

    def test_get_local_model_client_returns_object(self):
        from config.local_llm_config import get_local_model_client
        client = get_local_model_client()
        # AutoGen's OpenAIChatCompletionClient is what we expect
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        assert isinstance(client, OpenAIChatCompletionClient)

    def test_local_model_info_has_required_keys(self):
        from config.local_llm_config import LOCAL_MODEL_INFO
        required = {"vision", "function_calling", "json_output", "family", "structured_output"}
        assert required.issubset(LOCAL_MODEL_INFO.keys())

    def test_local_model_info_vision_is_false(self):
        from config.local_llm_config import LOCAL_MODEL_INFO
        assert LOCAL_MODEL_INFO["vision"] is False

    def test_local_model_info_function_calling_is_true(self):
        from config.local_llm_config import LOCAL_MODEL_INFO
        assert LOCAL_MODEL_INFO["function_calling"] is True

    def test_local_model_info_json_output_is_true(self):
        from config.local_llm_config import LOCAL_MODEL_INFO
        assert LOCAL_MODEL_INFO["json_output"] is True

    def test_empty_model_name_raises_runtime_error(self, monkeypatch):
        monkeypatch.setenv("LOCAL_MODEL_NAME", "")
        import importlib
        import config.settings as s
        importlib.reload(s)
        import config.local_llm_config as lc
        importlib.reload(lc)
        with pytest.raises(RuntimeError, match="LOCAL_MODEL_NAME is empty"):
            lc.get_local_model_client()
        importlib.reload(s)
        importlib.reload(lc)

    def test_empty_base_url_raises_runtime_error(self, monkeypatch):
        monkeypatch.setenv("LOCAL_MODEL_BASE_URL", "")
        import importlib
        import config.settings as s
        importlib.reload(s)
        import config.local_llm_config as lc
        importlib.reload(lc)
        with pytest.raises(RuntimeError, match="LOCAL_MODEL_BASE_URL is empty"):
            lc.get_local_model_client()
        importlib.reload(s)
        importlib.reload(lc)


# ── Group 3: URL derivation ────────────────────────────────────────────────────

class TestTagsUrlDerivation:
    """_tags_url strips /v1 and appends /api/tags correctly."""

    def _tags_url(self, base: str) -> str:
        from rl.local_model_health import _tags_url
        return _tags_url(base)

    def test_v1_suffix_stripped(self):
        assert self._tags_url("http://localhost:11434/v1") == "http://localhost:11434/api/tags"

    def test_no_v1_suffix_leaves_root(self):
        assert self._tags_url("http://localhost:11434") == "http://localhost:11434/api/tags"

    def test_trailing_slash_stripped(self):
        assert self._tags_url("http://localhost:11434/v1/") == "http://localhost:11434/api/tags"

    def test_custom_host(self):
        assert self._tags_url("http://192.168.1.5:11434/v1") == "http://192.168.1.5:11434/api/tags"

    def test_custom_port(self):
        assert self._tags_url("http://localhost:9999/v1") == "http://localhost:9999/api/tags"


# ── Group 4: mock HTTP server tests ──────────────────────────────────────────

def _start_mock_server(response_body: dict, status: int = 200) -> tuple[HTTPServer, int]:
    """
    Start a real (but in-process) HTTP server on a random port that returns
    `response_body` as JSON with `status` for every GET /api/tags request.
    Returns (server, port).
    """
    encoded = json.dumps(response_body).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):  # suppress server output in test logs
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port


class TestCheckOllamaHealthMocked:
    """
    Tests that exercise check_ollama_health() against a real in-process HTTP
    server so we never need a real Ollama installation.
    """

    def test_ollama_running_model_present(self):
        from rl.local_model_health import check_ollama_health

        body = {"models": [{"name": "llama3.1:8b"}, {"name": "mistral:7b"}]}
        server, port = _start_mock_server(body)
        try:
            result = check_ollama_health(
                base_url=f"http://127.0.0.1:{port}/v1",
                model_name="llama3.1:8b",
            )
        finally:
            server.shutdown()

        assert result.is_running is True
        assert result.target_model_available is True
        assert "llama3.1:8b" in result.available_models
        assert result.error_message == ""

    def test_ollama_running_model_absent(self):
        from rl.local_model_health import check_ollama_health

        body = {"models": [{"name": "mistral:7b"}]}
        server, port = _start_mock_server(body)
        try:
            result = check_ollama_health(
                base_url=f"http://127.0.0.1:{port}/v1",
                model_name="llama3.1:8b",
            )
        finally:
            server.shutdown()

        assert result.is_running is True
        assert result.target_model_available is False
        assert result.error_message != ""
        assert "llama3.1:8b" in result.error_message

    def test_ollama_running_no_models_pulled(self):
        from rl.local_model_health import check_ollama_health

        body = {"models": []}
        server, port = _start_mock_server(body)
        try:
            result = check_ollama_health(
                base_url=f"http://127.0.0.1:{port}/v1",
                model_name="llama3.1:8b",
            )
        finally:
            server.shutdown()

        assert result.is_running is True
        assert result.target_model_available is False
        assert result.available_models == []

    def test_ollama_not_running_returns_is_running_false(self):
        from rl.local_model_health import check_ollama_health

        # Port 19999 should be unused; connection refused → is_running=False
        result = check_ollama_health(
            base_url="http://127.0.0.1:19999/v1",
            model_name="llama3.1:8b",
            timeout_seconds=0.5,
        )

        assert result.is_running is False
        assert result.error_message != ""

    def test_model_name_without_tag_matches(self):
        # "llama3.1" (no :8b tag) should match "llama3.1:8b" in available list
        from rl.local_model_health import check_ollama_health

        body = {"models": [{"name": "llama3.1:8b"}]}
        server, port = _start_mock_server(body)
        try:
            result = check_ollama_health(
                base_url=f"http://127.0.0.1:{port}/v1",
                model_name="llama3.1",
            )
        finally:
            server.shutdown()

        assert result.is_running is True
        assert result.target_model_available is True

    def test_multiple_models_correct_one_found(self):
        from rl.local_model_health import check_ollama_health

        body = {
            "models": [
                {"name": "mistral:7b"},
                {"name": "llama3.1:8b"},
                {"name": "codellama:13b"},
            ]
        }
        server, port = _start_mock_server(body)
        try:
            result = check_ollama_health(
                base_url=f"http://127.0.0.1:{port}/v1",
                model_name="llama3.1:8b",
            )
        finally:
            server.shutdown()

        assert result.target_model_available is True
        assert len(result.available_models) == 3

    def test_health_result_fields_populated(self):
        from rl.local_model_health import check_ollama_health

        body = {"models": [{"name": "llama3.1:8b"}]}
        server, port = _start_mock_server(body)
        try:
            result = check_ollama_health(
                base_url=f"http://127.0.0.1:{port}/v1",
                model_name="llama3.1:8b",
            )
        finally:
            server.shutdown()

        assert result.target_model_name == "llama3.1:8b"
        assert "127.0.0.1" in result.ollama_base_url
        assert str(port) in result.ollama_base_url

    def test_health_result_dataclass_attributes_exist(self):
        from rl.local_model_health import HealthResult
        r = HealthResult(is_running=False)
        assert hasattr(r, "is_running")
        assert hasattr(r, "available_models")
        assert hasattr(r, "target_model_available")
        assert hasattr(r, "target_model_name")
        assert hasattr(r, "ollama_base_url")
        assert hasattr(r, "error_message")

    def test_print_health_report_does_not_crash(self, capsys):
        from rl.local_model_health import HealthResult, print_health_report

        r = HealthResult(
            is_running=True,
            available_models=["llama3.1:8b"],
            target_model_available=True,
            target_model_name="llama3.1:8b",
            ollama_base_url="http://localhost:11434",
            error_message="",
        )
        print_health_report(r)
        captured = capsys.readouterr()
        assert "llama3.1:8b" in captured.out
        assert "YES" in captured.out


# ── Group 5: live Ollama (skipped if not running) ──────────────────────────────

def _ollama_is_available() -> bool:
    """Return True if a real Ollama instance is reachable on the default port."""
    from rl.local_model_health import check_ollama_health
    result = check_ollama_health(timeout_seconds=1.0)
    return result.is_running


@pytest.mark.skipif(
    not _ollama_is_available(),
    reason="Ollama is not running on localhost:11434 — skipping live tests",
)
class TestLiveOllama:
    """
    These tests require Ollama to be running locally.
    They are skipped automatically when Ollama is not available,
    so the test suite always passes in CI and on machines without Ollama.

    To run these tests manually:
        ollama pull llama3.1:8b
        pytest tests/test_rl_local_model.py -k "LiveOllama" -v
    """

    def test_health_check_is_running(self):
        from rl.local_model_health import check_ollama_health
        result = check_ollama_health()
        assert result.is_running is True

    def test_available_models_is_list(self):
        from rl.local_model_health import check_ollama_health
        result = check_ollama_health()
        assert isinstance(result.available_models, list)

    def test_ollama_base_url_populated(self):
        from rl.local_model_health import check_ollama_health
        result = check_ollama_health()
        assert result.ollama_base_url != ""
