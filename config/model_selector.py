"""
config/model_selector.py
========================
Single integration point for switching inference backends at runtime.

The rest of the pipeline always calls `get_active_model_client()` when it
needs a model client.  This module reads `LLM_BACKEND` from settings and
returns the appropriate client — callers never need to know which backend
is active.

Supported backends
------------------
  "groq"   (default) — cloud Groq API via config/llm_config.py
  "local"             — local Ollama via config/local_llm_config.py

Usage
-----
    from config.model_selector import get_active_model_client
    client = get_active_model_client()
    # pass to run_pipeline(), run_baseline(), etc.

Why not modify existing callers?
---------------------------------
The existing pipeline callers (orchestrator/graph.py, evaluation/evaluator.py,
api.py) all import `get_model_client` from config/llm_config.py.  Changing
every call site creates unnecessary churn.  Instead, the RL evaluation layer
(evaluation/evaluator_rl.py) uses `get_active_model_client()` so the backend
switch is isolated to RL-specific code.  Existing code paths are untouched.
"""

from __future__ import annotations

from autogen_ext.models.openai import OpenAIChatCompletionClient

from config.settings import LLM_BACKEND

_VALID_BACKENDS = frozenset({"groq", "local"})


def get_active_model_client() -> OpenAIChatCompletionClient:
    """
    Return the model client for the currently configured backend.

    Reads `LLM_BACKEND` from config/settings.py (which reads from the env).

    Returns:
        OpenAIChatCompletionClient pointed at Groq or local Ollama.

    Raises:
        RuntimeError: if LLM_BACKEND is not one of "groq" or "local".
        RuntimeError: if the required env vars for the chosen backend are missing.
    """
    if LLM_BACKEND not in _VALID_BACKENDS:
        raise RuntimeError(
            f"Unknown LLM_BACKEND '{LLM_BACKEND}'. "
            f"Valid values: {sorted(_VALID_BACKENDS)}. "
            f"Set LLM_BACKEND in your .env file."
        )

    if LLM_BACKEND == "local":
        from config.local_llm_config import get_local_model_client
        return get_local_model_client()

    from config.llm_config import get_model_client
    return get_model_client()


def active_backend_name() -> str:
    """Return the current backend name string ("groq" or "local")."""
    return LLM_BACKEND


def is_local_backend() -> bool:
    """Return True when using the local Ollama backend."""
    return LLM_BACKEND == "local"
