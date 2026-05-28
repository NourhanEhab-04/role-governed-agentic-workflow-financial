"""
config/local_llm_config.py
===========================
AutoGen client factory for a locally running Ollama instance.

Ollama exposes an OpenAI-compatible REST API at:
  http://localhost:11434/v1/chat/completions

The `OpenAIChatCompletionClient` from autogen_ext can point at this endpoint
identically to how it points at Groq — only the base_url and api_key differ.

Ollama does not validate the API key; any non-empty string is accepted.
The canonical placeholder is "ollama".

Prerequisites
-------------
1. Install Ollama from https://ollama.com (Windows installer available).
2. Pull the target model:
       ollama pull llama3.1:8b
3. Verify Ollama is running (it starts automatically after install):
       curl http://localhost:11434/api/tags
4. Optionally override defaults via .env:
       LOCAL_MODEL_NAME=llama3.1:8b
       LOCAL_MODEL_BASE_URL=http://localhost:11434/v1
       LLM_BACKEND=local

Usage
-----
    from config.local_llm_config import get_local_model_client
    client = get_local_model_client()

The returned client has the same interface as the clients from config/llm_config.py
so it is a drop-in replacement for all AutoGen agents.

Model capabilities (llama3.1:8b via Ollama)
--------------------------------------------
  function_calling : True  (supported in Ollama ≥ 0.3.0 with llama3.1)
  json_output      : True  (via response_format={"type": "json_object"})
  vision           : False
  structured_output: False (Ollama does not support OpenAI structured_output schema)
"""

from __future__ import annotations

from autogen_ext.models.openai import OpenAIChatCompletionClient

# Ollama accepts any non-empty string as the API key.
_OLLAMA_PLACEHOLDER_KEY = "ollama"

# model_info tells AutoGen which capabilities this model supports.
# These match llama3.1:8b's actual capabilities when served via Ollama ≥ 0.3.0.
LOCAL_MODEL_INFO = {
    "vision": False,
    "function_calling": True,
    "json_output": True,
    "family": "unknown",
    "structured_output": False,
}


def get_local_model_client() -> OpenAIChatCompletionClient:
    """
    Return an AutoGen client pointing at the local Ollama instance.

    Reads LOCAL_MODEL_NAME and LOCAL_MODEL_BASE_URL from config/settings.py,
    which are themselves loaded from environment variables (with defaults).

    Raises:
        RuntimeError: if LOCAL_MODEL_NAME or LOCAL_MODEL_BASE_URL are empty
                      (should not happen with the defaults set in settings.py,
                      but guards against accidental override with empty string).
    """
    import config.settings as _settings
    model_name = _settings.LOCAL_MODEL_NAME
    base_url   = _settings.LOCAL_MODEL_BASE_URL

    if not model_name:
        raise RuntimeError(
            "LOCAL_MODEL_NAME is empty. "
            "Set LOCAL_MODEL_NAME in .env (e.g. 'llama3.1:8b') "
            "or remove the override to use the default."
        )
    if not base_url:
        raise RuntimeError(
            "LOCAL_MODEL_BASE_URL is empty. "
            "Set LOCAL_MODEL_BASE_URL in .env (e.g. 'http://localhost:11434/v1') "
            "or remove the override to use the default."
        )

    return OpenAIChatCompletionClient(
        model=model_name,
        api_key=_OLLAMA_PLACEHOLDER_KEY,
        base_url=base_url,
        model_info=LOCAL_MODEL_INFO,
        json_output=True,
    )
