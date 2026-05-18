# config/llm_config.py
# LLM model client configuration for the AutoGen pipeline.
# Loaded once at startup; all agents import from here.

import os
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Fast 8B — used by all profiling/classification/rule agents.
GROQ_MODEL = "llama-3.1-8b-instant"

# Larger 70B — used exclusively by the verifier.
# Different weights from the 8B agents means different failure modes.
GROQ_VERIFIER_MODEL = "llama-3.3-70b-versatile"

GROQ_MODEL_INFO = {
    "vision": False,
    "function_calling": True,
    "json_output": True,
    "family": "unknown",
    "structured_output": False,
}


def get_model_client() -> OpenAIChatCompletionClient:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set.")
    return OpenAIChatCompletionClient(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
        model_info=GROQ_MODEL_INFO,
        json_output=True,
    )


def get_verifier_client() -> OpenAIChatCompletionClient:
    """Return the Groq 70B client used exclusively by the verifier.
    Larger model with different weights = catches errors the 8B agents miss."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set.")
    return OpenAIChatCompletionClient(
        model=GROQ_VERIFIER_MODEL,
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
        model_info=GROQ_MODEL_INFO,
        json_output=True,
    )