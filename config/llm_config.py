# config/llm_config.py
# LLM model client configuration for the AutoGen pipeline.
# Loaded once at startup; all agents import from here.

import os
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Groq exposes an OpenAI-compatible endpoint.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.1-8b-instant"

# Required by OpenAIChatCompletionClient for non-OpenAI model names.
GROQ_MODEL_INFO = {
    "vision": False,
    "function_calling": True,
    "json_output": False,
    "family": "unknown",
    "structured_output": False,
}


def get_model_client() -> OpenAIChatCompletionClient:
    """Return a configured Groq model client for use by all agents.
    Raises RuntimeError if the API key is not set."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file or environment."
        )
    return OpenAIChatCompletionClient(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
        model_info=GROQ_MODEL_INFO,
    )