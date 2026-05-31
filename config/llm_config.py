# config/llm_config.py
# LLM model client configuration.
# Loaded once at startup; all agents import from here.
#
# Specialist agents (A1–A4) and the baseline use OpenAI gpt-4.1-nano.
# The verifier (AV) uses Groq llama-3.3-70b-versatile — a completely different
# model family (Meta Llama via Groq) so its failure modes are statistically
# independent from the OpenAI specialist agents it checks.
# The disclosure agent (A5) uses OpenAI gpt-4o-mini.

import os
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

GROQ_BASE_URL  = "https://api.groq.com/openai/v1"

# gpt-4.1-nano — used by all profiling/classification/rule agents and baseline.
OPENAI_MODEL = "gpt-4.1-nano"

# llama-3.3-70b-versatile via Groq — used exclusively by the verifier (AV).
# Different provider + different model family = independent failure modes.
GROQ_VERIFIER_MODEL = "llama-3.3-70b-versatile"

# gpt-4o-mini — used by the disclosure agent (A5).
OPENAI_DISCLOSURE_MODEL = "gpt-4o-mini"

# Keep this name so graph.py imports still resolve without changes.
OPENAI_VERIFIER_MODEL = GROQ_VERIFIER_MODEL

_NANO_MODEL_INFO = {
    "vision": False,
    "function_calling": True,
    "json_output": True,
    "family": "unknown",
    "structured_output": False,
}

_GROQ_MODEL_INFO = {
    "vision": False,
    "function_calling": True,
    "json_output": True,
    "family": "unknown",
    "structured_output": False,
}

_MINI_MODEL_INFO = {
    "vision": False,
    "function_calling": True,
    "json_output": True,
    "family": "unknown",
    "structured_output": False,
}


def get_model_client() -> OpenAIChatCompletionClient:
    """gpt-4.1-nano client for A1, A2, A3, A4, and the baseline.
    temperature=0 enforced so M3 (decision consistency) measures architecture
    stability, not model randomness."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_KEY is not set.")
    return OpenAIChatCompletionClient(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0,
        model_info=_NANO_MODEL_INFO,
    )


def get_verifier_client() -> OpenAIChatCompletionClient:
    """llama-3.3-70b-versatile client via Groq, used exclusively by AV.
    Groq (Meta Llama family) is a completely different provider and model family
    from the OpenAI specialist agents, ensuring statistically independent failure modes.
    temperature=0 for deterministic verification decisions."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set.")
    return OpenAIChatCompletionClient(
        model=GROQ_VERIFIER_MODEL,
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
        temperature=0,
        model_info=_GROQ_MODEL_INFO,
    )


def get_disclosure_client() -> OpenAIChatCompletionClient:
    """gpt-4o-mini client used exclusively by the A5 disclosure agent.
    temperature=0 for reproducible explanation text across evaluation runs."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_KEY is not set.")
    return OpenAIChatCompletionClient(
        model=OPENAI_DISCLOSURE_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0,
        model_info=_MINI_MODEL_INFO,
    )
