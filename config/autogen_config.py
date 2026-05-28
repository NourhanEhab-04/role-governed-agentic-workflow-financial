# config/autogen_config.py
# AutoGen-specific configuration helpers.
# All agents in this project use the OpenAIChatCompletionClient from
# autogen_ext, configured via config/llm_config.py.
# This module re-exports the relevant helpers for convenience.

from config.llm_config import get_model_client, get_verifier_client

__all__ = ["get_model_client", "get_verifier_client"]
