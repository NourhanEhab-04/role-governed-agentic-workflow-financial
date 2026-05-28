# config/settings.py
# Application-level settings loaded from environment variables.

import os
from dotenv import load_dotenv

load_dotenv()

# PostgreSQL connection string — optional; pipeline runs without a DB.
DATABASE_URL: str | None = os.getenv("DATABASE_URL")

# Evaluation settings
EVAL_RUNS_PER_SCENARIO: int = int(os.getenv("EVAL_RUNS_PER_SCENARIO", "3"))

# ── RL / Local model settings (Step 3) ────────────────────────────────────────
# LLM_BACKEND selects which inference backend agents use.
#   "groq"       — cloud Groq API (default, requires GROQ_API_KEY)
#   "local"      — local Ollama instance (no API key required)
LLM_BACKEND: str = os.getenv("LLM_BACKEND", "groq")

# Ollama model tag to use when LLM_BACKEND="local".
# Must match a model that has been pulled: `ollama pull llama3.1:8b`
LOCAL_MODEL_NAME: str = os.getenv("LOCAL_MODEL_NAME", "llama3.1:8b")

# Base URL for the Ollama OpenAI-compatible REST endpoint.
# Ollama exposes /v1/chat/completions at this address by default.
LOCAL_MODEL_BASE_URL: str = os.getenv("LOCAL_MODEL_BASE_URL", "http://localhost:11434/v1")
