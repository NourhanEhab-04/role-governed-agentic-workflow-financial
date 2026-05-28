"""
rl/local_model_health.py
========================
Health-check utility for the local Ollama instance.

This module is intentionally kept dependency-free: it uses only Python's
built-in `urllib.request` so it can be called from any environment without
needing httpx, requests, or any other third-party library.

Ollama exposes two endpoints we care about:
  GET  /api/tags          → lists all locally available models
  POST /api/show          → fetches detail for a specific model

The check_ollama_health() function:
  1. Hits /api/tags to verify Ollama is running.
  2. Inspects the returned model list for the configured LOCAL_MODEL_NAME.
  3. Returns a HealthResult dataclass with the outcome.

Usage
-----
    from rl.local_model_health import check_ollama_health

    result = check_ollama_health()
    if result.is_running and result.target_model_available:
        print("Ready to train locally.")
    else:
        print(result.error_message)

CLI
---
    python -m rl.local_model_health
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class HealthResult:
    """
    Result of an Ollama health check.

    Attributes:
        is_running:              True if Ollama responded to /api/tags.
        available_models:        List of model tag strings that Ollama has pulled.
                                 Empty when Ollama is not running.
        target_model_available:  True if LOCAL_MODEL_NAME appears in available_models.
        target_model_name:       The model name that was checked (from settings).
        ollama_base_url:         The base URL that was probed (without /v1 suffix).
        error_message:           Human-readable failure description; "" on success.
    """
    is_running: bool
    available_models: list[str] = field(default_factory=list)
    target_model_available: bool = False
    target_model_name: str = ""
    ollama_base_url: str = ""
    error_message: str = ""


def _tags_url(base_url: str) -> str:
    """
    Derive the /api/tags URL from LOCAL_MODEL_BASE_URL.

    LOCAL_MODEL_BASE_URL is the OpenAI-compatible endpoint: .../v1
    The Ollama management API lives at the root, not under /v1.

    Examples:
        "http://localhost:11434/v1"  → "http://localhost:11434/api/tags"
        "http://localhost:11434"     → "http://localhost:11434/api/tags"
    """
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return root + "/api/tags"


def check_ollama_health(
    *,
    base_url: str | None = None,
    model_name: str | None = None,
    timeout_seconds: float = 3.0,
) -> HealthResult:
    """
    Check whether Ollama is running and has the target model available.

    Args:
        base_url:        Override LOCAL_MODEL_BASE_URL from settings.
                         Useful for testing with a mock server.
        model_name:      Override LOCAL_MODEL_NAME from settings.
        timeout_seconds: HTTP request timeout. Defaults to 3 s.

    Returns:
        HealthResult with all fields populated.
    """
    from config.settings import LOCAL_MODEL_BASE_URL, LOCAL_MODEL_NAME

    effective_base_url = base_url if base_url is not None else LOCAL_MODEL_BASE_URL
    effective_model = model_name if model_name is not None else LOCAL_MODEL_NAME

    # Derive the Ollama management root (strip /v1 if present)
    root = effective_base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]

    tags_url = root + "/api/tags"

    result_base = HealthResult(
        is_running=False,
        target_model_name=effective_model,
        ollama_base_url=root,
    )

    try:
        req = urllib.request.Request(tags_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            if resp.status != 200:
                result_base.error_message = (
                    f"Ollama /api/tags returned HTTP {resp.status}"
                )
                return result_base

            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        result_base.error_message = (
            f"Cannot reach Ollama at {tags_url}: {exc.reason}. "
            f"Is Ollama installed and running? "
            f"Install from https://ollama.com then run: ollama pull {effective_model}"
        )
        return result_base
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        result_base.error_message = f"Ollama /api/tags returned unparseable response: {exc}"
        return result_base

    # Parse the model list from the /api/tags response.
    # Ollama returns: {"models": [{"name": "llama3.1:8b", ...}, ...]}
    raw_models: list[dict] = body.get("models", [])
    available: list[str] = [m.get("name", "") for m in raw_models if m.get("name")]

    # Ollama model names may have the ":latest" suffix stripped in some versions.
    # Normalise by checking if effective_model is a substring of any available name.
    target_found = any(
        effective_model == name or effective_model == name.split(":")[0]
        for name in available
    )

    return HealthResult(
        is_running=True,
        available_models=available,
        target_model_available=target_found,
        target_model_name=effective_model,
        ollama_base_url=root,
        error_message="" if target_found else (
            f"Model '{effective_model}' not found in Ollama. "
            f"Available: {available or ['(none pulled yet)']}. "
            f"Run: ollama pull {effective_model}"
        ),
    )


def print_health_report(result: HealthResult) -> None:
    """Print a human-readable health summary to stdout."""
    print(f"Ollama base URL : {result.ollama_base_url}")
    print(f"Ollama running  : {'YES' if result.is_running else 'NO'}")
    if result.is_running:
        print(f"Available models: {result.available_models or ['(none)']}")
        print(
            f"Target model    : {result.target_model_name} — "
            f"{'AVAILABLE' if result.target_model_available else 'NOT FOUND'}"
        )
    if result.error_message:
        print(f"Error           : {result.error_message}")


if __name__ == "__main__":
    health = check_ollama_health()
    print_health_report(health)
    raise SystemExit(0 if (health.is_running and health.target_model_available) else 1)
