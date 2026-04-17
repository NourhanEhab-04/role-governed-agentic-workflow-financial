"""
Mock-based tests for run_client_profiler.
No real LLM call is made — the model client is replaced with a mock
that returns a controlled string, letting us test the full pipeline
(agent → parse → validate) in isolation.
"""
import asyncio
import json
import pytest
import autogen_agentchat.agents as agents_module
from unittest.mock import AsyncMock, MagicMock, patch

from autogen_core.models import CreateResult, RequestUsage

from agents.client_profiler import run_client_profiler


VALID_PROFILE = {
    "financial_knowledge": "basic",
    "risk_tolerance_score": 4,
    "investment_horizon": 3,
    "liquid_assets": 8000.0,
    "income": 42000.0,
    "investment_amount": 5000.0,
    "can_afford_total_loss": False,
    "financial_vulnerability": "LOW",
}


def make_mock_client(reply: str):
    """Return a mock model client whose create() returns `reply` as content."""
    result = CreateResult(
        finish_reason="stop",
        content=reply,
        usage=RequestUsage(prompt_tokens=10, completion_tokens=20),
        cached=False,
        logprobs=None,
    )
    client = MagicMock()
    client.create = AsyncMock(return_value=result)
    client.model_info = {
        "vision": False,
        "function_calling": False,
        "json_output": False,
        "family": "unknown",
        "structured_output": False,
    }
    return client


@pytest.mark.asyncio
async def test_run_client_profiler_returns_valid_dict():
    """Happy path: mock returns clean JSON, function returns parsed dict."""
    mock_client = make_mock_client(json.dumps(VALID_PROFILE))
    result = await run_client_profiler("I am a client with basic knowledge.", mock_client)
    assert result == VALID_PROFILE


@pytest.mark.asyncio
async def test_run_client_profiler_handles_json_in_prose():
    """Agent wraps JSON in prose — parser must still extract it correctly."""
    reply = f"Here is the extracted profile:\n{json.dumps(VALID_PROFILE)}\nLet me know if correct."
    mock_client = make_mock_client(reply)
    result = await run_client_profiler("some input", mock_client)
    assert result["financial_knowledge"] == "basic"
    assert result["risk_tolerance_score"] == 4


@pytest.mark.asyncio
async def test_run_client_profiler_raises_on_missing_keys():
    """If agent omits required keys, ValueError is raised after parsing."""
    incomplete = {k: v for k, v in list(VALID_PROFILE.items())[:3]}
    mock_client = make_mock_client(json.dumps(incomplete))
    with pytest.raises(ValueError, match="missing required keys"):
        await run_client_profiler("some input", mock_client)


@pytest.mark.asyncio
async def test_run_client_profiler_raises_on_invalid_enum():
    """If agent returns an invalid enum value, ValueError is raised."""
    bad = {**VALID_PROFILE, "financial_knowledge": "guru"}
    mock_client = make_mock_client(json.dumps(bad))
    with pytest.raises(ValueError, match="Invalid financial_knowledge"):
        await run_client_profiler("some input", mock_client)


@pytest.mark.asyncio
async def test_run_client_profiler_raises_on_no_json():
    """If agent returns no JSON at all, ValueError is raised."""
    mock_client = make_mock_client("I could not extract a profile from this input.")
    with pytest.raises(ValueError, match="No JSON object found"):
        await run_client_profiler("some input", mock_client)


MOCK_VALID_RESPONSE = """{
    "financial_knowledge": "basic",
    "risk_tolerance_score": 3,
    "investment_horizon": 5,
    "liquid_assets": 12000.0,
    "income": 38000.0,
    "investment_amount": 4000.0,
    "can_afford_total_loss": false,
    "financial_vulnerability": "LOW"
}"""

MOCK_CLARIFICATION_RESPONSE = """{
    "status": "needs_clarification",
    "missing": ["investment_horizon", "liquid_assets"]
}"""


def _make_on_messages_mock(content: str):
    async def mock_on_messages(self, messages, cancellation_token):
        msg = MagicMock()
        msg.content = content
        resp = MagicMock()
        resp.chat_message = msg
        return resp
    return mock_on_messages


def test_valid_response_returns_clean_dict():
    """Patch on_messages directly — valid JSON response produces correct dict."""
    with patch.object(agents_module.AssistantAgent, "on_messages",
                      new=_make_on_messages_mock(MOCK_VALID_RESPONSE)):
        result = asyncio.run(
            run_client_profiler("I am 34, conservative investor", MagicMock())
        )
    assert result["financial_knowledge"] == "basic"
    assert result["risk_tolerance_score"] == 3
    assert result["can_afford_total_loss"] is False


def test_needs_clarification_response_raises_value_error():
    """Clarification response lacks required keys → parse raises ValueError."""
    with patch.object(agents_module.AssistantAgent, "on_messages",
                      new=_make_on_messages_mock(MOCK_CLARIFICATION_RESPONSE)):
        with pytest.raises(ValueError):
            asyncio.run(
                run_client_profiler("incomplete input", MagicMock())
            )
