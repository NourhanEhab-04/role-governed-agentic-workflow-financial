# tests/test_surgical_correction.py
"""
Step 4: Surgical field patching in the correction prompts.

Tests:
  1. Correction prompt includes LOCKED_FIELDS when some fields pass
  2. Correction prompt omits LOCKED_FIELDS when no fields pass
  3. LOCKED_FIELDS contains only passing fields, not failed ones
  4. LOCKED_FIELDS values match the previous_output values exactly
  5. FAILED FIELDS section lists only unsupported fields
  6. Same behaviour for A2 (product classifier)
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _verification_mixed():
    """Two fields fail, one passes."""
    return {
        "passed": False,
        "confidence": 0.5,
        "issues": ["risk_tolerance_score wrong", "income wrong"],
        "field_checks": {
            "risk_tolerance_score": {"supported": False, "reason": "Client said 3 but 7 returned."},
            "income": {"supported": False, "reason": "Net income not grossed up."},
            "financial_knowledge": {"supported": True, "reason": "Correct."},
        },
    }


def _verification_all_fail():
    """All checked fields fail."""
    return {
        "passed": False,
        "confidence": 0.2,
        "issues": ["everything wrong"],
        "field_checks": {
            "risk_tolerance_score": {"supported": False, "reason": "wrong"},
            "income": {"supported": False, "reason": "wrong"},
        },
    }


def _previous_output():
    return {
        "age": 40,
        "financial_knowledge": "moderate",
        "risk_tolerance_score": 7,
        "investment_horizon": 5,
        "liquid_assets": 20000.0,
        "income": 40000.0,
        "investment_amount": 5000.0,
        "can_afford_total_loss": False,
        "financial_vulnerability": "LOW",
    }


# ── A1 (client profiler) correction prompt tests ──────────────────────────────

@pytest.mark.asyncio
async def test_a1_correction_prompt_includes_locked_fields_when_some_pass():
    captured_prompts = []

    async def fake_on_messages(messages, cancellation_token):
        captured_prompts.extend(messages)
        result = MagicMock()
        result.chat_message.content = json.dumps(_previous_output())
        return result

    mock_agent = MagicMock()
    mock_agent.on_messages = fake_on_messages

    with patch("agents.client_profiler.AssistantAgent", return_value=mock_agent):
        from agents.client_profiler import run_client_profiler_with_feedback
        await run_client_profiler_with_feedback(
            "raw input", _previous_output(), _verification_mixed(), MagicMock()
        )

    prompt = captured_prompts[0].content
    assert "LOCKED_FIELDS" in prompt


@pytest.mark.asyncio
async def test_a1_correction_prompt_locked_fields_contains_passing_field():
    captured_prompts = []

    async def fake_on_messages(messages, cancellation_token):
        captured_prompts.extend(messages)
        result = MagicMock()
        result.chat_message.content = json.dumps(_previous_output())
        return result

    mock_agent = MagicMock()
    mock_agent.on_messages = fake_on_messages

    with patch("agents.client_profiler.AssistantAgent", return_value=mock_agent):
        from agents.client_profiler import run_client_profiler_with_feedback
        await run_client_profiler_with_feedback(
            "raw input", _previous_output(), _verification_mixed(), MagicMock()
        )

    prompt = captured_prompts[0].content
    locked_start = prompt.find("LOCKED_FIELDS")
    locked_section = prompt[locked_start:]
    assert "financial_knowledge" in locked_section
    assert "moderate" in locked_section


@pytest.mark.asyncio
async def test_a1_correction_prompt_locked_fields_excludes_failed_fields():
    captured_prompts = []

    async def fake_on_messages(messages, cancellation_token):
        captured_prompts.extend(messages)
        result = MagicMock()
        result.chat_message.content = json.dumps(_previous_output())
        return result

    mock_agent = MagicMock()
    mock_agent.on_messages = fake_on_messages

    with patch("agents.client_profiler.AssistantAgent", return_value=mock_agent):
        from agents.client_profiler import run_client_profiler_with_feedback
        await run_client_profiler_with_feedback(
            "raw input", _previous_output(), _verification_mixed(), MagicMock()
        )

    prompt = captured_prompts[0].content
    locked_start = prompt.find("LOCKED_FIELDS")
    locked_section = prompt[locked_start:locked_start + 500]
    # risk_tolerance_score and income are failed → must NOT appear in locked section
    assert "risk_tolerance_score" not in locked_section
    assert "income" not in locked_section


@pytest.mark.asyncio
async def test_a1_correction_prompt_omits_locked_fields_when_all_fail():
    captured_prompts = []

    async def fake_on_messages(messages, cancellation_token):
        captured_prompts.extend(messages)
        result = MagicMock()
        result.chat_message.content = json.dumps(_previous_output())
        return result

    mock_agent = MagicMock()
    mock_agent.on_messages = fake_on_messages

    with patch("agents.client_profiler.AssistantAgent", return_value=mock_agent):
        from agents.client_profiler import run_client_profiler_with_feedback
        await run_client_profiler_with_feedback(
            "raw input", _previous_output(), _verification_all_fail(), MagicMock()
        )

    prompt = captured_prompts[0].content
    assert "LOCKED_FIELDS" not in prompt


@pytest.mark.asyncio
async def test_a1_correction_prompt_failed_fields_section_present():
    captured_prompts = []

    async def fake_on_messages(messages, cancellation_token):
        captured_prompts.extend(messages)
        result = MagicMock()
        result.chat_message.content = json.dumps(_previous_output())
        return result

    mock_agent = MagicMock()
    mock_agent.on_messages = fake_on_messages

    with patch("agents.client_profiler.AssistantAgent", return_value=mock_agent):
        from agents.client_profiler import run_client_profiler_with_feedback
        await run_client_profiler_with_feedback(
            "raw input", _previous_output(), _verification_mixed(), MagicMock()
        )

    prompt = captured_prompts[0].content
    assert "FAILED FIELDS" in prompt
    assert "risk_tolerance_score" in prompt
    assert "income" in prompt


# ── A2 (product classifier) correction prompt tests ───────────────────────────

def _product_output():
    return {
        "product_name": "Global Equity ETF",
        "risk_class": 3,
        "complexity_tier": "NON-COMPLEX",
        "requires_knowledge_level": "basic",
        "minimum_horizon": 3,
        "potential_loss": "partial",
        "leverage": False,
    }


def _product_verification_mixed():
    return {
        "passed": False,
        "confidence": 0.5,
        "issues": ["risk_class wrong"],
        "field_checks": {
            "risk_class": {"supported": False, "reason": "Should be 4 not 3."},
            "complexity_tier": {"supported": True, "reason": "Correct."},
            "leverage": {"supported": True, "reason": "Correct."},
        },
    }


@pytest.mark.asyncio
async def test_a2_correction_prompt_includes_locked_fields():
    captured_prompts = []

    async def fake_on_messages(messages, cancellation_token):
        captured_prompts.extend(messages)
        result = MagicMock()
        result.chat_message.content = json.dumps(_product_output())
        return result

    mock_agent = MagicMock()
    mock_agent.on_messages = fake_on_messages

    with patch("agents.product_classifier.AssistantAgent", return_value=mock_agent):
        from agents.product_classifier import run_product_classifier_with_feedback
        await run_product_classifier_with_feedback(
            "product description", _product_output(), _product_verification_mixed(), MagicMock()
        )

    prompt = captured_prompts[0].content
    assert "LOCKED_FIELDS" in prompt
    locked_start = prompt.find("LOCKED_FIELDS")
    locked_section = prompt[locked_start:]
    assert "complexity_tier" in locked_section
    assert "leverage" in locked_section
    assert "risk_class" not in locked_section[:200]
