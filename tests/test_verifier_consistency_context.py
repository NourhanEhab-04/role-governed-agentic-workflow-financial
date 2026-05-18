# tests/test_verifier_consistency_context.py
"""
Step 2: Verify that consistency_issues are threaded through to the verifier message.

Tests:
  1. _build_verifier_message includes DETERMINISTIC_PRE_CHECKS when issues provided
  2. _build_verifier_message omits the section when issues is None
  3. _build_verifier_message omits the section when issues is []
  4. run_verifier_on_a1 passes consistency_issues into the message sent to the agent
  5. run_a1_feedback_loop passes consistency_issues into the verifier call
  6. run_a2_feedback_loop passes consistency_issues into the verifier call
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


# ── Unit tests for _build_verifier_message ────────────────────────────────────

def test_build_verifier_message_includes_issues_when_provided():
    from agents.verifier_agent import _build_verifier_message
    msg = _build_verifier_message(
        "raw input",
        {"field": "value"},
        consistency_issues=["income is 0 but vulnerability is LOW"],
    )
    assert "DETERMINISTIC_PRE_CHECKS" in msg
    assert "income is 0 but vulnerability is LOW" in msg


def test_build_verifier_message_omits_section_when_none():
    from agents.verifier_agent import _build_verifier_message
    msg = _build_verifier_message("raw input", {"field": "value"}, consistency_issues=None)
    assert "DETERMINISTIC_PRE_CHECKS" not in msg


def test_build_verifier_message_omits_section_when_empty_list():
    from agents.verifier_agent import _build_verifier_message
    msg = _build_verifier_message("raw input", {"field": "value"}, consistency_issues=[])
    assert "DETERMINISTIC_PRE_CHECKS" not in msg


def test_build_verifier_message_includes_all_issues():
    from agents.verifier_agent import _build_verifier_message
    issues = ["issue one", "issue two", "issue three"]
    msg = _build_verifier_message("input", {}, consistency_issues=issues)
    for issue in issues:
        assert issue in msg


# ── Integration: consistency_issues reach the verifier agent call ─────────────

RAW = "test client input"

def _good_profile():
    return {
        "financial_knowledge": "moderate",
        "risk_tolerance_score": 5,
        "investment_horizon": 5,
        "liquid_assets": 20000.0,
        "income": 50000.0,
        "investment_amount": 5000.0,
        "can_afford_total_loss": False,
        "financial_vulnerability": "LOW",
    }

def _ver_pass():
    return {
        "passed": True, "confidence": 0.95, "issues": [],
        "field_checks": {"financial_knowledge": {"supported": True, "reason": "ok"}},
    }


@pytest.mark.asyncio
async def test_run_verifier_on_a1_passes_consistency_issues_in_message():
    """The message sent to the LLM must contain the consistency issues text."""
    captured_messages = []

    async def fake_on_messages(messages, cancellation_token):
        captured_messages.extend(messages)
        result = MagicMock()
        result.chat_message.content = (
            '{"passed": true, "confidence": 0.9, "issues": [], "field_checks": {}}'
        )
        return result

    mock_agent = MagicMock()
    mock_agent.on_messages = fake_on_messages

    issues = ["income is 0 but vulnerability is LOW"]

    with patch("agents.verifier_agent.AssistantAgent", return_value=mock_agent):
        from agents.verifier_agent import run_verifier_on_a1
        await run_verifier_on_a1(RAW, _good_profile(), MagicMock(), consistency_issues=issues)

    assert len(captured_messages) == 1
    assert "DETERMINISTIC_PRE_CHECKS" in captured_messages[0].content
    assert "income is 0 but vulnerability is LOW" in captured_messages[0].content


@pytest.mark.asyncio
async def test_run_a1_feedback_loop_forwards_consistency_issues():
    """Consistency issues passed to the loop must reach run_verifier_on_a1."""
    issues = ["investment_amount exceeds liquid_assets"]
    captured_kwargs = {}

    async def spy_verifier(raw_input, profile, model_client, consistency_issues=None, prior_verification=None):
        captured_kwargs["consistency_issues"] = consistency_issues
        return _ver_pass()

    with patch("agents.verifier_agent.run_verifier_on_a1", side_effect=spy_verifier):
        from orchestrator.feedback_loops import run_a1_feedback_loop
        await run_a1_feedback_loop(RAW, _good_profile(), MagicMock(), consistency_issues=issues)

    assert captured_kwargs["consistency_issues"] == issues


@pytest.mark.asyncio
async def test_run_a2_feedback_loop_forwards_consistency_issues():
    """Consistency issues passed to the loop must reach run_verifier_on_a2."""
    product = {
        "product_name": "ETF",
        "risk_class": 4,
        "complexity_tier": "NON-COMPLEX",
        "requires_knowledge_level": "basic",
        "minimum_horizon": 3,
        "potential_loss": "partial",
        "leverage": False,
    }
    issues = ["risk_class 7 but leverage is False"]
    captured_kwargs = {}

    async def spy_verifier(raw_input, profile, model_client, consistency_issues=None, prior_verification=None):
        captured_kwargs["consistency_issues"] = consistency_issues
        return _ver_pass()

    with patch("agents.verifier_agent.run_verifier_on_a2", side_effect=spy_verifier):
        from orchestrator.feedback_loops import run_a2_feedback_loop
        await run_a2_feedback_loop(RAW, product, MagicMock(), consistency_issues=issues)

    assert captured_kwargs["consistency_issues"] == issues
