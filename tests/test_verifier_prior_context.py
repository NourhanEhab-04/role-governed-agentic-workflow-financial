# tests/test_verifier_prior_context.py
"""
Step 3: Context-aware re-verifier.

Tests:
  1. _build_verifier_message includes PRIOR_VERIFICATION_CONTEXT when provided
  2. _build_verifier_message omits the section when prior_verification is None
  3. Prior context includes the failed fields and issues from the previous run
  4. First verifier call gets prior_verification=None (no prior context)
  5. Second verifier call (after correction) gets the first verification as prior context
  6. Third verifier call gets the second verification as prior context
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


RAW = "test input"

def _client():
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

def _corrected_client():
    c = _client()
    c["risk_tolerance_score"] = 3
    return c

def _ver_pass():
    return {
        "passed": True, "confidence": 0.95, "issues": [],
        "field_checks": {"financial_knowledge": {"supported": True, "reason": "ok"}},
    }

def _ver_fail():
    return {
        "passed": False, "confidence": 0.45,
        "issues": ["risk_tolerance_score is wrong"],
        "field_checks": {
            "risk_tolerance_score": {"supported": False, "reason": "Client said 3 but 5 returned."},
            "financial_knowledge":  {"supported": True,  "reason": "ok"},
        },
    }


# ── Unit tests for _build_verifier_message with prior_verification ────────────

def test_build_verifier_message_includes_prior_context_when_provided():
    from agents.verifier_agent import _build_verifier_message
    prior = _ver_fail()
    msg = _build_verifier_message("input", {"f": "v"}, prior_verification=prior)
    assert "PRIOR_VERIFICATION_CONTEXT" in msg


def test_build_verifier_message_omits_prior_context_when_none():
    from agents.verifier_agent import _build_verifier_message
    msg = _build_verifier_message("input", {"f": "v"}, prior_verification=None)
    assert "PRIOR_VERIFICATION_CONTEXT" not in msg


def test_build_verifier_message_prior_context_includes_failed_fields():
    from agents.verifier_agent import _build_verifier_message
    prior = _ver_fail()
    msg = _build_verifier_message("input", {"f": "v"}, prior_verification=prior)
    assert "risk_tolerance_score" in msg
    assert "Client said 3 but 5 returned." in msg


def test_build_verifier_message_prior_context_includes_prior_issues():
    from agents.verifier_agent import _build_verifier_message
    prior = _ver_fail()
    msg = _build_verifier_message("input", {"f": "v"}, prior_verification=prior)
    assert "risk_tolerance_score is wrong" in msg


def test_build_verifier_message_prior_context_excludes_passing_fields():
    from agents.verifier_agent import _build_verifier_message
    prior = _ver_fail()
    msg = _build_verifier_message("input", {"f": "v"}, prior_verification=prior)
    # financial_knowledge passed → should not appear in the failed-fields block
    # (it will appear in PARSED_OUTPUT but not in the prior-failed-fields section)
    # We verify that the prior context block only lists risk_tolerance_score
    import json
    prior_section_start = msg.find("PRIOR_VERIFICATION_CONTEXT")
    prior_section = msg[prior_section_start:]
    assert "financial_knowledge" not in prior_section


# ── Integration: prior_verification threaded through the A1 feedback loop ─────

@pytest.mark.asyncio
async def test_first_verifier_call_has_no_prior_context():
    """On attempt 1 the verifier must receive prior_verification=None."""
    captured = []

    async def spy_verifier(raw, profile, mc, consistency_issues=None, prior_verification=None):
        captured.append(prior_verification)
        return _ver_pass()

    with patch("agents.verifier_agent.run_verifier_on_a1", side_effect=spy_verifier):
        from orchestrator.feedback_loops import run_a1_feedback_loop
        await run_a1_feedback_loop(RAW, _client(), MagicMock())

    assert captured[0] is None


@pytest.mark.asyncio
async def test_second_verifier_call_receives_first_result_as_prior():
    """On attempt 2 the verifier must receive the attempt-1 result as prior_verification."""
    captured = []

    async def spy_verifier(raw, profile, mc, consistency_issues=None, prior_verification=None):
        captured.append(prior_verification)
        if len(captured) == 1:
            return _ver_fail()
        return _ver_pass()

    fb_mock = AsyncMock(return_value=_corrected_client())

    with (
        patch("agents.verifier_agent.run_verifier_on_a1", side_effect=spy_verifier),
        patch("agents.client_profiler.run_client_profiler_with_feedback", new=fb_mock),
    ):
        from orchestrator.feedback_loops import run_a1_feedback_loop
        await run_a1_feedback_loop(RAW, _client(), MagicMock())

    assert len(captured) == 2
    assert captured[0] is None                          # first call: no prior
    assert captured[1] == _ver_fail()                   # second call: prior = first result


@pytest.mark.asyncio
async def test_third_verifier_call_receives_second_result_as_prior():
    """On attempt 3 the verifier must receive the attempt-2 result as prior_verification."""
    fail1 = _ver_fail()
    fail2 = dict(_ver_fail())
    fail2["issues"] = ["different issue on second pass"]
    captured = []

    async def spy_verifier(raw, profile, mc, consistency_issues=None, prior_verification=None):
        captured.append(prior_verification)
        if len(captured) == 1:
            return fail1
        if len(captured) == 2:
            return fail2
        return _ver_pass()

    fb_mock = AsyncMock(return_value=_corrected_client())

    with (
        patch("agents.verifier_agent.run_verifier_on_a1", side_effect=spy_verifier),
        patch("agents.client_profiler.run_client_profiler_with_feedback", new=fb_mock),
    ):
        from orchestrator.feedback_loops import run_a1_feedback_loop
        await run_a1_feedback_loop(RAW, _client(), MagicMock(), max_iterations=3)

    assert captured[0] is None
    assert captured[1] == fail1
    assert captured[2] == fail2
