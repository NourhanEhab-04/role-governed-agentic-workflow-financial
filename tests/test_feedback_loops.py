# tests/test_feedback_loops.py
"""
Unit tests for orchestrator/feedback_loops.py.

All LLM calls are mocked — no API calls.
Each test covers exactly one independent scenario of the loop.

A1 scenarios:
  1. Verifier passes on first attempt → no correction, no final_verification
  2. Verifier fails once, feedback succeeds, re-verify passes
  3. Verifier fails twice, two feedback runs, re-verify passes on 3rd attempt
  4. Max iterations reached without passing → loop exits, correction recorded
  5. Feedback run raises → non-fatal, original profile kept, error in loop_state
  6. Re-verify raises after feedback → non-fatal, final_verification has error
  7. First verifier raises → exception propagates to caller

A2 mirrors A1 in structure — one representative test each.
"""

import pytest
from unittest.mock import AsyncMock, patch


# ── Fixtures ──────────────────────────────────────────────────────────────────

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


def _product():
    return {
        "product_name": "Global Equity ETF",
        "risk_class": 4,
        "complexity_tier": "NON-COMPLEX",
        "requires_knowledge_level": "basic",
        "minimum_horizon": 3,
        "potential_loss": "partial",
        "leverage": False,
    }


def _ver_pass():
    return {
        "passed": True, "confidence": 0.95, "issues": [],
        "field_checks": {
            "financial_knowledge": {"supported": True, "reason": "ok"},
            "risk_tolerance_score": {"supported": True, "reason": "ok"},
        },
    }


def _ver_fail():
    return {
        "passed": False, "confidence": 0.45,
        "issues": ["risk_tolerance_score is wrong"],
        "field_checks": {
            "risk_tolerance_score": {"supported": False, "reason": "Client said 3 but 8 returned."},
            "financial_knowledge":  {"supported": True,  "reason": "ok"},
        },
    }


def _corrected_client():
    c = _client()
    c["risk_tolerance_score"] = 3
    return c


def _corrected_product():
    p = _product()
    p["risk_class"] = 5
    return p


RAW = '{"description": "test"}'
MOCK_CLIENT = object()  # sentinel — never actually called


# ── A1 loop tests ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a1_pass_on_first_attempt_no_correction():
    """Verifier passes immediately → loop_state has no correction or final_verification."""
    with (
        patch("agents.verifier_agent.run_verifier_on_a1", new=AsyncMock(return_value=_ver_pass())),
        patch("agents.client_profiler.run_client_profiler_with_feedback", new=AsyncMock()) as fb_mock,
    ):
        from orchestrator.feedback_loops import run_a1_feedback_loop
        profile, loop_state = await run_a1_feedback_loop(RAW, _client(), MOCK_CLIENT)

    assert loop_state["verification"]["passed"] is True
    assert "correction" not in loop_state
    assert "final_verification" not in loop_state
    fb_mock.assert_not_called()


@pytest.mark.asyncio
async def test_a1_pass_on_first_attempt_returns_original_profile():
    """Profile is unchanged when verifier passes on first attempt."""
    with patch("agents.verifier_agent.run_verifier_on_a1", new=AsyncMock(return_value=_ver_pass())):
        from orchestrator.feedback_loops import run_a1_feedback_loop
        profile, _ = await run_a1_feedback_loop(RAW, _client(), MOCK_CLIENT)

    assert profile == _client()


@pytest.mark.asyncio
async def test_a1_fail_then_pass_sets_correction_and_final_verification():
    """One failure → one feedback run → re-verify passes → correction + final_verification set."""
    ver_mock = AsyncMock(side_effect=[_ver_fail(), _ver_pass()])
    fb_mock  = AsyncMock(return_value=_corrected_client())

    with (
        patch("agents.verifier_agent.run_verifier_on_a1", new=ver_mock),
        patch("agents.client_profiler.run_client_profiler_with_feedback", new=fb_mock),
    ):
        from orchestrator.feedback_loops import run_a1_feedback_loop
        profile, loop_state = await run_a1_feedback_loop(RAW, _client(), MOCK_CLIENT)

    assert loop_state["verification"]["passed"] is False
    assert "correction" in loop_state
    assert loop_state["correction"]["original"] == _client()
    assert loop_state["correction"]["corrected"]["risk_tolerance_score"] == 3
    assert "risk_tolerance_score" in loop_state["correction"]["fields_fixed"]
    assert "financial_knowledge" not in loop_state["correction"]["fields_fixed"]
    assert loop_state["final_verification"]["passed"] is True
    assert profile["risk_tolerance_score"] == 3


@pytest.mark.asyncio
async def test_a1_profile_replaced_after_successful_feedback():
    """Returned profile must be the corrected version, not the original."""
    ver_mock = AsyncMock(side_effect=[_ver_fail(), _ver_pass()])
    fb_mock  = AsyncMock(return_value=_corrected_client())

    with (
        patch("agents.verifier_agent.run_verifier_on_a1", new=ver_mock),
        patch("agents.client_profiler.run_client_profiler_with_feedback", new=fb_mock),
    ):
        from orchestrator.feedback_loops import run_a1_feedback_loop
        profile, _ = await run_a1_feedback_loop(RAW, _client(), MOCK_CLIENT)

    assert profile["risk_tolerance_score"] == 3


@pytest.mark.asyncio
async def test_a1_two_failures_two_feedback_runs():
    """Two consecutive failures trigger two feedback runs before passing."""
    corrected_v2 = _corrected_client()
    corrected_v2["financial_knowledge"] = "basic"

    ver_mock = AsyncMock(side_effect=[_ver_fail(), _ver_fail(), _ver_pass()])
    fb_mock  = AsyncMock(side_effect=[_corrected_client(), corrected_v2])

    with (
        patch("agents.verifier_agent.run_verifier_on_a1", new=ver_mock),
        patch("agents.client_profiler.run_client_profiler_with_feedback", new=fb_mock),
    ):
        from orchestrator.feedback_loops import run_a1_feedback_loop
        profile, loop_state = await run_a1_feedback_loop(RAW, _client(), MOCK_CLIENT)

    assert fb_mock.call_count == 2
    assert loop_state["correction"]["attempts"] == 2
    assert loop_state["final_verification"]["passed"] is True


@pytest.mark.asyncio
async def test_a1_max_iterations_reached_accepts_last_result():
    """After max_iterations all failing, loop stops and returns the last profile."""
    ver_mock = AsyncMock(return_value=_ver_fail())
    fb_mock  = AsyncMock(return_value=_corrected_client())

    with (
        patch("agents.verifier_agent.run_verifier_on_a1", new=ver_mock),
        patch("agents.client_profiler.run_client_profiler_with_feedback", new=fb_mock),
    ):
        from orchestrator.feedback_loops import run_a1_feedback_loop
        profile, loop_state = await run_a1_feedback_loop(RAW, _client(), MOCK_CLIENT, max_iterations=3)

    # verifier: 3 calls (iterations 1,2,3); feedback: 2 calls (after iter 1 and 2)
    assert ver_mock.call_count == 3
    assert fb_mock.call_count == 2
    assert loop_state["final_verification"]["passed"] is False
    assert "correction" in loop_state


@pytest.mark.asyncio
async def test_a1_feedback_raises_non_fatal_keeps_original():
    """If the feedback run raises, the exception is caught, original profile kept."""
    ver_mock = AsyncMock(return_value=_ver_fail())
    fb_mock  = AsyncMock(side_effect=RuntimeError("LLM timeout"))

    with (
        patch("agents.verifier_agent.run_verifier_on_a1", new=ver_mock),
        patch("agents.client_profiler.run_client_profiler_with_feedback", new=fb_mock),
    ):
        from orchestrator.feedback_loops import run_a1_feedback_loop
        profile, loop_state = await run_a1_feedback_loop(RAW, _client(), MOCK_CLIENT)

    # Original profile returned unchanged
    assert profile == _client()
    # Error recorded in correction dict
    assert "error" in loop_state["correction"]
    assert "LLM timeout" in loop_state["correction"]["error"]
    # corrected stays None
    assert loop_state["correction"]["corrected"] is None


@pytest.mark.asyncio
async def test_a1_feedback_raises_no_final_verification():
    """When feedback raises and loop breaks, final_verification is not set."""
    ver_mock = AsyncMock(return_value=_ver_fail())
    fb_mock  = AsyncMock(side_effect=RuntimeError("timeout"))

    with (
        patch("agents.verifier_agent.run_verifier_on_a1", new=ver_mock),
        patch("agents.client_profiler.run_client_profiler_with_feedback", new=fb_mock),
    ):
        from orchestrator.feedback_loops import run_a1_feedback_loop
        _, loop_state = await run_a1_feedback_loop(RAW, _client(), MOCK_CLIENT)

    assert "final_verification" not in loop_state


@pytest.mark.asyncio
async def test_a1_reverify_raises_non_fatal():
    """Re-verify failure after a successful feedback run is non-fatal."""
    ver_mock = AsyncMock(side_effect=[_ver_fail(), RuntimeError("re-verify timeout")])
    fb_mock  = AsyncMock(return_value=_corrected_client())

    with (
        patch("agents.verifier_agent.run_verifier_on_a1", new=ver_mock),
        patch("agents.client_profiler.run_client_profiler_with_feedback", new=fb_mock),
    ):
        from orchestrator.feedback_loops import run_a1_feedback_loop
        profile, loop_state = await run_a1_feedback_loop(RAW, _client(), MOCK_CLIENT)

    # Corrected profile was already applied before re-verify failed
    assert profile["risk_tolerance_score"] == 3
    # final_verification records the error
    fv = loop_state["final_verification"]
    assert fv["passed"] is None
    assert any("Re-verify error" in i for i in fv["issues"])


@pytest.mark.asyncio
async def test_a1_first_verifier_raises_propagates():
    """First verification failure propagates to the caller (orchestrator catches it)."""
    ver_mock = AsyncMock(side_effect=RuntimeError("network error"))

    with patch("agents.verifier_agent.run_verifier_on_a1", new=ver_mock):
        from orchestrator.feedback_loops import run_a1_feedback_loop
        with pytest.raises(RuntimeError, match="network error"):
            await run_a1_feedback_loop(RAW, _client(), MOCK_CLIENT)


@pytest.mark.asyncio
async def test_a1_correction_attempts_counter():
    """correction.attempts tracks how many feedback runs actually completed."""
    ver_mock = AsyncMock(side_effect=[_ver_fail(), _ver_pass()])
    fb_mock  = AsyncMock(return_value=_corrected_client())

    with (
        patch("agents.verifier_agent.run_verifier_on_a1", new=ver_mock),
        patch("agents.client_profiler.run_client_profiler_with_feedback", new=fb_mock),
    ):
        from orchestrator.feedback_loops import run_a1_feedback_loop
        _, loop_state = await run_a1_feedback_loop(RAW, _client(), MOCK_CLIENT)

    assert loop_state["correction"]["attempts"] == 1


# ── A2 loop tests (representative subset) ────────────────────────────────────

@pytest.mark.asyncio
async def test_a2_pass_on_first_attempt_no_correction():
    with (
        patch("agents.verifier_agent.run_verifier_on_a2", new=AsyncMock(return_value=_ver_pass())),
        patch("agents.product_classifier.run_product_classifier_with_feedback", new=AsyncMock()) as fb_mock,
    ):
        from orchestrator.feedback_loops import run_a2_feedback_loop
        profile, loop_state = await run_a2_feedback_loop(RAW, _product(), MOCK_CLIENT)

    assert loop_state["verification"]["passed"] is True
    assert "correction" not in loop_state
    fb_mock.assert_not_called()


@pytest.mark.asyncio
async def test_a2_fail_then_pass_replaces_product_profile():
    ver_mock = AsyncMock(side_effect=[_ver_fail(), _ver_pass()])
    fb_mock  = AsyncMock(return_value=_corrected_product())

    with (
        patch("agents.verifier_agent.run_verifier_on_a2", new=ver_mock),
        patch("agents.product_classifier.run_product_classifier_with_feedback", new=fb_mock),
    ):
        from orchestrator.feedback_loops import run_a2_feedback_loop
        profile, loop_state = await run_a2_feedback_loop(RAW, _product(), MOCK_CLIENT)

    assert profile["risk_class"] == 5
    assert loop_state["correction"]["original"]["risk_class"] == 4
    assert loop_state["final_verification"]["passed"] is True


@pytest.mark.asyncio
async def test_a2_feedback_raises_keeps_original_product():
    ver_mock = AsyncMock(return_value=_ver_fail())
    fb_mock  = AsyncMock(side_effect=ValueError("parse error"))

    with (
        patch("agents.verifier_agent.run_verifier_on_a2", new=ver_mock),
        patch("agents.product_classifier.run_product_classifier_with_feedback", new=fb_mock),
    ):
        from orchestrator.feedback_loops import run_a2_feedback_loop
        profile, loop_state = await run_a2_feedback_loop(RAW, _product(), MOCK_CLIENT)

    assert profile == _product()
    assert "error" in loop_state["correction"]
    assert loop_state["correction"]["corrected"] is None


@pytest.mark.asyncio
async def test_a2_first_verifier_raises_propagates():
    ver_mock = AsyncMock(side_effect=RuntimeError("timeout"))

    with patch("agents.verifier_agent.run_verifier_on_a2", new=ver_mock):
        from orchestrator.feedback_loops import run_a2_feedback_loop
        with pytest.raises(RuntimeError, match="timeout"):
            await run_a2_feedback_loop(RAW, _product(), MOCK_CLIENT)
