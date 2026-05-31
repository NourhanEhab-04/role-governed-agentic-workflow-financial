# tests/test_corrector_wiring.py
"""
Tests for the feedback-loop wiring inside orchestrator/orchestrator.py.

After the corrector→feedback-loop refactor, corrections are driven by
run_client_profiler_with_feedback / run_product_classifier_with_feedback
instead of the old run_corrector_on_a1 / run_corrector_on_a2.

Covers:
  1. When AV passes → feedback never runs, no correction keys in state
  2. When AV fails  → feedback runs, a1_correction / a2_correction set
  3. When feedback succeeds → a1_final_verification set, client_profile replaced
  4. When feedback fails    → non-fatal, original profile kept, error recorded
  5. When re-verification fails after feedback → non-fatal, error recorded
  6. a2_correction mirrors the same logic for A2
  7. fields_fixed list is populated correctly from verifier field_checks
All LLM agents mocked — no API calls.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _good_client():
    return {
        "age": 40,
        "financial_knowledge": "moderate",
        "risk_tolerance_score": 5,
        "investment_horizon": 5,
        "liquid_assets": 20000.0,
        "income": 50000.0,
        "investment_amount": 5000.0,
        "can_afford_total_loss": False,
        "financial_vulnerability": "LOW",
    }


def _good_product():
    return {
        "product_name": "Global Equity ETF",
        "risk_class": 4,
        "complexity_tier": "NON-COMPLEX",
        "requires_knowledge_level": "basic",
        "minimum_horizon": 3,
        "potential_loss": "partial",
        "leverage": False,
    }


def _good_rule_verdict():
    return {
        "rules": {r: "PASS" for r in ["R1","R2","R3","R4","R5","R6","R7"]},
        "score": 100,
        "decision": "SUITABLE",
        "failed_rules": [],
    }


def _good_conflict():
    return {"flags": [], "escalate": False, "summary": "ok"}


def _good_suitability():
    return {
        "decision": "SUITABLE",
        "summary": "ok",
        "rule_findings": [
            {"rule_id": r, "status": "PASS", "explanation": "ok"}
            for r in ["R1","R2","R3","R4","R5","R6","R7"]
        ],
        "flags_addressed": [],
        "regulatory_basis": "Article 25(2)",
        "client_facing_summary": "Suitable.",
    }


def _ver_pass():
    return {
        "passed": True,
        "confidence": 0.95,
        "issues": [],
        "field_checks": {
            "financial_knowledge": {"supported": True, "reason": "ok"},
        },
    }


def _ver_fail():
    return {
        "passed": False,
        "confidence": 0.45,
        "issues": ["risk_tolerance_score is wrong"],
        "field_checks": {
            "risk_tolerance_score": {"supported": False, "reason": "Client said 3 but 8 returned."},
            "financial_knowledge":  {"supported": True,  "reason": "ok"},
        },
    }


def _corrected_client():
    c = _good_client()  # includes "age": 40
    c["risk_tolerance_score"] = 3
    return c


def _corrected_product():
    p = _good_product()
    p["risk_class"] = 5
    return p


# ── Pipeline runner helper ────────────────────────────────────────────────────

async def _run(
    a1_ver=None, a2_ver=None,
    a1_corrector=None, a2_corrector=None,
    a1_reverify=None, a2_reverify=None,
    a1_ver_raises=None, a1_corrector_raises=None, a1_reverify_raises=None,
    a2_ver_raises=None, a2_corrector_raises=None, a2_reverify_raises=None,
    sample_rate=1.0,
):
    a1_ver      = a1_ver      or _ver_pass()
    a2_ver      = a2_ver      or _ver_pass()
    a1_reverify = a1_reverify or _ver_pass()
    a2_reverify = a2_reverify or _ver_pass()

    def _make_fb_mock(result, raises):
        if raises:
            return AsyncMock(side_effect=raises)
        return AsyncMock(return_value=result)

    # run_verifier_on_a1 is called up to twice (first verify + re-verify)
    if a1_ver_raises:
        a1_ver_mock = AsyncMock(side_effect=a1_ver_raises)
    elif a1_reverify_raises:
        a1_ver_mock = AsyncMock(side_effect=[a1_ver, a1_reverify_raises])
    else:
        a1_ver_mock = AsyncMock(side_effect=[a1_ver, a1_reverify])

    if a2_ver_raises:
        a2_ver_mock = AsyncMock(side_effect=a2_ver_raises)
    elif a2_reverify_raises:
        a2_ver_mock = AsyncMock(side_effect=[a2_ver, a2_reverify_raises])
    else:
        a2_ver_mock = AsyncMock(side_effect=[a2_ver, a2_reverify])

    # Feedback mocks replace the old corrector mocks
    a1_fb_mock = _make_fb_mock(a1_corrector or _corrected_client(), a1_corrector_raises)
    a2_fb_mock = _make_fb_mock(a2_corrector or _corrected_product(), a2_corrector_raises)

    with (
        patch("agents.client_profiler.run_client_profiler",     new=AsyncMock(return_value=_good_client())),
        patch("agents.product_classifier.run_product_classifier", new=AsyncMock(return_value=_good_product())),
        patch("agents.rule_engine_agent.run_rule_engine_agent",  new=AsyncMock(return_value=_good_rule_verdict())),
        patch("agents.conflict_detector.run_conflict_detector",  new=AsyncMock(return_value=(_good_conflict(), []))),
        patch("agents.conflict_detector.check_rule_engine_agreement", new=MagicMock(return_value={"agreed": True, "details": ""})),
        patch("agents.disclosure_agent.run_disclosure_agent",    new=AsyncMock(return_value=_good_suitability())),
        patch("orchestrator.pre_check_tool.run_pre_check",       new=MagicMock(return_value={"decision": "SUITABLE", "triggered_rules": []})),
        patch("orchestrator.orchestrator._VERIFIER_SAMPLE_RATE", sample_rate),
        patch("agents.verifier_agent.run_verifier_on_a1",                          new=a1_ver_mock),
        patch("agents.verifier_agent.run_verifier_on_a2",                          new=a2_ver_mock),
        patch("agents.client_profiler.run_client_profiler_with_feedback",           new=a1_fb_mock),
        patch("agents.product_classifier.run_product_classifier_with_feedback",    new=a2_fb_mock),
    ):
        from orchestrator.orchestrator import run_pipeline
        return await run_pipeline(
            client_input='{"description": "moderate investor"}',
            product_input='{"description": "equity ETF"}',
            model_client=MagicMock(),
        )


# ── Tests: feedback not triggered when AV passes ─────────────────────────────

@pytest.mark.asyncio
async def test_no_correction_when_a1_passes():
    state, _ = await _run(a1_ver=_ver_pass())
    assert "a1_correction" not in state

@pytest.mark.asyncio
async def test_no_correction_when_a2_passes():
    state, _ = await _run(a2_ver=_ver_pass())
    assert "a2_correction" not in state

@pytest.mark.asyncio
async def test_no_final_verification_when_a1_passes():
    state, _ = await _run(a1_ver=_ver_pass())
    assert "a1_final_verification" not in state


# ── Tests: feedback triggered when AV fails ──────────────────────────────────

@pytest.mark.asyncio
async def test_a1_correction_set_when_av_fails():
    state, _ = await _run(a1_ver=_ver_fail())
    assert "a1_correction" in state

@pytest.mark.asyncio
async def test_a2_correction_set_when_av_fails():
    state, _ = await _run(a2_ver=_ver_fail())
    assert "a2_correction" in state

@pytest.mark.asyncio
async def test_a1_correction_has_original_and_corrected():
    state, _ = await _run(a1_ver=_ver_fail(), a1_corrector=_corrected_client())
    corr = state["a1_correction"]
    assert "original" in corr
    assert "corrected" in corr
    assert corr["corrected"]["risk_tolerance_score"] == 3

@pytest.mark.asyncio
async def test_a1_correction_replaces_client_profile():
    """After correction, state["client_profile"] must be the corrected version."""
    state, _ = await _run(a1_ver=_ver_fail(), a1_corrector=_corrected_client())
    assert state["client_profile"]["risk_tolerance_score"] == 3

@pytest.mark.asyncio
async def test_a2_correction_replaces_product_profile():
    state, _ = await _run(a2_ver=_ver_fail(), a2_corrector=_corrected_product())
    assert state["product_profile"]["risk_class"] == 5

@pytest.mark.asyncio
async def test_a1_final_verification_set_after_correction():
    state, _ = await _run(a1_ver=_ver_fail())
    assert "a1_final_verification" in state

@pytest.mark.asyncio
async def test_a2_final_verification_set_after_correction():
    state, _ = await _run(a2_ver=_ver_fail())
    assert "a2_final_verification" in state

@pytest.mark.asyncio
async def test_fields_fixed_populated_from_failed_field_checks():
    state, _ = await _run(a1_ver=_ver_fail())
    assert "risk_tolerance_score" in state["a1_correction"]["fields_fixed"]
    # passing field must not be listed
    assert "financial_knowledge" not in state["a1_correction"]["fields_fixed"]


# ── Tests: feedback failure is non-fatal ─────────────────────────────────────

@pytest.mark.asyncio
async def test_feedback_failure_is_non_fatal():
    """Feedback crash must not halt the pipeline."""
    state, _ = await _run(
        a1_ver=_ver_fail(),
        a1_corrector_raises=RuntimeError("LLM timeout"),
    )
    assert "suitability_report" in state
    assert state.get("halt") is not True

@pytest.mark.asyncio
async def test_feedback_failure_records_error():
    state, _ = await _run(
        a1_ver=_ver_fail(),
        a1_corrector_raises=RuntimeError("LLM timeout"),
    )
    assert "error" in state["a1_correction"]
    assert "LLM timeout" in state["a1_correction"]["error"]

@pytest.mark.asyncio
async def test_feedback_failure_keeps_original_profile():
    """When feedback fails, original profile must remain in state."""
    state, _ = await _run(
        a1_ver=_ver_fail(),
        a1_corrector_raises=RuntimeError("timeout"),
    )
    assert state["client_profile"]["risk_tolerance_score"] == 5  # original value

@pytest.mark.asyncio
async def test_feedback_failure_corrected_field_is_none():
    state, _ = await _run(
        a1_ver=_ver_fail(),
        a1_corrector_raises=ValueError("bad output"),
    )
    assert state["a1_correction"]["corrected"] is None


# ── Tests: re-verification failure is non-fatal ───────────────────────────────

@pytest.mark.asyncio
async def test_reverify_failure_is_non_fatal():
    """Re-verify crash after feedback must not halt pipeline."""
    state, _ = await _run(
        a1_ver=_ver_fail(),
        a1_reverify_raises=RuntimeError("re-verify timeout"),
    )
    assert "suitability_report" in state

@pytest.mark.asyncio
async def test_reverify_failure_records_error_in_final_verification():
    state, _ = await _run(
        a1_ver=_ver_fail(),
        a1_reverify_raises=RuntimeError("re-verify timeout"),
    )
    v = state["a1_final_verification"]
    assert v["passed"] is None
    assert any("Re-verify error" in i for i in v["issues"])


# ── Tests: pipeline still completes in all scenarios ─────────────────────────

@pytest.mark.asyncio
async def test_pipeline_completes_av_pass():
    state, _ = await _run()
    assert "suitability_report" in state

@pytest.mark.asyncio
async def test_pipeline_completes_av_fail_correction_success():
    state, _ = await _run(a1_ver=_ver_fail(), a2_ver=_ver_fail())
    assert "suitability_report" in state

@pytest.mark.asyncio
async def test_pipeline_completes_av_fail_correction_fails():
    state, _ = await _run(
        a1_ver=_ver_fail(),
        a1_corrector_raises=RuntimeError("x"),
        a2_ver=_ver_fail(),
        a2_corrector_raises=RuntimeError("x"),
    )
    assert "suitability_report" in state
