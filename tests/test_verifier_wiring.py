# tests/test_verifier_wiring.py
"""
Tests for the validation-layer wiring in orchestrator/orchestrator.py.

All LLM agents are mocked so these tests are fast and offline.
We verify:
  1. a1_consistency_issues is always set in state after A1
  2. a2_consistency_issues is always set in state after A2
  3. cross_consistency_issues is set before pre_check
  4. a1_verification is set when consistency issues exist
  5. a2_verification is set when consistency issues exist
  6. verifier errors are non-fatal (pipeline continues)
  7. verifier is skipped when no issues and sample rate is 0
"""

import pytest
import random
from unittest.mock import AsyncMock, patch, MagicMock


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _good_client():
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


def _good_conflict_report():
    return {"flags": [], "escalate": False, "summary": "No conflicts."}


def _good_suitability_report():
    return {
        "decision": "SUITABLE",
        "summary": "Client is suitable.",
        "rule_findings": [
            {"rule_id": r, "status": "PASS", "explanation": "ok"}
            for r in ["R1","R2","R3","R4","R5","R6","R7"]
        ],
        "flags_addressed": [],
        "regulatory_basis": "Article 25(2) MiFID II",
        "client_facing_summary": "You are suitable for this product.",
    }


def _good_verification():
    return {
        "passed": True,
        "confidence": 0.95,
        "issues": [],
        "field_checks": {
            "financial_knowledge": {"supported": True, "reason": "Clear from input."}
        },
    }


def _good_pre_check():
    return {"decision": "SUITABLE", "triggered_rules": []}


def _good_audit_verdict():
    return {"agreed": True, "details": ""}


# ── Helper to run the pipeline with all agents mocked ────────────────────────

async def _run_with_mocks(
    client_profile=None,
    product_profile=None,
    a1_verifier_result=None,
    a2_verifier_result=None,
    a1_verifier_raises=None,
    a2_verifier_raises=None,
    sample_rate=0.0,
):
    """
    Run run_pipeline with all LLM agents replaced by mocks.
    Returns (state, audit_log).
    """
    client_profile = client_profile or _good_client()
    product_profile = product_profile or _good_product()

    with (
        patch("agents.client_profiler.run_client_profiler", new=AsyncMock(return_value=client_profile)),
        patch("agents.product_classifier.run_product_classifier", new=AsyncMock(return_value=product_profile)),
        patch("agents.rule_engine_agent.run_rule_engine_agent", new=AsyncMock(return_value=_good_rule_verdict())),
        patch("agents.conflict_detector.run_conflict_detector", new=AsyncMock(return_value=_good_conflict_report())),
        patch("agents.conflict_detector.check_rule_engine_agreement", new=MagicMock(return_value=_good_audit_verdict())),
        patch("agents.disclosure_agent.run_disclosure_agent", new=AsyncMock(return_value=_good_suitability_report())),
        patch("orchestrator.orchestrator.run_pre_check", new=MagicMock(return_value=_good_pre_check())),
        patch("orchestrator.orchestrator._VERIFIER_SAMPLE_RATE", sample_rate),
    ):
        if a1_verifier_raises:
            a1_mock = AsyncMock(side_effect=a1_verifier_raises)
        else:
            a1_mock = AsyncMock(return_value=a1_verifier_result or _good_verification())

        if a2_verifier_raises:
            a2_mock = AsyncMock(side_effect=a2_verifier_raises)
        else:
            a2_mock = AsyncMock(return_value=a2_verifier_result or _good_verification())

        with (
            patch("agents.verifier_agent.run_verifier_on_a1", new=a1_mock),
            patch("agents.verifier_agent.run_verifier_on_a2", new=a2_mock),
        ):
            from orchestrator.orchestrator import run_pipeline
            return await run_pipeline(
                client_input='{"description": "moderate investor, 5yr horizon, 5000 EUR"}',
                product_input='{"description": "diversified equity ETF, medium risk"}',
                model_client=MagicMock(),
            )


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a1_consistency_issues_always_in_state():
    """a1_consistency_issues must always be present in state."""
    state, _ = await _run_with_mocks()
    assert "a1_consistency_issues" in state
    assert isinstance(state["a1_consistency_issues"], list)


@pytest.mark.asyncio
async def test_a2_consistency_issues_always_in_state():
    """a2_consistency_issues must always be present in state."""
    state, _ = await _run_with_mocks()
    assert "a2_consistency_issues" in state
    assert isinstance(state["a2_consistency_issues"], list)


@pytest.mark.asyncio
async def test_cross_consistency_issues_always_in_state():
    """cross_consistency_issues must always be present in state."""
    state, _ = await _run_with_mocks()
    assert "cross_consistency_issues" in state
    assert isinstance(state["cross_consistency_issues"], list)


@pytest.mark.asyncio
async def test_clean_inputs_no_consistency_issues():
    """A clean client+product pair should produce empty issue lists."""
    state, _ = await _run_with_mocks()
    assert state["a1_consistency_issues"] == []
    assert state["a2_consistency_issues"] == []
    assert state["cross_consistency_issues"] == []


@pytest.mark.asyncio
async def test_verifier_called_when_a1_has_consistency_issues():
    """a1_verification must be set when a1_consistency_issues is non-empty."""
    # zero income → HIGH vulnerability mismatch
    bad_client = _good_client()
    bad_client["income"] = 0.0
    bad_client["financial_vulnerability"] = "LOW"

    state, _ = await _run_with_mocks(client_profile=bad_client, sample_rate=0.0)

    assert len(state["a1_consistency_issues"]) > 0
    assert "a1_verification" in state
    assert state["a1_verification"]["passed"] is True  # mocked verifier returns good result


@pytest.mark.asyncio
async def test_verifier_called_when_a2_has_consistency_issues():
    """a2_verification must be set when a2_consistency_issues is non-empty."""
    # risk_class 7 without leverage → consistency issue
    bad_product = _good_product()
    bad_product["risk_class"] = 7
    bad_product["leverage"] = False
    bad_product["potential_loss"] = "total"
    bad_product["complexity_tier"] = "COMPLEX"
    bad_product["requires_knowledge_level"] = "advanced"
    bad_product["minimum_horizon"] = 1

    state, _ = await _run_with_mocks(product_profile=bad_product, sample_rate=0.0)

    assert len(state["a2_consistency_issues"]) > 0
    assert "a2_verification" in state


@pytest.mark.asyncio
async def test_verifier_skipped_when_no_issues_and_zero_sample_rate():
    """With clean inputs and sample_rate=0, verifier should NOT be called."""
    state, _ = await _run_with_mocks(sample_rate=0.0)

    assert state["a1_consistency_issues"] == []
    assert state["a2_consistency_issues"] == []
    assert "a1_verification" not in state
    assert "a2_verification" not in state


@pytest.mark.asyncio
async def test_verifier_called_by_sampling_with_full_sample_rate():
    """With sample_rate=1.0, verifier must be called even with no issues."""
    state, _ = await _run_with_mocks(sample_rate=1.0)

    assert state["a1_consistency_issues"] == []
    assert "a1_verification" in state
    assert "a2_verification" in state


@pytest.mark.asyncio
async def test_a1_verifier_error_is_non_fatal():
    """A verifier exception must not halt the pipeline — state records the error."""
    bad_client = _good_client()
    bad_client["income"] = 0.0
    bad_client["financial_vulnerability"] = "LOW"  # triggers verifier

    state, _ = await _run_with_mocks(
        client_profile=bad_client,
        a1_verifier_raises=RuntimeError("API timeout"),
        sample_rate=0.0,
    )

    # Pipeline must still complete (suitability_report present)
    assert "suitability_report" in state
    # Error recorded in a1_verification
    assert "a1_verification" in state
    assert any("Verifier error" in issue for issue in state["a1_verification"]["issues"])


@pytest.mark.asyncio
async def test_a2_verifier_error_is_non_fatal():
    """A2 verifier exception must not halt the pipeline."""
    bad_product = _good_product()
    bad_product["risk_class"] = 7
    bad_product["leverage"] = False
    bad_product["potential_loss"] = "total"
    bad_product["complexity_tier"] = "COMPLEX"
    bad_product["requires_knowledge_level"] = "advanced"
    bad_product["minimum_horizon"] = 1

    state, _ = await _run_with_mocks(
        product_profile=bad_product,
        a2_verifier_raises=RuntimeError("timeout"),
        sample_rate=0.0,
    )

    assert "suitability_report" in state
    assert "a2_verification" in state
    assert any("Verifier error" in issue for issue in state["a2_verification"]["issues"])


@pytest.mark.asyncio
async def test_verification_result_structure_when_present():
    """When a1_verification is present, it must have the required keys."""
    bad_client = _good_client()
    bad_client["income"] = 0.0
    bad_client["financial_vulnerability"] = "LOW"

    state, _ = await _run_with_mocks(client_profile=bad_client, sample_rate=0.0)

    v = state["a1_verification"]
    for key in ("passed", "confidence", "issues", "field_checks"):
        assert key in v, f"Missing key: {key}"


@pytest.mark.asyncio
async def test_pipeline_completes_with_all_clean_inputs():
    """Full pipeline with clean inputs should complete and have suitability_report."""
    state, audit = await _run_with_mocks(sample_rate=0.0)
    assert "suitability_report" in state
    assert state.get("halt") is not True


# ── Sample-rate gating tests ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_verifier_always_runs_when_consistency_issues_found():
    """Consistency issues must trigger the verifier regardless of sample_rate=0."""
    bad_client = _good_client()
    bad_client["income"] = 0.0
    bad_client["financial_vulnerability"] = "LOW"  # income=0 → HIGH expected

    state, _ = await _run_with_mocks(client_profile=bad_client, sample_rate=0.0)

    assert len(state["a1_consistency_issues"]) > 0
    assert "a1_verification" in state, "Verifier must run when consistency issues are present"


@pytest.mark.asyncio
async def test_verifier_skipped_for_clean_inputs_at_zero_rate():
    """With clean inputs and sample_rate=0, no LLM verifier call should happen."""
    state, _ = await _run_with_mocks(sample_rate=0.0)

    assert state["a1_consistency_issues"] == []
    assert state["a2_consistency_issues"] == []
    assert "a1_verification" not in state
    assert "a2_verification" not in state


@pytest.mark.asyncio
async def test_default_sample_rate_is_below_one():
    """Default _VERIFIER_SAMPLE_RATE must be < 1.0 (cost reduction requirement)."""
    from orchestrator.orchestrator import _VERIFIER_SAMPLE_RATE
    assert _VERIFIER_SAMPLE_RATE < 1.0, (
        f"Sample rate is {_VERIFIER_SAMPLE_RATE}; set it below 1.0 to reduce LLM costs"
    )
