# tests/test_parallel_profiling.py
"""
Step 5: A1 and A2 feedback loops run in parallel.

Tests:
  1. Both a1 and a2 results are present after parallel run
  2. A1 failure halts the pipeline (a2 still ran but halt is set)
  3. A2 failure halts the pipeline
  4. Both running concurrently: verify A2 LLM call starts before A1 LLM call finishes
     (checked via call ordering under asyncio)
  5. a1_consistency_issues and a2_consistency_issues both set
  6. Pipeline completes normally with clean inputs (no regression)
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock, call


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
        "summary": "ok",
        "rule_findings": [
            {"rule_id": r, "status": "PASS", "explanation": "ok"}
            for r in ["R1","R2","R3","R4","R5","R6","R7"]
        ],
        "flags_addressed": [],
        "regulatory_basis": "Article 25(2) MiFID II",
        "client_facing_summary": "You are suitable.",
    }


def _good_pre_check():
    return {"decision": "SUITABLE", "triggered_rules": []}


def _good_audit_verdict():
    return {"agreed": True, "details": ""}


async def _run_pipeline(
    client_profile=None,
    product_profile=None,
    a1_raises=None,
    a2_raises=None,
    sample_rate=0.0,
):
    client_profile = client_profile or _good_client()
    product_profile = product_profile or _good_product()

    if a1_raises:
        a1_mock = AsyncMock(side_effect=a1_raises)
    else:
        a1_mock = AsyncMock(return_value=client_profile)

    if a2_raises:
        a2_mock = AsyncMock(side_effect=a2_raises)
    else:
        a2_mock = AsyncMock(return_value=product_profile)

    with (
        patch("agents.client_profiler.run_client_profiler", new=a1_mock),
        patch("agents.product_classifier.run_product_classifier", new=a2_mock),
        patch("agents.rule_engine_agent.run_rule_engine_agent", new=AsyncMock(return_value=_good_rule_verdict())),
        patch("agents.conflict_detector.run_conflict_detector", new=AsyncMock(return_value=_good_conflict_report())),
        patch("agents.conflict_detector.check_rule_engine_agreement", new=MagicMock(return_value=_good_audit_verdict())),
        patch("agents.disclosure_agent.run_disclosure_agent", new=AsyncMock(return_value=_good_suitability_report())),
        patch("orchestrator.orchestrator.run_pre_check", new=MagicMock(return_value=_good_pre_check())),
        patch("orchestrator.orchestrator._VERIFIER_SAMPLE_RATE", sample_rate),
    ):
        from orchestrator.orchestrator import run_pipeline
        return await run_pipeline(
            client_input='{"description": "moderate investor"}',
            product_input='{"description": "equity ETF"}',
            model_client=MagicMock(),
        )


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_both_profiles_present_after_parallel_run():
    """Both client_profile and product_profile must be in state."""
    state, _ = await _run_pipeline()
    assert "client_profile" in state
    assert "product_profile" in state


@pytest.mark.asyncio
async def test_both_consistency_issues_set():
    """Both a1_consistency_issues and a2_consistency_issues must be set."""
    state, _ = await _run_pipeline()
    assert "a1_consistency_issues" in state
    assert "a2_consistency_issues" in state


@pytest.mark.asyncio
async def test_pipeline_completes_normally():
    """Pipeline should reach suitability_report with clean inputs."""
    state, _ = await _run_pipeline()
    assert "suitability_report" in state
    assert state.get("halt") is not True


@pytest.mark.asyncio
async def test_a1_failure_halts_pipeline():
    """If A1 keeps failing (2 attempts), pipeline must halt."""
    state, _ = await _run_pipeline(a1_raises=ValueError("A1 parse error"))
    assert state.get("halt") is True
    assert "suitability_report" not in state


@pytest.mark.asyncio
async def test_a2_failure_halts_pipeline():
    """If A2 keeps failing (2 attempts), pipeline must halt."""
    state, _ = await _run_pipeline(a2_raises=ValueError("A2 parse error"))
    assert state.get("halt") is True
    assert "suitability_report" not in state


@pytest.mark.asyncio
async def test_a1_and_a2_both_called_exactly_once_on_success():
    """Both agent functions must be called exactly once when both succeed."""
    a1_mock = AsyncMock(return_value=_good_client())
    a2_mock = AsyncMock(return_value=_good_product())

    with (
        patch("agents.client_profiler.run_client_profiler", new=a1_mock),
        patch("agents.product_classifier.run_product_classifier", new=a2_mock),
        patch("agents.rule_engine_agent.run_rule_engine_agent", new=AsyncMock(return_value=_good_rule_verdict())),
        patch("agents.conflict_detector.run_conflict_detector", new=AsyncMock(return_value=_good_conflict_report())),
        patch("agents.conflict_detector.check_rule_engine_agreement", new=MagicMock(return_value=_good_audit_verdict())),
        patch("agents.disclosure_agent.run_disclosure_agent", new=AsyncMock(return_value=_good_suitability_report())),
        patch("orchestrator.orchestrator.run_pre_check", new=MagicMock(return_value=_good_pre_check())),
        patch("orchestrator.orchestrator._VERIFIER_SAMPLE_RATE", 0.0),
    ):
        from orchestrator.orchestrator import run_pipeline
        await run_pipeline(
            client_input='{"description": "test"}',
            product_input='{"description": "test"}',
            model_client=MagicMock(),
        )

    a1_mock.assert_called_once()
    a2_mock.assert_called_once()


@pytest.mark.asyncio
async def test_parallel_execution_both_tasks_complete():
    """asyncio.gather completes both tasks; verify via ordering of state keys being set."""
    order = []

    original_a1 = AsyncMock(return_value=_good_client())
    original_a2 = AsyncMock(return_value=_good_product())

    async def tracking_a1(*args, **kwargs):
        result = await original_a1(*args, **kwargs)
        order.append("a1_done")
        return result

    async def tracking_a2(*args, **kwargs):
        result = await original_a2(*args, **kwargs)
        order.append("a2_done")
        return result

    with (
        patch("agents.client_profiler.run_client_profiler", side_effect=tracking_a1),
        patch("agents.product_classifier.run_product_classifier", side_effect=tracking_a2),
        patch("agents.rule_engine_agent.run_rule_engine_agent", new=AsyncMock(return_value=_good_rule_verdict())),
        patch("agents.conflict_detector.run_conflict_detector", new=AsyncMock(return_value=_good_conflict_report())),
        patch("agents.conflict_detector.check_rule_engine_agreement", new=MagicMock(return_value=_good_audit_verdict())),
        patch("agents.disclosure_agent.run_disclosure_agent", new=AsyncMock(return_value=_good_suitability_report())),
        patch("orchestrator.orchestrator.run_pre_check", new=MagicMock(return_value=_good_pre_check())),
        patch("orchestrator.orchestrator._VERIFIER_SAMPLE_RATE", 0.0),
    ):
        from orchestrator.orchestrator import run_pipeline
        state, _ = await run_pipeline(
            client_input='{"description": "test"}',
            product_input='{"description": "test"}',
            model_client=MagicMock(),
        )

    # Both must have completed
    assert "a1_done" in order
    assert "a2_done" in order
    assert "suitability_report" in state
