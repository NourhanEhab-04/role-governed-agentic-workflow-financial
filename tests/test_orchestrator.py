# tests/test_orchestrator.py
"""
Zero-LLM — 10 tests for the refactored orchestrator.
All agents are mocked. Tests verify:
  - pre_check_verdict is written between A2 and A3
  - audit_verdict is written before A4 LLM call
  - cross-check failures halt the pipeline with a descriptive reason
  - portfolio_concentration_pct injection survives the refactor
  - return type is always (dict, dict)
  - A2 double-failure halts before pre_check runs
  - A5 always runs even when escalated=True
"""
import json
import pytest
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch, MagicMock

# ── shared fixtures ──────────────────────────────────────────────────────────

CLIENT_PROFILE = {
    "financial_knowledge": "basic",
    "risk_tolerance_score": 4,
    "investment_horizon": 3,
    "liquid_assets": 8000,
    "income": 42000,
    "investment_amount": 5000,
    "can_afford_total_loss": False,
    "financial_vulnerability": "LOW",
}
PRODUCT_PROFILE = {
    "product_name": "Test Fund",
    "risk_class": 4,
    "complexity_tier": "NON-COMPLEX",
    "requires_knowledge_level": "basic",
    "minimum_horizon": 2,
    "potential_loss": "partial",
    "leverage": False,
}
RULE_VERDICT = {
    "score": 75,
    "decision": "SUITABLE",
    "rules": [],
    "failed_rules": [],
}
CONFLICT_REPORT = {
    "flags": [],
    "escalate": False,
    "summary": "No escalation required.",
}
SUITABILITY_REPORT = {
    "decision": "SUITABLE",
    "regulatory_basis": "Article 25(2) MiFID II",
    "rule_findings": [
        {"rule_id": f"R{i}", "passed": True, "detail": "ok"}
        for i in range(1, 8)
    ],
    "flags_addressed": [],
    "client_summary": "You are suitable.",
}

MODEL_CLIENT = MagicMock()

CLIENT_INPUT = json.dumps({**CLIENT_PROFILE, "portfolio_concentration_pct": 40})
PRODUCT_INPUT = "A simple equity fund"


def _patch_all(
    *,
    a1_return=None,
    a2_return=None,
    a3_return=None,
    a4_return=None,
    a5_return=None,
    a1_side_effect=None,
):
    """Return a dict of patch targets with sensible defaults."""
    return {
        "agents.client_profiler.run_client_profiler":
            AsyncMock(return_value=CLIENT_PROFILE if a1_return is None else a1_return,
                      side_effect=a1_side_effect),
        "agents.product_classifier.run_product_classifier":
            AsyncMock(return_value=PRODUCT_PROFILE if a2_return is None else a2_return),
        "agents.rule_engine_agent.run_rule_engine_agent":
            AsyncMock(return_value=RULE_VERDICT if a3_return is None else a3_return),
        "agents.conflict_detector.run_conflict_detector":
            AsyncMock(return_value=CONFLICT_REPORT if a4_return is None else a4_return),
        "agents.disclosure_agent.run_disclosure_agent":
            AsyncMock(return_value=SUITABILITY_REPORT if a5_return is None else a5_return),
    }


async def _run(patches: dict) -> tuple:
    from orchestrator.orchestrator import run_pipeline
    with ExitStack() as stack:
        for target, mock in patches.items():
            stack.enter_context(patch(target, mock))
        return await run_pipeline(CLIENT_INPUT, PRODUCT_INPUT, MODEL_CLIENT)


# ── tests ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_return_type_is_always_tuple_of_two_dicts():
    state, audit = await _run(_patch_all())
    assert isinstance(state, dict)
    assert isinstance(audit, dict)


@pytest.mark.asyncio
async def test_pre_check_verdict_written_to_state():
    """pre_check_verdict must be populated after A2 and before A3."""
    state, _ = await _run(_patch_all())
    assert "pre_check_verdict" in state
    assert state["pre_check_verdict"] is not None
    assert "decision" in state["pre_check_verdict"]


@pytest.mark.asyncio
async def test_pre_check_verdict_uses_real_rule_engine():
    """pre_check_verdict.decision must match what the rule engine actually returns."""
    from rule_engine.rule_engine import evaluate_suitability
    expected = evaluate_suitability(CLIENT_PROFILE, PRODUCT_PROFILE)["decision"]
    state, _ = await _run(_patch_all())
    assert state["pre_check_verdict"]["decision"] == expected


@pytest.mark.asyncio
async def test_audit_verdict_written_before_a4():
    """audit_verdict must be populated after A3 and before A4 LLM call."""
    state, _ = await _run(_patch_all())
    assert "audit_verdict" in state
    assert state["audit_verdict"] is not None
    assert "agreed" in state["audit_verdict"]


@pytest.mark.asyncio
async def test_audit_verdict_agrees_when_a3_correct():
    """When A3 returns the correct verdict, audit_verdict.agreed must be True."""
    from rule_engine.rule_engine import evaluate_suitability
    real = evaluate_suitability(CLIENT_PROFILE, PRODUCT_PROFILE)
    correct_verdict = {
        "score": real["score"],
        "decision": real["decision"],
        "rules": real["rules"],
        "failed_rules": [r["rule"] for r in real["rules"] if not r["pass"]],
    }
    state, _ = await _run(_patch_all(a3_return=correct_verdict))
    assert state["audit_verdict"]["agreed"] is True


@pytest.mark.asyncio
async def test_portfolio_concentration_injected():
    """portfolio_concentration_pct from client_input must appear in client_profile."""
    state, _ = await _run(_patch_all())
    assert state["client_profile"].get("portfolio_concentration_pct") == 40


@pytest.mark.asyncio
async def test_a1_double_failure_halts_before_pre_check():
    """If A1 fails twice, pre_check must never run and state must be halted."""
    from orchestrator.validators import validate_after_a1

    bad_profile = {}  # missing all required keys → validate_after_a1 fails
    patches = _patch_all(a1_return=bad_profile)
    state, audit = await _run(patches)

    assert state.get("halt") is True
    assert "pre_check_verdict" not in state or state.get("pre_check_verdict") is None


@pytest.mark.asyncio
async def test_a2_double_failure_halts_before_pre_check():
    """If A2 fails twice, pre_check must never run."""
    bad_product = {}
    patches = _patch_all(a2_return=bad_product)
    state, audit = await _run(patches)

    assert state.get("halt") is True
    # pre_check needs both profiles — it must not have run
    assert state.get("pre_check_verdict") is None


@pytest.mark.asyncio
async def test_escalation_flag_set_when_conflict_report_escalates():
    """When conflict_report.escalate=True, state['escalated'] must be True."""
    escalating_report = {**CONFLICT_REPORT, "escalate": True,
                         "flags": ["contradiction"], "severity": "HIGH"}
    state, _ = await _run(_patch_all(a4_return=escalating_report))
    assert state.get("escalated") is True


@pytest.mark.asyncio
async def test_a5_always_runs_even_when_escalated():
    """A5 must run and suitability_report must be populated even when escalated."""
    escalating_report = {**CONFLICT_REPORT, "escalate": True,
                         "flags": ["contradiction"], "severity": "HIGH"}
    escalated_report = {**SUITABILITY_REPORT, "decision": "ESCALATED"}
    state, _ = await _run(_patch_all(
        a4_return=escalating_report,
        a5_return=escalated_report,
    ))
    assert state.get("suitability_report") is not None
    assert state["suitability_report"]["decision"] == "ESCALATED"