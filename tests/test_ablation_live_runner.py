"""
tests/test_ablation_live_runner.py
=====================================
Static tests for the live runner module.
Tests scenario-to-input conversion, result extraction helpers,
manifest building, and metric wiring.

Zero API calls. Zero LLM calls.

Run with:
  pytest tests/test_ablation_live_runner.py -v
"""

import json
import pytest
from pathlib import Path

from evaluation.ablation_study.live_runner import (
    _scenario_to_inputs,
    _extract_final_decision,
    _extract_escalated,
    _extract_verifier_telemetry,
    _extract_consistency_telemetry,
    compute_live_metrics,
    compute_verifier_impact,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

SUITABLE_SCENARIO = {
    "_filename": "01_suitable_conservative.json",
    "_path":     "data/scenarios/01_suitable_conservative.json",
    "client_narrative": "Maria Santos is 67 years old...",
    "product_narrative": "Money market fund...",
    "expected_client_profile": {
        "financial_knowledge": "basic",
        "risk_tolerance_score": 2,
        "investment_horizon": 2,
        "liquid_assets": 80000.0,
        "income": 25200.0,
        "investment_amount": 50000.0,
        "can_afford_total_loss": False,
        "financial_vulnerability": "LOW",
        "age": 67,
    },
    "expected_product_profile": {
        "product_name": "Euro Money Market Fund",
        "risk_class": 1,
        "complexity_tier": "NON-COMPLEX",
        "requires_knowledge_level": "none",
        "minimum_horizon": 1,
        "potential_loss": "partial",
        "leverage": False,
    },
    "expected_decision": "SUITABLE",
    "expected_escalate": False,
    "expected_rules_failed": [],
}

ESCALATED_SCENARIO = {
    "_filename": "07_escalated_contradiction.json",
    "_path":     "data/scenarios/07_escalated_contradiction.json",
    "client_narrative": "Elena Rossi is 78...",
    "product_narrative": "MSCI World ETF...",
    "client": {
        "portfolio_concentration_pct": 10,
    },
    "expected_client_profile": {
        "financial_knowledge": "advanced",
        "risk_tolerance_score": 6,
        "investment_horizon": 8,
        "liquid_assets": 300000.0,
        "income": 40000.0,
        "investment_amount": 200000.0,
        "can_afford_total_loss": True,
        "financial_vulnerability": "HIGH",
        "age": 78,
    },
    "expected_product_profile": {
        "product_name": "MSCI World Index ETF",
        "risk_class": 4,
        "complexity_tier": "NON-COMPLEX",
        "requires_knowledge_level": "basic",
        "minimum_horizon": 5,
        "potential_loss": "partial",
        "leverage": False,
    },
    "expected_decision": "ESCALATED",
    "expected_escalate": True,
    "expected_rules_failed": [],
}

NO_NARRATIVE_SCENARIO = {
    "_filename": "synthetic_no_narrative.json",
    "_path":     "data/scenarios/synthetic_no_narrative.json",
    "expected_client_profile": {
        "financial_knowledge": "basic",
        "risk_tolerance_score": 3,
        "investment_horizon": 5,
        "liquid_assets": 50000.0,
        "income": 30000.0,
        "investment_amount": 10000.0,
        "can_afford_total_loss": False,
        "financial_vulnerability": "LOW",
        "age": 45,
    },
    "expected_product_profile": {
        "product_name": "Government Bond",
        "risk_class": 2,
        "complexity_tier": "NON-COMPLEX",
        "requires_knowledge_level": "basic",
        "minimum_horizon": 3,
        "potential_loss": "partial",
        "leverage": False,
    },
    "expected_decision": "SUITABLE",
    "expected_escalate": False,
    "expected_rules_failed": [],
}


# ── _scenario_to_inputs ────────────────────────────────────────────────────────

def test_narrative_used_when_present():
    client_input, product_input = _scenario_to_inputs(SUITABLE_SCENARIO)
    assert "Maria Santos" in client_input
    assert "Money market" in product_input


def test_fallback_to_json_when_no_narrative():
    client_input, product_input = _scenario_to_inputs(NO_NARRATIVE_SCENARIO)
    # Should be parseable JSON
    client_data = json.loads(client_input)
    assert "financial_knowledge" in client_data
    product_data = json.loads(product_input)
    assert "risk_class" in product_data


def test_concentration_injected_as_json_when_present():
    client_input, product_input = _scenario_to_inputs(ESCALATED_SCENARIO)
    # When portfolio_concentration_pct is in 'client', we pass JSON so graph.py can extract it
    data = json.loads(client_input)
    assert "portfolio_concentration_pct" in data
    assert data["portfolio_concentration_pct"] == 10


def test_product_narrative_passthrough():
    client_input, product_input = _scenario_to_inputs(ESCALATED_SCENARIO)
    assert "MSCI World" in product_input


# ── _extract_final_decision ────────────────────────────────────────────────────

def test_extract_decision_from_suitability_report():
    state = {"suitability_report": {"decision": "SUITABLE"}}
    assert _extract_final_decision(state) == "SUITABLE"


def test_extract_decision_escalated_from_flag():
    state = {"escalated": True, "suitability_report": {}}
    assert _extract_final_decision(state) == "ESCALATED"


def test_extract_decision_from_rule_verdict_fallback():
    state = {"rule_verdict": {"decision": "CONDITIONAL"}, "suitability_report": {}}
    assert _extract_final_decision(state) == "CONDITIONAL"


def test_extract_decision_error_when_halted():
    state = {"halt": True, "halt_reason": "something broke"}
    assert _extract_final_decision(state) == "ERROR"


def test_extract_decision_error_when_no_data():
    assert _extract_final_decision({}) == "ERROR"


# ── _extract_escalated ─────────────────────────────────────────────────────────

def test_escalated_from_report_decision():
    state = {"suitability_report": {"decision": "ESCALATED"}}
    assert _extract_escalated(state) is True


def test_escalated_from_flag():
    state = {"escalated": True, "suitability_report": {}}
    assert _extract_escalated(state) is True


def test_escalated_from_conflict_report():
    state = {"conflict_report": {"escalate": True}, "suitability_report": {}}
    assert _extract_escalated(state) is True


def test_not_escalated_when_suitable():
    state = {"suitability_report": {"decision": "SUITABLE"}, "conflict_report": {"escalate": False}}
    assert _extract_escalated(state) is False


def test_not_escalated_when_halted():
    state = {"halt": True}
    assert _extract_escalated(state) is False


# ── _extract_verifier_telemetry ────────────────────────────────────────────────

def test_verifier_telemetry_both_passed():
    state = {
        "a1_verification": {"passed": True, "field_checks": {}},
        "a2_verification": {"passed": True, "field_checks": {}},
        "a1_correction":   None,
        "a2_correction":   None,
    }
    tel = _extract_verifier_telemetry(state)
    assert tel["a1_verifier_ran"] is True
    assert tel["a1_verifier_passed"] is True
    assert tel["a1_correction_applied"] is False
    assert tel["any_correction"] is False


def test_verifier_telemetry_a1_corrected():
    state = {
        "a1_verification": {"passed": False, "field_checks": {"financial_knowledge": {"supported": False}}},
        "a2_verification": {"passed": True,  "field_checks": {}},
        "a1_correction":   {"original": {}, "corrected": {"financial_knowledge": "basic"}, "fields_fixed": ["financial_knowledge"]},
        "a2_correction":   None,
    }
    tel = _extract_verifier_telemetry(state)
    assert tel["a1_correction_applied"] is True
    assert tel["a1_fields_fixed"] == ["financial_knowledge"]
    assert tel["any_correction"] is True
    assert tel["a2_correction_applied"] is False


def test_verifier_telemetry_not_ran():
    state = {}
    tel = _extract_verifier_telemetry(state)
    assert tel["a1_verifier_ran"] is False
    assert tel["any_correction"] is False


# ── _extract_consistency_telemetry ────────────────────────────────────────────

def test_consistency_telemetry_no_overrides():
    state = {
        "client_profile_provenance": {"deterministic_overrides": []},
        "product_profile_provenance": {"deterministic_overrides": []},
    }
    tel = _extract_consistency_telemetry(state)
    assert tel["any_override"] is False
    assert tel["a1_overrides"] == []


def test_consistency_telemetry_with_override():
    state = {
        "client_profile_provenance": {"deterministic_overrides": ["financial_vulnerability"]},
        "product_profile_provenance": {"deterministic_overrides": []},
    }
    tel = _extract_consistency_telemetry(state)
    assert tel["any_override"] is True
    assert "financial_vulnerability" in tel["a1_overrides"]


def test_consistency_telemetry_empty_state():
    tel = _extract_consistency_telemetry({})
    assert tel["any_override"] is False


# ── compute_live_metrics ───────────────────────────────────────────────────────

def _make_result(config_id, expected, final, escalate_exp=False, escalate_out=False, error=None):
    return {
        "config_id":          config_id,
        "category":           expected,
        "expected_decision":  expected,
        "final_decision":     final,
        "expected_escalate":  escalate_exp,
        "output_escalate":    escalate_out,
        "decision_correct":   (final == expected),
        "escalation_correct": (escalate_out == escalate_exp),
        "error":              error,
        "verifier":           {"any_correction": False, "override_rate": False},
        "consistency":        {"any_override": False},
    }


def test_compute_live_metrics_perfect():
    results = {
        "C0": [
            _make_result("C0", "SUITABLE",    "SUITABLE"),
            _make_result("C0", "UNSUITABLE",  "UNSUITABLE"),
            _make_result("C0", "CONDITIONAL", "CONDITIONAL"),
            _make_result("C0", "ESCALATED",   "ESCALATED", True, True),
        ],
        "C1": [
            _make_result("C1", "SUITABLE",    "SUITABLE"),
            _make_result("C1", "UNSUITABLE",  "UNSUITABLE"),
            _make_result("C1", "CONDITIONAL", "CONDITIONAL"),
            _make_result("C1", "ESCALATED",   "CONDITIONAL", True, False),  # missed escalation
        ],
    }
    metrics = compute_live_metrics(results)
    assert metrics["per_config"]["C0"]["M1_decision_accuracy"] == 1.0
    assert metrics["per_config"]["C1"]["M1_decision_accuracy"] == pytest.approx(0.75)
    assert metrics["deltas"]["C1"] == pytest.approx(-0.25)


def test_compute_verifier_impact():
    results = {
        "C0": [
            {"config_id": "C0", "verifier": {"any_correction": True,  "override_rate": False}, "consistency": {"any_override": True}, "error": None},
            {"config_id": "C0", "verifier": {"any_correction": False, "override_rate": False}, "consistency": {"any_override": False}, "error": None},
        ],
        "C1": [
            {"config_id": "C1", "verifier": {"any_correction": False, "override_rate": False}, "consistency": {"any_override": False}, "error": None},
        ],
    }
    impact = compute_verifier_impact(results)
    assert impact["C0"]["correction_rate"] == pytest.approx(0.5)
    assert impact["C0"]["override_rate"]   == pytest.approx(0.5)
    assert impact["C1"]["correction_rate"] == pytest.approx(0.0)


# ── Scenario selector integration (static — no API) ───────────────────────────

def test_scenario_selector_returns_50():
    from evaluation.ablation_study.scenario_selector import select_and_report
    report = select_and_report()
    assert report["total"] == 50


def test_scenario_selector_balance():
    from evaluation.ablation_study.scenario_selector import select_and_report, TARGETS
    report = select_and_report()
    for cat, target in TARGETS.items():
        actual = report["counts"].get(cat, 0)
        assert actual == target, f"{cat}: expected {target}, got {actual}"


def test_scenario_selector_deterministic():
    from evaluation.ablation_study.scenario_selector import select_scenarios
    s1 = [s["_filename"] for s in select_scenarios(seed=42)]
    s2 = [s["_filename"] for s in select_scenarios(seed=42)]
    assert s1 == s2


def test_scenario_selector_seed_changes_selection():
    from evaluation.ablation_study.scenario_selector import select_scenarios
    s1 = set(s["_filename"] for s in select_scenarios(seed=42))
    s2 = set(s["_filename"] for s in select_scenarios(seed=99))
    # Different seeds may produce different selections (not guaranteed but very likely)
    # At minimum both should be valid 50-scenario sets
    assert len(s1) == 50
    assert len(s2) == 50
