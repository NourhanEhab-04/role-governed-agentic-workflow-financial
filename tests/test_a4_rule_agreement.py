# tests/test_a4_rule_agreement.py
"""
Zero-LLM tests for A4's independent rule engine cross-check.
"""
import pytest
from agents.conflict_detector import check_rule_engine_agreement

CLIENT = {
    "financial_knowledge": "basic",
    "risk_tolerance_score": 4,
    "investment_horizon": 3,
    "liquid_assets": 8000,
    "income": 42000,
    "investment_amount": 5000,
    "can_afford_total_loss": False,
    "financial_vulnerability": "LOW",
}

PRODUCT = {
    "product_name": "Test Fund",
    "risk_class": 4,
    "complexity_tier": "NON-COMPLEX",
    "requires_knowledge_level": "basic",
    "minimum_horizon": 2,
    "potential_loss": "partial",
    "leverage": False,
}


def _make_a3_verdict(decision: str, failed_rules: list) -> dict:
    return {"decision": decision, "failed_rules": failed_rules}


def test_agreement_when_a3_correct():
    """A3 reported the same decision as what A4's independent call produces."""
    from rule_engine.rule_engine import evaluate_suitability
    real = evaluate_suitability(CLIENT, PRODUCT)
    failed = sorted(r["rule"] for r in real["rules"] if not r["pass"])
    a3_verdict = _make_a3_verdict(real["decision"], failed)
    result = check_rule_engine_agreement(CLIENT, PRODUCT, a3_verdict)
    assert result["agreed"] is True


def test_disagreement_when_a3_wrong_decision():
    """A3 reported SUITABLE but A4's re-run returns UNSUITABLE."""
    bad_a3 = _make_a3_verdict("SUITABLE", [])  # lying verdict
    risky_client = {**CLIENT, "financial_knowledge": "none",
                    "risk_tolerance_score": 1, "investment_horizon": 1,
                    "can_afford_total_loss": False, "financial_vulnerability": "HIGH"}
    risky_product = {**PRODUCT, "risk_class": 7, "complexity_tier": "COMPLEX",
                     "requires_knowledge_level": "advanced", "minimum_horizon": 10,
                     "potential_loss": "total", "leverage": True}
    result = check_rule_engine_agreement(risky_client, risky_product, bad_a3)
    assert result["agreed"] is False
    assert result["a4_decision"] == "UNSUITABLE"
    assert result["a3_decision"] == "SUITABLE"


def test_disagreement_when_failed_rules_differ():
    """Same decision string but different failed rules → still disagreement."""
    from rule_engine.rule_engine import evaluate_suitability
    real = evaluate_suitability(CLIENT, PRODUCT)
    # Correct decision, wrong failed rules list
    a3_verdict = _make_a3_verdict(real["decision"], ["R1", "R2", "R99"])
    result = check_rule_engine_agreement(CLIENT, PRODUCT, a3_verdict)
    assert result["agreed"] is False


def test_result_contains_all_keys():
    a3 = _make_a3_verdict("CONDITIONAL", ["R1"])
    result = check_rule_engine_agreement(CLIENT, PRODUCT, a3)
    for key in ("agreed", "a4_decision", "a3_decision",
                "a4_failed_rules", "a3_failed_rules", "detail"):
        assert key in result


def test_a4_failed_rules_sorted():
    """Failed rules list is always sorted for deterministic comparison."""
    a3 = _make_a3_verdict("CONDITIONAL", [])
    result = check_rule_engine_agreement(CLIENT, PRODUCT, a3)
    assert result["a4_failed_rules"] == sorted(result["a4_failed_rules"])


def test_detail_contains_disagreement_word_when_disagree():
    a3 = _make_a3_verdict("SUITABLE", [])
    risky_client = {**CLIENT, "financial_knowledge": "none",
                    "risk_tolerance_score": 1, "investment_horizon": 1,
                    "can_afford_total_loss": False, "financial_vulnerability": "HIGH"}
    risky_product = {**PRODUCT, "risk_class": 7, "complexity_tier": "COMPLEX",
                     "requires_knowledge_level": "advanced", "minimum_horizon": 10,
                     "potential_loss": "total", "leverage": True}
    result = check_rule_engine_agreement(risky_client, risky_product, a3)
    assert "DISAGREEMENT" in result["detail"]








