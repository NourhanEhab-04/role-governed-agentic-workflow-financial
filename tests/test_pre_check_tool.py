# tests/test_pre_check_tool.py
"""
Zero-LLM tests for the pre_check_tool wrapper.
Verifies it delegates correctly to the rule engine and raises on bad input.
"""
import pytest
from orchestrator.pre_check_tool import run_pre_check

GOOD_CLIENT = {
    "financial_knowledge": "basic",
    "risk_tolerance_score": 4,
    "investment_horizon": 3,
    "liquid_assets": 8000,
    "income": 42000,
    "investment_amount": 5000,
    "can_afford_total_loss": False,
    "financial_vulnerability": "LOW",
}

GOOD_PRODUCT = {
    "product_name": "Test Fund",
    "risk_class": 4,
    "complexity_tier": "NON-COMPLEX",
    "requires_knowledge_level": "basic",
    "minimum_horizon": 2,
    "potential_loss": "partial",
    "leverage": False,
}


def test_pre_check_returns_score():
    result = run_pre_check(GOOD_CLIENT, GOOD_PRODUCT)
    assert "score" in result
    assert isinstance(result["score"], int)


def test_pre_check_returns_decision():
    result = run_pre_check(GOOD_CLIENT, GOOD_PRODUCT)
    assert result["decision"] in {"SUITABLE", "CONDITIONAL", "UNSUITABLE"}


def test_pre_check_returns_seven_rules():
    result = run_pre_check(GOOD_CLIENT, GOOD_PRODUCT)
    assert len(result["rules"]) == 7


def test_pre_check_rule_ids():
    result = run_pre_check(GOOD_CLIENT, GOOD_PRODUCT)
    ids = {r["rule"] for r in result["rules"]}
    assert ids == {"R1", "R2", "R3", "R4", "R5", "R6", "R7"}


def test_pre_check_suitable_case():
    """Conservative client + low-risk product → SUITABLE."""
    client = {**GOOD_CLIENT, "financial_knowledge": "advanced",
              "risk_tolerance_score": 8, "can_afford_total_loss": True}
    product = {**GOOD_PRODUCT, "risk_class": 2, "requires_knowledge_level": "none"}
    result = run_pre_check(client, product)
    assert result["decision"] == "SUITABLE"


def test_pre_check_unsuitable_case():
    """Many rules fail → UNSUITABLE."""
    client = {**GOOD_CLIENT, "financial_knowledge": "none",
              "risk_tolerance_score": 2, "investment_horizon": 1,
              "can_afford_total_loss": False, "financial_vulnerability": "HIGH"}
    product = {**GOOD_PRODUCT, "risk_class": 7, "complexity_tier": "COMPLEX",
               "requires_knowledge_level": "advanced", "minimum_horizon": 10,
               "potential_loss": "total", "leverage": True}
    result = run_pre_check(client, product)
    assert result["decision"] == "UNSUITABLE"


def test_pre_check_missing_client_key_raises():
    bad_client = {k: v for k, v in GOOD_CLIENT.items() if k != "risk_tolerance_score"}
    with pytest.raises(ValueError, match="client dict is missing"):
        run_pre_check(bad_client, GOOD_PRODUCT)


def test_pre_check_missing_product_key_raises():
    bad_product = {k: v for k, v in GOOD_PRODUCT.items() if k != "risk_class"}
    with pytest.raises(ValueError, match="product dict is missing"):
        run_pre_check(GOOD_CLIENT, bad_product)