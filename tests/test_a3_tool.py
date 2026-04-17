import json
import pytest
from agents.rule_engine_agent import build_rule_engine_tool
from rule_engine.rule_engine import evaluate_suitability

BASE_CLIENT = {
    "financial_knowledge": "moderate",
    "risk_tolerance_score": 5,
    "investment_horizon": 5,
    "liquid_assets": 20000.0,
    "income": 60000.0,
    "investment_amount": 3000.0,
    "can_afford_total_loss": True,
    "financial_vulnerability": "LOW",
}

BASE_PRODUCT = {
    "risk_class": 4,
    "complexity_tier": "NON-COMPLEX",
    "requires_knowledge_level": "basic",
    "minimum_horizon": 2,
    "potential_loss": "partial",
    "leverage": False,
}


def test_tool_builds_without_error():
    tool = build_rule_engine_tool()
    assert tool is not None


def test_tool_has_correct_name():
    tool = build_rule_engine_tool()
    assert tool.name == "evaluate_suitability_tool"


def test_tool_output_matches_direct_rule_engine_call():
    """Core test: tool output must have same score/decision as direct call,
    with rules converted to the dict format (rule_id -> PASS/FAIL)."""
    tool = build_rule_engine_tool()
    tool_result = tool._func(client_profile=BASE_CLIENT, product_profile=BASE_PRODUCT)
    direct_result = evaluate_suitability(BASE_CLIENT, BASE_PRODUCT)

    assert tool_result["score"] == direct_result["score"]
    assert tool_result["decision"] == direct_result["decision"]
    # Verify each rule maps correctly
    for rule in direct_result["rules"]:
        rule_id = {
            "R1": "R1_knowledge", "R2": "R2_risk", "R3": "R3_horizon",
            "R4": "R4_afford", "R5": "R5_vuln", "R6": "R6_leverage", "R7": "R7_complexity",
        }[rule["rule"]]
        expected = "PASS" if rule["pass"] else "FAIL"
        assert tool_result["rules"][rule_id] == expected


def test_tool_output_is_deterministic():
    """Same input through tool must produce identical output every time."""
    tool = build_rule_engine_tool()
    results = [
        json.dumps(
            tool._func(client_profile=BASE_CLIENT, product_profile=BASE_PRODUCT),
            sort_keys=True,
        )
        for _ in range(10)
    ]
    assert len(set(results)) == 1  # all 10 identical


def test_tool_raises_on_missing_client_keys():
    tool = build_rule_engine_tool()
    with pytest.raises(ValueError, match="missing required keys"):
        tool._func(client_profile={}, product_profile=BASE_PRODUCT)


def test_tool_raises_on_missing_product_keys():
    tool = build_rule_engine_tool()
    with pytest.raises(ValueError, match="missing required keys"):
        tool._func(client_profile=BASE_CLIENT, product_profile={})


def test_tool_unsuitable_case():
    """Verify tool correctly surfaces UNSUITABLE for a clearly failing profile."""
    bad_client = {**BASE_CLIENT, "can_afford_total_loss": False, "risk_tolerance_score": 2}
    bad_product = {**BASE_PRODUCT, "potential_loss": "total", "risk_class": 6, "leverage": True}

    tool = build_rule_engine_tool()
    result = tool._func(client_profile=bad_client, product_profile=bad_product)

    assert result["decision"] == "UNSUITABLE"
    assert result["score"] < 40
