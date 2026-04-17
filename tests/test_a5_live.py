# tests/test_a5_live.py

import pytest
from config.llm_config import get_model_client
from agents.disclosure_agent import run_disclosure_agent


@pytest.fixture(scope="module")
def model_client():
    return get_model_client()


def all_pass_verdict(score=100, decision="SUITABLE"):
    return {
        "rules": {r: "PASS" for r in ["R1","R2","R3","R4","R5","R6","R7"]},
        "score": score, "decision": decision, "failed_rules": [],
    }

def failed_verdict(failed_rules, score, decision):
    rules = {r: "PASS" for r in ["R1","R2","R3","R4","R5","R6","R7"]}
    for r in failed_rules:
        rules[r] = "FAIL"
    return {"rules": rules, "score": score, "decision": decision, "failed_rules": failed_rules}

def clean_conflict(escalate=False, triggered_flags=None):
    base_flags = [
        {"rule_id": "BORDERLINE",    "triggered": False, "severity": "LOW",  "message": "ok"},
        {"rule_id": "CONCENTRATION", "triggered": False, "severity": "LOW",  "message": "ok"},
        {"rule_id": "CONTRADICTION", "triggered": False, "severity": "LOW",  "message": "ok"},
        {"rule_id": "ESCALATION",    "triggered": False, "severity": "LOW",  "message": "ok"},
    ]
    flags = triggered_flags if triggered_flags is not None else base_flags
    return {"flags": flags, "escalate": escalate, "summary": "audit done"}

CLIENT_STANDARD = {
    "knowledge_level": "moderate", "risk_tolerance_score": 5,
    "investment_horizon_years": 5, "can_afford_total_loss": True,
    "vulnerability": "NONE", "portfolio_concentration_pct": 10,
}
PRODUCT_STANDARD = {
    "name": "Global ETF", "risk_class": 4, "is_complex": False,
    "is_leveraged": False, "total_loss_potential": False,
    "minimum_horizon_years": 3, "required_knowledge_level": "basic",
}


# 1. SUITABLE clean
@pytest.mark.asyncio
async def test_live_suitable_clean(model_client):
    result = await run_disclosure_agent(
        CLIENT_STANDARD, PRODUCT_STANDARD,
        all_pass_verdict(), clean_conflict(), model_client
    )
    assert result["decision"] == "SUITABLE"
    assert len(result["rule_findings"]) == 7
    assert "Article 25" in result["regulatory_basis"]
    assert result["flags_addressed"] == []


# 2. CONDITIONAL borderline — flags_addressed must mention BORDERLINE
@pytest.mark.asyncio
async def test_live_conditional_borderline(model_client):
    conflict = clean_conflict(triggered_flags=[
        {"rule_id": "BORDERLINE",    "triggered": True,  "severity": "LOW",  "message": "borderline"},
        {"rule_id": "CONCENTRATION", "triggered": False, "severity": "LOW",  "message": "ok"},
        {"rule_id": "CONTRADICTION", "triggered": False, "severity": "LOW",  "message": "ok"},
        {"rule_id": "ESCALATION",    "triggered": False, "severity": "LOW",  "message": "ok"},
    ])
    result = await run_disclosure_agent(
        CLIENT_STANDARD, PRODUCT_STANDARD,
        all_pass_verdict(score=45, decision="CONDITIONAL"),
        conflict, model_client
    )
    assert result["decision"] == "CONDITIONAL"
    assert len(result["rule_findings"]) == 7
    flag_ids = [f["rule_id"] for f in result["flags_addressed"]]
    assert "BORDERLINE" in flag_ids


# 3. UNSUITABLE multi-rule failure
@pytest.mark.asyncio
async def test_live_unsuitable_multi_rule(model_client):
    client = {
        **CLIENT_STANDARD,
        "knowledge_level": "none", "risk_tolerance_score": 2,
        "can_afford_total_loss": False,
    }
    product = {
        **PRODUCT_STANDARD,
        "risk_class": 7, "is_complex": True, "is_leveraged": True,
        "total_loss_potential": True, "required_knowledge_level": "advanced",
    }
    result = await run_disclosure_agent(
        client, product,
        failed_verdict(["R1","R2","R4","R6","R7"], 0, "UNSUITABLE"),
        clean_conflict(), model_client
    )
    assert result["decision"] == "UNSUITABLE"
    assert len(result["rule_findings"]) == 7
    failed = [f for f in result["rule_findings"] if f["status"] == "FAIL"]
    assert len(failed) >= 3


# 4. ESCALATED — contradiction
@pytest.mark.asyncio
async def test_live_escalated_contradiction(model_client):
    client = {**CLIENT_STANDARD, "vulnerability": "HIGH"}
    conflict = {
        "flags": [
            {"rule_id": "CONTRADICTION", "triggered": True,  "severity": "HIGH", "message": "contradiction"},
            {"rule_id": "ESCALATION",    "triggered": True,  "severity": "HIGH", "message": "escalate"},
            {"rule_id": "BORDERLINE",    "triggered": False, "severity": "LOW",  "message": "ok"},
            {"rule_id": "CONCENTRATION", "triggered": False, "severity": "LOW",  "message": "ok"},
        ],
        "escalate": True,
        "summary": "Escalation required due to contradiction.",
    }
    result = await run_disclosure_agent(
        client, PRODUCT_STANDARD,
        all_pass_verdict(), conflict, model_client
    )
    assert result["decision"] == "ESCALATED"
    assert len(result["rule_findings"]) == 7
    flag_ids = [f["rule_id"] for f in result["flags_addressed"]]
    assert "CONTRADICTION" in flag_ids


# 5. ESCALATED — leveraged high-risk
@pytest.mark.asyncio
async def test_live_escalated_leveraged(model_client):
    client = {
        **CLIENT_STANDARD,
        "risk_tolerance_score": 4,
        "can_afford_total_loss": False,
        "vulnerability": "HIGH",
        "portfolio_concentration_pct": 55,
    }
    product = {
        **PRODUCT_STANDARD,
        "risk_class": 7, "is_leveraged": True,
        "total_loss_potential": True, "is_complex": True,
        "required_knowledge_level": "advanced",
    }
    conflict = {
        "flags": [
            {"rule_id": "CONCENTRATION", "triggered": True,  "severity": "HIGH", "message": "55%"},
            {"rule_id": "CONTRADICTION", "triggered": True,  "severity": "HIGH", "message": "contradiction"},
            {"rule_id": "ESCALATION",    "triggered": True,  "severity": "HIGH", "message": "escalate"},
            {"rule_id": "BORDERLINE",    "triggered": False, "severity": "LOW",  "message": "ok"},
        ],
        "escalate": True,
        "summary": "Two HIGH flags. Escalation required.",
    }
    result = await run_disclosure_agent(
        client, product,
        failed_verdict(["R2","R4","R5","R6"], 10, "UNSUITABLE"),
        conflict, model_client
    )
    assert result["decision"] == "ESCALATED"
    assert len(result["flags_addressed"]) >= 2