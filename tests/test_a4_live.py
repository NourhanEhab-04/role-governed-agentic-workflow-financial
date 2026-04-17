# tests/test_a4_live.py
# Run with: pytest tests/test_a4_live.py -v
# Requires GEMINI_API_KEY in .env

import pytest
from config.llm_config import get_model_client 
from agents.conflict_detector import run_conflict_detector


@pytest.fixture(scope="module")
def model_client():
    return get_model_client()


def make_verdict(score, decision, failed=None):
    rules = {f"R{i}": "PASS" for i in range(1, 8)}
    for r in (failed or []):
        rules[r] = "FAIL"
    return {
        "rules": rules,
        "score": score,
        "decision": decision,
        "failed_rules": failed or [],
    }


# 1. Clean suitable — no flags
@pytest.mark.asyncio
async def test_live_clean_suitable(model_client):
    client = {
        "knowledge_level": "advanced", "risk_tolerance_score": 6,
        "investment_horizon_years": 7, "can_afford_total_loss": True,
        "vulnerability": "NONE", "portfolio_concentration_pct": 15,
    }
    product = {
        "risk_class": 4, "is_complex": False, "is_leveraged": False,
        "total_loss_potential": False, "minimum_horizon_years": 3,
        "required_knowledge_level": "basic",
    }
    result = await run_conflict_detector(client, product, make_verdict(100, "SUITABLE"), model_client)
    assert result["escalate"] is False


# 2. Borderline CONDITIONAL — borderline flag only
@pytest.mark.asyncio
async def test_live_borderline_conditional(model_client):
    client = {
        "knowledge_level": "basic", "risk_tolerance_score": 4,
        "investment_horizon_years": 3, "can_afford_total_loss": True,
        "vulnerability": "NONE", "portfolio_concentration_pct": 10,
    }
    product = {
        "risk_class": 4, "is_complex": False, "is_leveraged": False,
        "total_loss_potential": False, "minimum_horizon_years": 3,
        "required_knowledge_level": "basic",
    }
    result = await run_conflict_detector(client, product, make_verdict(45, "CONDITIONAL"), model_client)
    assert result["escalate"] is False
    borderline = next(f for f in result["flags"] if f["rule_id"] == "BORDERLINE")
    assert borderline["triggered"] is True


# 3. HIGH vulnerability + SUITABLE verdict — contradiction → escalation
@pytest.mark.asyncio
async def test_live_contradiction_escalates(model_client):
    client = {
        "knowledge_level": "advanced", "risk_tolerance_score": 7,
        "investment_horizon_years": 10, "can_afford_total_loss": True,
        "vulnerability": "HIGH", "portfolio_concentration_pct": 5,
    }
    product = {
        "risk_class": 3, "is_complex": False, "is_leveraged": False,
        "total_loss_potential": False, "minimum_horizon_years": 2,
        "required_knowledge_level": "basic",
    }
    result = await run_conflict_detector(client, product, make_verdict(100, "SUITABLE"), model_client)
    assert result["escalate"] is True
    contradiction = next(f for f in result["flags"] if f["rule_id"] == "CONTRADICTION")
    assert contradiction["triggered"] is True


# 4. Concentration > 40% — concentration flag
@pytest.mark.asyncio
async def test_live_concentration_flag(model_client):
    client = {
        "knowledge_level": "moderate", "risk_tolerance_score": 5,
        "investment_horizon_years": 5, "can_afford_total_loss": True,
        "vulnerability": "NONE", "portfolio_concentration_pct": 65,
    }
    product = {
        "risk_class": 4, "is_complex": False, "is_leveraged": False,
        "total_loss_potential": False, "minimum_horizon_years": 3,
        "required_knowledge_level": "basic",
    }
    result = await run_conflict_detector(client, product, make_verdict(100, "SUITABLE"), model_client)
    concentration = next(f for f in result["flags"] if f["rule_id"] == "CONCENTRATION")
    assert concentration["triggered"] is True


# 5. Leveraged product + low risk tolerance — should fail R6, test escalation from two HIGH flags
@pytest.mark.asyncio
async def test_live_leveraged_low_tolerance_escalation(model_client):
    client = {
        "knowledge_level": "moderate", "risk_tolerance_score": 4,
        "investment_horizon_years": 5, "can_afford_total_loss": False,
        "vulnerability": "HIGH", "portfolio_concentration_pct": 50,
    }
    product = {
        "risk_class": 7, "is_complex": True, "is_leveraged": True,
        "total_loss_potential": True, "minimum_horizon_years": 1,
        "required_knowledge_level": "moderate",
    }
    verdict = make_verdict(35, "UNSUITABLE", failed=["R2", "R4", "R6"])
    result = await run_conflict_detector(client, product, verdict, model_client)
    # concentration (>40%) + R6 mismatch context → two HIGH flags → escalation
    assert result["escalate"] is True


# 6. All clean, low score unsuitable — no flags, no escalation
@pytest.mark.asyncio
async def test_live_unsuitable_no_flags(model_client):
    client = {
        "knowledge_level": "none", "risk_tolerance_score": 2,
        "investment_horizon_years": 1, "can_afford_total_loss": False,
        "vulnerability": "NONE", "portfolio_concentration_pct": 5,
    }
    product = {
        "risk_class": 7, "is_complex": True, "is_leveraged": True,
        "total_loss_potential": True, "minimum_horizon_years": 5,
        "required_knowledge_level": "advanced",
    }
    verdict = make_verdict(0, "UNSUITABLE", failed=["R1", "R2", "R3", "R4", "R6", "R7"])
    result = await run_conflict_detector(client, product, verdict, model_client)
    assert result["escalate"] is False