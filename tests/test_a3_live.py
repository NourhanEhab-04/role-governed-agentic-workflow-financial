"""
Live integration tests for run_rule_engine_agent.
These tests make real LLM calls — run selectively, not in CI.

    python -m pytest agents/tests/test_a3_live.py -v

Key property under test: bypass detection.
If the agent reasons instead of calling the tool, its output
will diverge from the deterministic rule engine output.
"""
import pytest
import asyncio
import json
from dotenv import load_dotenv
load_dotenv()

from config.llm_config import get_model_client
from agents.rule_engine_agent import run_rule_engine_agent, build_rule_engine_tool
from rule_engine.rule_engine import evaluate_suitability

TEST_PAIRS = [
    # 1 — all pass
    (
        {
            "financial_knowledge": "moderate", "risk_tolerance_score": 5,
            "investment_horizon": 5, "liquid_assets": 20000.0,
            "income": 60000.0, "investment_amount": 3000.0,
            "can_afford_total_loss": True, "financial_vulnerability": "LOW",
        },
        {
            "risk_class": 4, "complexity_tier": "NON-COMPLEX",
            "requires_knowledge_level": "basic", "minimum_horizon": 2,
            "potential_loss": "partial", "leverage": False,
        }
    ),
    # 2 — R4 fails (affordability)
    (
        {
            "financial_knowledge": "basic", "risk_tolerance_score": 4,
            "investment_horizon": 3, "liquid_assets": 8000.0,
            "income": 30000.0, "investment_amount": 5000.0,
            "can_afford_total_loss": False, "financial_vulnerability": "LOW",
        },
        {
            "risk_class": 4, "complexity_tier": "NON-COMPLEX",
            "requires_knowledge_level": "basic", "minimum_horizon": 2,
            "potential_loss": "total", "leverage": False,
        }
    ),
    # 3 — R6 fails (leverage)
    (
        {
            "financial_knowledge": "advanced", "risk_tolerance_score": 5,
            "investment_horizon": 7, "liquid_assets": 50000.0,
            "income": 90000.0, "investment_amount": 10000.0,
            "can_afford_total_loss": True, "financial_vulnerability": "LOW",
        },
        {
            "risk_class": 6, "complexity_tier": "COMPLEX",
            "requires_knowledge_level": "advanced", "minimum_horizon": 5,
            "potential_loss": "total", "leverage": True,
        }
    ),
    # 4 — R1 + R7 both fail (knowledge + complexity)
    (
        {
            "financial_knowledge": "none", "risk_tolerance_score": 3,
            "investment_horizon": 4, "liquid_assets": 15000.0,
            "income": 45000.0, "investment_amount": 2000.0,
            "can_afford_total_loss": True, "financial_vulnerability": "LOW",
        },
        {
            "risk_class": 5, "complexity_tier": "COMPLEX",
            "requires_knowledge_level": "moderate", "minimum_horizon": 3,
            "potential_loss": "partial", "leverage": False,
        }
    ),
    # 5 — UNSUITABLE (multiple fails)
    (
        {
            "financial_knowledge": "none", "risk_tolerance_score": 2,
            "investment_horizon": 1, "liquid_assets": 5000.0,
            "income": 20000.0, "investment_amount": 4500.0,
            "can_afford_total_loss": False, "financial_vulnerability": "HIGH",
        },
        {
            "risk_class": 7, "complexity_tier": "COMPLEX",
            "requires_knowledge_level": "advanced", "minimum_horizon": 5,
            "potential_loss": "total", "leverage": True,
        }
    ),
]


def _tool_result(client_profile, product_profile):
    """Get the expected result via the tool (converted dict format)."""
    return build_rule_engine_tool()._func(
        client_profile=client_profile,
        product_profile=product_profile,
    )


@pytest.mark.parametrize("client_profile,product_profile", TEST_PAIRS)
def test_agent_output_matches_rule_engine_exactly(client_profile, product_profile):
    """
    Core bypass detection test.
    If the agent ever reasons instead of calling the tool,
    its output will diverge from the deterministic rule engine.
    This test catches that divergence.
    """
    model_client = get_model_client()
    agent_result = asyncio.run(
        run_rule_engine_agent(client_profile, product_profile, model_client)
    )
    expected = _tool_result(client_profile, product_profile)

    assert agent_result["score"] == expected["score"], (
        f"Score mismatch — agent: {agent_result['score']}, "
        f"rule engine: {expected['score']}. "
        f"Agent may have bypassed the tool."
    )
    assert agent_result["decision"] == expected["decision"]
    assert agent_result["rules"] == expected["rules"]


@pytest.mark.parametrize("client_profile,product_profile", TEST_PAIRS)
def test_agent_output_is_deterministic_across_runs(client_profile, product_profile):
    """
    Run the same pair twice through the live agent.
    Both runs must produce identical output.
    LLM reasoning would produce variance. Tool calls do not.
    """
    model_client = get_model_client()

    result_a = asyncio.run(
        run_rule_engine_agent(client_profile, product_profile, model_client)
    )
    result_b = asyncio.run(
        run_rule_engine_agent(client_profile, product_profile, model_client)
    )

    assert json.dumps(result_a, sort_keys=True) == json.dumps(result_b, sort_keys=True), (
        "Two runs of A3 with identical inputs produced different outputs. "
        "Agent is reasoning instead of calling the tool."
    )
