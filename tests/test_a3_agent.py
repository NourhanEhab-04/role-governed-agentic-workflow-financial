import pytest
import asyncio
import json
from unittest.mock import MagicMock, patch
import autogen_agentchat.agents as agents_module
from agents.rule_engine_agent import run_rule_engine_agent

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

EXPECTED_VERDICT = json.dumps({
    "score": 100,
    "decision": "SUITABLE",
    "rules": {
        "R1_knowledge": "PASS",
        "R2_risk": "PASS",
        "R3_horizon": "PASS",
        "R4_afford": "PASS",
        "R5_vuln": "PASS",
        "R6_leverage": "PASS",
        "R7_complexity": "PASS",
    }
})


def make_mock_on_messages(response_text: str):
    async def mock_on_messages(self, messages, cancellation_token):
        msg = MagicMock()
        msg.content = response_text
        resp = MagicMock()
        resp.chat_message = msg
        return resp
    return mock_on_messages


def test_valid_verdict_returned():
    with patch.object(agents_module.AssistantAgent, "on_messages",
                      new=make_mock_on_messages(EXPECTED_VERDICT)):
        result = asyncio.run(
            run_rule_engine_agent(BASE_CLIENT, BASE_PRODUCT, MagicMock())
        )
    assert result["score"] == 100
    assert result["decision"] == "SUITABLE"
    assert all(v == "PASS" for v in result["rules"].values())


def test_tool_output_matches_direct_rule_engine():
    """
    Most important test: agent output score/decision must match rule_engine.py
    directly. Rules are in converted dict format (rule_id -> PASS/FAIL).
    """
    from rule_engine.rule_engine import evaluate_suitability
    from agents.rule_engine_agent import build_rule_engine_tool

    direct_result = evaluate_suitability(BASE_CLIENT, BASE_PRODUCT)
    # Use the tool's converted output as the mock response
    tool_result = build_rule_engine_tool()._func(
        client_profile=BASE_CLIENT, product_profile=BASE_PRODUCT
    )
    tool_as_json = json.dumps(tool_result)

    with patch.object(agents_module.AssistantAgent, "on_messages",
                      new=make_mock_on_messages(tool_as_json)):
        agent_result = asyncio.run(
            run_rule_engine_agent(BASE_CLIENT, BASE_PRODUCT, MagicMock())
        )

    assert agent_result["score"] == direct_result["score"]
    assert agent_result["decision"] == direct_result["decision"]
    assert all(v in {"PASS", "FAIL"} for v in agent_result["rules"].values())


def test_tool_is_registered_on_agent():
    """
    Verify the tool is attached to the agent before any message is sent.
    Intercept agent instantiation and inspect the tools argument.
    """
    captured_tools = []
    original_init = agents_module.AssistantAgent.__init__

    def capturing_init(self, *args, **kwargs):
        if "tools" in kwargs:
            captured_tools.extend(kwargs["tools"])
        original_init(self, *args, **kwargs)

    async def noop_on_messages(self, messages, cancellation_token):
        msg = MagicMock()
        msg.content = EXPECTED_VERDICT
        resp = MagicMock()
        resp.chat_message = msg
        return resp

    with patch.object(agents_module.AssistantAgent, "__init__", capturing_init), \
         patch.object(agents_module.AssistantAgent, "on_messages", noop_on_messages):
        asyncio.run(
            run_rule_engine_agent(BASE_CLIENT, BASE_PRODUCT, MagicMock())
        )

    assert len(captured_tools) == 1, "Expected exactly one tool registered on A3"
    assert captured_tools[0].name == "evaluate_suitability_tool"


def test_unsuitable_verdict_parsed_correctly():
    unsuitable_verdict = json.dumps({
        "score": 35,
        "decision": "UNSUITABLE",
        "rules": {
            "R1_knowledge": "PASS",
            "R2_risk": "FAIL",
            "R3_horizon": "PASS",
            "R4_afford": "FAIL",
            "R5_vuln": "PASS",
            "R6_leverage": "PASS",
            "R7_complexity": "PASS",
        }
    })

    with patch.object(agents_module.AssistantAgent, "on_messages",
                      new=make_mock_on_messages(unsuitable_verdict)):
        result = asyncio.run(
            run_rule_engine_agent(BASE_CLIENT, BASE_PRODUCT, MagicMock())
        )

    assert result["decision"] == "UNSUITABLE"
    assert result["score"] == 35
    assert result["rules"]["R2_risk"] == "FAIL"
    assert result["rules"]["R4_afford"] == "FAIL"
