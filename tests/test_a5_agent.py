# tests/test_a5_agent.py

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agents.disclosure_agent import run_disclosure_agent


# --- shared fixtures ---

def make_findings(statuses=None):
    statuses = statuses or {r: "PASS" for r in ["R1","R2","R3","R4","R5","R6","R7"]}
    return [
        {"rule_id": r, "status": s, "explanation": f"{r} {'passed' if s=='PASS' else 'failed'}."}
        for r, s in statuses.items()
    ]

CLIENT = {
    "knowledge_level": "moderate", "risk_tolerance_score": 5,
    "investment_horizon_years": 5, "can_afford_total_loss": True,
    "vulnerability": "NONE", "portfolio_concentration_pct": 10,
}
PRODUCT = {
    "risk_class": 4, "is_complex": False, "is_leveraged": False,
    "total_loss_potential": False, "minimum_horizon_years": 3,
    "required_knowledge_level": "basic",
}
VERDICT_SUITABLE = {
    "rules": {r: "PASS" for r in ["R1","R2","R3","R4","R5","R6","R7"]},
    "score": 100, "decision": "SUITABLE", "failed_rules": [],
}
CONFLICT_CLEAN = {
    "flags": [
        {"rule_id": "BORDERLINE",    "triggered": False, "severity": "LOW",  "message": "ok"},
        {"rule_id": "CONCENTRATION", "triggered": False, "severity": "LOW",  "message": "ok"},
        {"rule_id": "CONTRADICTION", "triggered": False, "severity": "LOW",  "message": "ok"},
        {"rule_id": "ESCALATION",    "triggered": False, "severity": "LOW",  "message": "ok"},
    ],
    "escalate": False,
    "summary": "No issues.",
}
CONFLICT_ESCALATED = {
    **CONFLICT_CLEAN,
    "flags": [
        {"rule_id": "CONTRADICTION", "triggered": True, "severity": "HIGH", "message": "contradiction"},
        {"rule_id": "ESCALATION",    "triggered": True, "severity": "HIGH", "message": "escalate"},
    ],
    "escalate": True,
    "summary": "Escalation required.",
}


def make_mock_response(payload: dict):
    msg = MagicMock()
    msg.chat_message.content = json.dumps(payload)
    return msg

def patch_agent(payload: dict):
    return patch(
        "autogen_agentchat.agents.AssistantAgent.on_messages",
        new_callable=AsyncMock,
        return_value=make_mock_response(payload),
    )


# --- tests ---

@pytest.mark.asyncio
async def test_suitable_no_flags():
    mock_report = {
        "decision": "SUITABLE",
        "summary": "Client is suitable.",
        "rule_findings": make_findings(),
        "flags_addressed": [],
        "regulatory_basis": "MiFID II Article 25(2).",
        "client_facing_summary": "This product is suitable for you.",
    }
    with patch_agent(mock_report):
        result = await run_disclosure_agent(
            CLIENT, PRODUCT, VERDICT_SUITABLE, CONFLICT_CLEAN, MagicMock()
        )
    assert result["decision"] == "SUITABLE"
    assert len(result["rule_findings"]) == 7
    assert result["flags_addressed"] == []


@pytest.mark.asyncio
async def test_conditional_with_borderline_flag_addressed():
    verdict = {**VERDICT_SUITABLE, "score": 45, "decision": "CONDITIONAL"}
    conflict = {
        **CONFLICT_CLEAN,
        "flags": [
            {"rule_id": "BORDERLINE", "triggered": True, "severity": "LOW",
             "message": "Score is borderline."},
            {"rule_id": "CONCENTRATION", "triggered": False, "severity": "LOW", "message": "ok"},
            {"rule_id": "CONTRADICTION", "triggered": False, "severity": "LOW", "message": "ok"},
            {"rule_id": "ESCALATION",    "triggered": False, "severity": "LOW", "message": "ok"},
        ],
        "escalate": False,
        "summary": "Borderline score.",
    }
    mock_report = {
        "decision": "CONDITIONAL",
        "summary": "Conditional — borderline score.",
        "rule_findings": make_findings(),
        "flags_addressed": [
            {"rule_id": "BORDERLINE", "explanation": "Score 45 is close to the threshold."}
        ],
        "regulatory_basis": "MiFID II Article 25(2).",
        "client_facing_summary": "This product may be suitable under certain conditions.",
    }
    with patch_agent(mock_report):
        result = await run_disclosure_agent(
            CLIENT, PRODUCT, verdict, conflict, MagicMock()
        )
    assert result["decision"] == "CONDITIONAL"
    assert any(f["rule_id"] == "BORDERLINE" for f in result["flags_addressed"])


@pytest.mark.asyncio
async def test_escalate_true_sets_decision_escalated():
    mock_report = {
        "decision": "ESCALATED",
        "summary": "Escalated due to contradiction.",
        "rule_findings": make_findings(),
        "flags_addressed": [
            {"rule_id": "CONTRADICTION", "explanation": "Human review required."}
        ],
        "regulatory_basis": "MiFID II Article 25(2).",
        "client_facing_summary": "Your case has been referred to a human advisor.",
    }
    with patch_agent(mock_report):
        result = await run_disclosure_agent(
            CLIENT, PRODUCT, VERDICT_SUITABLE, CONFLICT_ESCALATED, MagicMock()
        )
    assert result["decision"] == "ESCALATED"


@pytest.mark.asyncio
async def test_malformed_json_raises_value_error():
    bad_msg = MagicMock()
    bad_msg.chat_message.content = "not json"
    with patch(
        "autogen_agentchat.agents.AssistantAgent.on_messages",
        new_callable=AsyncMock,
        return_value=bad_msg,
    ):
        with pytest.raises(ValueError, match="not valid JSON"):
            await run_disclosure_agent(
                CLIENT, PRODUCT, VERDICT_SUITABLE, CONFLICT_CLEAN, MagicMock()
            )