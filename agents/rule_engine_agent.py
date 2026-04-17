# agents/rule_engine_agent.py

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core.tools import FunctionTool
from schemas.rule_verdict import REQUIRED_VERDICT_KEYS
from rule_engine.rule_engine import evaluate_suitability
from pydantic import BaseModel
from typing import Literal
from agents.parsing import extract_json_object
import json


class ClientProfile(BaseModel):
    financial_knowledge: Literal["none", "basic", "moderate", "advanced"]
    risk_tolerance_score: int
    investment_horizon: int
    liquid_assets: float
    income: float
    investment_amount: float
    can_afford_total_loss: bool
    financial_vulnerability: Literal["LOW", "MEDIUM", "HIGH"]


class ProductProfile(BaseModel):
    risk_class: int
    complexity_tier: Literal["NON-COMPLEX", "COMPLEX"]
    requires_knowledge_level: Literal["none", "basic", "moderate", "advanced"]
    minimum_horizon: int
    potential_loss: Literal["none", "partial", "total"]
    leverage: bool


RULE_ENGINE_AGENT_SYSTEM_PROMPT = """
You are A3, the Rule Engine Agent in a MiFID II suitability assessment pipeline.

YOUR ONLY JOB: Format the client_profile and product_profile you receive into
the correct input structure, call the evaluate_suitability_tool tool exactly once,
and return its JSON output verbatim.

CRITICAL PROHIBITION:
You are STRICTLY FORBIDDEN from reasoning about suitability yourself.
Do not analyze the client profile. Do not analyze the product profile.
Do not form any opinion about whether the product is suitable.
Do not produce a suitability score or decision from your own reasoning.
If you find yourself thinking about whether something is suitable,
STOP — call the tool instead.

REQUIRED PROCESS — follow this exactly:
1. Receive the client_profile dict and product_profile dict
2. Call the evaluate_suitability_tool with these two dicts as arguments
3. Take the exact JSON the tool returns
4. Output that JSON and nothing else

REQUIRED OUTPUT FORMAT — copy the tool output verbatim:
{
    "score": <integer>,
    "decision": "<SUITABLE | CONDITIONAL | UNSUITABLE>",
    "rules": {
        "R1_knowledge": "<PASS | FAIL>",
        "R2_risk": "<PASS | FAIL>",
        "R3_horizon": "<PASS | FAIL>",
        "R4_afford": "<PASS | FAIL>",
        "R5_vuln": "<PASS | FAIL>",
        "R6_leverage": "<PASS | FAIL>",
        "R7_complexity": "<PASS | FAIL>"
    }
}

OUTPUT RULES:
- Output only the JSON object. No preamble, no explanation, no commentary.
- Do not modify, interpret, or summarise the tool output.
- Do not add or remove any fields.
- If the tool raises an error, output:
  {"error": "<exact error message from tool>"}
"""


def parse_rule_verdict(raw_text: str) -> dict:
    """Extract and return a rule_verdict dict from agent text output.
    Raises ValueError if JSON is missing or schema is incomplete."""
    from schemas.rule_verdict import VALID_DECISIONS, VALID_RULE_IDS

    verdict = extract_json_object(raw_text)

    # Step 3: required top-level keys
    missing = REQUIRED_VERDICT_KEYS - verdict.keys()
    if missing:
        raise ValueError(f"rule_verdict missing required keys: {missing}")

    # Step 4: score is int
    if not isinstance(verdict["score"], int):
        raise ValueError(
            f"score must be an integer, got: {type(verdict['score'])}"
        )

    # Step 5: decision is valid enum
    if verdict["decision"] not in VALID_DECISIONS:
        raise ValueError(
            f"Invalid decision: '{verdict['decision']}'. "
            f"Must be one of {VALID_DECISIONS}"
        )

    # Step 6: rules is a dict with all 7 rule IDs
    if not isinstance(verdict["rules"], dict):
        raise ValueError("rules must be a dict")

    missing_rules = VALID_RULE_IDS - verdict["rules"].keys()
    if missing_rules:
        raise ValueError(f"rules dict missing rule IDs: {missing_rules}")

    # Step 7: each rule result is PASS or FAIL
    for rule_id, result in verdict["rules"].items():
        if result not in {"PASS", "FAIL"}:
            raise ValueError(
                f"Rule {rule_id} has invalid result '{result}'. "
                f"Must be PASS or FAIL"
            )

    return verdict


def build_rule_engine_tool() -> FunctionTool:
    """Wrap evaluate_suitability as an AutoGen FunctionTool."""
    # Rule ID mapping: rule engine uses "R1"…"R7", A3 schema uses descriptive IDs
    RULE_ID_MAP = {
        "R1": "R1_knowledge",
        "R2": "R2_risk",
        "R3": "R3_horizon",
        "R4": "R4_afford",
        "R5": "R5_vuln",
        "R6": "R6_leverage",
        "R7": "R7_complexity",
    }

    def evaluate_suitability_tool(
        client_profile: ClientProfile,
        product_profile: ProductProfile,
    ) -> dict:
        """
        Evaluates MiFID II suitability for a client/product pair.
        Returns a dict with keys: score (int), decision (str), rules (dict).
        Always call this tool — never reason about suitability yourself.
        """
        cp = client_profile if isinstance(client_profile, ClientProfile) else ClientProfile(**client_profile)
        pp = product_profile if isinstance(product_profile, ProductProfile) else ProductProfile(**product_profile)
        raw = evaluate_suitability(cp.model_dump(), pp.model_dump())
        # Convert rules list → {rule_id: "PASS"/"FAIL"} dict for A3 schema
        rules_dict = {
            RULE_ID_MAP[r["rule"]]: "PASS" if r["pass"] else "FAIL"
            for r in raw["rules"]
        }
        return {
            "score": raw["score"],
            "decision": raw["decision"],
            "rules": rules_dict,
        }

    return FunctionTool(
        evaluate_suitability_tool,
        description=(
            "MiFID II suitability rule engine. "
            "Takes client_profile dict and product_profile dict. "
            "Returns score, decision, and per-rule PASS/FAIL results. "
            "This is the ONLY permitted way to determine suitability."
        )
    )


RULE_ID_MAP = {
    "R1": "R1_knowledge",
    "R2": "R2_risk",
    "R3": "R3_horizon",
    "R4": "R4_afford",
    "R5": "R5_vuln",
    "R6": "R6_leverage",
    "R7": "R7_complexity",
}


async def run_rule_engine_agent(
    client_profile: dict,
    product_profile: dict,
    model_client
) -> dict:
    """Run A3 deterministically — call evaluate_suitability directly, no LLM."""
    cp = ClientProfile(**{k: client_profile[k] for k in ClientProfile.model_fields})
    pp = ProductProfile(**{k: product_profile[k] for k in ProductProfile.model_fields})
    raw = evaluate_suitability(cp.model_dump(), pp.model_dump())
    rules_dict = {
        RULE_ID_MAP[r["rule"]]: "PASS" if r["pass"] else "FAIL"
        for r in raw["rules"]
    }
    return {
        "score": raw["score"],
        "decision": raw["decision"],
        "rules": rules_dict,
    }
