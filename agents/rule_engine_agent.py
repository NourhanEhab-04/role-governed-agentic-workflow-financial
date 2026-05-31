# agents/rule_engine_agent.py
"""
A3 — Rule Engine Agent

Design: the LLM agent receives client_profile + product_profile and MUST
call the evaluate_suitability_tool exactly once.  It is forbidden from
reasoning about suitability itself.

This preserves the three-point integrity architecture:
  pre_check  (deterministic, before A3) ──┐
  A3 LLM  (calls tool — should agree)  ──┤── all three must agree
  audit    (deterministic, before A4)  ──┘

If A3 produces a different result than pre_check, validate_after_a3 halts
the pipeline, indicating the LLM either bypassed the tool or mutated inputs.
"""

import json as _json

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core.tools import FunctionTool
from pydantic import BaseModel
from typing import Literal

from agents.parsing import extract_json_object
from rule_engine.rule_engine import evaluate_suitability


# ── Pydantic models used by the tool schema ────────────────────────────────────

class ClientProfile(BaseModel):
    age:                     int
    financial_knowledge:     Literal["none", "basic", "moderate", "advanced"]
    risk_tolerance_score:    int
    investment_horizon:      int
    liquid_assets:           float
    income:                  float
    investment_amount:       float
    can_afford_total_loss:   bool
    financial_vulnerability: Literal["LOW", "MEDIUM", "HIGH"]


class ProductProfile(BaseModel):
    risk_class:               int
    complexity_tier:          Literal["NON-COMPLEX", "COMPLEX"]
    requires_knowledge_level: Literal["none", "basic", "moderate", "advanced"]
    minimum_horizon:          int
    potential_loss:           Literal["none", "partial", "total"]
    leverage:                 bool


# ── System prompt ─────────────────────────────────────────────────────────────

RULE_ENGINE_AGENT_SYSTEM_PROMPT = """
You are A3, the Rule Engine Agent in a MiFID II suitability assessment pipeline.

YOUR ONLY JOB: Receive the client_profile and product_profile JSON objects,
call the evaluate_suitability_tool tool EXACTLY ONCE with those objects as
arguments, then return its JSON output verbatim.

CRITICAL PROHIBITION:
You are STRICTLY FORBIDDEN from reasoning about suitability yourself.
Do not analyse the client profile. Do not analyse the product profile.
Do not form any opinion about whether the product is suitable.
Do not produce a suitability score or decision from your own reasoning.
If you find yourself thinking about whether something is suitable — STOP.
Call the tool instead.

REQUIRED PROCESS:
1. Receive the client_profile dict and product_profile dict from the message.
2. Call evaluate_suitability_tool with client_profile and product_profile.
3. Copy the exact JSON the tool returns — do not modify it.
4. Output that JSON and nothing else.

REQUIRED OUTPUT FORMAT — copy the tool output verbatim:
{
    "score": <integer>,
    "decision": "<SUITABLE | CONDITIONAL | UNSUITABLE>",
    "hard_failed_rules": [...],
    "rules": {
        "R1": "<PASS | FAIL>",
        "R2": "<PASS | FAIL>",
        "R3": "<PASS | FAIL>",
        "R4": "<PASS | FAIL>",
        "R5": "<PASS | FAIL>",
        "R6": "<PASS | FAIL>",
        "R7": "<PASS | FAIL>"
    }
}

OUTPUT RULES:
- Output ONLY the JSON object. No preamble, no explanation, no markdown.
- Do not modify, interpret, or summarise the tool output.
- Do not add or remove any fields.
- If the tool raises an error, output: {"error": "<exact error message from tool>"}
"""


# ── Tool builder ───────────────────────────────────────────────────────────────

def build_rule_engine_tool() -> FunctionTool:
    """Wrap evaluate_suitability as an AutoGen FunctionTool for A3."""

    def evaluate_suitability_tool(
        client_profile: ClientProfile,
        product_profile: ProductProfile,
    ) -> dict:
        """
        MiFID II suitability rule engine (ESMA35-43-3172).
        Takes client_profile dict and product_profile dict.
        Returns score, decision, hard_failed_rules, and per-rule PASS/FAIL.
        This is the ONLY permitted way to determine suitability — never reason yourself.
        """
        from pydantic import ValidationError as _VE
        try:
            cp = (client_profile if isinstance(client_profile, ClientProfile)
                  else ClientProfile(**client_profile))
        except (_VE, Exception) as exc:
            raise ValueError(f"client_profile invalid: {exc}") from exc
        try:
            pp = (product_profile if isinstance(product_profile, ProductProfile)
                  else ProductProfile(**product_profile))
        except (_VE, Exception) as exc:
            raise ValueError(f"product_profile invalid: {exc}") from exc

        raw = evaluate_suitability(cp.model_dump(), pp.model_dump())
        rules_dict = {r["rule"]: "PASS" if r["pass"] else "FAIL" for r in raw["rules"]}
        return {
            "score":             raw["score"],
            "decision":          raw["decision"],
            "hard_failed_rules": raw["hard_failed_rules"],
            "rules":             rules_dict,
        }

    return FunctionTool(
        evaluate_suitability_tool,
        description=(
            "MiFID II suitability rule engine per ESMA35-43-3172. "
            "Takes client_profile dict and product_profile dict. "
            "Returns score, decision, hard_failed_rules, and per-rule PASS/FAIL. "
            "This is the ONLY permitted way to determine suitability."
        ),
    )


# ── Parser ────────────────────────────────────────────────────────────────────

def parse_rule_verdict(raw_text: str) -> dict:
    """Extract and return a validated rule_verdict dict from agent text output."""
    from schemas.output_models import RuleVerdictModel
    from pydantic import ValidationError
    data = extract_json_object(raw_text)
    try:
        return RuleVerdictModel.model_validate(data).model_dump()
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


# ── Agent runner ──────────────────────────────────────────────────────────────

async def run_rule_engine_agent(
    client_profile: dict,
    product_profile: dict,
    model_client,
) -> dict:
    """
    Run A3: the LLM agent receives the profiles and MUST call the
    evaluate_suitability_tool.  The result is validated by parse_rule_verdict.

    The three-point integrity check (pre_check == A3 == audit) is enforced
    by validate_after_a3 in orchestrator/validators.py.  Any disagreement
    between A3 and pre_check indicates the LLM bypassed or corrupted the
    tool call — this halts the pipeline.
    """
    tool = build_rule_engine_tool()

    agent = AssistantAgent(
        name="RuleEngineAgent",
        model_client=model_client,
        system_message=RULE_ENGINE_AGENT_SYSTEM_PROMPT,
        tools=[tool],
    )

    payload = _json.dumps({
        "client_profile":  client_profile,
        "product_profile": product_profile,
    })

    response = await agent.run(
        task=payload,
    )

    # Extract the last text message from the agent's response
    raw_text = ""
    for msg in reversed(response.messages):
        if hasattr(msg, "content") and isinstance(msg.content, str):
            raw_text = msg.content
            break

    try:
        verdict = parse_rule_verdict(raw_text)
    except ValueError:
        # LLM output was unparseable (e.g., 'N/A' rule values, missing fields).
        # Fall back to direct engine call — the rule engine result is authoritative.
        raw_fb = evaluate_suitability(client_profile, product_profile)
        rules_fb = {r["rule"]: "PASS" if r["pass"] else "FAIL" for r in raw_fb["rules"]}
        verdict = {
            "score":             raw_fb["score"],
            "decision":          raw_fb["decision"],
            "hard_failed_rules": raw_fb["hard_failed_rules"],
            "rules":             rules_fb,
        }

    # Always attach rule_details and override hard_failed_rules from the
    # deterministic engine — these are never trusted from LLM text output.
    raw = evaluate_suitability(client_profile, product_profile)
    rule_details = {r["rule"]: r["detail"] for r in raw["rules"]}
    verdict["rule_details"]      = rule_details
    verdict["hard_failed_rules"] = raw["hard_failed_rules"]
    return verdict
