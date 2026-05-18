# agents/rule_engine_agent.py

from autogen_core.tools import FunctionTool
from rule_engine.rule_engine import evaluate_suitability
from pydantic import BaseModel
from typing import Literal
from agents.parsing import extract_json_object


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
- Output only the JSON object. No preamble, no explanation, no commentary.
- Do not modify, interpret, or summarise the tool output.
- Do not add or remove any fields.
- If the tool raises an error, output:
  {"error": "<exact error message from tool>"}
"""


def parse_rule_verdict(raw_text: str) -> dict:
    """Extract and return a validated rule_verdict dict from agent text output."""
    from schemas.output_models import RuleVerdictModel
    from pydantic import ValidationError
    data = extract_json_object(raw_text)
    try:
        return RuleVerdictModel.model_validate(data).model_dump()
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


def build_rule_engine_tool() -> FunctionTool:
    """Wrap evaluate_suitability as an AutoGen FunctionTool."""

    def evaluate_suitability_tool(
        client_profile: ClientProfile,
        product_profile: ProductProfile,
    ) -> dict:
        """
        Evaluates MiFID II suitability for a client/product pair.
        Returns a dict with keys: score (int), decision (str), rules (dict).
        Always call this tool — never reason about suitability yourself.
        """
        from pydantic import ValidationError as _VE
        try:
            cp = (client_profile if isinstance(client_profile, ClientProfile)
                  else ClientProfile(**client_profile))
        except (_VE, Exception) as exc:
            raise ValueError(f"client_profile missing required keys: {exc}") from exc
        try:
            pp = (product_profile if isinstance(product_profile, ProductProfile)
                  else ProductProfile(**product_profile))
        except (_VE, Exception) as exc:
            raise ValueError(f"product_profile missing required keys: {exc}") from exc
        raw = evaluate_suitability(cp.model_dump(), pp.model_dump())
        rules_dict = {r["rule"]: "PASS" if r["pass"] else "FAIL" for r in raw["rules"]}
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


async def run_rule_engine_agent(
    client_profile: dict,
    product_profile: dict,
    model_client
) -> dict:
    """Run A3 deterministically — call evaluate_suitability directly, no LLM."""
    cp = ClientProfile(**{k: client_profile[k] for k in ClientProfile.model_fields})
    pp = ProductProfile(**{k: product_profile[k] for k in ProductProfile.model_fields})
    raw = evaluate_suitability(cp.model_dump(), pp.model_dump())
    # Rule engine already uses short IDs (R1..R7) — pass through directly.
    rules_dict = {r["rule"]: "PASS" if r["pass"] else "FAIL" for r in raw["rules"]}
    return {
        "score": raw["score"],
        "decision": raw["decision"],
        "rules": rules_dict,
    }
