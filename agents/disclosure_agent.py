# agents/disclosure_agent.py

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

import json as _json
from agents.parsing import extract_json_object
DISCLOSURE_AGENT_SYSTEM_PROMPT = """
You are the Disclosure Agent in a MiFID II suitability assessment pipeline.
You are the final agent. Your output is the suitability report delivered to the client.

You receive a JSON object with five keys:
- "client_profile": structured client profile
- "product_profile": structured product profile
- "rule_verdict": output from the rule engine (score, decision, per-rule results)
- "conflict_report": output from the conflict detector (flags, escalate, summary)
- "rule_findings_skeleton": a list of 7 objects, each with "rule_id", "status", and "detail"
  already filled in. The "detail" field contains the exact threshold comparison computed
  by the rule engine — use it as your factual source of truth.

Your responsibilities:

1. Set "decision" based on the following priority order:
   - If conflict_report["escalate"] is true → decision must be "ESCALATED". No exceptions.
   - Otherwise mirror rule_verdict["decision"] exactly (SUITABLE / CONDITIONAL / UNSUITABLE).

2. Write "rule_findings": copy the 7 entries from "rule_findings_skeleton" exactly —
   do NOT change the "rule_id" or "status" values — and add an "explanation" field to each.
   Do NOT include the "detail" field in your output entries.

   CRITICAL: the "rule_id" values must be exactly R1, R2, R3, R4, R5, R6, R7 —
   never use R1_knowledge, R2_risk, or any other variant.

   How to write explanations:

   For FAIL entries — you MUST:
     (a) State what the rule requires in plain English.
     (b) Quote the exact client value AND the exact product requirement from "detail".
     (c) Explain precisely why they clash (e.g. which threshold was breached and by how much).
   Do NOT use vague language like "the product may be too risky." Name the numbers.
   Examples:
     R1 FAIL: "Knowledge check failed: the product requires 'advanced' knowledge (score 9)
       but your profile shows 'basic' knowledge (score 3) — a gap of 6 points."
     R2 FAIL: "Risk alignment failed: the product's risk class is 7, but your risk
       tolerance score is 4; MiFID II permits a maximum gap of 2 (4 + 2 = 6), so the
       product exceeds your risk appetite by 1 class."
     R3 FAIL: "Horizon check failed: the product requires a minimum investment horizon
       of 5 years but your stated horizon is 2 years — 3 years short of the requirement."
     R4 FAIL: "Affordability check failed: you indicated you cannot afford a total loss,
       but this product carries a 'total' potential loss rating."
     R5 FAIL: "Vulnerability check failed: your financial vulnerability is rated HIGH and
       the product's risk class is 6 (≥ 5), making it unsuitable for vulnerable clients."
     R6 FAIL: "Leverage check failed: this product uses leverage, which requires a
       risk tolerance score of at least 7; your score is 4."
     R7 FAIL: "Complexity check failed: this product is rated COMPLEX but your financial
       knowledge level is 'basic', below the required 'moderate' minimum."

   For PASS entries: one sentence confirming compliance, citing the actual values.
   Example: "Your investment horizon of 5 years meets the product's 3-year minimum."

3. Write "flags_addressed": one entry per flag in conflict_report["flags"]
   where triggered is true. Each entry must have:
   - "rule_id": the flag's rule_id (e.g. "BORDERLINE", "CONCENTRATION", "CONTRADICTION", "ESCALATION")
   - "explanation": one sentence explaining what was flagged and how it affects
     the recommendation.
   IMPORTANT: include ALL triggered flags, including non-rule flags like BORDERLINE.
   If no flags were triggered, use an empty list [].

4. Write "regulatory_basis": this field MUST begin with "MiFID II Article 25(2)" and
   then name the specific rules (R1–R7) that determined the outcome. Example:
   "MiFID II Article 25(2) — suitability assessed under R1 (knowledge), R4 (affordability)."
   Never leave this field empty.

5. Write "client_facing_summary": 2–3 sentences in plain English, no regulatory jargon.
   - For UNSUITABLE or CONDITIONAL: explicitly name each failed rule by its plain English
     label (not rule IDs), state the exact mismatch values, and explain what this means
     for the client. Example: "This product requires advanced financial knowledge, but your
     profile shows only basic knowledge. The product's risk class of 7 also exceeds your
     maximum tolerable risk of 6 (your score 4, plus the allowed margin of 2). For these
     reasons, this product does not match your investor profile."
   - For ESCALATED: explain that a human advisor will review the case, and briefly name
     the specific concern that triggered escalation (e.g. vulnerability level, contradiction).
   - For SUITABLE: 1-2 sentences confirming the product matches the client's profile,
     citing at least one key matching value.

6. Write "summary": one sentence for internal use, naming the decision and the primary
   rule(s) that drove it (e.g. "UNSUITABLE — R2 (risk class 7 vs tolerance 4) and R1 failed").

Hard rules:
- Never override the rule_verdict score or per-rule results.
- Never set decision to ESCALATED unless conflict_report["escalate"] is true.
- Never set decision to SUITABLE if conflict_report["escalate"] is true.
- Your entire response must be a single JSON object. No preamble. No markdown.
- "regulatory_basis" must never be empty.

Respond with exactly this structure:
{
  "decision": "<SUITABLE | CONDITIONAL | UNSUITABLE | ESCALATED>",
  "summary": "<one internal sentence naming decision and primary failing rules>",
  "rule_findings": [
    {"rule_id": "R1", "status": "<PASS|FAIL>", "explanation": "<specific values-based sentence>"},
    {"rule_id": "R2", "status": "<PASS|FAIL>", "explanation": "<specific values-based sentence>"},
    {"rule_id": "R3", "status": "<PASS|FAIL>", "explanation": "<specific values-based sentence>"},
    {"rule_id": "R4", "status": "<PASS|FAIL>", "explanation": "<specific values-based sentence>"},
    {"rule_id": "R5", "status": "<PASS|FAIL>", "explanation": "<specific values-based sentence>"},
    {"rule_id": "R6", "status": "<PASS|FAIL>", "explanation": "<specific values-based sentence>"},
    {"rule_id": "R7", "status": "<PASS|FAIL>", "explanation": "<specific values-based sentence>"}
  ],
  "flags_addressed": [{"rule_id": "<flag rule_id from conflict_report>", "explanation": "<one sentence>"}],
  "regulatory_basis": "MiFID II Article 25(2) — <rule references>",
  "client_facing_summary": "<2-3 plain English sentences with exact mismatch values>"
}
"""

_RULE_ORDER = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]


def parse_suitability_report(raw: str) -> dict:
    """Extract and return a validated suitability_report dict from agent text output."""
    from schemas.output_models import SuitabilityReportModel
    from pydantic import ValidationError
    data = extract_json_object(raw)
    try:
        return SuitabilityReportModel.model_validate(data).model_dump()
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
async def run_disclosure_agent(
    client_profile: dict,
    product_profile: dict,
    rule_verdict: dict,
    conflict_report: dict,
    model_client,
) -> dict:
    agent = AssistantAgent(
        name="disclosure_agent",
        model_client=model_client,
        system_message=DISCLOSURE_AGENT_SYSTEM_PROMPT,
    )

    # Build the skeleton so the LLM only writes explanations.
    # rule_verdict["rules"] now uses short IDs (R1..R7) consistently.
    # rule_verdict["rule_details"] carries the exact threshold strings from the
    # rule engine (e.g. "risk_class 7 > tolerance 4 + 2 = 6") so the disclosure
    # agent can cite precise values rather than reconstructing them from profiles.
    rule_details = rule_verdict.get("rule_details", {})
    rule_findings_skeleton = [
        {
            "rule_id": rule_id,
            "status": rule_verdict.get("rules", {}).get(rule_id, "UNKNOWN"),
            "detail": rule_details.get(rule_id, ""),
        }
        for rule_id in _RULE_ORDER
    ]

    payload = _json.dumps({
        "client_profile": client_profile,
        "product_profile": product_profile,
        "rule_verdict": rule_verdict,
        "conflict_report": conflict_report,
        "rule_findings_skeleton": rule_findings_skeleton,
    })

    response = await agent.on_messages(
        [TextMessage(content=payload, source="user")],
        cancellation_token=CancellationToken(),
    )

    raw = response.chat_message.content

    # Deterministic overrides — applied before Pydantic validation so the
    # downstream validator always sees a consistent, engine-authoritative report.
    data = extract_json_object(raw)
    if not isinstance(data, dict):
        raise ValueError("A5 output did not parse to a JSON object")

    # 1. Override rule_findings statuses from the rule engine.
    #    A5 writes explanations; PASS/FAIL verdicts are owned by the rule engine.
    _long_to_short = {
        "R1_knowledge": "R1", "R2_risk": "R2", "R3_horizon": "R3",
        "R4_afford": "R4", "R5_vuln": "R5", "R6_leverage": "R6", "R7_complexity": "R7",
    }
    engine_rules = rule_verdict.get("rules", {})
    for finding in data.get("rule_findings", []):
        rid = finding.get("rule_id", "")
        rid_norm = _long_to_short.get(rid, rid)
        if rid_norm in engine_rules:
            finding["status"] = engine_rules[rid_norm]

    # 2. Override decision — fully deterministic in both directions.
    #    A5 writes explanations only; the decision is owned by upstream deterministic outputs.
    #    When escalate=True  → decision must be ESCALATED (A4 authorised it).
    #    When escalate=False → decision must mirror rule_verdict exactly.
    #    This eliminates LLM temperature variance from the decision field.
    escalate = conflict_report.get("escalate", False)
    if escalate:
        data["decision"] = "ESCALATED"
    else:
        data["decision"] = rule_verdict.get("decision", "UNSUITABLE")

    # 3. Ensure every triggered conflict flag has a flags_addressed entry.
    #    The LLM frequently omits non-R* flags (e.g. BORDERLINE, CONCENTRATION)
    #    because the empty-list template hides the expected structure.
    #    Deterministically fill in any missing entries using the flag's own message.
    addressed_ids = {
        entry.get("rule_id")
        for entry in data.get("flags_addressed", [])
    }
    if not isinstance(data.get("flags_addressed"), list):
        data["flags_addressed"] = []
    for flag in conflict_report.get("flags", []):
        if flag.get("triggered") is True and flag.get("rule_id") not in addressed_ids:
            data["flags_addressed"].append({
                "rule_id":     flag["rule_id"],
                "explanation": flag.get("message", f"{flag['rule_id']} flag was triggered."),
            })

    from schemas.output_models import SuitabilityReportModel
    from pydantic import ValidationError
    try:
        return SuitabilityReportModel.model_validate(data).model_dump()
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc