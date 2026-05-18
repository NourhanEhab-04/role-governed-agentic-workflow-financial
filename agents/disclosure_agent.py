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
- "rule_findings_skeleton": a list of 7 objects, each with "rule_id" and "status" already filled in

Your responsibilities:

1. Set "decision" based on the following priority order:
   - If conflict_report["escalate"] is true → decision must be "ESCALATED". No exceptions.
   - Otherwise mirror rule_verdict["decision"] exactly (SUITABLE / CONDITIONAL / UNSUITABLE).

2. Write "rule_findings": copy the 7 entries from "rule_findings_skeleton" exactly —
   do NOT change the "rule_id" or "status" values — and add an "explanation" field to
   each: one plain English sentence explaining what the rule checks and why it
   passed or failed for this specific client and product.
   CRITICAL: the "rule_id" values must be exactly R1, R2, R3, R4, R5, R6, R7 —
   never use R1_knowledge, R2_risk, or any other variant.

3. Write "flags_addressed": one entry per flag in conflict_report["flags"]
   where triggered is true. Each entry must have:
   - "rule_id": the flag's rule_id
   - "explanation": one sentence explaining what was flagged and how it affects
     the recommendation.
   If no flags were triggered, use an empty list [].

4. Write "regulatory_basis": this field MUST begin with "MiFID II Article 25(2)" and
   then name the specific rules (R1–R7) that determined the outcome. Example:
   "MiFID II Article 25(2) — suitability assessed under R1 (knowledge), R4 (affordability)."
   Never leave this field empty.

5. Write "client_facing_summary": 2–3 sentences in plain English, no regulatory
   jargon. Explain the outcome and what it means for the client. If ESCALATED,
   explain that a human advisor will review the case.

6. Write "summary": one sentence for internal use, summarising the decision
   and the key reason.

Hard rules:
- Never override the rule_verdict score or per-rule results.
- Never set decision to ESCALATED unless conflict_report["escalate"] is true.
- Never set decision to SUITABLE if conflict_report["escalate"] is true.
- Your entire response must be a single JSON object. No preamble. No markdown.
- "regulatory_basis" must never be empty.

Respond with exactly this structure:
{
  "decision": "<SUITABLE | CONDITIONAL | UNSUITABLE | ESCALATED>",
  "summary": "<one internal sentence>",
  "rule_findings": [
    {"rule_id": "R1", "status": "<PASS|FAIL>", "explanation": "<sentence>"},
    {"rule_id": "R2", "status": "<PASS|FAIL>", "explanation": "<sentence>"},
    {"rule_id": "R3", "status": "<PASS|FAIL>", "explanation": "<sentence>"},
    {"rule_id": "R4", "status": "<PASS|FAIL>", "explanation": "<sentence>"},
    {"rule_id": "R5", "status": "<PASS|FAIL>", "explanation": "<sentence>"},
    {"rule_id": "R6", "status": "<PASS|FAIL>", "explanation": "<sentence>"},
    {"rule_id": "R7", "status": "<PASS|FAIL>", "explanation": "<sentence>"}
  ],
  "flags_addressed": [],
  "regulatory_basis": "MiFID II Article 25(2) — <rule references>",
  "client_facing_summary": "<2-3 plain English sentences>"
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
    rule_findings_skeleton = [
        {"rule_id": rule_id, "status": rule_verdict.get("rules", {}).get(rule_id, "UNKNOWN")}
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
    return parse_suitability_report(raw)