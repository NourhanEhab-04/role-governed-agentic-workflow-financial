# agents/disclosure_agent.py

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

from schemas.client_profile import REQUIRED_CLIENT_KEYS
from schemas.product_profile import REQUIRED_PRODUCT_KEYS
from schemas.rule_verdict import REQUIRED_VERDICT_KEYS

import json as _json
from agents.parsing import extract_json_object
from schemas.suitability_report import (
    VALID_DECISIONS,
    VALID_RULE_IDS,
    REQUIRED_RULE_FINDING_KEYS,
    REQUIRED_REPORT_KEYS,
)
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

# Maps verbose rule_verdict keys → the short IDs required in rule_findings.
# The LLM sometimes echoes the long-form keys it sees in rule_verdict["rules"]
# despite the skeleton; we normalise them here so the parser is resilient.
_LONG_TO_SHORT_RULE_ID = {
    "R1_knowledge": "R1", "R2_risk": "R2", "R3_horizon": "R3",
    "R4_afford": "R4",   "R5_vuln": "R5",  "R6_leverage": "R6",
    "R7_complexity": "R7",
}


def parse_suitability_report(raw: str) -> dict:
    data = extract_json_object(raw)

    # top-level keys
    for key in REQUIRED_REPORT_KEYS:
        if key not in data:
            raise ValueError(f"Missing required key in suitability report: '{key}'")

    # decision enum
    if data["decision"] not in VALID_DECISIONS:
        raise ValueError(
            f"Invalid decision '{data['decision']}'. Must be one of {VALID_DECISIONS}"
        )

    # rule_findings must be a list
    if not isinstance(data["rule_findings"], list):
        raise ValueError("'rule_findings' must be a list")

    # Normalise long-form IDs (R1_knowledge → R1) before validation.
    # Gemini sometimes echoes the keys it sees in rule_verdict["rules"] rather
    # than using the short-form IDs from the skeleton we supplied.
    for finding in data["rule_findings"]:
        if isinstance(finding, dict) and "rule_id" in finding:
            finding["rule_id"] = _LONG_TO_SHORT_RULE_ID.get(
                finding["rule_id"], finding["rule_id"]
            )

    # exactly 7 entries
    if len(data["rule_findings"]) != 7:
        raise ValueError(
            f"'rule_findings' must contain exactly 7 entries, got {len(data['rule_findings'])}"
        )

    # each finding has required keys and a valid rule_id
    seen_ids = set()
    for i, finding in enumerate(data["rule_findings"]):
        for key in REQUIRED_RULE_FINDING_KEYS:
            if key not in finding:
                raise ValueError(f"rule_findings[{i}] missing required key: '{key}'")
        if finding["rule_id"] not in VALID_RULE_IDS:
            raise ValueError(
                f"rule_findings[{i}] has invalid rule_id '{finding['rule_id']}'. "
                f"Must be one of {VALID_RULE_IDS}"
            )
        if finding["rule_id"] in seen_ids:
            raise ValueError(f"Duplicate rule_id '{finding['rule_id']}' in rule_findings")
        seen_ids.add(finding["rule_id"])

    # flags_addressed must be a list (may be empty)
    if not isinstance(data["flags_addressed"], list):
        raise ValueError("'flags_addressed' must be a list")

    # regulatory_basis must contain "Article 25".  When the LLM omits it,
    # returns null, or returns an empty string (common under token pressure),
    # we construct it from the rule_findings we already validated rather than
    # failing the entire A5 stage.  This field is a deterministic citation, not
    # LLM reasoning, so building it programmatically is correct.
    reg_basis = data.get("regulatory_basis") or ""
    if not isinstance(reg_basis, str):
        reg_basis = ""
    if "Article 25" not in reg_basis:
        passed = [f["rule_id"] for f in data["rule_findings"] if f.get("status") == "PASS"]
        failed = [f["rule_id"] for f in data["rule_findings"] if f.get("status") == "FAIL"]
        all_rules = ", ".join(passed + failed) if (passed or failed) else "R1–R7"
        decision = data.get("decision", "")
        data["regulatory_basis"] = (
            f"MiFID II Article 25(2) — suitability assessed under {all_rules}. "
            f"Decision: {decision}."
        )

    return data
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

    # Pre-map rule IDs: rule_verdict["rules"] uses "R1_knowledge" etc.;
    # rule_findings must use "R1"–"R7". Build the skeleton here so the LLM
    # only needs to write explanations — not figure out the ID mapping.
    _VERDICT_KEY_TO_ID = {
        "R1_knowledge": "R1", "R2_risk": "R2", "R3_horizon": "R3",
        "R4_afford": "R4",   "R5_vuln": "R5",  "R6_leverage": "R6",
        "R7_complexity": "R7",
    }
    rule_findings_skeleton = [
        {"rule_id": short_id, "status": rule_verdict.get("rules", {}).get(long_key, "UNKNOWN")}
        for long_key, short_id in _VERDICT_KEY_TO_ID.items()
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