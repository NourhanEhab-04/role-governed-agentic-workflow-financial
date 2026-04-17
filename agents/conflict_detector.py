# agents/conflict_detector.py

import json as _json
from agents.parsing import extract_json_object

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from rule_engine.rule_engine import evaluate_suitability
from schemas.client_profile import REQUIRED_CLIENT_KEYS
from schemas.product_profile import REQUIRED_PRODUCT_KEYS
from schemas.rule_verdict import REQUIRED_VERDICT_KEYS
VALID_SEVERITIES = {"LOW", "HIGH"}
REQUIRED_FLAG_KEYS = {"rule_id", "triggered", "severity", "message"}

CONFLICT_DETECTOR_SYSTEM_PROMPT = """
You are the Conflict Detector in a MiFID II suitability assessment pipeline.

You receive a JSON object with three keys:
- "client_profile": the structured client profile from the profiler agent
- "product_profile": the structured product profile from the classifier agent
- "rule_verdict": the verdict produced by the rule engine, including score, decision, and per-rule results

Your job is to audit the rule verdict for four specific concerns:

1. BORDERLINE — Is the score in the range [40, 55]? If so, the decision is technically valid
   but sits close to a threshold boundary. severity: "LOW".

2. CONCENTRATION — Does client_profile contain a "portfolio_concentration_pct" field
   above 40%? This indicates the client is heavily concentrated in a single asset class,
   which increases risk exposure beyond what the rule engine captures. severity: "HIGH".

3. CONTRADICTION — Is the rule_verdict decision "SUITABLE" while the client's
   "vulnerability" field is "HIGH"? A suitable recommendation for a highly vulnerable
   client is a regulatory contradiction. severity: "HIGH".

4. ESCALATION — Does the above analysis produce at least one triggered HIGH-severity flag
   (CONCENTRATION or CONTRADICTION)? If yes, set triggered=true. severity: always "HIGH".
   Set "escalate" to true if and only if the ESCALATION flag is triggered.

Rules:
- Report all four checks in the "flags" list, even if not triggered.
- Never modify, override, or reinterpret the rule_verdict.
- Never add flags beyond the four listed above.
- The "severity" field must be exactly "LOW" or "HIGH" — never "N/A" or any other value.
- BORDERLINE is always "LOW". CONCENTRATION, CONTRADICTION, and ESCALATION are always "HIGH".
- Your entire response must be a single JSON object. No preamble. No explanation outside the JSON.

Respond with exactly this structure:
{
  "flags": [
    {"rule_id": "BORDERLINE",     "triggered": <true|false>, "severity": "LOW",  "message": "<explanation>"},
    {"rule_id": "CONCENTRATION",  "triggered": <true|false>, "severity": "HIGH", "message": "<explanation>"},
    {"rule_id": "CONTRADICTION",  "triggered": <true|false>, "severity": "HIGH", "message": "<explanation>"},
    {"rule_id": "ESCALATION",     "triggered": <true|false>, "severity": "HIGH", "message": "<explanation>"}
  ],
  "escalate": <true | false>,
  "summary": "<one sentence summarising the audit outcome>"
}
"""

def check_borderline(rule_verdict: dict) -> dict:
    score = rule_verdict["score"]
    triggered = 40 <= score <= 55
    return {
        "rule_id": "BORDERLINE",
        "triggered": triggered,
        "severity": "LOW",
        "message": (
            f"Score {score} is in the borderline zone [40–55]. "
            "Human review recommended." if triggered
            else f"Score {score} is outside borderline zone."
        ),
    }


def check_concentration_risk(client_profile: dict) -> dict:
    concentration = client_profile.get("portfolio_concentration_pct", 0)
    triggered = concentration > 40
    return {
        "rule_id": "CONCENTRATION",
        "triggered": triggered,
        "severity": "HIGH",
        "message": (
            f"Single-asset concentration {concentration}% exceeds 40% threshold."
            if triggered
            else f"Concentration {concentration}% within acceptable range."
        ),
    }


def check_contradiction(client_profile: dict, rule_verdict: dict) -> dict:
    vulnerability = client_profile.get("financial_vulnerability", "NONE")
    decision = rule_verdict["decision"]
    triggered = vulnerability == "HIGH" and decision == "SUITABLE"
    return {
        "rule_id": "CONTRADICTION",
        "triggered": triggered,
        "severity": "HIGH",
        "message": (
            "Contradiction: SUITABLE verdict issued for a HIGH-vulnerability client. "
            "Escalation required."
            if triggered
            else "No contradiction detected between vulnerability status and verdict."
        ),
    }


def check_escalation_trigger(flags: list[dict]) -> dict:
    high_flags = [f for f in flags if f["triggered"] and f["severity"] == "HIGH"]
    triggered = len(high_flags) >= 1  # any HIGH flag triggers escalation
    return {
        "rule_id": "ESCALATION",
        "triggered": triggered,
        "severity": "HIGH",
        "message": (
            f"{len(high_flags)} HIGH-severity flag(s) detected — escalation required."
            if triggered
            else "No escalation trigger conditions met."
        ),
    }




def parse_conflict_report(raw: dict) -> dict:
    for key in ("flags", "escalate", "summary"):
        if key not in raw:
            raise ValueError(f"conflict_report missing required key: '{key}'")

    if not isinstance(raw["escalate"], bool):
        raise ValueError("conflict_report 'escalate' must be a bool")

    if not isinstance(raw["flags"], list):
        raise ValueError("conflict_report 'flags' must be a list")

    if raw["escalate"] and len(raw["flags"]) == 0:
        raise ValueError("escalate=True requires at least one flag")

    for i, flag in enumerate(raw["flags"]):
        missing = REQUIRED_FLAG_KEYS - flag.keys()
        if missing:
            raise ValueError(f"Flag {i} missing keys: {missing}")
        if flag["severity"] not in VALID_SEVERITIES:
            raise ValueError(
                f"Flag {i} has invalid severity '{flag['severity']}'. "
                f"Must be one of {VALID_SEVERITIES}"
            )
        if not isinstance(flag["triggered"], bool):
            raise ValueError(f"Flag {i} 'triggered' must be a bool")

    return raw
import json as _json


async def run_conflict_detector(
    client_profile: dict,
    product_profile: dict,
    rule_verdict: dict,
    model_client,
) -> dict:
    borderline   = check_borderline(rule_verdict)
    concentration = check_concentration_risk(client_profile)
    contradiction = check_contradiction(client_profile, rule_verdict)
    flags = [borderline, concentration, contradiction]
    escalation = check_escalation_trigger(flags)
    flags.append(escalation)

    report = {
        "flags": flags,
        "escalate": escalation["triggered"],
        "summary": (
            "Escalation required: HIGH-severity flag(s) detected."
            if escalation["triggered"]
            else "No escalation required."
        ),
    }
    return parse_conflict_report(report)

def check_rule_engine_agreement(
    client_profile: dict,
    product_profile: dict,
    a3_verdict: dict,
) -> dict:
    """
    A4's independent rule engine call — the third point of contact.
    Re-runs evaluate_suitability and compares decision + failed rules to A3's verdict.
    
    Returns:
        {
            "agreed": bool,
            "a4_decision": str,
            "a3_decision": str,
            "a4_failed_rules": list[str],
            "a3_failed_rules": list[str],
            "detail": str,
        }
    """
    a4_result = evaluate_suitability(client_profile, product_profile)
    a4_decision = a4_result["decision"]
    a4_failed = sorted(r["rule"] for r in a4_result["rules"] if not r["pass"])

    a3_decision = a3_verdict.get("decision", "UNKNOWN")
    a3_failed = sorted(a3_verdict.get("failed_rules", []))

    agreed = (a4_decision == a3_decision) and (a4_failed == a3_failed)

    detail = (
        f"A4 and A3 agree: decision={a4_decision}, failed_rules={a4_failed}"
        if agreed else
        f"DISAGREEMENT — A4 decision={a4_decision} failed={a4_failed} | "
        f"A3 decision={a3_decision} failed={a3_failed}"
    )

    return {
        "agreed": agreed,
        "a4_decision": a4_decision,
        "a3_decision": a3_decision,
        "a4_failed_rules": a4_failed,
        "a3_failed_rules": a3_failed,
        "detail": detail,
    }