# agents/conflict_detector.py
"""
A4 — Conflict Detector

Design: the LLM agent receives client_profile, product_profile, and rule_verdict
and reasons about four conflict types.  Its output is then validated by a
deterministic override layer (check_rule_engine_agreement) that enforces the
hard invariant: any HIGH-severity flag triggers escalation.

Three-point architecture:
  pre_check → A3 (LLM + tool) → A4 (LLM + deterministic validation)
                                 ↑
                         audit_verdict injected before A4 runs
If audit_verdict.agreed is False, validate_after_a4 forces escalation.
"""

import json as _json

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

from agents.parsing import extract_json_object
from rule_engine.rule_engine import evaluate_suitability


# ── System prompt ─────────────────────────────────────────────────────────────

CONFLICT_DETECTOR_SYSTEM_PROMPT = """
You are A4, the Conflict Detector in a MiFID II suitability assessment pipeline.

You receive a JSON object with three keys:
- "client_profile": structured client profile from the profiler agent
- "product_profile": structured product profile from the classifier agent
- "rule_verdict": the verdict from the rule engine
  (score, decision, hard_failed_rules, per-rule PASS/FAIL results)

Your job: audit the rule verdict for FOUR specific conflict types and produce
a structured conflict report.  Reason carefully about EACH flag before setting
triggered = true or false.

════════════════════════════════════════
FOUR CONFLICT TYPES TO ASSESS
════════════════════════════════════════

1. BORDERLINE  (severity: LOW)
   Trigger when: rule_verdict["score"] is between 40 and 55 (inclusive).
   Meaning: the decision is technically valid but sits close to a threshold
   boundary — human review is advisable to avoid borderline misclassification.

2. CONCENTRATION  (severity: HIGH)
   Trigger when ALL of the following are met AND rule_verdict["decision"] != "UNSUITABLE":
   (An UNSUITABLE verdict already blocks the investment — escalation adds nothing.)
   Condition (a): client_profile["portfolio_concentration_pct"] > 60
                  AND product_profile["risk_class"] >= 6
                  (extreme concentration in a high-risk product amplifies tail risk)
   Condition (b): client_profile["portfolio_concentration_pct"] > 40
                  AND client_profile["financial_vulnerability"] == "HIGH"
                  AND product_profile["risk_class"] >= 2
                  (ESMA §53-56: stricter limits for vulnerable clients; risk_class=1
                   money-market products are safe enough even for vulnerable clients)
   Meaning: concentration risk warrants escalation only when it intersects with an
   elevated product risk class or a client needing enhanced protection — and only
   when the base verdict is not already UNSUITABLE.

3. CONTRADICTION  (severity: HIGH)
   Trigger when: client_profile["financial_vulnerability"] == "HIGH"
                 AND rule_verdict["decision"] == "SUITABLE"
                 AND product_profile["risk_class"] >= 4.
   Meaning: per ESMA GL §86-88, HIGH-vulnerability clients require enhanced
   protection.  A SUITABLE verdict on a product with risk_class >= 4 warrants
   mandatory human review, even when R5 (hard-fail at risk_class >= 5) did
   not fire.  Very-low-risk products (risk_class <= 3, e.g. government bonds)
   are appropriate for vulnerable clients and do NOT trigger this flag.

4. ESCALATION  (severity: HIGH)
   Trigger when: at least one CONCENTRATION or CONTRADICTION flag is triggered.
   This is a derived flag — it aggregates HIGH-severity findings.
   Set "escalate" to true if and only if ESCALATION is triggered.

════════════════════════════════════════
HARD RULES (non-negotiable)
════════════════════════════════════════
- Report ALL FOUR flags — even those not triggered.
- "severity" must be exactly "LOW" or "HIGH" — never anything else.
- BORDERLINE is always severity "LOW".
- CONCENTRATION, CONTRADICTION, and ESCALATION are always severity "HIGH".
- Never modify or reinterpret the rule_verdict.
- "escalate" must be a boolean — exactly true or false.
- Your entire response must be a single JSON object. No preamble. No markdown.

════════════════════════════════════════
REQUIRED OUTPUT FORMAT
════════════════════════════════════════
{
  "flags": [
    {"rule_id": "BORDERLINE",    "triggered": <true|false>, "severity": "LOW",  "message": "<concise explanation citing the score value>"},
    {"rule_id": "CONCENTRATION", "triggered": <true|false>, "severity": "HIGH", "message": "<concise explanation citing the percentage>"},
    {"rule_id": "CONTRADICTION", "triggered": <true|false>, "severity": "HIGH", "message": "<concise explanation citing the verdict and vulnerability>"},
    {"rule_id": "ESCALATION",    "triggered": <true|false>, "severity": "HIGH", "message": "<concise explanation naming which HIGH flags triggered this>"}
  ],
  "escalate": <true | false>,
  "summary": "<one sentence summarising the audit outcome>"
}
"""


# ── Deterministic check functions (validation layer) ─────────────────────────
# These run AFTER the LLM produces its output to enforce hard invariants.
# An LLM output that misses a clear trigger is corrected here.

def check_borderline(rule_verdict: dict) -> dict:
    score = rule_verdict.get("score", 0)
    triggered = 40 <= score <= 55
    return {
        "rule_id":   "BORDERLINE",
        "triggered": triggered,
        "severity":  "LOW",
        "message": (
            f"Score {score} is in the borderline zone [40–55]. Human review recommended."
            if triggered else
            f"Score {score} is outside borderline zone."
        ),
    }


def check_concentration_risk(
    client_profile: dict,
    product_profile: dict = {},
    rule_verdict: dict = {},
) -> dict:
    concentration = client_profile.get("portfolio_concentration_pct", 0)
    vulnerability = client_profile.get("financial_vulnerability", "LOW")
    risk_class    = product_profile.get("risk_class", 0)
    decision      = rule_verdict.get("decision", "")

    # An UNSUITABLE verdict already blocks the investment — escalation adds nothing.
    if decision == "UNSUITABLE":
        triggered = False
    else:
        # MiFID II proportionality (ESMA §53-56):
        # (a) extreme concentration in a high-risk product (risk_class ≥ 6)
        # (b) above-moderate concentration for HIGH-vulnerability client in any
        #     product above money-market level (risk_class ≥ 2).
        #     risk_class=1 (money market) is safe enough even for vulnerable clients.
        triggered = (concentration > 60 and risk_class >= 6) or \
                    (concentration > 40 and vulnerability == "HIGH" and risk_class >= 2)

    if triggered:
        if vulnerability == "HIGH":
            msg = (f"Concentration {concentration}% exceeds 40% for a HIGH-vulnerability "
                   f"client in a risk_class={risk_class} product (ESMA §53-56).")
        else:
            msg = (f"Concentration {concentration}% exceeds 60% in a risk_class={risk_class} "
                   f"product (≥6 tail-risk amplification threshold).")
    else:
        msg = (f"Concentration {concentration}% does not trigger escalation: "
               f"decision={decision or 'N/A'}, vulnerability={vulnerability}, "
               f"risk_class={risk_class}.")
    return {
        "rule_id":   "CONCENTRATION",
        "triggered": triggered,
        "severity":  "HIGH",
        "message":   msg,
    }


def check_contradiction(client_profile: dict, rule_verdict: dict, product_profile: dict = {}) -> dict:
    vulnerability = client_profile.get("financial_vulnerability", "NONE")
    decision      = rule_verdict.get("decision", "")
    risk_class    = product_profile.get("risk_class", 0)
    # Per ESMA GL §86-88: HIGH-vulnerability clients need enhanced protection.
    # SUITABLE on risk_class >= 4 requires human review (risk_class >= 5 is
    # already a hard fail via R5; risk_class 4 is the enhanced-protection zone).
    # Very-low-risk products (risk_class <= 3) are appropriate for vulnerable clients.
    triggered = (vulnerability == "HIGH" and decision == "SUITABLE" and risk_class >= 4)
    return {
        "rule_id":   "CONTRADICTION",
        "triggered": triggered,
        "severity":  "HIGH",
        "message": (
            f"HIGH-vulnerability client received SUITABLE on a risk_class={risk_class} product "
            f"(≥ 4 enhanced-protection threshold). Escalation required per ESMA GL §86-88."
            if triggered else
            f"No contradiction: vulnerability={vulnerability}, decision={decision}, "
            f"risk_class={risk_class} — enhanced-protection threshold not breached."
        ),
    }


def check_escalation_trigger(flags: list[dict]) -> dict:
    high_flags = [f for f in flags if f["triggered"] and f["severity"] == "HIGH"]
    triggered  = len(high_flags) >= 1
    return {
        "rule_id":   "ESCALATION",
        "triggered": triggered,
        "severity":  "HIGH",
        "message": (
            f"{len(high_flags)} HIGH-severity flag(s) detected — escalation required."
            if triggered else
            "No escalation trigger conditions met."
        ),
    }


def _apply_deterministic_overrides(
    llm_report: dict, client_profile: dict, rule_verdict: dict, product_profile: dict = {}
) -> tuple[dict, list[dict]]:
    """
    Enforce hard invariants on the LLM's conflict report.

    The LLM may miss or mis-set flags.  These deterministic checks
    override incorrect values to guarantee correctness.

    Returns
    -------
    (corrected_report, overrides_applied)
      corrected_report  : the validated conflict report dict
      overrides_applied : list of {rule_id, llm_triggered, corrected_triggered}
                          for every flag where the LLM value was wrong
    """
    borderline    = check_borderline(rule_verdict)
    concentration = check_concentration_risk(client_profile, product_profile, rule_verdict)
    contradiction = check_contradiction(client_profile, rule_verdict, product_profile)

    det = {
        "BORDERLINE":    borderline,
        "CONCENTRATION": concentration,
        "CONTRADICTION": contradiction,
    }

    overrides_applied: list[dict] = []
    corrected_flags = []
    for flag in llm_report.get("flags", []):
        rid = flag.get("rule_id", "")
        if rid in det:
            corrected = dict(flag)
            llm_triggered = flag.get("triggered")
            det_triggered = det[rid]["triggered"]
            corrected["triggered"] = det_triggered
            corrected["severity"]  = det[rid]["severity"]
            if det_triggered and not llm_triggered:
                corrected["message"] = det[rid]["message"]
            if llm_triggered != det_triggered:
                overrides_applied.append({
                    "rule_id":          rid,
                    "llm_triggered":    llm_triggered,
                    "corrected_triggered": det_triggered,
                    "reason":           det[rid]["message"],
                })
            corrected_flags.append(corrected)
        else:
            corrected_flags.append(flag)  # ESCALATION — handled below

    # Recompute ESCALATION from the corrected non-escalation flags
    non_esc = [f for f in corrected_flags if f.get("rule_id") != "ESCALATION"]
    escalation = check_escalation_trigger(non_esc)

    # Replace or append ESCALATION flag
    final_flags = [f for f in corrected_flags if f.get("rule_id") != "ESCALATION"]
    final_flags.append(escalation)

    # Generate summary deterministically from the corrected flags.
    triggered_high = [f["rule_id"] for f in final_flags if f["triggered"] and f["severity"] == "HIGH"]
    triggered_low  = [f["rule_id"] for f in final_flags if f["triggered"] and f["severity"] == "LOW"]

    if triggered_high:
        high_names = ", ".join(triggered_high)
        if triggered_low:
            low_names = ", ".join(triggered_low)
            summary = (
                f"Escalation required — HIGH-severity flag(s) triggered: {high_names}. "
                f"LOW-severity flag(s) also triggered: {low_names}."
            )
        else:
            summary = f"Escalation required — HIGH-severity flag(s) triggered: {high_names}."
    elif triggered_low:
        low_names = ", ".join(triggered_low)
        summary = (
            f"No escalation required. LOW-severity flag(s) noted for review: {low_names}."
        )
    else:
        summary = "No conflict flags triggered. Assessment proceeds under standard review."

    report = {
        "flags":    final_flags,
        "escalate": escalation["triggered"],
        "summary":  summary,
    }
    return report, overrides_applied


# ── Parser ─────────────────────────────────────────────────────────────────────

def parse_conflict_report(raw: dict) -> dict:
    """Validate a conflict_report dict using the Pydantic model."""
    from schemas.output_models import ConflictReportModel
    from pydantic import ValidationError
    try:
        return ConflictReportModel.model_validate(raw).model_dump()
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


# ── Agent runner ───────────────────────────────────────────────────────────────

async def run_conflict_detector(
    client_profile: dict,
    product_profile: dict,
    rule_verdict: dict,
    model_client,
) -> tuple[dict, list[dict]]:
    """
    Run A4: the LLM reasons about four conflict types, then a deterministic
    override layer enforces hard invariants before the report is validated.

    Returns
    -------
    (report, overrides_applied)
      report            : validated conflict_report dict
      overrides_applied : list of flag-level overrides the deterministic layer made
    """
    agent = AssistantAgent(
        name="ConflictDetector",
        model_client=model_client,
        system_message=CONFLICT_DETECTOR_SYSTEM_PROMPT,
    )

    payload = _json.dumps({
        "client_profile":  client_profile,
        "product_profile": product_profile,
        "rule_verdict":    rule_verdict,
    })

    response = await agent.on_messages(
        [TextMessage(content=payload, source="user")],
        cancellation_token=CancellationToken(),
    )

    raw_text = response.chat_message.content
    data = extract_json_object(raw_text)

    corrected, overrides_applied = _apply_deterministic_overrides(
        data, client_profile, rule_verdict, product_profile
    )
    return parse_conflict_report(corrected), overrides_applied


# ── Audit check (independent rule engine call for three-point integrity) ───────

def check_rule_engine_agreement(
    client_profile: dict,
    product_profile: dict,
    a3_verdict: dict,
) -> dict:
    """
    Third independent rule engine contact — runs deterministically before A4.

    Re-runs evaluate_suitability and compares decision + failed rules to A3's
    LLM-tool-call verdict.  Any disagreement indicates A3 bypassed or
    corrupted the tool call.

    Returns:
        {
            "agreed":         bool,
            "a4_decision":    str,
            "a3_decision":    str,
            "a4_failed_rules": list[str],
            "a3_failed_rules": list[str],
            "detail":         str,
        }
    """
    a4_result   = evaluate_suitability(client_profile, product_profile)
    a4_decision = a4_result["decision"]
    a4_failed   = sorted(r["rule"] for r in a4_result["rules"] if not r["pass"])

    a3_decision = a3_verdict.get("decision", "UNKNOWN")
    if "rules" in a3_verdict and isinstance(a3_verdict["rules"], dict):
        a3_failed = sorted(k for k, v in a3_verdict["rules"].items() if v == "FAIL")
    else:
        a3_failed = sorted(a3_verdict.get("failed_rules", []))

    agreed = (a4_decision == a3_decision) and (a4_failed == a3_failed)

    detail = (
        f"A4 and A3 agree: decision={a4_decision}, failed_rules={a4_failed}"
        if agreed else
        f"DISAGREEMENT — A4 decision={a4_decision} failed={a4_failed} | "
        f"A3 decision={a3_decision} failed={a3_failed}"
    )

    return {
        "agreed":          agreed,
        "a4_decision":     a4_decision,
        "a3_decision":     a3_decision,
        "a4_failed_rules": a4_failed,
        "a3_failed_rules": a3_failed,
        "detail":          detail,
    }
