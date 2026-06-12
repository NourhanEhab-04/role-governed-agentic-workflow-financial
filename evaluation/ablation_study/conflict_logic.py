"""
evaluation/ablation_study/conflict_logic.py
============================================
Deterministic conflict detection for the ablation study.

Imports the already-implemented deterministic check functions from
agents/conflict_detector.py and adds ablation support (flag disabling).

No LLM calls. Pure functions.
"""

from __future__ import annotations

from agents.conflict_detector import (
    check_borderline,
    check_concentration_risk,
    check_contradiction,
)


def _make_disabled_flag(rule_id: str, severity: str) -> dict:
    return {
        "rule_id":   rule_id,
        "triggered": False,
        "severity":  severity,
        "message":   f"[ABLATION: {rule_id} flag disabled]",
    }


def run_conflict_detection(
    client_profile: dict,
    product_profile: dict,
    rule_verdict: dict,
    disabled_flags: list[str] | None = None,
) -> dict:
    """
    Run all four conflict checks deterministically with optional flag disabling.

    Parameters
    ----------
    client_profile  : must include financial_vulnerability; optionally portfolio_concentration_pct
    product_profile : must include risk_class
    rule_verdict    : output of evaluate_suitability / evaluate_suitability_ablated
    disabled_flags  : flag IDs to suppress, e.g. ["BORDERLINE", "CONCENTRATION"]

    Returns
    -------
    {
      "flags": [
        {"rule_id": "BORDERLINE",    "triggered": bool, "severity": "LOW",  "message": str},
        {"rule_id": "CONCENTRATION", "triggered": bool, "severity": "HIGH", "message": str},
        {"rule_id": "CONTRADICTION", "triggered": bool, "severity": "HIGH", "message": str},
        {"rule_id": "ESCALATION",    "triggered": bool, "severity": "HIGH", "message": str},
      ],
      "escalate": bool,
      "summary":  str,
    }
    """
    disabled = set(disabled_flags or [])

    # ── BORDERLINE ────────────────────────────────────────────────────────────
    if "BORDERLINE" in disabled:
        borderline = _make_disabled_flag("BORDERLINE", "LOW")
    else:
        borderline = check_borderline(rule_verdict)

    # ── CONCENTRATION ─────────────────────────────────────────────────────────
    if "CONCENTRATION" in disabled:
        concentration = _make_disabled_flag("CONCENTRATION", "HIGH")
    else:
        concentration = check_concentration_risk(client_profile, product_profile, rule_verdict)

    # ── CONTRADICTION ─────────────────────────────────────────────────────────
    if "CONTRADICTION" in disabled:
        contradiction = _make_disabled_flag("CONTRADICTION", "HIGH")
    else:
        contradiction = check_contradiction(client_profile, rule_verdict, product_profile)

    # ── ESCALATION (derived) ──────────────────────────────────────────────────
    # ESCALATION is triggered when at least one HIGH-severity flag fires.
    # If the ESCALATION flag itself is disabled, force no escalation.
    high_triggered = (
        concentration["triggered"] or contradiction["triggered"]
    )

    if "ESCALATION" in disabled or "BORDERLINE" in disabled and "CONCENTRATION" in disabled and "CONTRADICTION" in disabled:
        escalation_triggered = False
    else:
        escalation_triggered = high_triggered

    escalation = {
        "rule_id":   "ESCALATION",
        "triggered": escalation_triggered,
        "severity":  "HIGH",
        "message": (
            "[ABLATION: ESCALATION flag disabled]"
            if "ESCALATION" in disabled
            else (
                "ESCALATION triggered: at least one HIGH-severity flag is active."
                if escalation_triggered
                else "No HIGH-severity flags triggered — no escalation."
            )
        ),
    }

    escalate = escalation_triggered
    flags = [borderline, concentration, contradiction, escalation]

    active = [f["rule_id"] for f in flags if f["triggered"]]
    summary = (
        f"Conflict detection complete. Active flags: {active}. "
        f"Escalate: {escalate}."
        if active
        else "No conflict flags triggered."
    )

    return {
        "flags":    flags,
        "escalate": escalate,
        "summary":  summary,
    }


def derive_final_decision(rule_verdict: dict, conflict_report: dict) -> str:
    """
    Combine rule engine verdict and conflict report into the final decision.

    If conflict_report["escalate"] is True, decision = "ESCALATED".
    Otherwise, decision = rule_verdict["decision"].
    """
    if conflict_report.get("escalate", False):
        return "ESCALATED"
    return rule_verdict["decision"]
