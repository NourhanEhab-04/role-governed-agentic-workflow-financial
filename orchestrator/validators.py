# orchestrator/validators.py

from schemas.suitability_report import VALID_DECISIONS



# orchestrator/validators.py


def validate_after_a1(state: dict) -> tuple:
    from schemas.client_profile import REQUIRED_CLIENT_KEYS
    profile = state.get("client_profile")
    if not isinstance(profile, dict):
        return False, "client_profile is missing or not a dict"
    missing = REQUIRED_CLIENT_KEYS - profile.keys()
    if missing:
        return False, f"client_profile missing keys: {missing}"
    return True, ""


def validate_after_a2(state: dict) -> tuple:
    from schemas.product_profile import REQUIRED_PRODUCT_KEYS
    profile = state.get("product_profile")
    if not isinstance(profile, dict):
        return False, "product_profile is missing or not a dict"
    missing = REQUIRED_PRODUCT_KEYS - profile.keys()
    if missing:
        return False, f"product_profile missing keys: {missing}"
    return True, ""


def validate_after_a3(state: dict) -> tuple:
    """
    Structural + integrity validation for A3 output.

    HALT conditions (return False):
      - rule_verdict missing or wrong type
      - required keys absent
      - decision value invalid
      - decision differs between pre_check and A3 (data mutation detected)

    Score drift is NOT a halt condition — it is a warning (see check_a3_score_drift).
    Two deterministic engine calls on the same profiles must produce the same decision
    and the same rule results.  A score difference with the same decision indicates
    A3 passed profiles with minor float differences to the tool, which is suspicious
    but not an integrity failure.
    """
    from schemas.rule_verdict import REQUIRED_VERDICT_KEYS
    verdict = state.get("rule_verdict")
    if not isinstance(verdict, dict):
        return False, "rule_verdict is missing or not a dict"
    missing = REQUIRED_VERDICT_KEYS - verdict.keys()
    if missing:
        return False, f"rule_verdict missing keys: {missing}"
    valid_decisions = {"SUITABLE", "CONDITIONAL", "UNSUITABLE"}
    if verdict.get("decision") not in valid_decisions:
        return False, f"rule_verdict has invalid decision: '{verdict.get('decision')}'"
    pre = state.get("pre_check_verdict")
    if pre is not None:
        if pre.get("decision") != verdict.get("decision"):
            return (
                False,
                f"System integrity error: deterministic rule engine produced "
                f"different DECISIONS between pre-check ('{pre.get('decision')}') "
                f"and A3 ('{verdict.get('decision')}') — possible data mutation "
                f"or tool bypass between checkpoints.",
            )
        # Failed-rules set must also agree — different failed rules with the same
        # decision would indicate the LLM corrupted a profile field used only in
        # a passing rule, which cannot happen if the tool was called correctly.
        pre_failed  = set(r["rule"] for r in pre.get("rules", []) if not r.get("pass", True))
        a3_rules    = verdict.get("rules", {})
        a3_failed   = set(k for k, v in a3_rules.items() if v == "FAIL")
        if pre_failed != a3_failed:
            return (
                False,
                f"System integrity error: failed rule sets differ between "
                f"pre-check ({sorted(pre_failed)}) and A3 ({sorted(a3_failed)}) "
                f"despite agreeing on decision '{verdict.get('decision')}' — "
                f"possible profile mutation in A3 tool call.",
            )
    return True, ""


def check_a3_score_drift(state: dict) -> dict | None:
    """
    Detect numeric score differences between the pre-check and A3 verdicts.

    Both are deterministic calls on the same profiles, so scores should be
    identical.  A difference means A3 passed slightly different profile data
    to the tool (e.g. float rounding, extra fields ignored by the engine).
    This is a warning, not a halt — the decision and failed rules already agree
    (validated by validate_after_a3), so the compliance outcome is unaffected.

    Returns a warning dict if scores differ, None if they are identical.
    """
    pre     = state.get("pre_check_verdict", {})
    verdict = state.get("rule_verdict", {})
    pre_score = pre.get("score")
    a3_score  = verdict.get("score")
    if pre_score is None or a3_score is None:
        return None
    if pre_score == a3_score:
        return None
    return {
        "stage":           "A3",
        "event":           "score_drift",
        "pre_check_score": pre_score,
        "a3_score":        a3_score,
        "delta":           a3_score - pre_score,
        "decision":        verdict.get("decision"),
        "detail": (
            f"Pre-check score {pre_score} ≠ A3 score {a3_score} "
            f"(delta {a3_score - pre_score:+d}). Decision '{verdict.get('decision')}' "
            f"is consistent. Possible cause: A3 tool call received profiles with "
            f"minor float differences from the pre-check profiles."
        ),
    }


def validate_after_a4(state: dict) -> tuple:
    report = state.get("conflict_report")
    if not isinstance(report, dict):
        return False, "conflict_report is missing or not a dict"
    for key in ("flags", "escalate", "summary"):
        if key not in report:
            return False, f"conflict_report missing key: '{key}'"
    if not isinstance(report["flags"], list):
        return False, "'flags' must be a list"
    if not isinstance(report["escalate"], bool):
        return False, "'escalate' must be a boolean"
    audit = state.get("audit_verdict")
    if audit is not None and audit.get("agreed") is False:
        if not report.get("escalate"):
            return (
                False,
                "audit_verdict shows rule engine disagreement but "
                "conflict_report.escalate is False — inconsistent state",
            )
    return True, ""


def validate_after_a5(state: dict) -> tuple:
    VALID_DECISIONS = {"SUITABLE", "CONDITIONAL", "UNSUITABLE", "ESCALATED"}
    report = state.get("suitability_report")
    if not isinstance(report, dict):
        return False, "suitability_report is missing or not a dict"
    if report.get("decision") not in VALID_DECISIONS:
        return False, f"suitability_report has invalid decision: '{report.get('decision')}'"
    if not isinstance(report.get("rule_findings"), list):
        return False, "'rule_findings' must be a list"
    if len(report.get("rule_findings", [])) != 7:
        return False, f"rule_findings must have 7 entries, got {len(report.get('rule_findings', []))}"
    reg_basis = report.get("regulatory_basis") or ""
    if "Article 25" not in reg_basis:
        return False, f"regulatory_basis must contain 'Article 25', got: '{reg_basis}'"

    # Semantic integrity: A5 output must agree with upstream deterministic verdicts.
    ok, err = validate_a5_semantic_correctness(state)
    if not ok:
        return False, err

    return True, ""


def validate_a5_semantic_correctness(state: dict) -> tuple:
    """
    Verify that A5's suitability_report is semantically consistent with the
    upstream deterministic outputs (rule_verdict and conflict_report).

    This closes the governance gap where A5 (a freeform LLM) could write
    rule_findings statuses that contradict the deterministic rule engine, or
    set a decision that contradicts the escalation flag.

    Three checks:
      1. Every rule_finding.status must match rule_verdict.rules[rule_id].
      2. decision == "ESCALATED" iff conflict_report.escalate is True.
      3. Every triggered flag must have a corresponding entry in flags_addressed.
    """
    report        = state.get("suitability_report", {})
    rule_verdict  = state.get("rule_verdict", {})
    conflict_report = state.get("conflict_report", {})

    # ── Check 1: rule_findings statuses agree with rule_verdict.rules ─────────
    engine_rules = rule_verdict.get("rules", {})  # {"R1": "PASS", ..., "R7": "FAIL"}
    if engine_rules:
        for finding in report.get("rule_findings", []):
            rid    = finding.get("rule_id")
            status = finding.get("status")
            if rid in engine_rules and engine_rules[rid] != status:
                return (
                    False,
                    f"A5 semantic error: rule_findings[{rid}].status='{status}' "
                    f"contradicts rule_verdict.rules[{rid}]='{engine_rules[rid]}'. "
                    f"A5 must copy statuses from the rule engine output — it may not override them."
                )

    # ── Check 2: decision matches escalate flag ───────────────────────────────
    decision  = report.get("decision")
    escalate  = conflict_report.get("escalate")
    if escalate is True and decision != "ESCALATED":
        return (
            False,
            f"A5 semantic error: conflict_report.escalate=True but "
            f"suitability_report.decision='{decision}'. "
            f"When escalate=True the decision MUST be 'ESCALATED'."
        )
    if escalate is False and decision == "ESCALATED":
        return (
            False,
            f"A5 semantic error: suitability_report.decision='ESCALATED' but "
            f"conflict_report.escalate=False. "
            f"'ESCALATED' is only permitted when the conflict detector sets escalate=True."
        )

    # ── Check 3: every triggered flag is addressed ────────────────────────────
    triggered_flag_ids = {
        f["rule_id"]
        for f in conflict_report.get("flags", [])
        if f.get("triggered") is True
    }
    addressed_ids = {
        entry.get("rule_id")
        for entry in report.get("flags_addressed", [])
    }
    missing = triggered_flag_ids - addressed_ids
    if missing:
        return (
            False,
            f"A5 semantic error: flags_addressed is missing entries for triggered "
            f"flag(s): {sorted(missing)}. Every triggered conflict flag must be addressed."
        )

    return True, ""