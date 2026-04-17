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
                f"pre_check_verdict decision '{pre.get('decision')}' "
                f"!= rule_verdict decision '{verdict.get('decision')}' — "
                f"possible bypass attempt",
            )
    return True, ""


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
    return True, ""