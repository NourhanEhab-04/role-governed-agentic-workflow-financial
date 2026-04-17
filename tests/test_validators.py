# tests/test_validators.py

from orchestrator.validators import (
    validate_after_a1,
    validate_after_a2,
    validate_after_a3,
    validate_after_a4,
    validate_after_a5,
)


def _findings():
    return [
        {"rule_id": r, "status": "PASS", "explanation": "ok"}
        for r in ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
    ]


# ── A1 ──────────────────────────────────────────────

def test_a1_valid():
    state = {"client_profile": {
        "financial_knowledge": "moderate",
        "risk_tolerance_score": 5,
        "investment_horizon": 5,
        "liquid_assets": 10000.0,
        "income": 50000.0,
        "investment_amount": 5000.0,
        "can_afford_total_loss": True,
        "financial_vulnerability": "LOW",
    }}
    ok, msg = validate_after_a1(state)
    assert ok is True
    assert msg == ""

def test_a1_missing_key():
    state = {"client_profile": {"knowledge_level": "moderate"}}
    ok, msg = validate_after_a1(state)
    assert ok is False
    assert msg != ""

def test_a1_not_a_dict():
    ok, msg = validate_after_a1({"client_profile": "wrong type"})
    assert ok is False


# ── A2 ──────────────────────────────────────────────

def test_a2_valid():
    state = {"product_profile": {
        "product_name": "Example Fund",
        "risk_class": 4,
        "complexity_tier": "NON-COMPLEX",
        "requires_knowledge_level": "basic",
        "minimum_horizon": 3,
        "potential_loss": "partial",
        "leverage": False,
    }}
    ok, msg = validate_after_a2(state)
    assert ok is True

def test_a2_missing_key():
    state = {"product_profile": {"risk_class": 4}}
    ok, msg = validate_after_a2(state)
    assert ok is False

def test_a2_missing_entirely():
    ok, msg = validate_after_a2({})
    assert ok is False


# ── A3 ──────────────────────────────────────────────

def test_a3_valid():
    state = {"rule_verdict": {
        "rules": {r: "PASS" for r in ["R1","R2","R3","R4","R5","R6","R7"]},
        "score": 100,
        "decision": "SUITABLE",
        "failed_rules": [],
    }}
    ok, msg = validate_after_a3(state)
    assert ok is True

def test_a3_invalid_decision():
    state = {"rule_verdict": {
        "rules": {},
        "score": 100,
        "decision": "MAYBE",
        "failed_rules": [],
    }}
    ok, msg = validate_after_a3(state)
    assert ok is False
    assert "decision" in msg

def test_a3_missing_score():
    state = {"rule_verdict": {
        "rules": {},
        "decision": "SUITABLE",
        "failed_rules": [],
    }}
    ok, msg = validate_after_a3(state)
    assert ok is False


# ── A4 ──────────────────────────────────────────────

def test_a4_valid():
    state = {"conflict_report": {
        "flags": [],
        "escalate": False,
        "summary": "ok",
    }}
    ok, msg = validate_after_a4(state)
    assert ok is True

def test_a4_missing_escalate():
    state = {"conflict_report": {"flags": [], "summary": "ok"}}
    ok, msg = validate_after_a4(state)
    assert ok is False
    assert "escalate" in msg

def test_a4_escalate_wrong_type():
    state = {"conflict_report": {
        "flags": [], "escalate": "yes", "summary": "ok"
    }}
    ok, msg = validate_after_a4(state)
    assert ok is False


# ── A5 ──────────────────────────────────────────────

def test_a5_valid():
    state = {"suitability_report": {
        "decision": "SUITABLE",
        "summary": "ok",
        "rule_findings": _findings(),
        "flags_addressed": [],
        "regulatory_basis": "Article 25(2)",
        "client_facing_summary": "You are suitable.",
    }}
    ok, msg = validate_after_a5(state)
    assert ok is True

def test_a5_invalid_decision():
    state = {"suitability_report": {
        "decision": "UNKNOWN",
        "summary": "x",
        "rule_findings": _findings(),
        "flags_addressed": [],
        "regulatory_basis": "x",
        "client_facing_summary": "x",
    }}
    ok, msg = validate_after_a5(state)
    assert ok is False

def test_a5_wrong_rule_findings_count():
    state = {"suitability_report": {
        "decision": "SUITABLE",
        "summary": "x",
        "rule_findings": _findings()[:3],
        "flags_addressed": [],
        "regulatory_basis": "x",
        "client_facing_summary": "x",
    }}
    ok, msg = validate_after_a5(state)
    assert ok is False
    assert "7" in msg