# tests/test_a5_parse.py

import json
import pytest
from agents.disclosure_agent import parse_suitability_report


def make_findings(rule_ids=None):
    ids = rule_ids or ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
    return [
        {"rule_id": r, "status": "PASS", "explanation": f"{r} passed."}
        for r in ids
    ]


def make_report(**overrides):
    base = {
        "decision": "SUITABLE",
        "summary": "Client is suitable for the product.",
        "rule_findings": make_findings(),
        "flags_addressed": [],
        "regulatory_basis": "MiFID II Article 25(2) — suitability assessment.",
        "client_facing_summary": "Based on your profile, this product suits your needs.",
    }
    base.update(overrides)
    return json.dumps(base)


# --- valid cases ---

def test_valid_suitable_report():
    result = parse_suitability_report(make_report())
    assert result["decision"] == "SUITABLE"
    assert len(result["rule_findings"]) == 7

def test_valid_escalated_with_flags():
    report = make_report(
        decision="ESCALATED",
        flags_addressed=[
            {"rule_id": "CONTRADICTION", "explanation": "Referred for human review."}
        ]
    )
    result = parse_suitability_report(report)
    assert result["decision"] == "ESCALATED"
    assert len(result["flags_addressed"]) == 1

def test_valid_empty_flags_addressed():
    result = parse_suitability_report(make_report(flags_addressed=[]))
    assert result["flags_addressed"] == []


# --- invalid decision ---

def test_missing_decision_key():
    raw = json.dumps({
        "summary": "x", "rule_findings": make_findings(),
        "flags_addressed": [], "regulatory_basis": "x",
        "client_facing_summary": "x"
    })
    with pytest.raises(ValueError, match="decision"):
        parse_suitability_report(raw)

def test_wrong_decision_value():
    with pytest.raises(ValueError, match="Invalid decision"):
        parse_suitability_report(make_report(decision="MAYBE"))


# --- rule_findings validation ---

def test_fewer_than_7_rule_findings():
    with pytest.raises(ValueError, match="exactly 7"):
        parse_suitability_report(make_report(rule_findings=make_findings(["R1", "R2"])))

def test_duplicate_rule_ids():
    findings = make_findings(["R1", "R1", "R2", "R3", "R4", "R5", "R6"])
    with pytest.raises(ValueError, match="Duplicate"):
        parse_suitability_report(make_report(rule_findings=findings))

def test_invalid_rule_id():
    findings = make_findings(["R1", "R2", "R3", "R4", "R5", "R6", "R8"])
    with pytest.raises(ValueError, match="invalid rule_id"):
        parse_suitability_report(make_report(rule_findings=findings))

def test_finding_missing_explanation_key():
    findings = make_findings()
    findings[0] = {"rule_id": "R1", "status": "PASS"}  # missing explanation
    with pytest.raises(ValueError, match="explanation"):
        parse_suitability_report(make_report(rule_findings=findings))

def test_invalid_json():
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_suitability_report("not json")