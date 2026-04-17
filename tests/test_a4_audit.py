# tests/test_a4_audit.py

import pytest
from agents.conflict_detector import (
    check_borderline,
    check_concentration_risk,
    check_contradiction,
    check_escalation_trigger,
)


# --- check_borderline ---

def test_borderline_triggered_at_lower_bound():
    verdict = {"score": 40, "decision": "CONDITIONAL"}
    result = check_borderline(verdict)
    assert result["triggered"] is True
    assert result["severity"] == "LOW"
    assert result["rule_id"] == "BORDERLINE"

def test_borderline_triggered_mid_zone():
    verdict = {"score": 50, "decision": "CONDITIONAL"}
    assert check_borderline(verdict)["triggered"] is True

def test_borderline_not_triggered_above_zone():
    verdict = {"score": 56, "decision": "CONDITIONAL"}
    assert check_borderline(verdict)["triggered"] is False

def test_borderline_not_triggered_unsuitable():
    verdict = {"score": 35, "decision": "UNSUITABLE"}
    assert check_borderline(verdict)["triggered"] is False


# --- check_concentration_risk ---

def test_concentration_triggered_above_threshold():
    client = {"single_asset_concentration_pct": 41}
    result = check_concentration_risk(client)
    assert result["triggered"] is True
    assert result["severity"] == "LOW"

def test_concentration_not_triggered_at_threshold():
    client = {"single_asset_concentration_pct": 40}
    assert check_concentration_risk(client)["triggered"] is False

def test_concentration_not_triggered_missing_key():
    # missing key → defaults to 0, no trigger
    assert check_concentration_risk({})["triggered"] is False


# --- check_contradiction ---

def test_contradiction_triggered_high_vuln_suitable():
    client = {"vulnerability_status": "HIGH"}
    verdict = {"score": 75, "decision": "SUITABLE"}
    result = check_contradiction(client, verdict)
    assert result["triggered"] is True
    assert result["severity"] == "HIGH"

def test_contradiction_not_triggered_high_vuln_unsuitable():
    client = {"vulnerability_status": "HIGH"}
    verdict = {"score": 30, "decision": "UNSUITABLE"}
    assert check_contradiction(client, verdict)["triggered"] is False

def test_contradiction_not_triggered_normal_client_suitable():
    client = {"vulnerability_status": "NONE"}
    verdict = {"score": 80, "decision": "SUITABLE"}
    assert check_contradiction(client, verdict)["triggered"] is False


# --- check_escalation_trigger ---

def test_escalation_triggered_by_one_high_flag():
    flags = [
        {"rule_id": "CONTRADICTION", "triggered": True, "severity": "HIGH", "message": "x"},
    ]
    result = check_escalation_trigger(flags)
    assert result["triggered"] is True

def test_escalation_not_triggered_by_low_flags_only():
    flags = [
        {"rule_id": "BORDERLINE", "triggered": True, "severity": "LOW", "message": "x"},
        {"rule_id": "CONCENTRATION", "triggered": True, "severity": "LOW", "message": "x"},
    ]
    assert check_escalation_trigger(flags)["triggered"] is False