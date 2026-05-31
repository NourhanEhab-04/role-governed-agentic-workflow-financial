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
    # HIGH-vuln client + concentration > 40 + risk_class >= 2 → triggered
    client = {"portfolio_concentration_pct": 41, "financial_vulnerability": "HIGH"}
    product = {"risk_class": 3}
    result = check_concentration_risk(client, product)
    assert result["triggered"] is True
    assert result["severity"] == "HIGH"

def test_concentration_triggered_extreme_concentration_high_risk():
    # Extreme concentration (> 60) in risk_class >= 6 product → triggered regardless of vulnerability
    client = {"portfolio_concentration_pct": 65, "financial_vulnerability": "LOW"}
    product = {"risk_class": 6}
    result = check_concentration_risk(client, product)
    assert result["triggered"] is True

def test_concentration_not_triggered_at_threshold():
    # Exactly 60% concentration is NOT strictly greater than 60 → no trigger
    client = {"portfolio_concentration_pct": 60, "financial_vulnerability": "MEDIUM"}
    product = {"risk_class": 6}
    assert check_concentration_risk(client, product)["triggered"] is False

def test_concentration_not_triggered_missing_key():
    # missing key → defaults to 0, no trigger
    assert check_concentration_risk({})["triggered"] is False


# --- check_contradiction ---

def test_contradiction_triggered_high_vuln_suitable():
    # HIGH-vuln + SUITABLE verdict on risk_class >= 4 product → contradiction
    client = {"financial_vulnerability": "HIGH"}
    verdict = {"score": 75, "decision": "SUITABLE"}
    product = {"risk_class": 4}
    result = check_contradiction(client, verdict, product)
    assert result["triggered"] is True
    assert result["severity"] == "HIGH"

def test_contradiction_not_triggered_high_vuln_suitable_low_risk():
    # HIGH-vuln + SUITABLE on risk_class 3 (low risk) → no contradiction; safe product
    client = {"financial_vulnerability": "HIGH"}
    verdict = {"score": 85, "decision": "SUITABLE"}
    product = {"risk_class": 3}
    assert check_contradiction(client, verdict, product)["triggered"] is False

def test_contradiction_not_triggered_high_vuln_unsuitable():
    client = {"financial_vulnerability": "HIGH"}
    verdict = {"score": 30, "decision": "UNSUITABLE"}
    product = {"risk_class": 5}
    assert check_contradiction(client, verdict, product)["triggered"] is False

def test_contradiction_not_triggered_normal_client_suitable():
    client = {"financial_vulnerability": "LOW"}
    verdict = {"score": 80, "decision": "SUITABLE"}
    product = {"risk_class": 5}
    assert check_contradiction(client, verdict, product)["triggered"] is False


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