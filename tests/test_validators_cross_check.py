# tests/test_validators_cross_check.py
"""
Zero-LLM tests for the new cross-check logic in validate_after_a3.
"""
import pytest
from orchestrator.validators import validate_after_a3

GOOD_VERDICT = {
    "score": 75,
    "decision": "SUITABLE",
    "rules": [],
    "failed_rules": [],
}

PRE_CHECK_MATCHING = {"decision": "SUITABLE", "score": 75}
PRE_CHECK_MISMATCHING = {"decision": "UNSUITABLE", "score": 20}


def test_pass_when_both_agree():
    state = {
        "rule_verdict": GOOD_VERDICT,
        "pre_check_verdict": PRE_CHECK_MATCHING,
    }
    ok, err = validate_after_a3(state)
    assert ok is True


def test_fail_when_decisions_disagree():
    state = {
        "rule_verdict": GOOD_VERDICT,
        "pre_check_verdict": PRE_CHECK_MISMATCHING,
    }
    ok, err = validate_after_a3(state)
    assert ok is False
    assert "bypass" in err.lower() or "!=" in err


def test_pass_when_pre_check_absent():
    """If A0 couldn't run pre_check, cross-check is skipped — not a hard failure."""
    state = {
        "rule_verdict": GOOD_VERDICT,
        "pre_check_verdict": None,
    }
    ok, err = validate_after_a3(state)
    assert ok is True


def test_fail_when_rule_verdict_missing():
    state = {"rule_verdict": None, "pre_check_verdict": PRE_CHECK_MATCHING}
    ok, err = validate_after_a3(state)
    assert ok is False


def test_fail_when_decision_invalid():
    state = {
        "rule_verdict": {**GOOD_VERDICT, "decision": "MAYBE"},
        "pre_check_verdict": PRE_CHECK_MATCHING,
    }
    ok, err = validate_after_a3(state)
    assert ok is False