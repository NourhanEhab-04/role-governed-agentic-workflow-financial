# tests/test_validators_audit_cross_check.py
"""Zero-LLM tests for validate_after_a4 audit_verdict cross-check."""
import pytest
from orchestrator.validators import validate_after_a4

GOOD_REPORT = {
    "flags": [],
    "escalate": False,
    "summary": "No escalation required.",
}

ESCALATED_REPORT = {
    "flags": ["rule_engine_disagreement"],
    "escalate": True,
    "summary": "Escalation required.",
}


def test_pass_when_agreed_and_no_escalation():
    state = {
        "conflict_report": GOOD_REPORT,
        "audit_verdict": {"agreed": True},
    }
    ok, err = validate_after_a4(state)
    assert ok is True


def test_pass_when_disagreed_and_escalated():
    state = {
        "conflict_report": ESCALATED_REPORT,
        "audit_verdict": {"agreed": False},
    }
    ok, err = validate_after_a4(state)
    assert ok is True


def test_fail_when_disagreed_but_not_escalated():
    state = {
        "conflict_report": GOOD_REPORT,  # escalate=False
        "audit_verdict": {"agreed": False},
    }
    ok, err = validate_after_a4(state)
    assert ok is False
    assert "disagreement" in err.lower()


def test_pass_when_audit_verdict_absent():
    """audit_verdict not yet written — cross-check skipped."""
    state = {
        "conflict_report": GOOD_REPORT,
        "audit_verdict": None,
    }
    ok, err = validate_after_a4(state)
    assert ok is True


def test_fail_when_conflict_report_missing():
    state = {"conflict_report": None, "audit_verdict": {"agreed": True}}
    ok, err = validate_after_a4(state)
    assert ok is False