# tests/test_a4_parse.py

import pytest
from agents.conflict_detector import parse_conflict_report


def _valid_report(**overrides):
    base = {
        "flags": [
            {
                "rule_id": "BORDERLINE",
                "triggered": True,
                "severity": "LOW",
                "message": "Score 48 is in borderline zone.",
            }
        ],
        "escalate": False,
        "summary": "One low-severity flag detected.",
    }
    base.update(overrides)
    return base


def test_parse_valid_full_report():
    result = parse_conflict_report(_valid_report())
    assert result["escalate"] is False
    assert len(result["flags"]) == 1


def test_parse_empty_flags_no_escalate_is_valid():
    report = _valid_report(flags=[], escalate=False, summary="Clean run.")
    result = parse_conflict_report(report)
    assert result["flags"] == []


def test_parse_missing_escalate_raises():
    report = _valid_report()
    del report["escalate"]
    with pytest.raises(ValueError, match="escalate"):
        parse_conflict_report(report)


def test_parse_missing_flags_raises():
    report = _valid_report()
    del report["flags"]
    with pytest.raises(ValueError, match="flags"):
        parse_conflict_report(report)


def test_parse_invalid_severity_raises():
    report = _valid_report(flags=[{
        "rule_id": "X", "triggered": True,
        "severity": "MEDIUM",  # invalid
        "message": "test",
    }])
    with pytest.raises(ValueError, match="severity"):
        parse_conflict_report(report)


def test_parse_escalate_true_with_empty_flags_raises():
    report = _valid_report(flags=[], escalate=True)
    with pytest.raises(ValueError, match="escalate=True"):
        parse_conflict_report(report)


def test_parse_flag_missing_key_raises():
    report = _valid_report(flags=[{
        "rule_id": "X",
        "triggered": True,
        # missing severity and message
    }])
    with pytest.raises(ValueError, match="missing keys"):
        parse_conflict_report(report)


def test_parse_non_bool_escalate_raises():
    report = _valid_report(escalate="yes")
    with pytest.raises(ValueError, match="bool"):
        parse_conflict_report(report)