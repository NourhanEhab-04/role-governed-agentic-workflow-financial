# tests/test_verifier_parse.py
"""
Unit tests for parse_verification_result and validate_verification_result.
No LLM calls — all inputs are hand-crafted strings.
"""

import json
import pytest

from agents.verifier_agent import parse_verification_result
from schemas.verification_result import validate_verification_result, CONFIDENCE_THRESHOLD


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_raw(passed=True, confidence=0.9, issues=None, field_checks=None):
    """Return a raw JSON string that looks like verifier output."""
    obj = {
        "passed": passed,
        "confidence": confidence,
        "issues": issues if issues is not None else [],
        "field_checks": field_checks if field_checks is not None else {
            "financial_knowledge": {"supported": True, "reason": "Client said 'basic stocks'."},
            "risk_tolerance_score": {"supported": True, "reason": "Client said score 5."},
        },
    }
    return json.dumps(obj)


# ── validate_verification_result ─────────────────────────────────────────────

def test_validate_ok():
    result = json.loads(_make_raw())
    ok, msg = validate_verification_result(result)
    assert ok is True
    assert msg == ""


def test_validate_not_dict():
    ok, msg = validate_verification_result("not a dict")
    assert ok is False
    assert "not a dict" in msg


def test_validate_missing_key():
    result = {"passed": True, "confidence": 0.9, "issues": []}
    # missing field_checks
    ok, msg = validate_verification_result(result)
    assert ok is False
    assert "field_checks" in msg


def test_validate_passed_wrong_type():
    result = json.loads(_make_raw())
    result["passed"] = "yes"
    ok, msg = validate_verification_result(result)
    assert ok is False
    assert "passed" in msg


def test_validate_confidence_out_of_range_high():
    result = json.loads(_make_raw(confidence=1.5))
    ok, msg = validate_verification_result(result)
    assert ok is False
    assert "confidence" in msg


def test_validate_confidence_out_of_range_low():
    result = json.loads(_make_raw(confidence=-0.1))
    ok, msg = validate_verification_result(result)
    assert ok is False
    assert "confidence" in msg


def test_validate_issues_wrong_type():
    result = json.loads(_make_raw())
    result["issues"] = "not a list"
    ok, msg = validate_verification_result(result)
    assert ok is False
    assert "issues" in msg


def test_validate_field_checks_wrong_type():
    result = json.loads(_make_raw())
    result["field_checks"] = ["list instead of dict"]
    ok, msg = validate_verification_result(result)
    assert ok is False
    assert "field_checks" in msg


def test_validate_field_check_entry_missing_supported():
    result = json.loads(_make_raw())
    result["field_checks"]["financial_knowledge"] = {"reason": "ok"}
    ok, msg = validate_verification_result(result)
    assert ok is False
    assert "supported" in msg


def test_validate_field_check_entry_supported_wrong_type():
    result = json.loads(_make_raw())
    result["field_checks"]["financial_knowledge"] = {"supported": "yes", "reason": "ok"}
    ok, msg = validate_verification_result(result)
    assert ok is False
    assert "supported" in msg


def test_validate_field_check_entry_missing_reason():
    result = json.loads(_make_raw())
    result["field_checks"]["financial_knowledge"] = {"supported": True}
    ok, msg = validate_verification_result(result)
    assert ok is False
    assert "reason" in msg


# ── parse_verification_result ─────────────────────────────────────────────────

def test_parse_clean_json():
    result = parse_verification_result(_make_raw())
    assert result["passed"] is True
    assert result["confidence"] == 0.9
    assert isinstance(result["issues"], list)
    assert isinstance(result["field_checks"], dict)


def test_parse_json_inside_markdown_fence():
    raw = f"```json\n{_make_raw()}\n```"
    result = parse_verification_result(raw)
    assert result["passed"] is True


def test_parse_json_with_surrounding_prose():
    raw = f"Here is my verification:\n{_make_raw()}\nDone."
    result = parse_verification_result(raw)
    assert result["confidence"] == 0.9


def test_parse_no_json_raises():
    with pytest.raises(ValueError, match="No JSON object found"):
        parse_verification_result("I could not verify this.")


def test_parse_malformed_json_raises():
    # json_repair may partially recover the JSON; Pydantic then rejects the
    # invalid field values — either way a ValueError must be raised.
    with pytest.raises(ValueError):
        parse_verification_result('{"passed": true, "confidence":}')


def test_parse_missing_field_checks_defaults_to_empty_dict():
    # field_checks is optional in the model (default_factory=dict)
    bad = {"passed": True, "confidence": 0.9, "issues": []}
    result = parse_verification_result(json.dumps(bad))
    assert result["field_checks"] == {}


def test_parse_low_confidence_forces_passed_false():
    """If confidence < threshold, passed must be forced to False even if LLM says True."""
    raw = _make_raw(passed=True, confidence=CONFIDENCE_THRESHOLD - 0.01)
    result = parse_verification_result(raw)
    assert result["passed"] is False
    assert any("forced" in issue for issue in result["issues"])


def test_parse_low_confidence_already_false_unchanged():
    """If confidence < threshold and passed is already False, issues list unchanged by force."""
    raw = _make_raw(passed=False, confidence=0.5, issues=["something wrong"])
    result = parse_verification_result(raw)
    assert result["passed"] is False
    # Only the original issue should be present (no duplicate "forced" message)
    assert len(result["issues"]) == 1


def test_parse_high_confidence_passed_true_unchanged():
    raw = _make_raw(passed=True, confidence=0.95)
    result = parse_verification_result(raw)
    assert result["passed"] is True
    assert result["issues"] == []


def test_parse_with_issues_list():
    raw = _make_raw(passed=False, confidence=0.5, issues=["income seems wrong"])
    result = parse_verification_result(raw)
    assert result["passed"] is False
    assert "income seems wrong" in result["issues"]


def test_parse_field_checks_preserved():
    field_checks = {
        "risk_tolerance_score": {"supported": False, "reason": "Client said 3 but 7 was returned."}
    }
    raw = _make_raw(passed=False, confidence=0.4, field_checks=field_checks)
    result = parse_verification_result(raw)
    assert result["field_checks"]["risk_tolerance_score"]["supported"] is False
