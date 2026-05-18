import pytest
import json
from agents.rule_engine_agent import parse_rule_verdict

VALID_VERDICT = """{
    "score": 80,
    "decision": "SUITABLE",
    "rules": {
        "R1": "PASS",
        "R2": "PASS",
        "R3": "FAIL",
        "R4": "PASS",
        "R5": "PASS",
        "R6": "PASS",
        "R7": "PASS"
    }
}"""


def test_clean_verdict_parses_correctly():
    result = parse_rule_verdict(VALID_VERDICT)
    assert result["score"] == 80
    assert result["decision"] == "SUITABLE"
    assert result["rules"]["R3"] == "FAIL"
    assert len(result["rules"]) == 7


def test_verdict_inside_markdown_fence():
    fenced = f"Tool result:\n```json\n{VALID_VERDICT}\n```"
    result = parse_rule_verdict(fenced)
    assert result["score"] == 80


def test_missing_top_level_key_raises():
    incomplete = json.loads(VALID_VERDICT)
    del incomplete["score"]
    with pytest.raises(ValueError, match="score"):
        parse_rule_verdict(json.dumps(incomplete))


def test_no_json_raises():
    with pytest.raises(ValueError, match="No JSON object found"):
        parse_rule_verdict("The tool returned an error.")


def test_malformed_json_raises():
    # json_repair may partially recover the JSON; Pydantic then rejects the
    # invalid field values — either way a ValueError must be raised.
    with pytest.raises(ValueError):
        parse_rule_verdict('{"score": 80, "decision":}')


def test_invalid_decision_raises():
    bad = json.loads(VALID_VERDICT)
    bad["decision"] = "MAYBE"
    with pytest.raises(ValueError, match="Invalid decision"):
        parse_rule_verdict(json.dumps(bad))


def test_score_not_int_raises():
    bad = json.loads(VALID_VERDICT)
    bad["score"] = "80"
    with pytest.raises(ValueError, match="score must be an integer"):
        parse_rule_verdict(json.dumps(bad))


def test_missing_rule_id_raises():
    bad = json.loads(VALID_VERDICT)
    del bad["rules"]["R7"]
    with pytest.raises(ValueError, match="missing rule IDs"):
        parse_rule_verdict(json.dumps(bad))


def test_invalid_rule_result_raises():
    bad = json.loads(VALID_VERDICT)
    bad["rules"]["R1"] = "MAYBE"
    with pytest.raises(ValueError, match="rules"):
        parse_rule_verdict(json.dumps(bad))


def test_rules_not_dict_raises():
    bad = json.loads(VALID_VERDICT)
    bad["rules"] = "all pass"
    with pytest.raises(ValueError, match="rules must be a dict"):
        parse_rule_verdict(json.dumps(bad))
