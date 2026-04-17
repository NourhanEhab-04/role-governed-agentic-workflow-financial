import json
import pytest
from agents.client_profiler import parse_client_profile

VALID_JSON = """{
    "financial_knowledge": "basic",
    "risk_tolerance_score": 4,
    "investment_horizon": 3,
    "liquid_assets": 8000.0,
    "income": 42000.0,
    "investment_amount": 5000.0,
    "can_afford_total_loss": false,
    "financial_vulnerability": "LOW"
}"""


def test_clean_json_parses_correctly():
    result = parse_client_profile(VALID_JSON)
    assert result["financial_knowledge"] == "basic"
    assert result["risk_tolerance_score"] == 4
    assert result["can_afford_total_loss"] is False


def test_json_inside_markdown_fence():
    fenced = f"Here is the profile:\n```json\n{VALID_JSON}\n```"
    result = parse_client_profile(fenced)
    assert result["investment_horizon"] == 3


def test_json_with_surrounding_prose():
    prose = f"Based on the input, I extracted:\n{VALID_JSON}\nLet me know if anything is wrong."
    result = parse_client_profile(prose)
    assert result["liquid_assets"] == 8000.0


def test_missing_key_raises_value_error():
    incomplete = json.loads(VALID_JSON)
    del incomplete["financial_knowledge"]
    with pytest.raises(ValueError, match="missing required keys"):
        parse_client_profile(json.dumps(incomplete))


def test_no_json_raises_value_error():
    with pytest.raises(ValueError, match="No JSON object found"):
        parse_client_profile("Sorry, I could not extract a profile.")


def test_malformed_json_raises_value_error():
    with pytest.raises(ValueError, match="malformed JSON"):
        parse_client_profile('{"financial_knowledge": "basic", "risk_tolerance_score":}')


def test_invalid_knowledge_enum_raises_value_error():
    bad = json.loads(VALID_JSON)
    bad["financial_knowledge"] = "expert"
    with pytest.raises(ValueError, match="Invalid financial_knowledge"):
        parse_client_profile(json.dumps(bad))


def test_invalid_vulnerability_enum_raises_value_error():
    bad = json.loads(VALID_JSON)
    bad["financial_vulnerability"] = "VERY_HIGH"
    with pytest.raises(ValueError, match="Invalid financial_vulnerability"):
        parse_client_profile(json.dumps(bad))
