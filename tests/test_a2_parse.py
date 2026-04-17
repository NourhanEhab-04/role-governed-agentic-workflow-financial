import json
import pytest
from agents.product_classifier import parse_product_profile

VALID_JSON = """{
    "product_name": "Vanguard FTSE All-World ETF",
    "risk_class": 4,
    "complexity_tier": "NON-COMPLEX",
    "requires_knowledge_level": "basic",
    "minimum_horizon": 3,
    "potential_loss": "partial",
    "leverage": false
}"""


def test_clean_json_parses_correctly():
    result = parse_product_profile(VALID_JSON)
    assert result["product_name"] == "Vanguard FTSE All-World ETF"
    assert result["risk_class"] == 4
    assert result["complexity_tier"] == "NON-COMPLEX"
    assert result["leverage"] is False


def test_json_inside_markdown_fence():
    fenced = f"Here is the product classification:\n```json\n{VALID_JSON}\n```"
    result = parse_product_profile(fenced)
    assert result["minimum_horizon"] == 3


def test_json_with_surrounding_prose():
    prose = f"Based on the product description:\n{VALID_JSON}\nClassification complete."
    result = parse_product_profile(prose)
    assert result["potential_loss"] == "partial"


def test_missing_key_raises_value_error():
    incomplete = json.loads(VALID_JSON)
    del incomplete["risk_class"]
    with pytest.raises(ValueError, match="missing required keys"):
        parse_product_profile(json.dumps(incomplete))


def test_no_json_raises_value_error():
    with pytest.raises(ValueError, match="No JSON object found"):
        parse_product_profile("I could not classify this product.")


def test_malformed_json_raises_value_error():
    with pytest.raises(ValueError, match="malformed JSON"):
        parse_product_profile('{"risk_class": 4, "complexity_tier":}')


def test_invalid_complexity_tier_raises_value_error():
    bad = json.loads(VALID_JSON)
    bad["complexity_tier"] = "SEMI-COMPLEX"
    with pytest.raises(ValueError, match="Invalid complexity_tier"):
        parse_product_profile(json.dumps(bad))


def test_invalid_knowledge_level_raises_value_error():
    bad = json.loads(VALID_JSON)
    bad["requires_knowledge_level"] = "expert"
    with pytest.raises(ValueError, match="Invalid requires_knowledge_level"):
        parse_product_profile(json.dumps(bad))


def test_invalid_potential_loss_raises_value_error():
    bad = json.loads(VALID_JSON)
    bad["potential_loss"] = "severe"
    with pytest.raises(ValueError, match="Invalid potential_loss"):
        parse_product_profile(json.dumps(bad))


def test_risk_class_out_of_range_raises_value_error():
    bad = json.loads(VALID_JSON)
    bad["risk_class"] = 8
    with pytest.raises(ValueError, match="risk_class must be an integer between 1 and 7"):
        parse_product_profile(json.dumps(bad))


def test_leverage_not_bool_raises_value_error():
    bad = json.loads(VALID_JSON)
    bad["leverage"] = "yes"
    with pytest.raises(ValueError, match="leverage must be a boolean"):
        parse_product_profile(json.dumps(bad))


def test_risk_class_as_float_raises_value_error():
    bad = json.loads(VALID_JSON)
    bad["risk_class"] = 4.5
    with pytest.raises(ValueError, match="risk_class must be an integer"):
        parse_product_profile(json.dumps(bad))
