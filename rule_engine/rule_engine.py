from schemas.client_profile import REQUIRED_CLIENT_KEYS
from schemas.product_profile import REQUIRED_PRODUCT_KEYS as _ALL_PRODUCT_KEYS

# Rule engine only needs the fields it evaluates — product_name is metadata
REQUIRED_PRODUCT_KEYS = _ALL_PRODUCT_KEYS - {"product_name"}

KNOWLEDGE_LEVELS = {
    "none": 1,
    "basic": 3,
    "moderate": 6,
    "advanced": 9,
}


def _validate_inputs(client: dict, product: dict) -> None:
    missing_client = REQUIRED_CLIENT_KEYS - client.keys()
    if missing_client:
        raise ValueError(f"client dict is missing required keys: {missing_client}")
    missing_product = REQUIRED_PRODUCT_KEYS - product.keys()
    if missing_product:
        raise ValueError(f"product dict is missing required keys: {missing_product}")


def _r1_knowledge(client: dict, product: dict) -> dict:
    client_level = KNOWLEDGE_LEVELS[client["financial_knowledge"]]
    required_level = KNOWLEDGE_LEVELS[product["requires_knowledge_level"]]
    passed = client_level >= required_level
    return {
        "rule": "R1",
        "pass": passed,
        "penalty": 0 if passed else -25,
        "detail": (
            f"client knowledge '{client['financial_knowledge']}' ({client_level}) "
            f">= product requirement '{product['requires_knowledge_level']}' ({required_level})"
            if passed else
            f"client knowledge '{client['financial_knowledge']}' ({client_level}) "
            f"< product requirement '{product['requires_knowledge_level']}' ({required_level})"
        ),
    }


def _r2_risk_alignment(client: dict, product: dict) -> dict:
    risk_class = product["risk_class"]
    tolerance = client["risk_tolerance_score"]
    passed = risk_class <= tolerance + 2
    return {
        "rule": "R2",
        "pass": passed,
        "penalty": 0 if passed else -30,
        "detail": (
            f"risk_class {risk_class} <= risk_tolerance_score {tolerance} + 2 ({tolerance + 2})"
            if passed else
            f"risk_class {risk_class} > risk_tolerance_score {tolerance} + 2 ({tolerance + 2})"
        ),
    }


def _r3_horizon(client: dict, product: dict) -> dict:
    horizon = client["investment_horizon"]
    min_horizon = product["minimum_horizon"]
    passed = horizon >= min_horizon
    return {
        "rule": "R3",
        "pass": passed,
        "penalty": 0 if passed else -20,
        "detail": (
            f"investment_horizon {horizon} >= minimum_horizon {min_horizon}"
            if passed else
            f"investment_horizon {horizon} < minimum_horizon {min_horizon}"
        ),
    }


def _r4_affordability(client: dict, product: dict) -> dict:
    passed = not (client["can_afford_total_loss"] is False and product["potential_loss"] == "total")
    return {
        "rule": "R4",
        "pass": passed,
        "penalty": 0 if passed else -35,
        "detail": (
            "client can afford total loss or product does not risk total loss"
            if passed else
            "client cannot afford total loss but product has potential_loss='total'"
        ),
    }


def _r5_vulnerability(client: dict, product: dict) -> dict:
    risk_class = product["risk_class"]
    passed = not (client["financial_vulnerability"] == "HIGH" and risk_class >= 5)
    return {
        "rule": "R5",
        "pass": passed,
        "penalty": 0 if passed else -25,
        "detail": (
            f"client is not HIGH vulnerability or product risk_class {risk_class} < 5"
            if passed else
            f"client is HIGH vulnerability and product risk_class {risk_class} >= 5"
        ),
    }


COMPLEX_MIN_KNOWLEDGE = {"moderate", "advanced"}


def _r7_complexity(client: dict, product: dict) -> dict:
    passed = not (
        product["complexity_tier"] == "COMPLEX"
        and client["financial_knowledge"] not in COMPLEX_MIN_KNOWLEDGE
    )
    return {
        "rule": "R7",
        "pass": passed,
        "penalty": 0 if passed else -20,
        "detail": (
            f"product complexity_tier='{product['complexity_tier']}' or "
            f"client knowledge '{client['financial_knowledge']}' meets COMPLEX threshold"
            if passed else
            f"product is COMPLEX but client knowledge '{client['financial_knowledge']}' "
            f"is below moderate"
        ),
    }


def _r6_leverage(client: dict, product: dict) -> dict:
    tolerance = client["risk_tolerance_score"]
    passed = not (product["leverage"] is True and tolerance < 7)
    return {
        "rule": "R6",
        "pass": passed,
        "penalty": 0 if passed else -30,
        "detail": (
            f"product is not leveraged or risk_tolerance_score {tolerance} >= 7"
            if passed else
            f"product is leveraged and risk_tolerance_score {tolerance} < 7"
        ),
    }


def _score_to_decision(score: int) -> str:
    if score >= 70:
        return "SUITABLE"
    elif score >= 40:
        return "CONDITIONAL"
    return "UNSUITABLE"


def evaluate_suitability(client: dict, product: dict) -> dict:
    _validate_inputs(client, product)

    rules = [
        _r1_knowledge(client, product),
        _r2_risk_alignment(client, product),
        _r3_horizon(client, product),
        _r4_affordability(client, product),
        _r5_vulnerability(client, product),
        _r6_leverage(client, product),
        _r7_complexity(client, product),
    ]

    score = 100 + sum(r["penalty"] for r in rules)

    return {
        "score": score,
        "decision": _score_to_decision(score),
        "rules": rules,
    }


if __name__ == "__main__":
    import json
    sample_client = {
        "financial_knowledge": "basic",
        "risk_tolerance_score": 4,
        "investment_horizon": 3,
        "liquid_assets": 8000,
        "income": 42000,
        "investment_amount": 5000,
        "can_afford_total_loss": False,
        "financial_vulnerability": "LOW",
    }
    sample_product = {
        "risk_class": 4,
        "complexity_tier": "NON-COMPLEX",
        "requires_knowledge_level": "basic",
        "minimum_horizon": 2,
        "potential_loss": "partial",
        "leverage": False,
    }
    print(json.dumps(evaluate_suitability(sample_client, sample_product), indent=2))
