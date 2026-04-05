import pytest
from rule_engine.rule_engine import evaluate_suitability


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_rule(rules: list, name: str) -> dict:
    return next(r for r in rules if r["rule"] == name)


# ---------------------------------------------------------------------------
# Fixtures — all rules pass by default
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return dict(
        financial_knowledge="advanced",   # 9 — beats any requirement
        risk_tolerance_score=8,           # 8+2=10 — beats any risk_class ≤10
        investment_horizon=36,
        liquid_assets=50000,
        income=80000,
        investment_amount=5000,
        can_afford_total_loss=True,
        financial_vulnerability="LOW",
    )


@pytest.fixture
def product():
    return dict(
        risk_class=1,
        complexity_tier=1,
        requires_knowledge_level="none",  # 1 — beaten by any client level
        minimum_horizon=6,
        potential_loss="partial",
        leverage=False,
    )


# ---------------------------------------------------------------------------
# 1. All-PASS
# ---------------------------------------------------------------------------

def test_all_rules_pass(client, product):
    r = evaluate_suitability(client, product)
    assert r["score"] == 100
    assert r["decision"] == "SUITABLE"
    assert all(rule["pass"] for rule in r["rules"])
    assert all(rule["penalty"] == 0 for rule in r["rules"])


# ---------------------------------------------------------------------------
# 2–7. Single-rule failures
# ---------------------------------------------------------------------------

def test_r1_fail_knowledge_below_requirement(client, product):
    # none (1) < advanced (9)  →  R1 FAIL,  100 - 25 = 75
    r = evaluate_suitability(
        {**client, "financial_knowledge": "none"},
        {**product, "requires_knowledge_level": "advanced"},
    )
    assert r["score"] == 75
    assert r["decision"] == "SUITABLE"
    assert get_rule(r["rules"], "R1")["pass"] is False
    assert get_rule(r["rules"], "R1")["penalty"] == -25


def test_r2_fail_risk_class_exceeds_tolerance(client, product):
    # risk_class 11 > tolerance 8 + 2 = 10  →  R2 FAIL,  100 - 30 = 70
    r = evaluate_suitability(client, {**product, "risk_class": 11})
    assert r["score"] == 70
    assert r["decision"] == "SUITABLE"
    assert get_rule(r["rules"], "R2")["pass"] is False
    assert get_rule(r["rules"], "R2")["penalty"] == -30


def test_r3_fail_horizon_too_short(client, product):
    # horizon 3 < min_horizon 12  →  R3 FAIL,  100 - 20 = 80
    r = evaluate_suitability(
        {**client, "investment_horizon": 3},
        {**product, "minimum_horizon": 12},
    )
    assert r["score"] == 80
    assert r["decision"] == "SUITABLE"
    assert get_rule(r["rules"], "R3")["pass"] is False
    assert get_rule(r["rules"], "R3")["penalty"] == -20


def test_r4_fail_cannot_afford_total_loss(client, product):
    # can_afford_total_loss=False + potential_loss='total'  →  R4 FAIL,  100 - 35 = 65
    r = evaluate_suitability(
        {**client, "can_afford_total_loss": False},
        {**product, "potential_loss": "total"},
    )
    assert r["score"] == 65
    assert r["decision"] == "CONDITIONAL"
    assert get_rule(r["rules"], "R4")["pass"] is False
    assert get_rule(r["rules"], "R4")["penalty"] == -35


def test_r5_fail_high_vulnerability_risky_product(client, product):
    # vulnerability=HIGH + risk_class 6 ≥ 5  →  R5 FAIL,  100 - 25 = 75
    r = evaluate_suitability(
        {**client, "financial_vulnerability": "HIGH"},
        {**product, "risk_class": 6},
    )
    assert r["score"] == 75
    assert r["decision"] == "SUITABLE"
    assert get_rule(r["rules"], "R5")["pass"] is False
    assert get_rule(r["rules"], "R5")["penalty"] == -25


def test_r6_fail_leveraged_low_tolerance(client, product):
    # leverage=True + tolerance 5 < 7  →  R6 FAIL,  100 - 30 = 70
    r = evaluate_suitability(
        {**client, "risk_tolerance_score": 5},
        {**product, "leverage": True},
    )
    assert r["score"] == 70
    assert r["decision"] == "SUITABLE"
    assert get_rule(r["rules"], "R6")["pass"] is False
    assert get_rule(r["rules"], "R6")["penalty"] == -30


# ---------------------------------------------------------------------------
# 8–11. Two-rule failure combinations
# ---------------------------------------------------------------------------

def test_r1_r2_fail(client, product):
    # 100 - 25 - 30 = 45  →  CONDITIONAL
    r = evaluate_suitability(
        {**client, "financial_knowledge": "none", "risk_tolerance_score": 5},
        {**product, "requires_knowledge_level": "advanced", "risk_class": 11},
    )
    assert r["score"] == 45
    assert r["decision"] == "CONDITIONAL"
    assert get_rule(r["rules"], "R1")["pass"] is False
    assert get_rule(r["rules"], "R2")["pass"] is False


def test_r1_r4_fail(client, product):
    # 100 - 25 - 35 = 40  →  CONDITIONAL
    r = evaluate_suitability(
        {**client, "financial_knowledge": "none", "can_afford_total_loss": False},
        {**product, "requires_knowledge_level": "advanced", "potential_loss": "total"},
    )
    assert r["score"] == 40
    assert r["decision"] == "CONDITIONAL"
    assert get_rule(r["rules"], "R1")["pass"] is False
    assert get_rule(r["rules"], "R4")["pass"] is False


def test_r2_r4_fail(client, product):
    # 100 - 30 - 35 = 35  →  UNSUITABLE
    r = evaluate_suitability(
        {**client, "risk_tolerance_score": 5, "can_afford_total_loss": False},
        {**product, "risk_class": 11, "potential_loss": "total"},
    )
    assert r["score"] == 35
    assert r["decision"] == "UNSUITABLE"
    assert get_rule(r["rules"], "R2")["pass"] is False
    assert get_rule(r["rules"], "R4")["pass"] is False


def test_r3_r5_fail(client, product):
    # 100 - 20 - 25 = 55  →  CONDITIONAL
    r = evaluate_suitability(
        {**client, "investment_horizon": 3, "financial_vulnerability": "HIGH"},
        {**product, "minimum_horizon": 12, "risk_class": 6},
    )
    assert r["score"] == 55
    assert r["decision"] == "CONDITIONAL"
    assert get_rule(r["rules"], "R3")["pass"] is False
    assert get_rule(r["rules"], "R5")["pass"] is False


# ---------------------------------------------------------------------------
# 12–14. Boundary cases
# ---------------------------------------------------------------------------

def test_boundary_score_70_suitable(client, product):
    # R6 fails alone: 100 - 30 = 70  →  lowest SUITABLE score
    r = evaluate_suitability(
        {**client, "risk_tolerance_score": 5},
        {**product, "leverage": True},
    )
    assert r["score"] == 70
    assert r["decision"] == "SUITABLE"


def test_boundary_score_40_conditional(client, product):
    # R2 + R6 fail: 100 - 30 - 30 = 40  →  lowest CONDITIONAL score
    r = evaluate_suitability(
        {**client, "risk_tolerance_score": 5},
        {**product, "risk_class": 11, "leverage": True},
    )
    assert r["score"] == 40
    assert r["decision"] == "CONDITIONAL"


def test_boundary_below_40_unsuitable(client, product):
    # R4 + R6 fail: 100 - 35 - 30 = 35  →  UNSUITABLE
    # Score 39 is unreachable — all penalties are multiples of 5;
    # 35 is the nearest achievable score below 40.
    r = evaluate_suitability(
        {**client, "risk_tolerance_score": 5, "can_afford_total_loss": False},
        {**product, "leverage": True, "potential_loss": "total"},
    )
    assert r["score"] == 35
    assert r["score"] < 40
    assert r["decision"] == "UNSUITABLE"


# ---------------------------------------------------------------------------
# 15–17. UNSUITABLE cases (score < 40)
# ---------------------------------------------------------------------------

def test_unsuitable_r1_r2_r3_fail(client, product):
    # 100 - 25 - 30 - 20 = 25
    r = evaluate_suitability(
        {**client, "financial_knowledge": "none", "risk_tolerance_score": 5, "investment_horizon": 3},
        {**product, "requires_knowledge_level": "advanced", "risk_class": 11, "minimum_horizon": 12},
    )
    assert r["score"] == 25
    assert r["decision"] == "UNSUITABLE"


def test_unsuitable_r1_r2_r4_fail(client, product):
    # 100 - 25 - 30 - 35 = 10
    r = evaluate_suitability(
        {**client, "financial_knowledge": "none", "risk_tolerance_score": 5, "can_afford_total_loss": False},
        {**product, "requires_knowledge_level": "advanced", "risk_class": 11, "potential_loss": "total"},
    )
    assert r["score"] == 10
    assert r["decision"] == "UNSUITABLE"


def test_unsuitable_r4_r5_r6_fail(client, product):
    # 100 - 35 - 25 - 30 = 10
    r = evaluate_suitability(
        {**client, "risk_tolerance_score": 5, "can_afford_total_loss": False, "financial_vulnerability": "HIGH"},
        {**product, "risk_class": 6, "potential_loss": "total", "leverage": True},
    )
    assert r["score"] == 10
    assert r["decision"] == "UNSUITABLE"


# ---------------------------------------------------------------------------
# 18–19. CONDITIONAL cases (40 ≤ score < 70)
# ---------------------------------------------------------------------------

def test_conditional_r2_r3_fail(client, product):
    # 100 - 30 - 20 = 50
    r = evaluate_suitability(
        {**client, "risk_tolerance_score": 5, "investment_horizon": 3},
        {**product, "risk_class": 11, "minimum_horizon": 12},
    )
    assert r["score"] == 50
    assert r["decision"] == "CONDITIONAL"


def test_conditional_r1_r5_fail(client, product):
    # 100 - 25 - 25 = 50
    r = evaluate_suitability(
        {**client, "financial_knowledge": "none", "financial_vulnerability": "HIGH"},
        {**product, "requires_knowledge_level": "advanced", "risk_class": 6},
    )
    assert r["score"] == 50
    assert r["decision"] == "CONDITIONAL"


# ---------------------------------------------------------------------------
# 20. All-rules-fail
# ---------------------------------------------------------------------------

def test_all_rules_fail(client, product):
    # 100 - 25 - 30 - 20 - 35 - 25 - 30 = -65
    r = evaluate_suitability(
        {
            **client,
            "financial_knowledge": "none",
            "risk_tolerance_score": 1,
            "investment_horizon": 1,
            "can_afford_total_loss": False,
            "financial_vulnerability": "HIGH",
        },
        {
            **product,
            "requires_knowledge_level": "advanced",
            "risk_class": 10,
            "minimum_horizon": 24,
            "potential_loss": "total",
            "leverage": True,
        },
    )
    assert r["score"] == -65
    assert r["decision"] == "UNSUITABLE"
    assert all(rule["pass"] is False for rule in r["rules"])
