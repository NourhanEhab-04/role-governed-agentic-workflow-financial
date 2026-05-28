"""
MiFID II Suitability Rule Engine
=================================
Implements Article 25(2) of MiFID II as formalised in:
  ESMA35-43-3172 — Guidelines on certain aspects of the MiFID II suitability
  requirements (September 2022, applicable from 3 October 2023).

Three mandatory assessment pillars (Art. 25(2)(a-c)):
  Pillar 1 — Knowledge & Experience  : R1, R7
  Pillar 2 — Financial Situation     : R3, R4
  Pillar 3 — Investment Objectives   : R2, R5, R6

Decision logic
--------------
HARD-FAIL rules (R1, R4, R5, R7) represent mandatory protection thresholds.
Failing ANY hard-fail rule → UNSUITABLE, regardless of total score.
Failing only SOFT rules (R2, R3, R6) → score-based decision.

Score (soft-fail penalties only, starting at 100):
  SUITABLE    : score >= 70  (all soft rules pass, or minor deviations)
  CONDITIONAL : score >= 40  (one or more soft rules fail but above floor)
  UNSUITABLE  : score <  40  (or any hard-fail rule failed)

Justification for hard vs. soft classification
-----------------------------------------------
Hard (UNSUITABLE regardless of score):
  R1 — Knowledge mismatch: ESMA GL §52 requires firms to ensure the client has
       the necessary knowledge and experience; this cannot be cured by disclosure.
  R4 — Affordability: Art. 25(2)(b) explicitly requires assessment of ability to
       bear losses; recommending a total-loss product to someone who cannot bear
       total loss is an unconditional breach.
  R5 — Vulnerability: ESMA GL §86-88 require heightened protection for vulnerable
       clients; assigning a high-risk product to a HIGH-vulnerability client is
       always unsuitable.
  R7 — Complexity: ESMA GL §58-60 require knowledge proportionate to product
       complexity; a client who cannot understand a COMPLEX product's risks cannot
       give informed consent.

Soft (CONDITIONAL if score permits):
  R2 — Risk alignment: minor over-tolerance can be addressed via enhanced suitability
       report disclosures per Art. 25(6).
  R3 — Horizon: a slightly short horizon may be addressed by disclosure of liquidity
       risk if the client has otherwise suitable characteristics.
  R6 — Leverage: leverage tolerance mismatch is a serious but quantitative criterion
       that, if borderline (e.g. score = 6 vs threshold 7), may allow conditional
       recommendation with explicit leverage-risk disclosure.
"""

from schemas.client_profile import REQUIRED_CLIENT_KEYS
from schemas.product_profile import REQUIRED_PRODUCT_KEYS as _ALL_PRODUCT_KEYS

# Rule engine only needs the fields it evaluates — product_name is metadata
REQUIRED_PRODUCT_KEYS = _ALL_PRODUCT_KEYS - {"product_name"}

# Semantic version of this rule set.
# Bump MAJOR on any change to hard-fail classification or threshold values.
# Bump MINOR on new rules or soft-penalty adjustments.
# Bump PATCH on documentation or detail-string fixes.
# Stored in every audit record so historical assessments remain reproducible.
RULE_ENGINE_VERSION = "1.0.0"

# ── MiFID II pillar classification ─────────────────────────────────────────────
# Source: ESMA35-43-3172, Art. 25(2)(a-c)
PILLAR_1_KNOWLEDGE   = {"R1", "R7"}   # Art. 25(2)(a) Knowledge & Experience
PILLAR_2_FINANCIAL   = {"R3", "R4"}   # Art. 25(2)(b) Financial Situation
PILLAR_3_OBJECTIVES  = {"R2", "R5", "R6"}  # Art. 25(2)(c) Investment Objectives

# Hard-fail rules: failing any of these → UNSUITABLE regardless of score.
# These correspond to mandatory regulatory protections that cannot be cured
# by disclosure or conditional recommendation.
HARD_FAIL_RULES = {"R1", "R4", "R5", "R7"}

# Soft-fail rules: these contribute to the numerical score.
# Failing one or more may yield CONDITIONAL if the score floor is met.
SOFT_FAIL_RULES = {"R2", "R3", "R6"}

KNOWLEDGE_LEVELS = {
    "none":     1,
    "basic":    3,
    "moderate": 6,
    "advanced": 9,
}

# ── Thresholds (score is computed from soft-fail penalties only) ───────────────
SUITABLE_THRESHOLD    = 70  # all or nearly all soft rules pass
CONDITIONAL_THRESHOLD = 40  # at least one soft rule fails but above floor


def _validate_inputs(client: dict, product: dict) -> None:
    missing_client = REQUIRED_CLIENT_KEYS - client.keys()
    if missing_client:
        raise ValueError(f"client dict is missing required keys: {missing_client}")
    missing_product = REQUIRED_PRODUCT_KEYS - product.keys()
    if missing_product:
        raise ValueError(f"product dict is missing required keys: {missing_product}")


# ── Rule functions ─────────────────────────────────────────────────────────────
# Each returns a dict: {rule, pass, hard_fail, penalty, pillar, detail}
# penalty is 0 for hard-fail rules (the gate logic handles their effect on
# the decision; only soft-fail penalties affect the numeric score).

def _r1_knowledge(client: dict, product: dict) -> dict:
    """
    Pillar 1 — Knowledge & Experience  (Art. 25(2)(a), ESMA GL §52)
    Hard fail: client must have knowledge >= product requirement.
    """
    client_level   = KNOWLEDGE_LEVELS[client["financial_knowledge"]]
    required_level = KNOWLEDGE_LEVELS[product["requires_knowledge_level"]]
    passed = client_level >= required_level
    return {
        "rule":      "R1",
        "pass":      passed,
        "hard_fail": True,
        "penalty":   0,          # hard-fail rules use gate logic, not score
        "pillar":    1,
        "detail": (
            f"client knowledge '{client['financial_knowledge']}' ({client_level}) "
            f">= product requirement '{product['requires_knowledge_level']}' ({required_level})"
            if passed else
            f"client knowledge '{client['financial_knowledge']}' ({client_level}) "
            f"< product requirement '{product['requires_knowledge_level']}' ({required_level}) "
            f"[HARD FAIL — Art. 25(2)(a)]"
        ),
    }


def _r2_risk_alignment(client: dict, product: dict) -> dict:
    """
    Pillar 3 — Investment Objectives  (Art. 25(2)(c), ESMA GL §67)
    Soft fail: risk class must not exceed client tolerance + 2.
    The +2 margin reflects industry practice for marginal over-tolerance cases
    where enhanced disclosure (Art. 25(6) suitability report) is sufficient.
    """
    risk_class = product["risk_class"]
    tolerance  = client["risk_tolerance_score"]
    passed = risk_class <= tolerance + 2
    return {
        "rule":      "R2",
        "pass":      passed,
        "hard_fail": False,
        "penalty":   0 if passed else -20,
        "pillar":    3,
        "detail": (
            f"risk_class {risk_class} <= risk_tolerance_score {tolerance} + 2 ({tolerance + 2})"
            if passed else
            f"risk_class {risk_class} > risk_tolerance_score {tolerance} + 2 ({tolerance + 2}) "
            f"[soft fail — score -20]"
        ),
    }


def _r3_horizon(client: dict, product: dict) -> dict:
    """
    Pillar 2 — Financial Situation  (Art. 25(2)(b), ESMA GL §72)
    Soft fail: client horizon should meet product minimum.
    Short horizon with liquidity risk may be disclosed rather than blocked.
    """
    horizon     = client["investment_horizon"]
    min_horizon = product["minimum_horizon"]
    passed = horizon >= min_horizon
    return {
        "rule":      "R3",
        "pass":      passed,
        "hard_fail": False,
        "penalty":   0 if passed else -15,
        "pillar":    2,
        "detail": (
            f"investment_horizon {horizon} >= minimum_horizon {min_horizon}"
            if passed else
            f"investment_horizon {horizon} < minimum_horizon {min_horizon} "
            f"[soft fail — score -15]"
        ),
    }


def _r4_affordability(client: dict, product: dict) -> dict:
    """
    Pillar 2 — Financial Situation  (Art. 25(2)(b))
    Hard fail: client cannot afford total loss but product has total loss potential.
    This is an unconditional breach of Art. 25(2)(b) — no disclosure can cure it.
    """
    passed = not (
        client["can_afford_total_loss"] is False
        and product["potential_loss"] == "total"
    )
    return {
        "rule":      "R4",
        "pass":      passed,
        "hard_fail": True,
        "penalty":   0,
        "pillar":    2,
        "detail": (
            "client can afford total loss or product does not carry total loss risk"
            if passed else
            "client cannot afford total loss but product has potential_loss='total' "
            "[HARD FAIL — Art. 25(2)(b)]"
        ),
    }


def _r5_vulnerability(client: dict, product: dict) -> dict:
    """
    Pillar 3 — Investment Objectives / Client Protection  (Art. 25(2)(b-c))
    Hard fail: HIGH-vulnerability client must not be placed in products
    with risk class >= 5.  ESMA GL §86-88 require enhanced protection.
    """
    risk_class = product["risk_class"]
    passed = not (
        client["financial_vulnerability"] == "HIGH" and risk_class >= 5
    )
    return {
        "rule":      "R5",
        "pass":      passed,
        "hard_fail": True,
        "penalty":   0,
        "pillar":    3,
        "detail": (
            f"client is not HIGH-vulnerability or product risk_class {risk_class} < 5"
            if passed else
            f"client is HIGH-vulnerability and product risk_class {risk_class} >= 5 "
            f"[HARD FAIL — ESMA GL §86-88]"
        ),
    }


def _r6_leverage(client: dict, product: dict) -> dict:
    """
    Pillar 3 — Investment Objectives  (Art. 25(2)(c))
    Soft fail: leveraged products require risk tolerance >= 7.
    A score of 6 vs threshold 7 may allow conditional recommendation
    with explicit leverage-risk disclosure.
    """
    tolerance = client["risk_tolerance_score"]
    passed = not (product["leverage"] is True and tolerance < 7)
    return {
        "rule":      "R6",
        "pass":      passed,
        "hard_fail": False,
        "penalty":   0 if passed else -25,
        "pillar":    3,
        "detail": (
            f"product is not leveraged or risk_tolerance_score {tolerance} >= 7"
            if passed else
            f"product is leveraged and risk_tolerance_score {tolerance} < 7 "
            f"[soft fail — score -25]"
        ),
    }


def _r7_complexity(client: dict, product: dict) -> dict:
    """
    Pillar 1 — Knowledge & Experience  (Art. 25(2)(a), ESMA GL §58-60)
    Hard fail: COMPLEX products require at least moderate knowledge.
    Understanding a complex product's risk profile is a prerequisite for
    informed consent — it cannot be addressed through disclosure alone.
    """
    COMPLEX_MIN_KNOWLEDGE = {"moderate", "advanced"}
    passed = not (
        product["complexity_tier"] == "COMPLEX"
        and client["financial_knowledge"] not in COMPLEX_MIN_KNOWLEDGE
    )
    return {
        "rule":      "R7",
        "pass":      passed,
        "hard_fail": True,
        "penalty":   0,
        "pillar":    1,
        "detail": (
            f"product complexity_tier='{product['complexity_tier']}' or "
            f"client knowledge '{client['financial_knowledge']}' meets COMPLEX threshold"
            if passed else
            f"product is COMPLEX but client knowledge '{client['financial_knowledge']}' "
            f"is below 'moderate' [HARD FAIL — ESMA GL §58-60]"
        ),
    }


# ── Decision logic ─────────────────────────────────────────────────────────────

def _compute_decision(rules: list[dict]) -> tuple[int, str, list[str]]:
    """
    Returns (score, decision, hard_failed_rules).

    Score is derived solely from soft-fail penalties (so it reflects the
    degree of objective misalignment when no hard protection is breached).

    Decision:
      UNSUITABLE  — any hard-fail rule failed (mandatory protection breached)
      UNSUITABLE  — score < CONDITIONAL_THRESHOLD (all soft fails too severe)
      CONDITIONAL — CONDITIONAL_THRESHOLD <= score < SUITABLE_THRESHOLD
      SUITABLE    — score >= SUITABLE_THRESHOLD and no hard fails
    """
    hard_failed = [r["rule"] for r in rules if r["hard_fail"] and not r["pass"]]
    soft_penalty = sum(r["penalty"] for r in rules if not r["hard_fail"])
    score = 100 + soft_penalty   # 100 is the base; only soft penalties deduct

    if hard_failed:
        return score, "UNSUITABLE", hard_failed
    if score >= SUITABLE_THRESHOLD:
        return score, "SUITABLE", []
    if score >= CONDITIONAL_THRESHOLD:
        return score, "CONDITIONAL", []
    return score, "UNSUITABLE", []


# ── Public API ─────────────────────────────────────────────────────────────────

def evaluate_suitability(client: dict, product: dict) -> dict:
    """
    Evaluate MiFID II suitability for a (client, product) pair.

    Returns:
        {
            "score":      int,
            "decision":   "SUITABLE" | "CONDITIONAL" | "UNSUITABLE",
            "hard_failed_rules": list[str],   # empty if no hard fails
            "rules": [
                {
                    "rule":      str,
                    "pass":      bool,
                    "hard_fail": bool,
                    "penalty":   int,
                    "pillar":    int,  # 1, 2, or 3
                    "detail":    str,
                },
                ...
            ]
        }
    """
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

    score, decision, hard_failed = _compute_decision(rules)

    return {
        "score":               score,
        "decision":            decision,
        "hard_failed_rules":   hard_failed,
        "rules":               rules,
        "rule_engine_version": RULE_ENGINE_VERSION,
    }


if __name__ == "__main__":
    import json
    sample_client = {
        "financial_knowledge":    "basic",
        "risk_tolerance_score":   4,
        "investment_horizon":     3,
        "liquid_assets":          8000.0,
        "income":                 42000.0,
        "investment_amount":      5000.0,
        "can_afford_total_loss":  False,
        "financial_vulnerability": "LOW",
    }
    sample_product = {
        "risk_class":               4,
        "complexity_tier":          "NON-COMPLEX",
        "requires_knowledge_level": "basic",
        "minimum_horizon":          2,
        "potential_loss":           "partial",
        "leverage":                 False,
    }
    result = evaluate_suitability(sample_client, sample_product)
    print(json.dumps(result, indent=2))
