# orchestrator/consistency_checks.py
"""
Layer 1 — Deterministic consistency checks (no LLM, zero API cost).

Three check functions:
  check_a1_consistency(client_profile)         -> list[str]   (issues)
  check_a2_consistency(product_profile)        -> list[str]   (issues)
  check_a1_a2_cross(client_profile, product_profile) -> list[str]

Each returns a (possibly empty) list of issue strings.
An empty list means no issues found.

These run immediately after the schema validators and before the LLM verifier,
so the LLM verifier only needs to be called on cases that already pass here.
"""

from __future__ import annotations

# Knowledge level ordering used for cross-checks
_KNOWLEDGE_ORDER = {"none": 0, "basic": 1, "moderate": 2, "advanced": 3}


def enforce_a1_consistency(profile: dict) -> dict:
    """
    Deterministically override A1 fields where the rule is unambiguous.
    Called after all LLM correction attempts — these rules cannot be wrong.
    """
    p = dict(profile)
    age = p.get("age")
    income = p.get("income")
    vulnerability = p.get("financial_vulnerability", "LOW")

    # Rule: age > 70 always → HIGH (MiFID II mandatory)
    if isinstance(age, int) and age > 70:
        p["financial_vulnerability"] = "HIGH"
        vulnerability = "HIGH"
    # Rule: zero income (unemployed, no pension) → HIGH
    elif income == 0.0:
        p["financial_vulnerability"] = "HIGH"
        vulnerability = "HIGH"

    # Rule: can_afford_total_loss cannot be True for HIGH-vulnerability clients
    if vulnerability == "HIGH" and p.get("can_afford_total_loss") is True:
        p["can_afford_total_loss"] = False

    # Rule: investment_horizon must be at least 1
    if isinstance(p.get("investment_horizon"), int) and p["investment_horizon"] < 1:
        p["investment_horizon"] = 1

    return p


def enforce_a2_consistency(profile: dict) -> dict:
    """
    Deterministically override A2 fields where ESMA rules are unambiguous.
    Called after all LLM correction attempts — these rules cannot be wrong.
    """
    p = dict(profile)
    risk_class = p.get("risk_class")

    if not isinstance(risk_class, int):
        return p  # type errors handled upstream; skip

    # Rule: class 7 mandatory fields (ESMA — no exceptions)
    if risk_class == 7:
        p["leverage"] = True
        p["potential_loss"] = "total"
        p["complexity_tier"] = "COMPLEX"
        p["requires_knowledge_level"] = "advanced"

    # Rule: leveraged product must be at least class 6
    if p.get("leverage") is True and risk_class < 6:
        p["risk_class"] = 6

    # Rule: COMPLEX products require at least moderate knowledge
    if p.get("complexity_tier") == "COMPLEX":
        knowledge = p.get("requires_knowledge_level", "none")
        if _KNOWLEDGE_ORDER.get(knowledge, 0) < _KNOWLEDGE_ORDER["moderate"]:
            p["requires_knowledge_level"] = "moderate"

    # Rule: class 1–4 cannot have total potential loss
    if risk_class <= 4 and p.get("potential_loss") == "total":
        p["potential_loss"] = "partial"

    # Rule: class 1–4 are not leveraged
    if risk_class <= 4 and p.get("leverage") is True:
        p["leverage"] = False

    # Rule: minimum_horizon must be at least 1
    if isinstance(p.get("minimum_horizon"), int) and p["minimum_horizon"] < 1:
        p["minimum_horizon"] = 1

    return p


def check_a1_consistency(client_profile: dict) -> list[str]:
    """
    Check internal consistency of the A1 client_profile output.
    Returns a list of issue strings (empty = no issues).
    """
    issues: list[str] = []
    p = client_profile

    income = p.get("income", None)
    vulnerability = p.get("financial_vulnerability", "")

    # ── age > 70 → vulnerability must be HIGH ────────────────────────────────
    age = p.get("age", None)
    if isinstance(age, int) and age > 70 and vulnerability != "HIGH":
        issues.append(
            f"age is {age} (over 70) but financial_vulnerability is '{vulnerability}' "
            "(expected HIGH — clients over 70 are always HIGH vulnerability under MiFID II rules)"
        )

    # ── income = 0 → vulnerability must be HIGH ──────────────────────────────
    if income == 0.0 and vulnerability != "HIGH":
        issues.append(
            f"income is 0 but financial_vulnerability is '{vulnerability}' "
            "(expected HIGH for unemployed/zero-income client)"
        )

    # ── investment_amount > liquid_assets → suspicious ────────────────────────
    investment_amount = p.get("investment_amount", 0.0)
    liquid_assets = p.get("liquid_assets", 0.0)
    if isinstance(investment_amount, (int, float)) and isinstance(liquid_assets, (int, float)):
        if investment_amount > liquid_assets:
            issues.append(
                f"investment_amount ({investment_amount}) exceeds liquid_assets ({liquid_assets}): "
                "client is investing more than their accessible cash"
            )

    # ── can_afford_total_loss = True AND vulnerability = HIGH/MEDIUM → contradiction ─
    can_afford = p.get("can_afford_total_loss", False)
    if can_afford is True and vulnerability == "HIGH":
        issues.append(
            "can_afford_total_loss is True but financial_vulnerability is HIGH: "
            "contradictory — a highly vulnerable client should not be marked as able to lose all"
        )
    if can_afford is True and vulnerability == "MEDIUM":
        issues.append(
            "can_afford_total_loss is True but financial_vulnerability is MEDIUM: "
            "suspicious — verify the client explicitly stated they can write off the entire amount, "
            "not just that they can tolerate partial loss"
        )

    # ── high risk tolerance but HIGH vulnerability → flag ────────────────────
    risk_score = p.get("risk_tolerance_score", 0)
    if isinstance(risk_score, int) and risk_score >= 8 and vulnerability == "HIGH":
        issues.append(
            f"risk_tolerance_score is {risk_score} (aggressive) but financial_vulnerability "
            "is HIGH — possible mismatch between stated tolerance and actual financial situation"
        )

    # ── investment_horizon = 0 → impossible ──────────────────────────────────
    horizon = p.get("investment_horizon", None)
    if isinstance(horizon, int) and horizon == 0:
        issues.append(
            "investment_horizon is 0 years — this is not a valid investment horizon"
        )

    # ── investment_amount <= 0 → invalid ─────────────────────────────────────
    if isinstance(investment_amount, (int, float)) and investment_amount <= 0:
        issues.append(
            f"investment_amount is {investment_amount} — must be a positive value"
        )

    return issues


def check_a2_consistency(product_profile: dict) -> list[str]:
    """
    Check internal consistency of the A2 product_profile output using ESMA rules.
    Returns a list of issue strings (empty = no issues).
    """
    issues: list[str] = []
    p = product_profile

    risk_class = p.get("risk_class", None)
    leverage = p.get("leverage", None)
    potential_loss = p.get("potential_loss", "")
    complexity = p.get("complexity_tier", "")
    knowledge = p.get("requires_knowledge_level", "")
    min_horizon = p.get("minimum_horizon", None)

    if not isinstance(risk_class, int):
        return issues  # type errors caught upstream; skip logic checks

    # ── leverage must be True for class 7 ────────────────────────────────────
    if risk_class == 7 and leverage is not True:
        issues.append(
            f"risk_class is 7 but leverage is {leverage}: "
            "ESMA class 7 products (CFDs, 3x ETFs, uncovered options) require leverage=true"
        )

    # ── potential_loss must be total for class 7 ─────────────────────────────
    if risk_class == 7 and potential_loss != "total":
        issues.append(
            f"risk_class is 7 but potential_loss is '{potential_loss}': "
            "ESMA class 7 always carries total potential loss"
        )

    # ── complexity must be COMPLEX for class 7 ───────────────────────────────
    if risk_class == 7 and complexity != "COMPLEX":
        issues.append(
            f"risk_class is 7 but complexity_tier is '{complexity}': "
            "ESMA class 7 products are always COMPLEX"
        )

    # ── leverage=True should imply risk_class >= 6 ───────────────────────────
    if leverage is True and isinstance(risk_class, int) and risk_class < 6:
        issues.append(
            f"leverage is True but risk_class is {risk_class}: "
            "leveraged products should be risk class 6 or higher"
        )

    # ── COMPLEX should imply knowledge >= moderate ────────────────────────────
    if complexity == "COMPLEX":
        knowledge_rank = _KNOWLEDGE_ORDER.get(knowledge, -1)
        if knowledge_rank < _KNOWLEDGE_ORDER["moderate"]:
            issues.append(
                f"complexity_tier is COMPLEX but requires_knowledge_level is '{knowledge}': "
                "COMPLEX products require at least moderate knowledge"
            )

    # ── low risk class should not have leverage ───────────────────────────────
    if isinstance(risk_class, int) and risk_class <= 2 and leverage is True:
        issues.append(
            f"risk_class is {risk_class} but leverage is True: "
            "low risk class products are never leveraged"
        )

    # ── minimum_horizon must be positive ─────────────────────────────────────
    if isinstance(min_horizon, int) and min_horizon <= 0:
        issues.append(
            f"minimum_horizon is {min_horizon}: must be at least 1 year"
        )

    # ── class 1-4 should not have potential_loss = total ─────────────────────
    if isinstance(risk_class, int) and risk_class <= 4 and potential_loss == "total":
        issues.append(
            f"risk_class is {risk_class} but potential_loss is 'total': "
            "diversified/low-risk products should have partial potential loss"
        )

    return issues


def check_a1_a2_cross(client_profile: dict, product_profile: dict) -> list[str]:
    """
    Cross-check client_profile against product_profile for suitability red flags.
    These are structural mismatches detectable before the rule engine runs.
    Returns a list of issue strings (empty = no issues).
    """
    issues: list[str] = []
    c = client_profile
    p = product_profile

    client_knowledge = c.get("financial_knowledge", "none")
    product_knowledge = p.get("requires_knowledge_level", "none")
    client_rank = _KNOWLEDGE_ORDER.get(client_knowledge, 0)
    product_rank = _KNOWLEDGE_ORDER.get(product_knowledge, 0)

    # ── client knowledge below product requirement ────────────────────────────
    if client_rank < product_rank:
        issues.append(
            f"client financial_knowledge is '{client_knowledge}' but product "
            f"requires_knowledge_level is '{product_knowledge}': "
            "client lacks the knowledge required for this product"
        )

    # ── client investment_horizon < product minimum_horizon ───────────────────
    client_horizon = c.get("investment_horizon", 0)
    product_min_horizon = p.get("minimum_horizon", 0)
    if (isinstance(client_horizon, int) and isinstance(product_min_horizon, int)
            and client_horizon < product_min_horizon):
        issues.append(
            f"client investment_horizon is {client_horizon} years but product "
            f"minimum_horizon is {product_min_horizon} years: "
            "client cannot hold this product for its recommended minimum period"
        )

    # ── high product risk vs low client tolerance ─────────────────────────────
    risk_class = p.get("risk_class", 0)
    risk_score = c.get("risk_tolerance_score", 5)
    if isinstance(risk_class, int) and isinstance(risk_score, int):
        if risk_class >= 6 and risk_score <= 3:
            issues.append(
                f"product risk_class is {risk_class} (high risk) but client "
                f"risk_tolerance_score is {risk_score} (conservative): "
                "extreme risk mismatch"
            )
        if risk_class == 7 and risk_score <= 5:
            issues.append(
                f"product risk_class is 7 (very high risk) but client "
                f"risk_tolerance_score is {risk_score}: "
                "product is speculative; client tolerance is insufficient"
            )

    # ── COMPLEX product + none/basic knowledge ────────────────────────────────
    complexity = p.get("complexity_tier", "")
    if complexity == "COMPLEX" and client_knowledge in ("none", "basic"):
        issues.append(
            f"product is COMPLEX but client financial_knowledge is '{client_knowledge}': "
            "client cannot adequately assess the risks of this product"
        )

    # ── total potential loss + can_afford_total_loss = False ──────────────────
    potential_loss = p.get("potential_loss", "")
    can_afford = c.get("can_afford_total_loss", False)
    if potential_loss == "total" and can_afford is False:
        issues.append(
            "product potential_loss is 'total' but client can_afford_total_loss is False: "
            "client cannot afford to lose all of their investment in this product"
        )

    return issues
