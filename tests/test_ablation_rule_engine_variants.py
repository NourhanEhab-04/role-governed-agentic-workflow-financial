"""
tests/test_ablation_rule_engine_variants.py
=============================================
Tests the ablated rule engine against known ground truth.

No LLM calls. No file I/O (profiles are inlined as fixtures).
"""

import pytest
from evaluation.ablation_study.rule_engine_variants import evaluate_suitability_ablated

# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def suitable_profiles():
    """Conservative client + money market fund. All rules pass, score=100."""
    client = {
        "financial_knowledge":     "basic",
        "risk_tolerance_score":    2,
        "investment_horizon":      5,
        "liquid_assets":           80000.0,
        "income":                  25200.0,
        "investment_amount":       50000.0,
        "can_afford_total_loss":   False,
        "financial_vulnerability": "LOW",
        "age":                     67,
    }
    product = {
        "product_name":              "Money Market Fund",
        "risk_class":                1,
        "complexity_tier":           "NON-COMPLEX",
        "requires_knowledge_level":  "none",
        "minimum_horizon":           1,
        "potential_loss":            "partial",
        "leverage":                  False,
    }
    return client, product


@pytest.fixture
def knowledge_fail_profiles():
    """Client knowledge=none, product requires advanced → R1 hard fail."""
    client = {
        "financial_knowledge":     "none",
        "risk_tolerance_score":    5,
        "investment_horizon":      5,
        "liquid_assets":           50000.0,
        "income":                  30000.0,
        "investment_amount":       10000.0,
        "can_afford_total_loss":   True,
        "financial_vulnerability": "LOW",
        "age":                     40,
    }
    product = {
        "product_name":              "Equity Derivative",
        "risk_class":                6,
        "complexity_tier":           "COMPLEX",
        "requires_knowledge_level":  "advanced",
        "minimum_horizon":           1,
        "potential_loss":            "total",
        "leverage":                  False,
    }
    return client, product


@pytest.fixture
def soft_fail_profiles():
    """Risk misalignment only (R2 soft fail, −20). Score=80 → SUITABLE still."""
    client = {
        "financial_knowledge":     "moderate",
        "risk_tolerance_score":    3,
        "investment_horizon":      10,
        "liquid_assets":           100000.0,
        "income":                  50000.0,
        "investment_amount":       20000.0,
        "can_afford_total_loss":   True,
        "financial_vulnerability": "LOW",
        "age":                     45,
    }
    product = {
        "product_name":              "EM Equity ETF",
        "risk_class":                6,   # tolerance(3)+2=5 < 6 → R2 fails
        "complexity_tier":           "NON-COMPLEX",
        "requires_knowledge_level":  "moderate",
        "minimum_horizon":           5,
        "potential_loss":            "partial",
        "leverage":                  False,
    }
    return client, product


@pytest.fixture
def all_soft_fail_profiles():
    """R2+R3+R6 all soft-fail → score=100-20-15-25=40 → CONDITIONAL boundary."""
    client = {
        "financial_knowledge":     "moderate",
        "risk_tolerance_score":    3,
        "investment_horizon":      2,    # product min=5 → R3 fails
        "liquid_assets":           60000.0,
        "income":                  36000.0,
        "investment_amount":       15000.0,
        "can_afford_total_loss":   True,
        "financial_vulnerability": "LOW",
        "age":                     42,
    }
    product = {
        "product_name":              "Leveraged ETF",
        "risk_class":                6,   # tolerance(3)+2=5 < 6 → R2 fails
        "complexity_tier":           "NON-COMPLEX",
        "requires_knowledge_level":  "moderate",
        "minimum_horizon":           5,   # horizon(2) < 5 → R3 fails
        "potential_loss":            "partial",
        "leverage":                  True,  # tolerance(3) < 7 → R6 fails
    }
    return client, product


# ── Control (no ablation) ──────────────────────────────────────────────────────

class TestControlBehaviour:
    def test_suitable_gives_suitable(self, suitable_profiles):
        client, product = suitable_profiles
        result = evaluate_suitability_ablated(client, product)
        assert result["decision"] == "SUITABLE"
        assert result["score"] == 100
        assert result["hard_failed_rules"] == []

    def test_r1_hard_fail_gives_unsuitable(self, knowledge_fail_profiles):
        client, product = knowledge_fail_profiles
        result = evaluate_suitability_ablated(client, product)
        assert result["decision"] == "UNSUITABLE"
        assert "R1" in result["hard_failed_rules"]

    def test_r2_soft_fail_reduces_score(self, soft_fail_profiles):
        client, product = soft_fail_profiles
        result = evaluate_suitability_ablated(client, product)
        assert result["score"] == 80

    def test_all_soft_fails_score_40(self, all_soft_fail_profiles):
        client, product = all_soft_fail_profiles
        result = evaluate_suitability_ablated(client, product)
        assert result["score"] == 40
        assert result["decision"] == "CONDITIONAL"

    def test_ablation_fields_present(self, suitable_profiles):
        client, product = suitable_profiles
        result = evaluate_suitability_ablated(client, product)
        assert "ablation_disabled_rules"        in result
        assert "ablation_suitable_threshold"    in result
        assert "ablation_conditional_threshold" in result
        assert result["ablation_disabled_rules"] == []


# ── Rule disabling ─────────────────────────────────────────────────────────────

class TestDisabledRules:
    def test_disabling_r1_removes_hard_fail(self, knowledge_fail_profiles):
        """With R1 off, a knowledge-gap client is no longer blocked."""
        client, product = knowledge_fail_profiles
        result = evaluate_suitability_ablated(client, product, disabled_rules=["R1"])
        # R1 forced to pass; R7 still fires (COMPLEX + knowledge=none)
        # so decision is still UNSUITABLE from R7
        assert "R1" not in result["hard_failed_rules"]

    def test_disabling_r1_and_r7_clears_hard_fails(self, knowledge_fail_profiles):
        """With both knowledge-related hard-fail rules off, unsuitable becomes possible
        only via R4 (affordability) — which passes here (can_afford_total_loss=True)."""
        client, product = knowledge_fail_profiles
        # R4 passes because can_afford_total_loss=True; R5 passes (LOW vulnerability)
        result = evaluate_suitability_ablated(
            client, product, disabled_rules=["R1", "R7"]
        )
        assert result["hard_failed_rules"] == []
        # Score = 100 (no soft fails)
        assert result["decision"] == "SUITABLE"

    def test_disabling_r2_removes_soft_penalty(self, soft_fail_profiles):
        """With R2 off, the −20 risk-alignment penalty is absent."""
        client, product = soft_fail_profiles
        result_with    = evaluate_suitability_ablated(client, product)
        result_without = evaluate_suitability_ablated(client, product, disabled_rules=["R2"])
        assert result_without["score"] == result_with["score"] + 20

    def test_disabling_r3_and_r6_raises_score(self, all_soft_fail_profiles):
        """With R3 and R6 off, only R2 fires → score = 100−20 = 80 → SUITABLE."""
        client, product = all_soft_fail_profiles
        result = evaluate_suitability_ablated(
            client, product, disabled_rules=["R3", "R6"]
        )
        assert result["score"] == 80
        assert result["decision"] == "SUITABLE"

    def test_disabling_all_rules_gives_suitable(self, knowledge_fail_profiles):
        """Disabling all 7 rules: every rule forced to pass → score=100 → SUITABLE."""
        client, product = knowledge_fail_profiles
        result = evaluate_suitability_ablated(
            client, product,
            disabled_rules=["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
        )
        assert result["decision"] == "SUITABLE"
        assert result["score"] == 100
        assert result["hard_failed_rules"] == []

    def test_disabled_rules_recorded_in_output(self, suitable_profiles):
        client, product = suitable_profiles
        result = evaluate_suitability_ablated(
            client, product, disabled_rules=["R2", "R5"]
        )
        assert set(result["ablation_disabled_rules"]) == {"R2", "R5"}


# ── Threshold variants ─────────────────────────────────────────────────────────

class TestThresholdVariants:
    def test_strict_threshold_downgrades_suitable_to_conditional(self, soft_fail_profiles):
        """With strict threshold (SUITABLE≥80), score=80 is exactly at boundary."""
        client, product = soft_fail_profiles
        # score=80 with default → SUITABLE; with strict threshold 80 → SUITABLE (>=80 passes)
        default = evaluate_suitability_ablated(client, product)
        strict  = evaluate_suitability_ablated(
            client, product, suitable_threshold=81
        )
        assert default["decision"] == "SUITABLE"
        assert strict["decision"]  == "CONDITIONAL"

    def test_lenient_threshold_upgrades_conditional_to_suitable(self, all_soft_fail_profiles):
        """With lenient threshold (SUITABLE≥60), score=40 is still CONDITIONAL."""
        client, product = all_soft_fail_profiles
        # score=40 < 60 → still CONDITIONAL even with lenient
        lenient = evaluate_suitability_ablated(
            client, product, suitable_threshold=60, conditional_threshold=30
        )
        assert lenient["decision"] == "CONDITIONAL"

    def test_lenient_threshold_lifts_r2_only_case(self, soft_fail_profiles):
        """score=80, lenient SUITABLE≥60 → still SUITABLE."""
        client, product = soft_fail_profiles
        result = evaluate_suitability_ablated(
            client, product, suitable_threshold=60, conditional_threshold=30
        )
        assert result["decision"] == "SUITABLE"

    def test_thresholds_recorded_in_output(self, suitable_profiles):
        client, product = suitable_profiles
        result = evaluate_suitability_ablated(
            client, product, suitable_threshold=75, conditional_threshold=45
        )
        assert result["ablation_suitable_threshold"]    == 75
        assert result["ablation_conditional_threshold"] == 45

    def test_hard_fail_overrides_any_threshold(self, knowledge_fail_profiles):
        """Hard-fail rules produce UNSUITABLE regardless of score thresholds."""
        client, product = knowledge_fail_profiles
        # Even with very lenient thresholds, a hard fail still → UNSUITABLE
        result = evaluate_suitability_ablated(
            client, product,
            disabled_rules=["R7"],   # only R1 fires
            suitable_threshold=10,
            conditional_threshold=5,
        )
        assert result["decision"] == "UNSUITABLE"
        assert "R1" in result["hard_failed_rules"]
