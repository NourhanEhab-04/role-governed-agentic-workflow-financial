# tests/test_consistency_checks.py
"""
Unit tests for the deterministic consistency checks in orchestrator/consistency_checks.py.
No LLM calls.
"""

import pytest
from orchestrator.consistency_checks import (
    check_a1_consistency,
    check_a2_consistency,
    check_a1_a2_cross,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _client(**overrides):
    base = {
        "financial_knowledge": "moderate",
        "risk_tolerance_score": 5,
        "investment_horizon": 5,
        "liquid_assets": 20000.0,
        "income": 50000.0,
        "investment_amount": 5000.0,
        "can_afford_total_loss": False,
        "financial_vulnerability": "LOW",
    }
    base.update(overrides)
    return base


def _product(**overrides):
    base = {
        "product_name": "Diversified Equity ETF",
        "risk_class": 4,
        "complexity_tier": "NON-COMPLEX",
        "requires_knowledge_level": "basic",
        "minimum_horizon": 3,
        "potential_loss": "partial",
        "leverage": False,
    }
    base.update(overrides)
    return base


# ── check_a1_consistency ──────────────────────────────────────────────────────

class TestA1Consistency:

    def test_clean_profile_no_issues(self):
        assert check_a1_consistency(_client()) == []

    def test_zero_income_not_high_vulnerability(self):
        issues = check_a1_consistency(_client(income=0.0, financial_vulnerability="LOW"))
        assert any("income is 0" in i for i in issues)

    def test_zero_income_with_high_vulnerability_ok(self):
        issues = check_a1_consistency(_client(income=0.0, financial_vulnerability="HIGH"))
        assert not any("income is 0" in i for i in issues)

    def test_investment_exceeds_liquid_assets(self):
        issues = check_a1_consistency(_client(investment_amount=30000.0, liquid_assets=20000.0))
        assert any("exceeds liquid_assets" in i for i in issues)

    def test_investment_equals_liquid_assets_ok(self):
        issues = check_a1_consistency(_client(investment_amount=20000.0, liquid_assets=20000.0))
        assert not any("exceeds liquid_assets" in i for i in issues)

    def test_can_afford_total_loss_and_high_vulnerability(self):
        issues = check_a1_consistency(_client(
            can_afford_total_loss=True,
            financial_vulnerability="HIGH",
        ))
        assert any("contradictory" in i for i in issues)

    def test_can_afford_total_loss_low_vulnerability_ok(self):
        issues = check_a1_consistency(_client(
            can_afford_total_loss=True,
            financial_vulnerability="LOW",
        ))
        assert not any("contradictory" in i for i in issues)

    def test_high_risk_score_with_high_vulnerability(self):
        issues = check_a1_consistency(_client(
            risk_tolerance_score=9,
            financial_vulnerability="HIGH",
        ))
        assert any("mismatch" in i for i in issues)

    def test_high_risk_score_low_vulnerability_ok(self):
        issues = check_a1_consistency(_client(
            risk_tolerance_score=9,
            financial_vulnerability="LOW",
        ))
        assert not any("mismatch" in i for i in issues)

    def test_zero_investment_horizon(self):
        issues = check_a1_consistency(_client(investment_horizon=0))
        assert any("investment_horizon is 0" in i for i in issues)

    def test_positive_investment_horizon_ok(self):
        issues = check_a1_consistency(_client(investment_horizon=1))
        assert not any("investment_horizon is 0" in i for i in issues)

    def test_zero_investment_amount(self):
        issues = check_a1_consistency(_client(investment_amount=0.0))
        assert any("must be a positive value" in i for i in issues)

    def test_negative_investment_amount(self):
        issues = check_a1_consistency(_client(investment_amount=-500.0))
        assert any("must be a positive value" in i for i in issues)

    def test_multiple_issues_returned(self):
        issues = check_a1_consistency(_client(
            income=0.0,
            financial_vulnerability="LOW",
            investment_amount=50000.0,
            liquid_assets=10000.0,
        ))
        assert len(issues) >= 2


# ── check_a2_consistency ──────────────────────────────────────────────────────

class TestA2Consistency:

    def test_clean_product_no_issues(self):
        assert check_a2_consistency(_product()) == []

    def test_class7_no_leverage(self):
        issues = check_a2_consistency(_product(
            risk_class=7, leverage=False,
            potential_loss="total", complexity_tier="COMPLEX",
            requires_knowledge_level="advanced",
        ))
        assert any("leverage=true" in i for i in issues)

    def test_class7_not_total_loss(self):
        issues = check_a2_consistency(_product(
            risk_class=7, leverage=True,
            potential_loss="partial", complexity_tier="COMPLEX",
            requires_knowledge_level="advanced",
        ))
        assert any("potential_loss" in i for i in issues)

    def test_class7_not_complex(self):
        issues = check_a2_consistency(_product(
            risk_class=7, leverage=True,
            potential_loss="total", complexity_tier="NON-COMPLEX",
            requires_knowledge_level="advanced",
        ))
        assert any("COMPLEX" in i for i in issues)

    def test_class7_fully_correct_no_issues(self):
        issues = check_a2_consistency(_product(
            risk_class=7, leverage=True,
            potential_loss="total", complexity_tier="COMPLEX",
            requires_knowledge_level="advanced",
            minimum_horizon=1,
        ))
        assert issues == []

    def test_leverage_true_low_risk_class(self):
        issues = check_a2_consistency(_product(risk_class=3, leverage=True))
        assert any("risk class 6 or higher" in i for i in issues)

    def test_leverage_true_class6_ok(self):
        issues = check_a2_consistency(_product(
            risk_class=6, leverage=True,
            complexity_tier="COMPLEX", requires_knowledge_level="advanced",
        ))
        assert not any("risk class 6 or higher" in i for i in issues)

    def test_complex_but_none_knowledge(self):
        issues = check_a2_consistency(_product(
            complexity_tier="COMPLEX", requires_knowledge_level="none",
        ))
        assert any("moderate knowledge" in i for i in issues)

    def test_complex_moderate_knowledge_ok(self):
        issues = check_a2_consistency(_product(
            complexity_tier="COMPLEX", requires_knowledge_level="moderate",
        ))
        assert not any("moderate knowledge" in i for i in issues)

    def test_class1_leverage_true(self):
        issues = check_a2_consistency(_product(risk_class=1, leverage=True))
        assert any("never leveraged" in i for i in issues)

    def test_zero_minimum_horizon(self):
        issues = check_a2_consistency(_product(minimum_horizon=0))
        assert any("minimum_horizon" in i for i in issues)

    def test_class3_total_loss(self):
        issues = check_a2_consistency(_product(risk_class=3, potential_loss="total"))
        assert any("partial potential loss" in i for i in issues)

    def test_class5_partial_loss_ok(self):
        issues = check_a2_consistency(_product(
            risk_class=5, potential_loss="partial",
            requires_knowledge_level="moderate",
        ))
        assert not any("partial potential loss" in i for i in issues)


# ── check_a1_a2_cross ─────────────────────────────────────────────────────────

class TestA1A2Cross:

    def test_clean_pair_no_issues(self):
        assert check_a1_a2_cross(_client(), _product()) == []

    def test_client_knowledge_below_requirement(self):
        issues = check_a1_a2_cross(
            _client(financial_knowledge="none"),
            _product(requires_knowledge_level="advanced"),
        )
        assert any("lacks the knowledge" in i for i in issues)

    def test_client_knowledge_meets_requirement_ok(self):
        issues = check_a1_a2_cross(
            _client(financial_knowledge="moderate"),
            _product(requires_knowledge_level="moderate"),
        )
        assert not any("lacks the knowledge" in i for i in issues)

    def test_horizon_too_short(self):
        issues = check_a1_a2_cross(
            _client(investment_horizon=1),
            _product(minimum_horizon=5),
        )
        assert any("minimum period" in i for i in issues)

    def test_horizon_meets_minimum_ok(self):
        issues = check_a1_a2_cross(
            _client(investment_horizon=5),
            _product(minimum_horizon=5),
        )
        assert not any("minimum period" in i for i in issues)

    def test_class6_with_very_low_tolerance(self):
        issues = check_a1_a2_cross(
            _client(risk_tolerance_score=2),
            _product(risk_class=6),
        )
        assert any("extreme risk mismatch" in i for i in issues)

    def test_class7_with_moderate_tolerance(self):
        issues = check_a1_a2_cross(
            _client(risk_tolerance_score=5),
            _product(risk_class=7, leverage=True, potential_loss="total",
                     complexity_tier="COMPLEX", requires_knowledge_level="advanced"),
        )
        assert any("speculative" in i for i in issues)

    def test_complex_product_none_knowledge(self):
        issues = check_a1_a2_cross(
            _client(financial_knowledge="none"),
            _product(complexity_tier="COMPLEX", requires_knowledge_level="moderate"),
        )
        assert any("cannot adequately assess" in i for i in issues)

    def test_total_loss_cannot_afford(self):
        issues = check_a1_a2_cross(
            _client(can_afford_total_loss=False),
            _product(potential_loss="total"),
        )
        assert any("cannot afford to lose all" in i for i in issues)

    def test_total_loss_can_afford_ok(self):
        issues = check_a1_a2_cross(
            _client(can_afford_total_loss=True),
            _product(potential_loss="total"),
        )
        assert not any("cannot afford to lose all" in i for i in issues)

    def test_multiple_cross_issues(self):
        issues = check_a1_a2_cross(
            _client(financial_knowledge="none", investment_horizon=1,
                    risk_tolerance_score=2, can_afford_total_loss=False),
            _product(risk_class=7, requires_knowledge_level="advanced",
                     minimum_horizon=5, potential_loss="total",
                     complexity_tier="COMPLEX", leverage=True),
        )
        assert len(issues) >= 3
