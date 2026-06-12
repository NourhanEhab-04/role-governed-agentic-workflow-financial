"""
tests/test_ablation_conflict_logic.py
=======================================
Tests the deterministic conflict detection layer used in the ablation study.

No LLM calls. No file I/O.
"""

import pytest
from evaluation.ablation_study.conflict_logic import (
    run_conflict_detection,
    derive_final_decision,
)


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def suitable_verdict():
    return {"decision": "SUITABLE", "score": 100, "hard_failed_rules": []}


@pytest.fixture
def conditional_borderline_verdict():
    """Score 48 → BORDERLINE triggers."""
    return {"decision": "CONDITIONAL", "score": 48, "hard_failed_rules": []}


@pytest.fixture
def conditional_safe_verdict():
    """Score 65 → no BORDERLINE (>55)."""
    return {"decision": "CONDITIONAL", "score": 65, "hard_failed_rules": []}


@pytest.fixture
def unsuitable_verdict():
    return {"decision": "UNSUITABLE", "score": 60, "hard_failed_rules": ["R1"]}


@pytest.fixture
def low_vuln_client():
    return {
        "financial_vulnerability": "LOW",
        "portfolio_concentration_pct": 20,
    }


@pytest.fixture
def high_vuln_client_low_conc():
    return {
        "financial_vulnerability": "HIGH",
        "portfolio_concentration_pct": 10,
    }


@pytest.fixture
def high_vuln_client_high_conc():
    """HIGH vulnerability + 65% concentration → CONCENTRATION triggers."""
    return {
        "financial_vulnerability": "HIGH",
        "portfolio_concentration_pct": 65,
    }


@pytest.fixture
def low_vuln_client_extreme_conc():
    """LOW vulnerability + 70% concentration + RC7 → CONCENTRATION triggers."""
    return {
        "financial_vulnerability": "LOW",
        "portfolio_concentration_pct": 70,
    }


@pytest.fixture
def rc4_product():
    return {"risk_class": 4, "complexity_tier": "NON-COMPLEX"}


@pytest.fixture
def rc7_product():
    return {"risk_class": 7, "complexity_tier": "NON-COMPLEX"}


# ── Full detection (control) ───────────────────────────────────────────────────

class TestFullDetection:
    def test_no_flags_clean_suitable(
        self, low_vuln_client, rc4_product, suitable_verdict
    ):
        report = run_conflict_detection(low_vuln_client, rc4_product, suitable_verdict)
        assert report["escalate"] is False
        triggered = [f for f in report["flags"] if f["triggered"]]
        assert triggered == []

    def test_borderline_triggers_on_score_48(
        self, low_vuln_client, rc4_product, conditional_borderline_verdict
    ):
        report = run_conflict_detection(
            low_vuln_client, rc4_product, conditional_borderline_verdict
        )
        borderline = next(f for f in report["flags"] if f["rule_id"] == "BORDERLINE")
        assert borderline["triggered"] is True
        # BORDERLINE is LOW severity — does not cause escalation on its own
        assert report["escalate"] is False

    def test_contradiction_triggers_high_vuln_suitable_rc4(
        self, high_vuln_client_low_conc, rc4_product, suitable_verdict
    ):
        report = run_conflict_detection(
            high_vuln_client_low_conc, rc4_product, suitable_verdict
        )
        contradiction = next(f for f in report["flags"] if f["rule_id"] == "CONTRADICTION")
        assert contradiction["triggered"] is True
        assert report["escalate"] is True

    def test_contradiction_does_not_trigger_on_unsuitable(
        self, high_vuln_client_low_conc, rc4_product, unsuitable_verdict
    ):
        """UNSUITABLE verdict with HIGH vulnerability: CONTRADICTION checks SUITABLE only."""
        report = run_conflict_detection(
            high_vuln_client_low_conc, rc4_product, unsuitable_verdict
        )
        contradiction = next(f for f in report["flags"] if f["rule_id"] == "CONTRADICTION")
        assert contradiction["triggered"] is False

    def test_concentration_triggers_low_vuln_extreme_conc_rc7(
        self, low_vuln_client_extreme_conc, rc7_product, suitable_verdict
    ):
        report = run_conflict_detection(
            low_vuln_client_extreme_conc, rc7_product, suitable_verdict
        )
        concentration = next(f for f in report["flags"] if f["rule_id"] == "CONCENTRATION")
        assert concentration["triggered"] is True
        assert report["escalate"] is True

    def test_escalation_requires_high_severity_flag(
        self, low_vuln_client, rc4_product, conditional_borderline_verdict
    ):
        """BORDERLINE alone (LOW severity) must not set escalate=True."""
        report = run_conflict_detection(
            low_vuln_client, rc4_product, conditional_borderline_verdict
        )
        escalation = next(f for f in report["flags"] if f["rule_id"] == "ESCALATION")
        assert escalation["triggered"] is False
        assert report["escalate"] is False

    def test_output_has_exactly_four_flags(
        self, low_vuln_client, rc4_product, suitable_verdict
    ):
        report = run_conflict_detection(low_vuln_client, rc4_product, suitable_verdict)
        assert len(report["flags"]) == 4
        flag_ids = {f["rule_id"] for f in report["flags"]}
        assert flag_ids == {"BORDERLINE", "CONCENTRATION", "CONTRADICTION", "ESCALATION"}

    def test_output_has_escalate_and_summary(
        self, low_vuln_client, rc4_product, suitable_verdict
    ):
        report = run_conflict_detection(low_vuln_client, rc4_product, suitable_verdict)
        assert "escalate" in report
        assert "summary"  in report
        assert isinstance(report["escalate"], bool)


# ── Flag disabling ─────────────────────────────────────────────────────────────

class TestFlagDisabling:
    def test_disabling_all_flags_gives_no_escalation(
        self, high_vuln_client_low_conc, rc4_product, suitable_verdict
    ):
        """Contradiction would fire, but all flags disabled → no escalation."""
        report = run_conflict_detection(
            high_vuln_client_low_conc, rc4_product, suitable_verdict,
            disabled_flags=["BORDERLINE", "CONCENTRATION", "CONTRADICTION", "ESCALATION"],
        )
        assert report["escalate"] is False
        for flag in report["flags"]:
            assert flag["triggered"] is False

    def test_disabling_contradiction_stops_escalation(
        self, high_vuln_client_low_conc, rc4_product, suitable_verdict
    ):
        """With CONTRADICTION disabled, a scenario that would escalate doesn't."""
        report = run_conflict_detection(
            high_vuln_client_low_conc, rc4_product, suitable_verdict,
            disabled_flags=["CONTRADICTION"],
        )
        contradiction = next(f for f in report["flags"] if f["rule_id"] == "CONTRADICTION")
        assert contradiction["triggered"] is False
        assert report["escalate"] is False

    def test_disabling_concentration_stops_concentration_escalation(
        self, low_vuln_client_extreme_conc, rc7_product, suitable_verdict
    ):
        report = run_conflict_detection(
            low_vuln_client_extreme_conc, rc7_product, suitable_verdict,
            disabled_flags=["CONCENTRATION"],
        )
        concentration = next(f for f in report["flags"] if f["rule_id"] == "CONCENTRATION")
        assert concentration["triggered"] is False
        assert report["escalate"] is False

    def test_disabling_borderline_does_not_affect_escalation(
        self, high_vuln_client_low_conc, rc4_product, suitable_verdict
    ):
        """Disabling BORDERLINE doesn't affect HIGH-severity escalation."""
        report = run_conflict_detection(
            high_vuln_client_low_conc, rc4_product, suitable_verdict,
            disabled_flags=["BORDERLINE"],
        )
        # CONTRADICTION still fires → escalation still happens
        assert report["escalate"] is True

    def test_disabled_flag_message_contains_ablation_marker(
        self, low_vuln_client, rc4_product, suitable_verdict
    ):
        report = run_conflict_detection(
            low_vuln_client, rc4_product, suitable_verdict,
            disabled_flags=["BORDERLINE"],
        )
        borderline = next(f for f in report["flags"] if f["rule_id"] == "BORDERLINE")
        assert "ABLATION" in borderline["message"]


# ── derive_final_decision ──────────────────────────────────────────────────────

class TestDeriveFinalDecision:
    def test_escalated_when_escalate_true(self):
        verdict = {"decision": "SUITABLE"}
        conflict = {"escalate": True}
        assert derive_final_decision(verdict, conflict) == "ESCALATED"

    def test_suitable_when_no_escalation(self):
        verdict = {"decision": "SUITABLE"}
        conflict = {"escalate": False}
        assert derive_final_decision(verdict, conflict) == "SUITABLE"

    def test_conditional_when_no_escalation(self):
        verdict = {"decision": "CONDITIONAL"}
        conflict = {"escalate": False}
        assert derive_final_decision(verdict, conflict) == "CONDITIONAL"

    def test_unsuitable_when_no_escalation(self):
        verdict = {"decision": "UNSUITABLE"}
        conflict = {"escalate": False}
        assert derive_final_decision(verdict, conflict) == "UNSUITABLE"
