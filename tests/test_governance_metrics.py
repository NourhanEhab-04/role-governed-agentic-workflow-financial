"""
tests/test_governance_metrics.py
==================================
Unit tests for evaluation/governance_metrics.py (M8-M15).

All tests are:
  - pure (no I/O, no LLM calls, no side effects)
  - independent (each test builds its own fixture data)
  - verifiable (assertions on exact values)

Run with: pytest tests/test_governance_metrics.py -v
"""

import pytest

from evaluation.governance_metrics import (
    override_rate,
    pipeline_halt_rate,
    three_point_integrity_rate,
    hard_rule_enforcement_rate,
    vulnerability_protection_rate,
    regulatory_citation_rate,
    explanation_completeness,
    decision_traceability_score,
    compute_governance_metrics,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _base_pipeline_result(**overrides) -> dict:
    """Minimal result dict representing a successful pipeline run."""
    r = {
        "output_decision":         "SUITABLE",
        "expected_decision":       "SUITABLE",
        "halted":                  False,
        "three_point_agreed":      True,
        "a1_overrides_applied":    False,
        "a2_overrides_applied":    False,
        "a4_overrides_applied":    False,
        "output_hard_failed_rules": [],
        "client_vulnerability":    "LOW",
        "regulatory_basis":        "MiFID II Article 25(2) — R1, R2, R3.",
        "output_rule_explanations": {
            "R1": "Knowledge check passed.",
            "R2": "Risk alignment check passed.",
            "R3": "Horizon check passed.",
            "R4": "Affordability check passed.",
            "R5": "Vulnerability check passed.",
            "R6": "Leverage check passed.",
            "R7": "Complexity check passed.",
        },
        "pre_check_present":    True,
        "rule_details_present": True,
        "conflict_present":     True,
        "expected_rules_failed": [],
    }
    r.update(overrides)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# M8 — Override Rate
# ─────────────────────────────────────────────────────────────────────────────

class TestOverrideRate:
    def test_no_overrides_any_agent(self):
        results = [_base_pipeline_result()] * 4
        rates = override_rate(results)
        assert rates["A1"] == 0.0
        assert rates["A2"] == 0.0
        assert rates["A4"] == 0.0

    def test_a1_all_overridden(self):
        results = [_base_pipeline_result(a1_overrides_applied=True)] * 3
        rates = override_rate(results)
        assert rates["A1"] == 1.0
        assert rates["A2"] == 0.0

    def test_partial_override_rate(self):
        results = [
            _base_pipeline_result(a1_overrides_applied=True),
            _base_pipeline_result(a1_overrides_applied=True),
            _base_pipeline_result(a1_overrides_applied=False),
            _base_pipeline_result(a1_overrides_applied=False),
        ]
        rates = override_rate(results)
        assert rates["A1"] == 0.5

    def test_a4_override_tracked_independently(self):
        results = [
            _base_pipeline_result(a4_overrides_applied=True),
            _base_pipeline_result(a4_overrides_applied=False),
        ]
        rates = override_rate(results)
        assert rates["A4"] == 0.5
        assert rates["A1"] == 0.0

    def test_missing_key_returns_none(self):
        # Results without override keys → None (not 0.0, to distinguish "absent" from "no overrides")
        results = [{"output_decision": "SUITABLE", "halted": False}] * 3
        rates = override_rate(results)
        assert rates["A1"] is None
        assert rates["A2"] is None
        assert rates["A4"] is None

    def test_empty_results(self):
        rates = override_rate([])
        assert rates == {"A1": None, "A2": None, "A4": None}

    def test_mixed_key_presence(self):
        # Some results have the key, some don't — count only those that have it
        results = [
            _base_pipeline_result(a2_overrides_applied=True),
            {"output_decision": "SUITABLE"},  # no a2_overrides_applied key
        ]
        rates = override_rate(results)
        assert rates["A2"] == 1.0  # 1/1 runs that have the key

    def test_rounding(self):
        results = [_base_pipeline_result(a1_overrides_applied=(i % 3 == 0)) for i in range(7)]
        rates = override_rate(results)
        # 3 out of 7 = 0.4286
        assert abs(rates["A1"] - round(3/7, 4)) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# M9 — Pipeline Halt Rate
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineHaltRate:
    def test_no_halts(self):
        results = [_base_pipeline_result()] * 5
        assert pipeline_halt_rate(results) == 0.0

    def test_all_halted(self):
        results = [_base_pipeline_result(halted=True)] * 5
        assert pipeline_halt_rate(results) == 1.0

    def test_partial_halts(self):
        results = [
            _base_pipeline_result(halted=True),
            _base_pipeline_result(halted=True),
            _base_pipeline_result(halted=False),
            _base_pipeline_result(halted=False),
        ]
        assert pipeline_halt_rate(results) == 0.5

    def test_empty_results(self):
        assert pipeline_halt_rate([]) == 0.0

    def test_missing_halted_key_treated_as_not_halted(self):
        # Absent key → not halted
        results = [{"output_decision": "SUITABLE"}] * 4
        assert pipeline_halt_rate(results) == 0.0

    def test_single_halt_in_many_runs(self):
        results = [_base_pipeline_result(halted=False)] * 9 + [_base_pipeline_result(halted=True)]
        assert pipeline_halt_rate(results) == 0.1


# ─────────────────────────────────────────────────────────────────────────────
# M10 — Three-Point Integrity Rate
# ─────────────────────────────────────────────────────────────────────────────

class TestThreePointIntegrityRate:
    def test_all_agreed(self):
        results = [_base_pipeline_result(three_point_agreed=True)] * 5
        assert three_point_integrity_rate(results) == 1.0

    def test_none_agreed(self):
        results = [_base_pipeline_result(three_point_agreed=False)] * 5
        assert three_point_integrity_rate(results) == 0.0

    def test_partial_agreement(self):
        results = [
            _base_pipeline_result(three_point_agreed=True),
            _base_pipeline_result(three_point_agreed=True),
            _base_pipeline_result(three_point_agreed=False),
        ]
        assert abs(three_point_integrity_rate(results) - round(2/3, 4)) < 1e-9

    def test_missing_key_returns_none(self):
        results = [{"output_decision": "SUITABLE"}] * 3
        assert three_point_integrity_rate(results) is None

    def test_empty_results_returns_none(self):
        assert three_point_integrity_rate([]) is None


# ─────────────────────────────────────────────────────────────────────────────
# M11 — Hard Rule Enforcement Rate
# ─────────────────────────────────────────────────────────────────────────────

class TestHardRuleEnforcementRate:
    def test_hard_fail_correctly_produces_unsuitable(self):
        results = [
            _base_pipeline_result(
                output_hard_failed_rules=["R4"],
                output_decision="UNSUITABLE",
            )
        ]
        assert hard_rule_enforcement_rate(results) == 1.0

    def test_hard_fail_correctly_produces_escalated(self):
        results = [
            _base_pipeline_result(
                output_hard_failed_rules=["R1"],
                output_decision="ESCALATED",
            )
        ]
        assert hard_rule_enforcement_rate(results) == 1.0

    def test_hard_fail_incorrectly_produces_suitable(self):
        results = [
            _base_pipeline_result(
                output_hard_failed_rules=["R3"],
                output_decision="SUITABLE",   # wrong!
            )
        ]
        assert hard_rule_enforcement_rate(results) == 0.0

    def test_no_hard_fails_returns_none(self):
        results = [_base_pipeline_result(output_hard_failed_rules=[])] * 5
        assert hard_rule_enforcement_rate(results) is None

    def test_mixed_enforcement(self):
        results = [
            _base_pipeline_result(output_hard_failed_rules=["R4"], output_decision="UNSUITABLE"),
            _base_pipeline_result(output_hard_failed_rules=["R1"], output_decision="SUITABLE"),  # violation
            _base_pipeline_result(output_hard_failed_rules=["R5"], output_decision="ESCALATED"),
        ]
        assert abs(hard_rule_enforcement_rate(results) - round(2/3, 4)) < 1e-9

    def test_empty_results(self):
        assert hard_rule_enforcement_rate([]) is None


# ─────────────────────────────────────────────────────────────────────────────
# M12 — Vulnerability Protection Rate
# ─────────────────────────────────────────────────────────────────────────────

class TestVulnerabilityProtectionRate:
    def test_high_vuln_gets_unsuitable(self):
        results = [
            _base_pipeline_result(client_vulnerability="HIGH", output_decision="UNSUITABLE")
        ]
        assert vulnerability_protection_rate(results) == 1.0

    def test_high_vuln_gets_escalated(self):
        results = [
            _base_pipeline_result(client_vulnerability="HIGH", output_decision="ESCALATED")
        ]
        assert vulnerability_protection_rate(results) == 1.0

    def test_high_vuln_gets_conditional_counts_as_protected(self):
        results = [
            _base_pipeline_result(client_vulnerability="HIGH", output_decision="CONDITIONAL")
        ]
        assert vulnerability_protection_rate(results) == 1.0

    def test_high_vuln_gets_suitable_is_violation(self):
        results = [
            _base_pipeline_result(client_vulnerability="HIGH", output_decision="SUITABLE")
        ]
        assert vulnerability_protection_rate(results) == 0.0

    def test_no_high_vuln_returns_none(self):
        results = [_base_pipeline_result(client_vulnerability="LOW")] * 5
        assert vulnerability_protection_rate(results) is None

    def test_mixed_protection(self):
        results = [
            _base_pipeline_result(client_vulnerability="HIGH", output_decision="UNSUITABLE"),
            _base_pipeline_result(client_vulnerability="HIGH", output_decision="SUITABLE"),  # violation
            _base_pipeline_result(client_vulnerability="LOW",  output_decision="SUITABLE"),  # not counted
        ]
        assert vulnerability_protection_rate(results) == 0.5

    def test_empty_results(self):
        assert vulnerability_protection_rate([]) is None


# ─────────────────────────────────────────────────────────────────────────────
# M13 — Regulatory Citation Rate
# ─────────────────────────────────────────────────────────────────────────────

class TestRegulatoryCitationRate:
    def test_full_citation_present(self):
        results = [
            _base_pipeline_result(regulatory_basis="MiFID II Article 25(2) — R1, R2.")
        ]
        assert regulatory_citation_rate(results) == 1.0

    def test_missing_article_25_not_cited(self):
        results = [
            _base_pipeline_result(regulatory_basis="MiFID II general rules apply.")
        ]
        assert regulatory_citation_rate(results) == 0.0

    def test_no_regulatory_basis_field_returns_none(self):
        results = [{"output_decision": "SUITABLE"}] * 3
        assert regulatory_citation_rate(results) is None

    def test_none_regulatory_basis_not_counted(self):
        results = [
            _base_pipeline_result(regulatory_basis=None),
            _base_pipeline_result(regulatory_basis="MiFID II Article 25(2) — R3."),
        ]
        # Only 1 run has a non-None basis, and it cites correctly → 1/1
        assert regulatory_citation_rate(results) == 1.0

    def test_mixed_citation(self):
        results = [
            _base_pipeline_result(regulatory_basis="MiFID II Article 25(2) — ok."),
            _base_pipeline_result(regulatory_basis="ESMA guidelines apply."),  # no Article 25
        ]
        assert regulatory_citation_rate(results) == 0.5

    def test_empty_results(self):
        assert regulatory_citation_rate([]) is None


# ─────────────────────────────────────────────────────────────────────────────
# M14 — Explanation Completeness
# ─────────────────────────────────────────────────────────────────────────────

class TestExplanationCompleteness:
    def test_all_failed_rules_explained(self):
        results = [
            _base_pipeline_result(
                expected_rules_failed=["R1", "R3"],
                output_rule_explanations={"R1": "Knowledge mismatch.", "R3": "Horizon too short."},
            )
        ]
        assert explanation_completeness(results) == 1.0

    def test_no_explanation_for_failed_rule(self):
        results = [
            _base_pipeline_result(
                expected_rules_failed=["R1"],
                output_rule_explanations={"R1": ""},  # empty explanation
            )
        ]
        assert explanation_completeness(results) == 0.0

    def test_partial_explanation_coverage(self):
        results = [
            _base_pipeline_result(
                expected_rules_failed=["R1", "R2", "R4"],
                output_rule_explanations={
                    "R1": "Knowledge mismatch.",
                    "R2": "",         # missing
                    "R4": "Cannot afford total loss.",
                },
            )
        ]
        # 2 out of 3 failed rules have explanations
        assert abs(explanation_completeness(results) - round(2/3, 4)) < 1e-9

    def test_no_failed_rules_returns_none(self):
        results = [_base_pipeline_result(expected_rules_failed=[])] * 5
        assert explanation_completeness(results) is None

    def test_missing_explanation_dict(self):
        results = [
            _base_pipeline_result(
                expected_rules_failed=["R1"],
                output_rule_explanations={},  # empty dict, R1 not present
            )
        ]
        assert explanation_completeness(results) == 0.0

    def test_average_across_multiple_runs(self):
        results = [
            _base_pipeline_result(
                expected_rules_failed=["R1"],
                output_rule_explanations={"R1": "Explanation."},
            ),
            _base_pipeline_result(
                expected_rules_failed=["R2", "R3"],
                output_rule_explanations={"R2": "Explanation.", "R3": ""},
            ),
        ]
        # run1 = 1.0, run2 = 0.5 → avg = 0.75
        assert explanation_completeness(results) == 0.75

    def test_empty_results(self):
        assert explanation_completeness([]) is None


# ─────────────────────────────────────────────────────────────────────────────
# M15 — Decision Traceability Score
# ─────────────────────────────────────────────────────────────────────────────

class TestDecisionTraceabilityScore:
    def test_fully_traceable_run(self):
        results = [_base_pipeline_result()]
        # All 5 components present → score = 1.0
        assert decision_traceability_score(results) == 1.0

    def test_no_traceability(self):
        results = [{
            "output_decision": "SUITABLE",
            "pre_check_present":    False,
            "rule_details_present": False,
            "conflict_present":     False,
            "regulatory_basis":     None,
            "output_rule_explanations": {},
        }]
        assert decision_traceability_score(results) == 0.0

    def test_partial_traceability_3_of_5(self):
        results = [{
            "pre_check_present":    True,
            "rule_details_present": True,
            "conflict_present":     True,
            "regulatory_basis":     None,                # missing
            "output_rule_explanations": {},              # missing
        }]
        assert decision_traceability_score(results) == 0.6

    def test_explanation_component_requires_nonempty_string(self):
        results = [{
            "pre_check_present":    False,
            "rule_details_present": False,
            "conflict_present":     False,
            "regulatory_basis":     None,
            "output_rule_explanations": {"R1": "   "},  # whitespace only → not counted
        }]
        assert decision_traceability_score(results) == 0.0

    def test_regulatory_basis_component_requires_mifid_article25(self):
        # "MiFID II" present but "25" absent → not counted
        results = [{
            "pre_check_present":    False,
            "rule_details_present": False,
            "conflict_present":     False,
            "regulatory_basis":     "MiFID II general rules",  # no "25"
            "output_rule_explanations": {},
        }]
        assert decision_traceability_score(results) == 0.0

    def test_average_across_runs(self):
        results = [
            _base_pipeline_result(),  # score = 1.0
            {
                "pre_check_present":    True,
                "rule_details_present": True,
                "conflict_present":     False,
                "regulatory_basis":     None,
                "output_rule_explanations": {},
            },  # score = 0.4
        ]
        expected = round((1.0 + 0.4) / 2, 4)
        assert decision_traceability_score(results) == expected

    def test_empty_results(self):
        assert decision_traceability_score([]) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_governance_metrics — integration
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeGovernanceMetrics:
    def test_pipeline_keys_present(self):
        results = [_base_pipeline_result()]
        summary = compute_governance_metrics(results, "pipeline")
        expected_keys = {
            "architecture",
            "M8_override_rate",
            "M9_pipeline_halt_rate",
            "M10_three_point_integrity",
            "M11_hard_rule_enforcement",
            "M12_vulnerability_protection",
            "M13_regulatory_citation_rate",
            "M14_explanation_completeness",
            "M15_decision_traceability",
        }
        assert set(summary.keys()) == expected_keys
        assert summary["architecture"] == "pipeline"

    def test_baseline_pipeline_only_metrics_are_na(self):
        results = [_base_pipeline_result()]
        summary = compute_governance_metrics(results, "baseline")
        assert summary["M8_override_rate"] == "N/A (baseline)"
        assert summary["M10_three_point_integrity"] == "N/A (baseline)"
        assert summary["M13_regulatory_citation_rate"] == "N/A (baseline)"
        assert summary["M14_explanation_completeness"] == "N/A (baseline)"
        assert summary["M15_decision_traceability"] == "N/A (baseline)"

    def test_baseline_universal_metrics_computed(self):
        results = [_base_pipeline_result(halted=False)]
        summary = compute_governance_metrics(results, "baseline")
        assert summary["M9_pipeline_halt_rate"] == 0.0

    def test_override_rate_in_summary_is_dict(self):
        results = [_base_pipeline_result(a1_overrides_applied=True)]
        summary = compute_governance_metrics(results, "pipeline")
        assert isinstance(summary["M8_override_rate"], dict)
        assert "A1" in summary["M8_override_rate"]

    def test_clean_all_pass_pipeline_run(self):
        results = [_base_pipeline_result()] * 10
        summary = compute_governance_metrics(results, "pipeline")
        assert summary["M9_pipeline_halt_rate"] == 0.0
        assert summary["M10_three_point_integrity"] == 1.0
        assert summary["M13_regulatory_citation_rate"] == 1.0
        assert summary["M15_decision_traceability"] == 1.0
