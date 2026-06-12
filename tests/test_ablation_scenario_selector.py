"""
tests/test_ablation_scenario_selector.py
==========================================
Tests that scenario selection is balanced, reproducible,
and produces valid scenario dicts.

Reads real files from data/scenarios/. No LLM calls.
"""

import pytest
from pathlib import Path
from evaluation.ablation_study.scenario_selector import (
    select_scenarios,
    select_and_report,
    TARGETS,
    SCENARIOS_DIR,
)

# Skip all tests if data/scenarios/ doesn't exist (CI without data)
pytestmark = pytest.mark.skipif(
    not SCENARIOS_DIR.exists(),
    reason="data/scenarios/ not found",
)


class TestSelectionBalance:
    def test_total_count_is_50(self):
        scenarios = select_scenarios()
        assert len(scenarios) == 50

    def test_suitable_count(self):
        scenarios = select_scenarios()
        count = sum(1 for s in scenarios if s["expected_decision"] == "SUITABLE")
        assert count == TARGETS["SUITABLE"]

    def test_conditional_count(self):
        scenarios = select_scenarios()
        count = sum(1 for s in scenarios if s["expected_decision"] == "CONDITIONAL")
        assert count == TARGETS["CONDITIONAL"]

    def test_unsuitable_count(self):
        scenarios = select_scenarios()
        count = sum(1 for s in scenarios if s["expected_decision"] == "UNSUITABLE")
        assert count == TARGETS["UNSUITABLE"]

    def test_escalated_count(self):
        scenarios = select_scenarios()
        count = sum(1 for s in scenarios if s["expected_decision"] == "ESCALATED")
        assert count == TARGETS["ESCALATED"]


class TestSelectionReproducibility:
    def test_same_seed_gives_same_scenarios(self):
        s1 = select_scenarios(seed=42)
        s2 = select_scenarios(seed=42)
        names1 = sorted(s["_filename"] for s in s1)
        names2 = sorted(s["_filename"] for s in s2)
        assert names1 == names2

    def test_different_seeds_give_different_scenarios(self):
        s1 = select_scenarios(seed=42)
        s2 = select_scenarios(seed=99)
        names1 = sorted(s["_filename"] for s in s1)
        names2 = sorted(s["_filename"] for s in s2)
        # Very likely to differ (not guaranteed but true for these seeds)
        assert names1 != names2


class TestScenarioSchema:
    def test_all_scenarios_have_expected_decision(self):
        scenarios = select_scenarios()
        for s in scenarios:
            assert "expected_decision" in s, f"{s.get('_filename')} missing expected_decision"

    def test_all_scenarios_have_client_profile(self):
        scenarios = select_scenarios()
        for s in scenarios:
            assert "expected_client_profile" in s, \
                f"{s.get('_filename')} missing expected_client_profile"

    def test_all_scenarios_have_product_profile(self):
        scenarios = select_scenarios()
        for s in scenarios:
            assert "expected_product_profile" in s, \
                f"{s.get('_filename')} missing expected_product_profile"

    def test_all_scenarios_have_path(self):
        scenarios = select_scenarios()
        for s in scenarios:
            assert "_path"     in s
            assert "_filename" in s
            assert Path(s["_path"]).exists(), f"File not found: {s['_path']}"

    def test_no_duplicate_filenames(self):
        scenarios = select_scenarios()
        filenames = [s["_filename"] for s in scenarios]
        assert len(filenames) == len(set(filenames)), \
            "Duplicate scenarios selected"


class TestReportStructure:
    def test_report_keys(self):
        report = select_and_report()
        assert "scenarios"  in report
        assert "counts"     in report
        assert "total"      in report
        assert "filenames"  in report
        assert "seed"       in report
        assert "targets"    in report

    def test_report_total_matches_scenarios(self):
        report = select_and_report()
        assert report["total"] == len(report["scenarios"])

    def test_report_counts_sum_to_total(self):
        report = select_and_report()
        assert sum(report["counts"].values()) == report["total"]


class TestCustomTargets:
    def test_custom_targets_respected(self):
        custom = {"SUITABLE": 5, "CONDITIONAL": 5, "UNSUITABLE": 5, "ESCALATED": 5}
        scenarios = select_scenarios(targets=custom)
        assert len(scenarios) == 20
        for cat, n in custom.items():
            count = sum(1 for s in scenarios if s["expected_decision"] == cat)
            assert count == n, f"Expected {n} {cat}, got {count}"

    def test_insufficient_scenarios_raises_value_error(self):
        # Request more ESCALATED than exist
        custom = {"SUITABLE": 5, "CONDITIONAL": 5, "UNSUITABLE": 5, "ESCALATED": 9999}
        with pytest.raises(ValueError, match="Not enough 'ESCALATED' scenarios"):
            select_scenarios(targets=custom)
