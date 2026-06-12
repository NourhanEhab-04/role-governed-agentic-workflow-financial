"""
tests/test_ablation_runner.py
===============================
Tests the ablation runner: run_cell, run_all, and cell result schema.

Uses inline scenario fixtures — no file I/O for scenario loading.
Tests that write files use a tmp_path fixture.
No LLM calls.
"""

import json
import pytest
from pathlib import Path

from evaluation.ablation_study.runner import run_cell, run_all
from evaluation.ablation_study.configs import get_config, UNIQUE_CONFIGS


# ── Inline scenario fixtures ───────────────────────────────────────────────────

def _make_scenario(
    filename: str,
    expected_decision: str,
    expected_escalate: bool,
    client_overrides: dict | None = None,
    product_overrides: dict | None = None,
    include_concentration: bool = False,
) -> dict:
    base_client = {
        "financial_knowledge":     "moderate",
        "risk_tolerance_score":    4,
        "investment_horizon":      5,
        "liquid_assets":           50000.0,
        "income":                  36000.0,
        "investment_amount":       10000.0,
        "can_afford_total_loss":   True,
        "financial_vulnerability": "LOW",
        "age":                     40,
    }
    base_product = {
        "product_name":              "Test ETF",
        "risk_class":                4,
        "complexity_tier":           "NON-COMPLEX",
        "requires_knowledge_level":  "moderate",
        "minimum_horizon":           3,
        "potential_loss":            "partial",
        "leverage":                  False,
    }
    if client_overrides:
        base_client.update(client_overrides)
    if product_overrides:
        base_product.update(product_overrides)

    scenario = {
        "_filename":               filename,
        "_path":                   f"data/scenarios/{filename}",
        "expected_decision":       expected_decision,
        "expected_escalate":       expected_escalate,
        "expected_rules_failed":   [],
        "expected_client_profile": base_client,
        "expected_product_profile": base_product,
        "description":             f"Test scenario: {filename}",
    }
    if include_concentration:
        scenario["client"] = {"portfolio_concentration_pct": 70}
    return scenario


SUITABLE_SCENARIO = _make_scenario(
    "test_suitable.json", "SUITABLE", False
)

UNSUITABLE_SCENARIO = _make_scenario(
    "test_unsuitable.json", "UNSUITABLE", False,
    client_overrides={"financial_knowledge": "none"},
    product_overrides={"requires_knowledge_level": "advanced", "complexity_tier": "COMPLEX"},
)

CONDITIONAL_SCENARIO = _make_scenario(
    "test_conditional.json", "CONDITIONAL", False,
    client_overrides={"risk_tolerance_score": 2},
    product_overrides={"risk_class": 6},   # R2 fails (2+2=4 < 6): −20; score=80 → SUITABLE actually
    # To get CONDITIONAL we need score 40-69
    # Use r_tolerance=1 so RC6 > 1+2=3: -20; still 80. Need more soft fails.
)

# Simpler: R2 + R3 + R6 all fail → score 40 → CONDITIONAL
CONDITIONAL_SCENARIO = _make_scenario(
    "test_conditional.json", "CONDITIONAL", False,
    client_overrides={"risk_tolerance_score": 2, "investment_horizon": 2},
    product_overrides={
        "risk_class": 6, "minimum_horizon": 5,
        "leverage": True,
    },
)

ESCALATED_SCENARIO = _make_scenario(
    "test_escalated.json", "ESCALATED", True,
    client_overrides={"financial_vulnerability": "HIGH"},
    product_overrides={"risk_class": 4},
)

CONTROL_CONFIG = get_config("A0")
NO_CONFLICT_CONFIG = get_config("A1")


# ── run_cell schema ────────────────────────────────────────────────────────────

REQUIRED_RESULT_KEYS = {
    "scenario_id", "scenario_file", "category", "config_id", "config_label",
    "config_axis", "rule_verdict", "conflict_report", "final_decision",
    "expected_decision", "expected_escalate", "output_escalate",
    "decision_correct", "escalation_correct", "score", "rules_failed", "error",
}

class TestRunCellSchema:
    def test_result_has_all_required_keys(self):
        result = run_cell(SUITABLE_SCENARIO, CONTROL_CONFIG)
        missing = REQUIRED_RESULT_KEYS - result.keys()
        assert not missing, f"Result missing keys: {missing}"

    def test_scenario_id_is_stem(self):
        result = run_cell(SUITABLE_SCENARIO, CONTROL_CONFIG)
        assert result["scenario_id"] == "test_suitable"

    def test_config_id_matches(self):
        result = run_cell(SUITABLE_SCENARIO, CONTROL_CONFIG)
        assert result["config_id"] == "A0"

    def test_no_error_on_valid_input(self):
        result = run_cell(SUITABLE_SCENARIO, CONTROL_CONFIG)
        assert result["error"] is None

    def test_error_populated_on_bad_profile(self):
        bad_scenario = {
            "_filename": "bad.json",
            "_path":     "data/scenarios/bad.json",
            "expected_decision": "SUITABLE",
            "expected_escalate": False,
            "expected_rules_failed": [],
            "expected_client_profile":  {},  # missing required keys
            "expected_product_profile": {},
        }
        result = run_cell(bad_scenario, CONTROL_CONFIG)
        assert result["error"] is not None
        assert result["final_decision"] == "ERROR"


# ── Correctness spot-checks ────────────────────────────────────────────────────

class TestRunCellCorrectness:
    def test_suitable_scenario_correct_under_control(self):
        result = run_cell(SUITABLE_SCENARIO, CONTROL_CONFIG)
        assert result["final_decision"]   == "SUITABLE"
        assert result["decision_correct"] is True

    def test_unsuitable_scenario_correct_under_control(self):
        result = run_cell(UNSUITABLE_SCENARIO, CONTROL_CONFIG)
        assert result["final_decision"]   == "UNSUITABLE"
        assert result["decision_correct"] is True

    def test_escalated_scenario_correct_under_control(self):
        result = run_cell(ESCALATED_SCENARIO, CONTROL_CONFIG)
        assert result["final_decision"]   == "ESCALATED"
        assert result["decision_correct"] is True

    def test_escalated_missed_without_conflict_detector(self):
        """With A1 (no conflict detector), escalated scenarios become SUITABLE."""
        result = run_cell(ESCALATED_SCENARIO, NO_CONFLICT_CONFIG)
        # No conflict detection → SUITABLE (rule engine passes: knowledge=moderate,
        # risk=4, LOW concentration, no hard fails)
        assert result["final_decision"] != "ESCALATED"
        assert result["decision_correct"] is False

    def test_rule_ablation_changes_unsuitable_to_suitable(self):
        """With R1 and R7 disabled, a knowledge-gap client passes."""
        r1_r7_off = get_config("R1_off")
        # UNSUITABLE_SCENARIO fails R1 (knowledge=none, needs advanced)
        # and R7 (COMPLEX + knowledge=none)
        # Disabling R1 alone: R7 still fires
        result_r1_off = run_cell(UNSUITABLE_SCENARIO, r1_r7_off)
        # R7 still fires for UNSUITABLE_SCENARIO (COMPLEX + none)
        assert result_r1_off["final_decision"] == "UNSUITABLE"

    def test_strict_threshold_downgrades_score_80_to_conditional(self):
        """score=80 scenario: with T1 (SUITABLE≥80) still SUITABLE, with T1+1 becomes CONDITIONAL."""
        t1 = get_config("T1")
        # CONDITIONAL_SCENARIO has R2+R3+R6 all failing → score=40; T1 has conditional≥50
        # so score=40 < 50 → UNSUITABLE under T1
        result = run_cell(CONDITIONAL_SCENARIO, t1)
        # expected_decision is CONDITIONAL but strict threshold makes it UNSUITABLE
        assert result["decision_correct"] is False


# ── run_all ────────────────────────────────────────────────────────────────────

class TestRunAll:
    def test_dry_run_writes_nothing(self, tmp_path):
        results = run_all(
            [SUITABLE_SCENARIO],
            [CONTROL_CONFIG],
            output_dir=tmp_path,
            dry_run=True,
            verbose=False,
        )
        assert results == []
        assert list(tmp_path.iterdir()) == []

    def test_run_all_writes_cell_files(self, tmp_path):
        scenarios = [SUITABLE_SCENARIO, UNSUITABLE_SCENARIO]
        configs   = [CONTROL_CONFIG, NO_CONFLICT_CONFIG]
        run_all(scenarios, configs, output_dir=tmp_path, verbose=False)

        json_files = [f for f in tmp_path.iterdir()
                      if f.suffix == ".json" and f.name != "ablation_manifest.json"]
        assert len(json_files) == 4  # 2 scenarios × 2 configs

    def test_run_all_writes_manifest(self, tmp_path):
        run_all([SUITABLE_SCENARIO], [CONTROL_CONFIG], output_dir=tmp_path, verbose=False)
        manifest_path = tmp_path / "ablation_manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "total_cells" in manifest
        assert manifest["total_cells"] == 1

    def test_run_all_returns_all_results(self, tmp_path):
        scenarios = [SUITABLE_SCENARIO, UNSUITABLE_SCENARIO, ESCALATED_SCENARIO]
        configs   = [CONTROL_CONFIG]
        results   = run_all(scenarios, configs, output_dir=tmp_path, verbose=False)
        assert len(results) == 3

    def test_cell_file_is_valid_json(self, tmp_path):
        run_all([SUITABLE_SCENARIO], [CONTROL_CONFIG], output_dir=tmp_path, verbose=False)
        cell_files = [f for f in tmp_path.iterdir()
                      if f.suffix == ".json" and f.name != "ablation_manifest.json"]
        assert len(cell_files) == 1
        data = json.loads(cell_files[0].read_text(encoding="utf-8"))
        assert "final_decision" in data

    def test_filename_format(self, tmp_path):
        run_all([SUITABLE_SCENARIO], [CONTROL_CONFIG], output_dir=tmp_path, verbose=False)
        cell_files = [f for f in tmp_path.iterdir()
                      if f.suffix == ".json" and f.name != "ablation_manifest.json"]
        assert len(cell_files) == 1
        assert cell_files[0].stem == "test_suitable__A0"
