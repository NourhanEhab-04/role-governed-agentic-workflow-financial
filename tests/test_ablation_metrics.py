"""
tests/test_ablation_metrics.py
================================
Tests metric computation on known synthetic result sets.

No LLM calls. No file I/O.
"""

import math
import pytest
from evaluation.ablation_study.metrics import (
    decision_accuracy,
    per_category_accuracy,
    unsuitable_recall,
    unsuitable_precision,
    escalation_recall,
    escalation_precision,
    escalation_f1,
    category_parity_gap,
    compute_config_metrics,
    compute_all_metrics,
    CATEGORIES,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _r(final: str, expected: str, escalate_out: bool = False) -> dict:
    """Build a minimal result dict."""
    return {
        "final_decision":    final,
        "expected_decision": expected,
        "category":          expected,
        "output_escalate":   escalate_out,
        "decision_correct":  final == expected,
        "error":             None,
    }


PERFECT_RESULTS = [
    _r("SUITABLE",    "SUITABLE"),
    _r("SUITABLE",    "SUITABLE"),
    _r("CONDITIONAL", "CONDITIONAL"),
    _r("UNSUITABLE",  "UNSUITABLE"),
    _r("ESCALATED",   "ESCALATED", escalate_out=True),
]

ALL_WRONG_RESULTS = [
    _r("UNSUITABLE",  "SUITABLE"),
    _r("SUITABLE",    "CONDITIONAL"),
    _r("SUITABLE",    "UNSUITABLE"),
    _r("SUITABLE",    "ESCALATED"),
]

MIXED_RESULTS = [
    _r("SUITABLE",    "SUITABLE"),     # correct
    _r("CONDITIONAL", "SUITABLE"),     # wrong
    _r("UNSUITABLE",  "UNSUITABLE"),   # correct
    _r("SUITABLE",    "UNSUITABLE"),   # wrong (missed unsuitable)
    _r("ESCALATED",   "ESCALATED", escalate_out=True),   # correct
    _r("SUITABLE",    "ESCALATED"),    # wrong (missed escalation)
]


# ── M1 Decision Accuracy ───────────────────────────────────────────────────────

class TestDecisionAccuracy:
    def test_perfect_results(self):
        assert decision_accuracy(PERFECT_RESULTS) == 1.0

    def test_all_wrong_results(self):
        assert decision_accuracy(ALL_WRONG_RESULTS) == 0.0

    def test_mixed_results(self):
        # 3 correct out of 6
        assert decision_accuracy(MIXED_RESULTS) == pytest.approx(3 / 6)

    def test_empty_list(self):
        assert decision_accuracy([]) == 0.0

    def test_errors_excluded(self):
        results = [
            _r("SUITABLE", "SUITABLE"),
            {"final_decision": "ERROR", "expected_decision": "SUITABLE",
             "category": "SUITABLE", "decision_correct": False, "error": "boom"},
        ]
        # Only the non-error result counts
        assert decision_accuracy(results) == 1.0


# ── M1c Per-category Accuracy ──────────────────────────────────────────────────

class TestPerCategoryAccuracy:
    def test_perfect_all_categories(self):
        acc = per_category_accuracy(PERFECT_RESULTS)
        assert acc["SUITABLE"]    == 1.0
        assert acc["CONDITIONAL"] == 1.0
        assert acc["UNSUITABLE"]  == 1.0
        assert acc["ESCALATED"]   == 1.0

    def test_missing_category_is_nan(self):
        results = [_r("SUITABLE", "SUITABLE")]
        acc = per_category_accuracy(results)
        assert math.isnan(acc["CONDITIONAL"])
        assert math.isnan(acc["UNSUITABLE"])
        assert math.isnan(acc["ESCALATED"])

    def test_partial_accuracy_per_category(self):
        results = [
            _r("SUITABLE", "SUITABLE"),
            _r("UNSUITABLE", "SUITABLE"),   # wrong
        ]
        acc = per_category_accuracy(results)
        assert acc["SUITABLE"] == pytest.approx(0.5)

    def test_returns_all_categories(self):
        acc = per_category_accuracy(PERFECT_RESULTS)
        assert set(acc.keys()) == set(CATEGORIES)


# ── M2 Unsuitable recall / precision ──────────────────────────────────────────

class TestUnsuitableMetrics:
    def test_perfect_recall(self):
        assert unsuitable_recall(PERFECT_RESULTS) == 1.0

    def test_zero_recall_when_all_missed(self):
        results = [_r("SUITABLE", "UNSUITABLE")]
        assert unsuitable_recall(results) == 0.0

    def test_perfect_precision(self):
        assert unsuitable_precision(PERFECT_RESULTS) == 1.0

    def test_zero_precision_when_false_positives(self):
        results = [_r("UNSUITABLE", "SUITABLE")]
        assert unsuitable_precision(results) == 0.0

    def test_no_unsuitable_scenarios_recall_is_zero(self):
        results = [_r("SUITABLE", "SUITABLE")]
        assert unsuitable_recall(results) == 0.0

    def test_no_unsuitable_output_precision_is_zero(self):
        results = [_r("SUITABLE", "SUITABLE")]
        assert unsuitable_precision(results) == 0.0


# ── M3 Escalation metrics ──────────────────────────────────────────────────────

class TestEscalationMetrics:
    def test_perfect_escalation_recall(self):
        assert escalation_recall(PERFECT_RESULTS) == 1.0

    def test_zero_escalation_recall_when_all_missed(self):
        results = [_r("SUITABLE", "ESCALATED")]
        assert escalation_recall(results) == 0.0

    def test_escalation_f1_perfect(self):
        assert escalation_f1(PERFECT_RESULTS) == 1.0

    def test_escalation_f1_zero_when_no_hits(self):
        results = [_r("SUITABLE", "ESCALATED"), _r("ESCALATED", "SUITABLE")]
        # recall=0, precision=0 → f1=0
        assert escalation_f1(results) == 0.0

    def test_escalation_f1_mixed(self):
        results = [
            _r("ESCALATED", "ESCALATED"),   # TP
            _r("SUITABLE",  "ESCALATED"),   # FN
        ]
        rec = 0.5
        pre = 1.0
        expected_f1 = 2 * rec * pre / (rec + pre)
        assert escalation_f1(results) == pytest.approx(expected_f1)


# ── M6 Category Parity Gap ─────────────────────────────────────────────────────

class TestCategoryParityGap:
    def test_zero_gap_when_all_perfect(self):
        assert category_parity_gap(PERFECT_RESULTS) == 0.0

    def test_gap_is_max_minus_min(self):
        results = [
            _r("SUITABLE",    "SUITABLE"),    # cat SUITABLE: 1 correct / 1
            _r("UNSUITABLE",  "UNSUITABLE"),  # cat UNSUITABLE: 1 correct / 1
            _r("SUITABLE",    "CONDITIONAL"), # cat CONDITIONAL: 0 correct / 1 = 0.0
        ]
        gap = category_parity_gap(results)
        # max=1.0, min=0.0 → gap=1.0
        assert gap == pytest.approx(1.0)

    def test_single_category_returns_zero(self):
        results = [_r("SUITABLE", "SUITABLE"), _r("SUITABLE", "SUITABLE")]
        gap = category_parity_gap(results)
        # Only SUITABLE present; other categories are NaN; only 1 valid value
        assert math.isnan(gap) or gap == 0.0


# ── compute_config_metrics ─────────────────────────────────────────────────────

class TestComputeConfigMetrics:
    def test_returns_all_metric_keys(self):
        m = compute_config_metrics(PERFECT_RESULTS)
        expected_keys = {
            "M1_decision_accuracy",
            "M1c_per_category",
            "M2_unsuitable_recall",
            "M2p_unsuitable_precision",
            "M3_escalation_recall",
            "M3p_escalation_precision",
            "M3f_escalation_f1",
            "M6_parity_gap",
            "n_cells",
            "n_errors",
        }
        assert expected_keys <= m.keys()

    def test_n_cells_excludes_errors(self):
        results = [
            _r("SUITABLE", "SUITABLE"),
            {"final_decision": "ERROR", "expected_decision": "SUITABLE",
             "category": "SUITABLE", "decision_correct": False, "error": "oops"},
        ]
        m = compute_config_metrics(results)
        assert m["n_cells"]  == 1
        assert m["n_errors"] == 1


# ── compute_all_metrics ────────────────────────────────────────────────────────

class TestComputeAllMetrics:
    def test_structure(self):
        by_config = {
            "A0": PERFECT_RESULTS,
            "A1": ALL_WRONG_RESULTS,
        }
        metrics = compute_all_metrics(by_config, control_id="A0")
        assert "per_config" in metrics
        assert "deltas"     in metrics
        assert "control_id" in metrics
        assert "A0" in metrics["per_config"]
        assert "A1" in metrics["per_config"]

    def test_control_has_no_delta(self):
        by_config = {"A0": PERFECT_RESULTS, "A1": ALL_WRONG_RESULTS}
        metrics = compute_all_metrics(by_config, control_id="A0")
        # A0 is control; its delta is not in the deltas dict
        assert "A0" not in metrics["deltas"]

    def test_negative_delta_when_ablation_hurts(self):
        by_config = {"A0": PERFECT_RESULTS, "A1": ALL_WRONG_RESULTS}
        metrics = compute_all_metrics(by_config, control_id="A0")
        # A1 is worse than A0 → negative delta
        assert metrics["deltas"]["A1"] < 0

    def test_positive_delta_impossible_when_control_perfect(self):
        by_config = {"A0": PERFECT_RESULTS, "A1": PERFECT_RESULTS}
        metrics = compute_all_metrics(by_config, control_id="A0")
        assert metrics["deltas"]["A1"] == pytest.approx(0.0)
