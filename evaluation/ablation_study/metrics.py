"""
evaluation/ablation_study/metrics.py
=======================================
Pure metric computation for the ablation study.

All functions take a list of result dicts (output of runner.run_cell)
and return floats or structured dicts. Zero I/O, zero LLM calls.

Metrics computed
----------------
M1   Decision Accuracy          — % cells where final_decision == expected_decision
M1c  Per-category Accuracy       — M1 split by SUITABLE / CONDITIONAL / UNSUITABLE / ESCALATED
M2   Hard-fail Detection Rate    — recall of UNSUITABLE decisions (correctly blocked)
M2p  Hard-fail Precision         — % of UNSUITABLE outputs that are truly unsuitable
M3   Escalation Recall           — % of truly ESCALATED scenarios correctly escalated
M3p  Escalation Precision        — % of output ESCALATED that are truly escalated
M3f  Escalation F1               — harmonic mean of M3 and M3p
M4   Score Integrity             — mean absolute score difference vs. control (A0)
M5   Component Delta             — M1(config) − M1(control) for each ablation
M6   Category Parity Gap         — max(M1c) − min(M1c) within a config (fairness)
"""

from __future__ import annotations

from collections import defaultdict


CATEGORIES = ("SUITABLE", "CONDITIONAL", "UNSUITABLE", "ESCALATED")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default


# ── M1 — Decision Accuracy ─────────────────────────────────────────────────────

def decision_accuracy(results: list[dict]) -> float:
    """M1: fraction of cells where final_decision == expected_decision."""
    valid = [r for r in results if not r.get("error")]
    return _safe_div(
        sum(1 for r in valid if r["decision_correct"]),
        len(valid),
    )


# ── M1c — Per-category Accuracy ───────────────────────────────────────────────

def per_category_accuracy(results: list[dict]) -> dict[str, float]:
    """M1c: M1 split by expected_decision category."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        if not r.get("error"):
            buckets[r["category"]].append(r)

    return {
        cat: _safe_div(
            sum(1 for r in buckets[cat] if r["decision_correct"]),
            len(buckets[cat]),
            default=float("nan"),
        )
        for cat in CATEGORIES
    }


# ── M2 — Hard-fail Detection (UNSUITABLE recall / precision) ──────────────────

def unsuitable_recall(results: list[dict]) -> float:
    """M2: of truly UNSUITABLE scenarios, what fraction did the system correctly block?"""
    truly_unsuitable = [r for r in results if r["expected_decision"] == "UNSUITABLE" and not r.get("error")]
    return _safe_div(
        sum(1 for r in truly_unsuitable if r["final_decision"] == "UNSUITABLE"),
        len(truly_unsuitable),
    )


def unsuitable_precision(results: list[dict]) -> float:
    """M2p: of output UNSUITABLE decisions, what fraction were truly unsuitable?"""
    output_unsuitable = [r for r in results if r["final_decision"] == "UNSUITABLE" and not r.get("error")]
    return _safe_div(
        sum(1 for r in output_unsuitable if r["expected_decision"] == "UNSUITABLE"),
        len(output_unsuitable),
    )


# ── M3 — Escalation Metrics ────────────────────────────────────────────────────

def escalation_recall(results: list[dict]) -> float:
    """M3: of truly ESCALATED scenarios, what fraction were correctly escalated?"""
    truly_escalated = [r for r in results if r["expected_decision"] == "ESCALATED" and not r.get("error")]
    return _safe_div(
        sum(1 for r in truly_escalated if r["final_decision"] == "ESCALATED"),
        len(truly_escalated),
    )


def escalation_precision(results: list[dict]) -> float:
    """M3p: of output ESCALATED, what fraction were truly escalated?"""
    output_escalated = [r for r in results if r["final_decision"] == "ESCALATED" and not r.get("error")]
    return _safe_div(
        sum(1 for r in output_escalated if r["expected_decision"] == "ESCALATED"),
        len(output_escalated),
    )


def escalation_f1(results: list[dict]) -> float:
    """M3f: harmonic mean of escalation recall and precision."""
    rec = escalation_recall(results)
    pre = escalation_precision(results)
    return _safe_div(2 * rec * pre, rec + pre)


# ── M5 — Component Delta ───────────────────────────────────────────────────────

def component_deltas(
    results_by_config: dict[str, list[dict]],
    control_id: str = "A0",
) -> dict[str, float]:
    """
    M5: M1(config) − M1(control) for each config.
    Negative values indicate accuracy loss from removing a component.
    """
    if control_id not in results_by_config:
        return {}
    control_acc = decision_accuracy(results_by_config[control_id])
    return {
        config_id: decision_accuracy(results) - control_acc
        for config_id, results in results_by_config.items()
        if config_id != control_id
    }


# ── M6 — Category Parity Gap ──────────────────────────────────────────────────

def category_parity_gap(results: list[dict]) -> float:
    """
    M6: max(per_category_accuracy) − min(per_category_accuracy).
    Measures fairness across decision categories.
    Lower is fairer. 0.0 means perfect parity.
    """
    cat_acc = per_category_accuracy(results)
    valid_vals = [v for v in cat_acc.values() if v == v]  # exclude NaN
    if len(valid_vals) < 2:
        return float("nan")
    return max(valid_vals) - min(valid_vals)


# ── Full metrics bundle ────────────────────────────────────────────────────────

def compute_config_metrics(results: list[dict]) -> dict:
    """Compute all metrics for a single config's results."""
    return {
        "M1_decision_accuracy":     decision_accuracy(results),
        "M1c_per_category":         per_category_accuracy(results),
        "M2_unsuitable_recall":     unsuitable_recall(results),
        "M2p_unsuitable_precision": unsuitable_precision(results),
        "M3_escalation_recall":     escalation_recall(results),
        "M3p_escalation_precision": escalation_precision(results),
        "M3f_escalation_f1":        escalation_f1(results),
        "M6_parity_gap":            category_parity_gap(results),
        "n_cells":                  len([r for r in results if not r.get("error")]),
        "n_errors":                 len([r for r in results if r.get("error")]),
    }


def compute_all_metrics(
    results_by_config: dict[str, list[dict]],
    control_id: str = "A0",
) -> dict:
    """
    Compute metrics for all configs and add cross-config deltas.

    Parameters
    ----------
    results_by_config : {config_id: [result, ...]}
    control_id        : config treated as the reference for delta computation

    Returns
    -------
    {
      "per_config": {config_id: {metric: value, ...}},
      "deltas":     {config_id: float},   — M5, relative to control
    }
    """
    per_config = {
        config_id: compute_config_metrics(results)
        for config_id, results in results_by_config.items()
    }
    deltas = component_deltas(results_by_config, control_id)

    return {
        "per_config": per_config,
        "deltas":     deltas,
        "control_id": control_id,
    }
