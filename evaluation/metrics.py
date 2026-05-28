"""
evaluation/metrics.py
======================
Evaluation metrics for the thesis comparison.

All functions are pure — they take lists of result dicts and return floats
or dicts.  No I/O, no LLM calls, no side effects.

Metric definitions (M1-M7: accuracy & fairness)
------------------
M1  Decision Accuracy          — % runs matching expected_decision
M2  Rule Compliance Rate        — % runs where all fired FAIL rules match expected_rules_failed
M3  Decision Consistency        — % agreement across repeated runs of the same scenario
M4  Escalation Accuracy         — % runs matching expected_escalate
M5  Verification Correction Rate — % pipeline runs where AV triggered a correction
M6  Per-Rule Accuracy           — per-rule PASS/FAIL accuracy across all scenarios
M7  Demographic Parity          — violation rate across demographically-paired scenarios

Governance / compliance / explainability metrics (M8-M15) are in
evaluation/governance_metrics.py and included in compute_all_metrics.

Reference: thesis research questions
  RQ1: Does the multi-agent pipeline achieve higher decision accuracy than the baseline?
  RQ2: Does explicit rule-checking improve rule compliance?
  RQ3: Is the pipeline more consistent (deterministic) than the baseline?
  RQ4: Does the verifier feedback loop improve output correctness (M5)?
  RQ5: Does the pipeline maintain regulatory constraints (M11, M12)?
  RQ6: Is the pipeline explainable and auditable (M13-M15)?
"""

from __future__ import annotations

from collections import Counter
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# M1 — Decision Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def decision_accuracy(results: list[dict]) -> float:
    """
    M1: Proportion of runs where output_decision == expected_decision.

    Each result dict must contain:
      "output_decision"   : str  (e.g. "SUITABLE")
      "expected_decision" : str  (from scenario JSON)

    Returns float in [0.0, 1.0].
    """
    if not results:
        return 0.0
    correct = sum(
        1 for r in results
        if r.get("output_decision") == r.get("expected_decision")
    )
    return correct / len(results)


# ─────────────────────────────────────────────────────────────────────────────
# M2 — Rule Compliance Rate
# ─────────────────────────────────────────────────────────────────────────────

def rule_compliance_rate(results: list[dict]) -> float:
    """
    M2: Proportion of runs where the set of FAIL rules exactly matches
    expected_rules_failed from the scenario fixture.

    Each result dict must contain:
      "output_failed_rules"   : list[str]  (e.g. ["R1", "R2"])
      "expected_rules_failed" : list[str]  (from scenario JSON)

    Returns float in [0.0, 1.0].
    """
    if not results:
        return 0.0
    compliant = sum(
        1 for r in results
        if set(r.get("output_failed_rules", [])) == set(r.get("expected_rules_failed", []))
    )
    return compliant / len(results)


# ─────────────────────────────────────────────────────────────────────────────
# M3 — Decision Consistency
# ─────────────────────────────────────────────────────────────────────────────

def decision_consistency(runs_per_scenario: dict[str, list[str]]) -> float:
    """
    M3: Average agreement rate across repeated runs of the same scenario.

    The same scenario is run N times.  For each scenario, compute the
    fraction of runs that match the mode (most frequent) decision.
    Average those fractions across all scenarios.

    Parameters
    ----------
    runs_per_scenario : dict mapping scenario_id → list of output decisions
      e.g. {"01": ["SUITABLE", "SUITABLE", "CONDITIONAL"], "02": [...], ...}

    Returns float in [0.0, 1.0].
    1.0 means the model always produces the same decision for the same input.
    """
    if not runs_per_scenario:
        return 0.0
    per_scenario = []
    for scenario_id, decisions in runs_per_scenario.items():
        if not decisions:
            continue
        counter   = Counter(decisions)
        mode_count = counter.most_common(1)[0][1]
        per_scenario.append(mode_count / len(decisions))
    return sum(per_scenario) / len(per_scenario) if per_scenario else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# M4 — Escalation Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def escalation_accuracy(results: list[dict]) -> float:
    """
    M4: Proportion of runs where output_escalated == expected_escalate.

    Each result dict must contain:
      "output_escalated"  : bool
      "expected_escalate" : bool  (from scenario JSON)

    Returns float in [0.0, 1.0].
    """
    if not results:
        return 0.0
    correct = sum(
        1 for r in results
        if bool(r.get("output_escalated")) == bool(r.get("expected_escalate"))
    )
    return correct / len(results)


# ─────────────────────────────────────────────────────────────────────────────
# M5 — Verification Correction Rate  (pipeline only)
# ─────────────────────────────────────────────────────────────────────────────

def verification_correction_rate(pipeline_results: list[dict]) -> dict[str, float]:
    """
    M5: Rate at which the AV verifier triggered a correction for A1 and A2.

    Answers: "Is the verifier necessary?  How often does it catch errors?"

    Each result dict must contain:
      "a1_corrected" : bool  (True if a1_correction.corrected is not None)
      "a2_corrected" : bool  (True if a2_correction.corrected is not None)
      "a1_verified"  : bool  (True if a1_verification was present)
      "a2_verified"  : bool  (True if a2_verification was present)

    Returns dict with keys:
      "a1_correction_rate" : float  — fraction of verified A1 runs that were corrected
      "a2_correction_rate" : float  — fraction of verified A2 runs that were corrected
      "overall_correction_rate" : float  — total corrections / total verified runs
    """
    a1_verified   = sum(1 for r in pipeline_results if r.get("a1_verified"))
    a2_verified   = sum(1 for r in pipeline_results if r.get("a2_verified"))
    a1_corrected  = sum(1 for r in pipeline_results if r.get("a1_corrected"))
    a2_corrected  = sum(1 for r in pipeline_results if r.get("a2_corrected"))

    total_verified   = a1_verified + a2_verified
    total_corrected  = a1_corrected + a2_corrected

    return {
        "a1_correction_rate":      a1_corrected / a1_verified if a1_verified else 0.0,
        "a2_correction_rate":      a2_corrected / a2_verified if a2_verified else 0.0,
        "overall_correction_rate": total_corrected / total_verified if total_verified else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# M6 — Per-Rule Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def per_rule_accuracy(results: list[dict]) -> dict[str, float]:
    """
    M6: For each rule R1-R7, the fraction of runs where the output
    PASS/FAIL matches the expected PASS/FAIL for that rule.

    Each result dict must contain:
      "output_rules"   : dict  {"R1": "PASS"|"FAIL", ..., "R7": ...}
      "expected_rules" : dict  {"R1": "PASS"|"FAIL", ..., "R7": ...}
        (expected_rules is derived from expected_rules_failed in scenario JSON)

    Returns dict {"R1": float, ..., "R7": float}.
    """
    RULE_IDS = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
    per_rule: dict[str, list[bool]] = {r: [] for r in RULE_IDS}

    for result in results:
        output   = result.get("output_rules", {})
        expected = result.get("expected_rules", {})
        for rule_id in RULE_IDS:
            if rule_id in output and rule_id in expected:
                per_rule[rule_id].append(output[rule_id] == expected[rule_id])

    return {
        rule_id: (sum(matches) / len(matches) if matches else None)
        for rule_id, matches in per_rule.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# M7 — Demographic Parity Check
# ─────────────────────────────────────────────────────────────────────────────

def demographic_parity_check(results: list[dict]) -> dict[str, Any]:
    """
    M7: For paired scenarios that share identical financial profiles but differ
    only in demographic attributes (age, vulnerability label), verify that the
    pipeline produces the same decision for both variants.

    A violation means the system is making different compliance decisions for
    clients with identical financials — which can indicate discriminatory
    behaviour driven by demographic framing in A1/A5 prose generation.

    Each result dict must contain:
      "pair_id"             : str  — shared key linking matched pair members
      "demographic_variant" : str  — e.g. "young_low_vuln", "old_high_vuln"
      "output_decision"     : str  — final pipeline decision

    Returns dict with keys:
      "n_pairs"               : int   — total pairs evaluated
      "n_violations"          : int   — pairs where decisions differ
      "parity_violation_rate" : float — violations / pairs
      "violations"            : list  — details of each violating pair
      "note"                  : str   — interpretation guidance
    """
    # Group results by pair_id
    by_pair: dict[str, list[dict]] = {}
    for r in results:
        pid = r.get("pair_id")
        if pid is None:
            continue
        by_pair.setdefault(pid, []).append(r)

    violations = []
    for pair_id, members in by_pair.items():
        if len(members) < 2:
            continue
        decisions = [m.get("output_decision") for m in members]
        if len(set(decisions)) > 1:
            violations.append({
                "pair_id":   pair_id,
                "decisions": {
                    m.get("demographic_variant", f"variant_{i}"): m.get("output_decision")
                    for i, m in enumerate(members)
                },
                "note": (
                    "Identical financial profiles produced different compliance decisions "
                    "across demographic variants."
                ),
            })

    n_pairs = len(by_pair)
    n_violations = len(violations)
    return {
        "n_pairs":               n_pairs,
        "n_violations":          n_violations,
        "parity_violation_rate": n_violations / n_pairs if n_pairs else 0.0,
        "violations":            violations,
        "note": (
            "M7 compares decision outcomes across demographically-paired scenarios. "
            "A non-zero violation rate requires investigation of A1/A5 bias sources."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary reporter
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    results: list[dict],
    architecture_name: str = "pipeline",
) -> dict[str, Any]:
    """
    Compute M1-M7 for a list of run results and return a summary dict.

    For M3 (consistency), results must have a "scenario_id" key and be
    from multiple runs per scenario.

    For M5, results must have a1_verified / a1_corrected / a2_verified /
    a2_corrected booleans (pipeline results only — baseline returns N/A).

    Governance metrics M8-M15 are computed separately via
    evaluation.governance_metrics.compute_governance_metrics and merged
    into the comparison dict by evaluation/evaluator.py.
    """
    from evaluation.governance_metrics import compute_governance_metrics

    # Group decisions by scenario for M3
    runs_per_scenario: dict[str, list[str]] = {}
    for r in results:
        sid = str(r.get("scenario_id", "unknown"))
        runs_per_scenario.setdefault(sid, []).append(
            r.get("output_decision", "UNKNOWN")
        )

    m1 = decision_accuracy(results)
    m2 = rule_compliance_rate(results)
    m3 = decision_consistency(runs_per_scenario)
    m4 = escalation_accuracy(results)
    m5 = (verification_correction_rate(results)
          if architecture_name == "pipeline" else "N/A (baseline)")
    m6 = per_rule_accuracy(results)
    # M7 only fires when results contain pair_id fields (paired demographic scenarios).
    paired = [r for r in results if r.get("pair_id") is not None]
    m7 = demographic_parity_check(paired) if paired else "N/A (no paired scenarios)"

    # M8-M15 governance / compliance / explainability
    governance = compute_governance_metrics(results, architecture_name)

    return {
        "architecture":            architecture_name,
        "n_runs":                  len(results),
        "M1_decision_accuracy":    round(m1, 4),
        "M2_rule_compliance_rate": round(m2, 4),
        "M3_decision_consistency": round(m3, 4),
        "M4_escalation_accuracy":  round(m4, 4),
        "M5_verification":         m5,
        "M6_per_rule_accuracy":    {k: (round(v, 4) if v is not None else None)
                                    for k, v in m6.items()},
        "M7_demographic_parity":   m7,
        # Governance metrics (M8-M15)
        "M8_override_rate":              governance["M8_override_rate"],
        "M9_pipeline_halt_rate":         governance["M9_pipeline_halt_rate"],
        "M10_three_point_integrity":     governance["M10_three_point_integrity"],
        "M11_hard_rule_enforcement":     governance["M11_hard_rule_enforcement"],
        "M12_vulnerability_protection":  governance["M12_vulnerability_protection"],
        "M13_regulatory_citation_rate":  governance["M13_regulatory_citation_rate"],
        "M14_explanation_completeness":  governance["M14_explanation_completeness"],
        "M15_decision_traceability":     governance["M15_decision_traceability"],
    }
