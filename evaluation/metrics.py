"""
evaluation/metrics.py
======================
Evaluation metrics for the thesis comparison.

All functions are pure — they take lists of result dicts and return floats
or dicts.  No I/O, no LLM calls, no side effects.

Metric definitions (M1-M7: accuracy & fairness)
------------------
M1   Decision Accuracy           — % runs matching expected_decision
M1a  Weighted Decision Accuracy  — cost-sensitive accuracy (UNSUITABLE→SUITABLE = 3× penalty)
M1b  Scenario-Level Accuracy     — majority-vote accuracy at scenario level
M2   Rule Compliance Rate         — % runs where FAIL rule set exactly matches expected
M2b  Rule Compliance Jaccard      — average Jaccard similarity (partial credit vs M2)
M3   Decision Consistency         — % agreement across repeated runs of the same scenario
M4   Escalation Accuracy          — sensitivity, specificity, precision, F1, MCC
M5   Verification Correction Rate — % pipeline runs where AV triggered a correction
M5b  Correction Precision         — fraction of verifier field-changes that improved accuracy
M6   Per-Rule Accuracy            — per-rule PASS/FAIL accuracy across all scenarios
M6b  Per-Rule Accuracy with CI    — same as M6 with 95% Wilson confidence intervals
M7   Demographic Parity           — violation rate excluding legitimate regulatory differentiation
M7b  Equalized Odds Gap           — TPR/FPR gap across demographic groups

Governance / compliance / explainability metrics (M8-M15) are in
evaluation/governance_metrics.py and included in compute_all_metrics.

Reference: thesis research questions
  RQ1: Does the multi-agent pipeline achieve higher decision accuracy than the baseline?
  RQ2: Does explicit rule-checking improve rule compliance?
  RQ3: Is the pipeline more consistent (deterministic) than the baseline?
  RQ4: Does the verifier feedback loop improve output correctness (M5, M5b)?
  RQ5: Does the pipeline maintain regulatory constraints (M11, M12)?
  RQ6: Is the pipeline explainable and auditable (M13-M15)?
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# M1 — Decision Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def decision_accuracy(results: list[dict]) -> float:
    """
    M1: Proportion of completed (non-halted) runs where output_decision == expected_decision.

    Halted runs are excluded from both numerator and denominator — halts are a
    governance event captured by M9, not a wrong decision.

    Each result dict must contain:
      "output_decision"   : str  (e.g. "SUITABLE")
      "expected_decision" : str  (from scenario JSON)
      "halted"            : bool (optional, default False)

    Returns float in [0.0, 1.0].
    """
    completed = [r for r in results if not r.get("halted", False)]
    if not completed:
        return 0.0
    correct = sum(
        1 for r in completed
        if r.get("output_decision") == r.get("expected_decision")
    )
    return correct / len(completed)


# ─────────────────────────────────────────────────────────────────────────────
# M1a — Weighted Decision Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def weighted_decision_accuracy(
    results: list[dict],
    mismatch_weights: "dict[tuple[str, str], float] | None" = None,
) -> dict[str, Any]:
    """
    M1a: Accuracy weighted by the asymmetric cost of misclassification errors.

    Under MiFID II the consequences of different errors are not equal:
      UNSUITABLE → SUITABLE  : 3× (product reaches a client it should be barred from)
      ESCALATED  → SUITABLE  : 3× (escalation flag stripped without human review)
      CONDITIONAL → SUITABLE : 2× (protective conditions silently removed)
      all other errors       : 1× (standard misclassification)

    weighted_accuracy = 1 − (total_penalty / total_weight)

    Halted runs excluded (consistent with M1).
    """
    _DEFAULTS: dict[tuple[str, str], float] = {
        ("UNSUITABLE",  "SUITABLE"): 3.0,
        ("ESCALATED",   "SUITABLE"): 3.0,
        ("CONDITIONAL", "SUITABLE"): 2.0,
    }
    weights = mismatch_weights if mismatch_weights is not None else _DEFAULTS

    completed = [r for r in results if not r.get("halted", False)]
    if not completed:
        return {"n_evaluated": 0, "weighted_accuracy": 0.0, "unweighted_accuracy": 0.0,
                "total_penalty": 0.0}

    total_weight  = 0.0
    total_penalty = 0.0
    n_wrong = 0
    for r in completed:
        expected  = r.get("expected_decision", "")
        predicted = r.get("output_decision",   "")
        if expected == predicted:
            total_weight += 1.0
        else:
            w = weights.get((expected, predicted), 1.0)
            total_weight  += w
            total_penalty += w
            n_wrong += 1

    return {
        "n_evaluated":         len(completed),
        "weighted_accuracy":   round(1.0 - total_penalty / total_weight, 4) if total_weight else 0.0,
        "unweighted_accuracy": round(1.0 - n_wrong / len(completed), 4),
        "total_penalty":       round(total_penalty, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# M1b — Scenario-Level Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def scenario_level_accuracy(results: list[dict]) -> dict[str, Any]:
    """
    M1b: Decision accuracy measured at scenario level using majority vote.

    Run-level M1 masks within-scenario inconsistency: a scenario correct 2/3
    runs contributes 0.67 to M1 yet the system cannot be trusted on that
    scenario.  This metric counts a scenario correct only when the majority
    vote across all its runs matches the expected decision.

    Requires results to carry "scenario_id".  Halted runs are excluded.
    """
    by_scenario: dict[str, list[dict]] = {}
    for r in results:
        if r.get("halted", False):
            continue
        sid = str(r.get("scenario_id", "unknown"))
        by_scenario.setdefault(sid, []).append(r)

    if not by_scenario:
        return {"n_scenarios": 0, "scenario_accuracy": 0.0, "split_vote_scenarios": 0}

    correct = split = 0
    for sid, runs in by_scenario.items():
        expected  = runs[0].get("expected_decision")
        decisions = [r.get("output_decision") for r in runs]
        counter   = Counter(decisions)
        top_count = counter.most_common(1)[0][1]
        majority  = counter.most_common(1)[0][0]
        # split: multiple decisions share the top count (no strict majority)
        if sum(1 for c in counter.values() if c == top_count) > 1:
            split += 1
        if majority == expected:
            correct += 1

    return {
        "n_scenarios":          len(by_scenario),
        "scenario_accuracy":    round(correct / len(by_scenario), 4),
        "split_vote_scenarios": split,
    }


# ─────────────────────────────────────────────────────────────────────────────
# M2 — Rule Compliance Rate
# ─────────────────────────────────────────────────────────────────────────────

def rule_compliance_rate(results: list[dict]) -> float:
    """
    M2: Proportion of completed (non-halted) runs where the set of FAIL rules
    exactly matches expected_rules_failed from the scenario fixture.

    Halted runs are excluded — they produced no rule verdict.

    Each result dict must contain:
      "output_failed_rules"   : list[str]  (e.g. ["R1", "R2"])
      "expected_rules_failed" : list[str]  (from scenario JSON)
      "halted"                : bool (optional, default False)

    Returns float in [0.0, 1.0].
    """
    completed = [r for r in results if not r.get("halted", False)]
    if not completed:
        return 0.0
    compliant = sum(
        1 for r in completed
        if set(r.get("output_failed_rules", [])) == set(r.get("expected_rules_failed", []))
    )
    return compliant / len(completed)


# ─────────────────────────────────────────────────────────────────────────────
# M2b — Rule Compliance Jaccard
# ─────────────────────────────────────────────────────────────────────────────

def rule_compliance_jaccard(results: list[dict]) -> dict[str, Any]:
    """
    M2b: Average Jaccard similarity of predicted vs expected failed-rule sets.

    M2's exact-set matching is all-or-nothing: a system that correctly identifies
    R1 and R3 but misses R2 scores 0, identically to a system that identifies
    nothing.  Jaccard gives partial credit proportional to overlap:

      Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    Both-empty sets → Jaccard = 1.0 (both agree: no rules fail).
    """
    completed = [r for r in results if not r.get("halted", False)]
    if not completed:
        return {"n_evaluated": 0, "mean_jaccard": 0.0, "exact_match_rate": 0.0}

    scores: list[float] = []
    exact_matches = 0
    for r in completed:
        predicted = set(r.get("output_failed_rules",   []))
        expected  = set(r.get("expected_rules_failed", []))
        union     = predicted | expected
        if not union:
            scores.append(1.0)
            exact_matches += 1
        else:
            j = len(predicted & expected) / len(union)
            scores.append(j)
            if j == 1.0:
                exact_matches += 1

    return {
        "n_evaluated":      len(completed),
        "mean_jaccard":     round(sum(scores) / len(scores), 4),
        "exact_match_rate": round(exact_matches / len(completed), 4),
    }


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
        counter    = Counter(decisions)
        mode_count = counter.most_common(1)[0][1]
        per_scenario.append(mode_count / len(decisions))
    return sum(per_scenario) / len(per_scenario) if per_scenario else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# M4 — Escalation Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def escalation_accuracy(results: list[dict]) -> dict[str, Any]:
    """
    M4: Escalation accuracy with sensitivity, specificity, precision, F1, and MCC.

    Overall accuracy alone is misleading when expected_escalate=True is rare
    (~8–20% of scenarios).  A system that never escalates scores ≥80% overall
    while having 0% sensitivity — the accuracy paradox.

    Added metrics:
    - precision : TP / (TP + FP) — of all escalated outputs, fraction truly needed
    - f1        : harmonic mean of precision and recall (sensitivity)
    - mcc       : Matthews Correlation Coefficient, robust to class imbalance;
                  MCC ∈ [−1, +1] where +1 = perfect, 0 = random, −1 = inverse

    Halted runs are excluded — the escalation flag was never finalised.

    Each result dict must contain:
      "output_escalated"  : bool
      "expected_escalate" : bool  (from scenario JSON)
      "halted"            : bool  (optional, default False)
    """
    completed = [r for r in results if not r.get("halted", False)]
    positive  = [r for r in completed if  bool(r.get("expected_escalate"))]
    negative  = [r for r in completed if not bool(r.get("expected_escalate"))]

    tp = sum(1 for r in positive if  bool(r.get("output_escalated")))
    tn = sum(1 for r in negative if not bool(r.get("output_escalated")))
    fp = sum(1 for r in negative if  bool(r.get("output_escalated")))
    fn = len(positive) - tp

    overall     = (tp + tn) / len(completed) if completed else 0.0
    sensitivity = tp / len(positive)         if positive  else 0.0   # recall
    specificity = tn / len(negative)         if negative  else 0.0
    precision   = tp / (tp + fp)             if (tp + fp) else 0.0
    f1          = (2 * precision * sensitivity / (precision + sensitivity)
                   if (precision + sensitivity) else 0.0)
    # MCC denominator is 0 when any marginal sum is 0 (degenerate class distribution)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc   = (tp * tn - fp * fn) / denom if denom else 0.0

    return {
        "overall_accuracy": round(overall,     4),
        "sensitivity":      round(sensitivity, 4),
        "specificity":      round(specificity, 4),
        "precision":        round(precision,   4),
        "f1":               round(f1,          4),
        "mcc":              round(mcc,          4),
        "n_positive":       len(positive),
        "n_negative":       len(negative),
    }


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
    a1_verified  = sum(1 for r in pipeline_results if r.get("a1_verified"))
    a2_verified  = sum(1 for r in pipeline_results if r.get("a2_verified"))
    a1_corrected = sum(1 for r in pipeline_results if r.get("a1_corrected"))
    a2_corrected = sum(1 for r in pipeline_results if r.get("a2_corrected"))

    total_verified  = a1_verified + a2_verified
    total_corrected = a1_corrected + a2_corrected

    return {
        "a1_correction_rate":      a1_corrected / a1_verified if a1_verified else 0.0,
        "a2_correction_rate":      a2_corrected / a2_verified if a2_verified else 0.0,
        "overall_correction_rate": total_corrected / total_verified if total_verified else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# M5b — Correction Precision  (pipeline only)
# ─────────────────────────────────────────────────────────────────────────────

def correction_precision(
    pipeline_results: list[dict],
    profile_key: str = "client",
) -> dict[str, Any]:
    """
    M5b: For each profile field changed by the verifier, whether the change
    improved (beneficial) or degraded (harmful) the extracted value relative
    to ground truth.

    A high M5 correction_rate with low correction_precision means the verifier
    introduces new errors while fixing others — a net negative despite high activity.

    Requires result dicts to carry:
      "a1_correction"           : {"original": {...}, "corrected": {...}}
        or
      "a2_correction"           : {"original": {...}, "corrected": {...}}
      "expected_client_profile" : dict  (for profile_key="client")
        or
      "expected_product_profile": dict  (for profile_key="product")

    profile_key : "client" (default, inspects a1_correction) or "product" (a2_correction).

    Returns dict with:
      correction_precision : beneficial / total_field_changes  (None when no data)
      correction_harm_rate : harmful    / total_field_changes
    """
    agent_key = "a1_correction" if profile_key == "client" else "a2_correction"
    exp_key   = f"expected_{profile_key}_profile"
    fields    = _A1_FIELDS if profile_key == "client" else _A2_FIELDS

    beneficial = harmful = neutral = 0

    for r in pipeline_results:
        correction = r.get(agent_key)
        if not correction or correction.get("corrected") is None:
            continue
        expected = r.get(exp_key)
        if not expected:
            continue
        original  = correction.get("original",  {})
        corrected = correction.get("corrected", {})
        for field in fields:
            orig_val = original.get(field)
            corr_val = corrected.get(field)
            if orig_val == corr_val:
                continue                    # field not changed by verifier
            exp_val  = expected.get(field)
            orig_ok  = _field_match(orig_val, exp_val)
            corr_ok  = _field_match(corr_val, exp_val)
            if not orig_ok and corr_ok:
                beneficial += 1             # fixed a wrong value → good
            elif orig_ok and not corr_ok:
                harmful    += 1             # broke a correct value → bad
            else:
                neutral    += 1             # wrong→different wrong, or correct→correct

    total = beneficial + harmful + neutral
    return {
        "beneficial_corrections": beneficial,
        "harmful_corrections":    harmful,
        "neutral_corrections":    neutral,
        "total_field_changes":    total,
        "correction_precision":   round(beneficial / total, 4) if total else None,
        "correction_harm_rate":   round(harmful    / total, 4) if total else None,
        "note": (
            "correction_precision = beneficial / total_field_changes. "
            "Returns None when no corrected runs with ground-truth profiles are present."
        ),
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
# M6b — Per-Rule Accuracy with Wilson Confidence Intervals
# ─────────────────────────────────────────────────────────────────────────────

def _wilson_ci(count: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for proportion count/n."""
    if n == 0:
        return (0.0, 0.0)
    p      = count / n
    denom  = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half   = z * math.sqrt(p * (1 - p) / n + z * z / (4.0 * n * n)) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def per_rule_accuracy_with_ci(results: list[dict]) -> dict[str, dict]:
    """
    M6b: Per-rule accuracy with 95% Wilson confidence intervals.

    Point estimates are misleading for rules that appear in few scenarios.
    If R6 (leverage) fires in only 12 runs, a 100% accuracy point estimate
    has a Wilson CI of roughly [74%, 100%] — far wider than the headline
    number suggests.  CIs allow readers to assess statistical uncertainty.
    """
    RULE_IDS = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
    per_rule: dict[str, list[bool]] = {r: [] for r in RULE_IDS}

    for result in results:
        output   = result.get("output_rules", {})
        expected = result.get("expected_rules", {})
        for rule_id in RULE_IDS:
            if rule_id in output and rule_id in expected:
                per_rule[rule_id].append(output[rule_id] == expected[rule_id])

    out: dict[str, dict] = {}
    for rule_id, matches in per_rule.items():
        n       = len(matches)
        correct = sum(matches)
        acc     = round(correct / n, 4) if n else None
        ci      = _wilson_ci(correct, n) if n else (None, None)
        out[rule_id] = {"accuracy": acc, "n_evaluated": n, "ci_95": ci}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# M7 — Demographic Parity Check
# ─────────────────────────────────────────────────────────────────────────────

def demographic_parity_check(results: list[dict]) -> dict[str, Any]:
    """
    M7: For paired scenarios sharing identical financial profiles but differing
    only in demographic attributes, verify that the pipeline produces the same
    decision for both variants — but ONLY when the gold-label expected_decision
    is identical for both variants.

    Critical distinction (MiFID II Art. 25):
    MiFID II legitimately requires different decisions for HIGH-vulnerability
    clients (R5 triggers when vulnerability=HIGH and product risk_class ≥ 5).
    Flagging such differences as demographic bias conflates regulatory compliance
    with discrimination.  The parity test must therefore apply only to pairs
    where the expected_decision is identical — otherwise a regulatory obligation
    is incorrectly scored as a bias violation.

    Pairs are classified into four states:
      parity_preserved     — same expected, same output          (correct)
      parity_violated      — same expected, different output     (potential bias)
      justified_diff       — different expected, different output (correct regulation)
      under_differentiated — different expected, same output     (missed regulatory req)

    parity_violation_rate is computed over same-expected pairs only.

    Each result dict must contain:
      "pair_id"             : str
      "demographic_variant" : str  (e.g. "young_low_vuln", "old_high_vuln")
      "output_decision"     : str
      "expected_decision"   : str
    """
    by_pair: dict[str, list[dict]] = {}
    for r in results:
        pid = r.get("pair_id")
        if pid is None:
            continue
        by_pair.setdefault(pid, []).append(r)

    violations:           list[dict] = []
    justified_diffs:      list[dict] = []
    under_differentiated: list[dict] = []
    same_expected_n = 0

    for pair_id, members in by_pair.items():
        if len(members) < 2:
            continue

        outputs_set  = set(m.get("output_decision")   for m in members)
        expected_set = set(m.get("expected_decision") for m in members)
        output_differs   = len(outputs_set)  > 1
        expected_differs = len(expected_set) > 1

        pair_info = {
            "pair_id": pair_id,
            "decisions": {
                m.get("demographic_variant", f"variant_{i}"): {
                    "output":   m.get("output_decision"),
                    "expected": m.get("expected_decision"),
                }
                for i, m in enumerate(members)
            },
        }

        if not expected_differs:
            # True parity test: both demographic variants have the same regulatory target
            same_expected_n += 1
            if output_differs:
                violations.append({
                    **pair_info,
                    "note": (
                        "Identical financials + same expected decision → different "
                        "outputs.  Potential demographic bias in A1/A5 narrative framing."
                    ),
                })
        else:
            # Regulatory differentiation is expected — exclude from parity test
            if output_differs:
                justified_diffs.append({
                    **pair_info,
                    "note": "Correctly differentiated by MiFID II rules (expected decisions differ).",
                })
            else:
                under_differentiated.append({
                    **pair_info,
                    "note": (
                        "Expected decisions differ (regulatory requirement) but system "
                        "produced identical output — regulatory under-differentiation."
                    ),
                })

    return {
        "n_pairs":                     len(by_pair),
        "n_pairs_same_expected":       same_expected_n,
        "n_violations":                len(violations),
        "parity_violation_rate":       (
            round(len(violations) / same_expected_n, 4) if same_expected_n else 0.0
        ),
        "n_justified_differentiation": len(justified_diffs),
        "n_under_differentiation":     len(under_differentiated),
        "violations":                  violations,
        "justified_diffs":             justified_diffs,
        "under_differentiated":        under_differentiated,
        "note": (
            "parity_violation_rate is computed only over pairs where both variants "
            "share the same expected_decision, to avoid penalising legitimate "
            "MiFID II regulatory differentiation as demographic bias."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# M7b — Equalized Odds Gap
# ─────────────────────────────────────────────────────────────────────────────

def equalized_odds_gap(results: list[dict]) -> dict[str, Any]:
    """
    M7b: Equalized odds gap across demographic groups in paired scenarios.

    Equalized odds requires equal TPR AND equal FPR across protected demographic
    groups.  This is stricter than demographic parity (equal positive rate) and
    is the fairness criterion recommended by EU AI Act implementation guidelines
    for high-risk classification systems.

    Positive class: UNSUITABLE or ESCALATED (regulatory rejection).
    Groups: inferred from demographic_variant field ("young*" vs "old*").

    A TPR gap or FPR gap > 0.05 indicates a meaningful disparity.
    """
    group_runs: dict[str, list[dict]] = {}
    for r in results:
        if r.get("pair_id") is None:
            continue
        variant = r.get("demographic_variant", "")
        group   = ("old"   if "old"   in variant else
                   "young" if "young" in variant else "other")
        group_runs.setdefault(group, []).append(r)

    POS = {"UNSUITABLE", "ESCALATED"}

    group_stats: dict[str, dict] = {}
    for group, runs in group_runs.items():
        completed = [r for r in runs if not r.get("halted", False)]
        pos = [r for r in completed if r.get("expected_decision") in POS]
        neg = [r for r in completed if r.get("expected_decision") not in POS]
        tp  = sum(1 for r in pos if r.get("output_decision") in POS)
        fp  = sum(1 for r in neg if r.get("output_decision") in POS)
        group_stats[group] = {
            "n":          len(completed),
            "n_positive": len(pos),
            "tpr":        round(tp / len(pos), 4) if pos else None,
            "fpr":        round(fp / len(neg), 4) if neg else None,
        }

    tpr_gap = fpr_gap = None
    if "young" in group_stats and "old" in group_stats:
        y, o = group_stats["young"], group_stats["old"]
        tpr_gap = (round(abs(y["tpr"] - o["tpr"]), 4)
                   if y["tpr"] is not None and o["tpr"] is not None else None)
        fpr_gap = (round(abs(y["fpr"] - o["fpr"]), 4)
                   if y["fpr"] is not None and o["fpr"] is not None else None)

    THRESHOLD = 0.05
    return {
        "group_stats":              group_stats,
        "tpr_gap":                  tpr_gap,
        "fpr_gap":                  fpr_gap,
        "equalized_odds_satisfied": (
            tpr_gap is not None and fpr_gap is not None
            and tpr_gap <= THRESHOLD and fpr_gap <= THRESHOLD
        ),
        "threshold": THRESHOLD,
        "note": "TPR/FPR gap ≤ 0.05 is the equalized-odds pass criterion.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# M_A1 / M_A2 — Extraction Accuracy  (pipeline, narrative scenarios only)
# ─────────────────────────────────────────────────────────────────────────────

# Fields A1 is expected to produce correctly from narrative text.
_A1_FIELDS = [
    "financial_knowledge",
    "risk_tolerance_score",
    "investment_horizon",
    "liquid_assets",
    "income",
    "investment_amount",
    "can_afford_total_loss",
    "financial_vulnerability",
    "age",
]

# Classification fields A2 is expected to produce (product_name excluded — free text).
_A2_FIELDS = [
    "risk_class",
    "complexity_tier",
    "requires_knowledge_level",
    "minimum_horizon",
    "potential_loss",
    "leverage",
]

# Relative tolerance for float comparisons (income, liquid_assets, investment_amount).
# A1 converting "€3,750/month" → 45000.0 is exact; allow 1% for edge-case rounding.
_FLOAT_TOLERANCE = 0.01


def _field_match(output_val: Any, expected_val: Any) -> bool:
    """
    Return True if output matches expected.
    For floats, allows up to _FLOAT_TOLERANCE relative difference.
    For all other types, uses strict equality.
    """
    if output_val is None and expected_val is None:
        return True
    if output_val is None or expected_val is None:
        return False
    if isinstance(expected_val, float) or isinstance(output_val, float):
        try:
            ev = float(expected_val)
            ov = float(output_val)
            if ev == 0.0:
                return ov == 0.0
            return abs(ov - ev) / abs(ev) <= _FLOAT_TOLERANCE
        except (TypeError, ValueError):
            return False
    return output_val == expected_val


def extraction_accuracy(
    results: list[dict],
    profile_key: str,       # "client" or "product"
) -> dict[str, Any]:
    """
    M_A1 / M_A2: Per-field and overall extraction accuracy for A1 (client) or A2 (product).

    Only results that carry both "output_<profile_key>_profile" and
    "expected_<profile_key>_profile" (non-None) are evaluated.

    Numeric fields (income, liquid_assets, investment_amount, minimum_horizon,
    potential_loss) are compared with ±1% relative tolerance via _field_match,
    not exact equality — A1/A2 may produce minor rounding differences when
    parsing narrative text (e.g. "€3,750/month" → 45,000 vs 45,010).

    Returns:
      "n_evaluated"             : int   — number of results compared
      "overall_exact_match_rate": float — % where ALL evaluated fields correct
      "per_field_accuracy"      : dict[field, float]
    """
    fields  = _A1_FIELDS if profile_key == "client" else _A2_FIELDS
    out_key = f"output_{profile_key}_profile"
    exp_key = f"expected_{profile_key}_profile"

    evaluated = [
        r for r in results
        if r.get(out_key) and r.get(exp_key)
    ]
    if not evaluated:
        return {"n_evaluated": 0, "overall_exact_match_rate": 0.0, "per_field_accuracy": {}}

    per_field:   dict[str, list[bool]] = {f: [] for f in fields}
    all_correct: list[bool]            = []

    for r in evaluated:
        output   = r[out_key]
        expected = r[exp_key]
        row_correct = True
        for field in fields:
            match = _field_match(output.get(field), expected.get(field))
            per_field[field].append(match)
            if not match:
                row_correct = False
        all_correct.append(row_correct)

    return {
        "n_evaluated":              len(evaluated),
        "overall_exact_match_rate": round(sum(all_correct) / len(all_correct), 4),
        "per_field_accuracy": {
            field: round(sum(matches) / len(matches), 4) if matches else None
            for field, matches in per_field.items()
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary reporter
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    results: list[dict],
    architecture_name: str = "pipeline",
) -> dict[str, Any]:
    """
    Compute M1-M7 (and variants) plus M_A1/M_A2 for a list of run results.

    For M3 (consistency), results must have a "scenario_id" key and be
    from multiple runs per scenario.

    For M5/M5b, results must have a1/a2 verification + correction fields
    (pipeline results only — baseline returns N/A).

    Governance metrics M8-M15 are computed separately via
    evaluation.governance_metrics.compute_governance_metrics and merged
    into the returned dict.
    """
    from evaluation.governance_metrics import compute_governance_metrics

    # Group decisions by scenario for M3 (halted runs contribute "HALT" so
    # consistent halting is still counted as consistent behaviour)
    runs_per_scenario: dict[str, list[str]] = {}
    for r in results:
        sid      = str(r.get("scenario_id", "unknown"))
        decision = "HALT" if r.get("halted", False) else r.get("output_decision", "UNKNOWN")
        runs_per_scenario.setdefault(sid, []).append(decision)

    is_pipeline = architecture_name == "pipeline"

    m1   = decision_accuracy(results)
    m1a  = weighted_decision_accuracy(results)
    m1b  = scenario_level_accuracy(results)
    m2   = rule_compliance_rate(results)
    m2b  = rule_compliance_jaccard(results)
    m3   = decision_consistency(runs_per_scenario)
    m4   = escalation_accuracy(results)
    m5   = verification_correction_rate(results)   if is_pipeline else "N/A (baseline)"
    m5b  = correction_precision(results, "client") if is_pipeline else "N/A (baseline)"
    m6   = per_rule_accuracy(results)
    m6b  = per_rule_accuracy_with_ci(results)

    # M7 / M7b only fire when results contain pair_id fields (paired demographic scenarios).
    paired = [r for r in results if r.get("pair_id") is not None]
    m7  = demographic_parity_check(paired) if paired else "N/A (no paired scenarios)"
    m7b = equalized_odds_gap(paired)        if paired else "N/A (no paired scenarios)"

    # M_A1 / M_A2: extraction accuracy on narrative scenarios (pipeline only).
    if is_pipeline:
        m_a1 = extraction_accuracy(results, "client")
        m_a2 = extraction_accuracy(results, "product")
    else:
        m_a1 = "N/A (baseline has no A1/A2 agents)"
        m_a2 = "N/A (baseline has no A1/A2 agents)"

    # M8-M15 governance / compliance / explainability
    governance = compute_governance_metrics(results, architecture_name)

    return {
        "architecture":                   architecture_name,
        "n_runs":                         len(results),
        # M1 family
        "M1_decision_accuracy":           round(m1, 4),
        "M1a_weighted_accuracy":          m1a,
        "M1b_scenario_level_accuracy":    m1b,
        # M2 family
        "M2_rule_compliance_rate":        round(m2, 4),
        "M2b_rule_compliance_jaccard":    m2b,
        # M3
        "M3_decision_consistency":        round(m3, 4),
        # M4
        "M4_escalation_accuracy":         m4,
        # M5 family
        "M5_verification":                m5,
        "M5b_correction_precision":       m5b,
        # M6 family
        "M6_per_rule_accuracy":           {k: (round(v, 4) if v is not None else None)
                                           for k, v in m6.items()},
        "M6b_per_rule_accuracy_with_ci":  m6b,
        # M7 family
        "M7_demographic_parity":          m7,
        "M7b_equalized_odds":             m7b,
        # Extraction accuracy on narrative scenarios
        "M_A1_extraction":                m_a1,
        "M_A2_extraction":                m_a2,
        # Governance metrics (M8-M15) merged from governance_metrics.py
        "M8_override_rate":               governance["M8_override_rate"],
        "M9_pipeline_halt_rate":          governance["M9_pipeline_halt_rate"],
        "M10_three_point_integrity":      governance["M10_three_point_integrity"],
        "M11_hard_rule_enforcement":      governance["M11_hard_rule_enforcement"],
        "M12_vulnerability_protection":   governance["M12_vulnerability_protection"],
        "M13_regulatory_citation_rate":   governance["M13_regulatory_citation_rate"],
        "M14_explanation_completeness":   governance["M14_explanation_completeness"],
        "M15_decision_traceability":      governance["M15_decision_traceability"],
        "M15b_decision_traceability_v2":  governance["M15b_decision_traceability_v2"],
    }
