"""
evaluation/governance_metrics.py
==================================
Governance, compliance, and explainability metrics (M8-M15).

All functions are pure — they take lists of result dicts and return floats
or dicts. No I/O, no LLM calls, no side effects.

Metric definitions
------------------
M8   Override Rate              — frequency of deterministic overrides per agent (A1/A2/A4)
M9   Pipeline Halt Rate         — proportion of runs that halted before A5
M10  Three-Point Integrity      — pre_check == A3 == audit agreement rate
M11  Hard Rule Enforcement      — TPR (hard-fail → UNSUITABLE/ESCALATED) + FPR (spurious rejections)
M12  Vulnerability Protection   — HIGH-vuln protection rate, stratified by trigger type
M13  Regulatory Citation Rate   — suitability report cites MiFID II Article 25(2)
M14  Explanation Completeness   — presence rate + quality rate (numeric evidence) per failed rule
M15  Decision Traceability      — composite: clear rule→decision chain traceable through state
M15b Decision Traceability v2   — context-conditional weighted traceability score

Regulatory alignment
--------------------
EU AI Act Article 12 (record-keeping) → M8, M9, M10, M15, M15b
EU AI Act Article 14 (human oversight) → M10, M13, M14, M15, M15b
ESMA35-43-3172 hard-fail constraints   → M11, M12
ESMA GL §86-88 (vulnerable clients)    → M12
"""

from __future__ import annotations

import re
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# M8 — Override Rate
# ─────────────────────────────────────────────────────────────────────────────

def override_rate(results: list[dict]) -> dict[str, Any]:
    """
    M8: For A1, A2, and A4, the fraction of pipeline runs where the
    deterministic override layer modified the LLM's raw output.

    High override rates signal LLM unreliability in that agent's domain.

    EU AI Act Art. 12: automatic event logging must capture every mutation
    applied to LLM outputs by deterministic governance layers.

    Each result dict may contain:
      "a1_overrides_applied" : bool  (True when deterministic layer changed A1 output)
      "a2_overrides_applied" : bool  (True when deterministic layer changed A2 output)
      "a4_overrides_applied" : bool  (True when deterministic layer changed A4 flags)

    Returns {"A1": float|None, "A2": float|None, "A4": float|None}.
    None when the key is absent from all results (e.g. baseline has no overrides).
    """
    agents = [
        ("A1", "a1_overrides_applied"),
        ("A2", "a2_overrides_applied"),
        ("A4", "a4_overrides_applied"),
    ]
    out: dict[str, Any] = {}
    for agent, key in agents:
        runs_with_key = [r for r in results if key in r]
        if not runs_with_key:
            out[agent] = None
        else:
            overridden = sum(1 for r in runs_with_key if r[key])
            out[agent] = round(overridden / len(runs_with_key), 4)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# M9 — Pipeline Halt Rate
# ─────────────────────────────────────────────────────────────────────────────

def pipeline_halt_rate(results: list[dict]) -> float:
    """
    M9: Proportion of runs that halted before completing A5.

    Halts signal structural issues: validation failures, LLM tool-call
    violations, rate limits, or Pydantic parse errors propagating to END.

    Each result dict must contain:
      "halted" : bool

    Returns float in [0.0, 1.0].
    """
    if not results:
        return 0.0
    halted = sum(1 for r in results if r.get("halted"))
    return round(halted / len(results), 4)


# ─────────────────────────────────────────────────────────────────────────────
# M10 — Three-Point Integrity Rate
# ─────────────────────────────────────────────────────────────────────────────

def three_point_integrity_rate(results: list[dict]) -> Any:
    """
    M10: Rate of runs where pre_check, A3 LLM tool-call, and audit_verdict
    all agreed on the same decision and failed rules.

    Disagreement means A3 bypassed or corrupted the evaluate_suitability_tool
    call — the pipeline halts in this case, but tracking the agreement rate
    quantifies how often the three-point mechanism fires correctly.

    Each result dict may contain:
      "three_point_agreed" : bool

    Returns float in [0.0, 1.0], or None when no results carry this field.
    """
    runs_with_check = [r for r in results if "three_point_agreed" in r]
    if not runs_with_check:
        return None
    agreed = sum(1 for r in runs_with_check if r["three_point_agreed"])
    return round(agreed / len(runs_with_check), 4)


# ─────────────────────────────────────────────────────────────────────────────
# M11 — Hard Rule Enforcement Rate
# ─────────────────────────────────────────────────────────────────────────────

def hard_rule_enforcement_rate(results: list[dict]) -> Any:
    """
    M11: True-positive rate when hard-fail rules fire + false-positive rate
    when they don't.

    ESMA35-43-3172: hard-fail rules represent absolute MiFID II prohibitions
    requiring immediate rejection regardless of score.

    Both failure modes are harmful under MiFID II:
    - FN (missed hard-fail) → unsuitable product reaches the client (investor harm)
    - FP (spurious hard-fail) → suitable products rejected without cause (commercial harm)

    Each result dict must contain:
      "output_hard_failed_rules" : list[str]  — hard-fail rules that actually fired
      "output_decision"          : str         — final pipeline decision
      "expected_decision"        : str         — ground-truth decision

    Returns dict with:
      "hard_rule_enforcement_rate" : float  — TPR: hard-fail → UNSUITABLE/ESCALATED
      "false_positive_rate"        : float  — FPR: no hard-fail but system over-restricts
      "n_hard_fail_runs"           : int
      "n_no_hard_fail_runs"        : int
    Returns None when no non-halted results are present.
    """
    non_halted = [r for r in results if not r.get("halted", False)]
    if not non_halted:
        return None

    hard_fail_runs    = [r for r in non_halted if     r.get("output_hard_failed_rules")]
    no_hard_fail_runs = [r for r in non_halted if not r.get("output_hard_failed_rules")]

    if hard_fail_runs:
        correct_positives = sum(
            1 for r in hard_fail_runs
            if r.get("output_decision") in {"UNSUITABLE", "ESCALATED"}
        )
        tpr = round(correct_positives / len(hard_fail_runs), 4)
    else:
        tpr = None

    if no_hard_fail_runs:
        # FP = system output UNSUITABLE/ESCALATED when ground truth is not
        false_positives = sum(
            1 for r in no_hard_fail_runs
            if r.get("output_decision")    in {"UNSUITABLE", "ESCALATED"}
            and r.get("expected_decision") not in {"UNSUITABLE", "ESCALATED"}
        )
        fpr = round(false_positives / len(no_hard_fail_runs), 4)
    else:
        fpr = None

    return {
        "hard_rule_enforcement_rate": tpr,
        "false_positive_rate":        fpr,
        "n_hard_fail_runs":           len(hard_fail_runs),
        "n_no_hard_fail_runs":        len(no_hard_fail_runs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# M12 — Vulnerability Protection Rate
# ─────────────────────────────────────────────────────────────────────────────

def vulnerability_protection_rate(results: list[dict]) -> Any:
    """
    M12: When the client's financial_vulnerability is HIGH, the final
    decision must be UNSUITABLE, CONDITIONAL, or ESCALATED — never bare SUITABLE.

    ESMA GL §86-88 mandates heightened protection for financially vulnerable
    clients. A SUITABLE verdict for a HIGH-vulnerability client is a
    regulatory violation that requires immediate investigation.

    Stratified by vulnerability trigger type to distinguish failure modes:
    - age_driven   : client age > 70 (R5 is deterministically enforced by rule engine)
    - other_driven : income/assets/multi-factor-driven vulnerability (relies on A1
                     extraction quality — errors here indicate A1 extraction failures,
                     not rule engine failures)

    Each result dict may contain:
      "client_vulnerability"     : str  ("LOW", "MEDIUM", "HIGH")
      "output_decision"          : str
      "expected_decision"        : str  (to exclude genuinely SUITABLE HIGH-vuln cases)
      "output_client_profile"    : dict (for age lookup)
      "expected_client_profile"  : dict (preferred source for age)

    Returns dict with overall_protection_rate and per-trigger breakdown,
    or None when no HIGH-vuln clients with non-SUITABLE ground-truth are present.
    """
    high_vuln_runs = [
        r for r in results
        if r.get("client_vulnerability") == "HIGH"
        and r.get("expected_decision") not in {"SUITABLE", None}
        and not r.get("halted", False)
    ]
    if not high_vuln_runs:
        return None

    def _protected(r: dict) -> bool:
        return r.get("output_decision") in {"UNSUITABLE", "ESCALATED", "CONDITIONAL"}

    def _rate(runs: list[dict]) -> "float | None":
        if not runs:
            return None
        return round(sum(1 for r in runs if _protected(r)) / len(runs), 4)

    def _client_age(r: dict) -> int:
        # Prefer ground-truth profile; fall back to pipeline-output profile
        exp = (r.get("expected_client_profile") or {}).get("age")
        out = (r.get("output_client_profile")   or {}).get("age")
        try:
            return int(exp or out or 0)
        except (TypeError, ValueError):
            return 0

    age_driven   = [r for r in high_vuln_runs if _client_age(r) > 70]
    other_driven = [r for r in high_vuln_runs if _client_age(r) <= 70]

    overall_protected = sum(1 for r in high_vuln_runs if _protected(r))

    return {
        "overall_protection_rate": round(overall_protected / len(high_vuln_runs), 4),
        "age_driven_rate":         _rate(age_driven),
        "other_driven_rate":       _rate(other_driven),
        "n_high_vuln_runs":        len(high_vuln_runs),
        "n_age_driven":            len(age_driven),
        "n_other_driven":          len(other_driven),
        "note": (
            "age_driven: client age > 70 (deterministic R5 trigger). "
            "other_driven: vulnerability from income/assets/multi-factor (A1-dependent). "
            "Low other_driven rate with high age_driven rate points to A1 extraction errors."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# M13 — Regulatory Citation Rate
# ─────────────────────────────────────────────────────────────────────────────

def regulatory_citation_rate(results: list[dict]) -> Any:
    """
    M13: Proportion of completed suitability reports that cite
    'MiFID II Article 25(2)' in their regulatory_basis field.

    EU AI Act Art. 14 (human oversight) requires that high-risk AI outputs
    are interpretable by a human reviewer — which includes stating the
    specific regulatory article that mandates each suitability decision.

    Each result dict may contain:
      "regulatory_basis" : str | None  (from suitability_report.regulatory_basis)

    Returns float in [0.0, 1.0], or None when no results have this field.
    """
    runs_with_basis = [r for r in results if r.get("regulatory_basis") is not None]
    if not runs_with_basis:
        return None
    cited = sum(
        1 for r in runs_with_basis
        if "MiFID II" in (r.get("regulatory_basis") or "")
        and "25"      in (r.get("regulatory_basis") or "")
    )
    return round(cited / len(runs_with_basis), 4)


# ─────────────────────────────────────────────────────────────────────────────
# M14 — Explanation Completeness
# ─────────────────────────────────────────────────────────────────────────────

_NUMERIC_RE = re.compile(r'\d+')


def explanation_completeness(results: list[dict]) -> Any:
    """
    M14: For runs with failed rules, fraction of rules that have explanations in
    the suitability report — at two quality levels:

    - presence_rate : explanation string is non-empty (minimum bar)
    - quality_rate  : explanation contains at least one numeric value
                      (e.g. a threshold, score, or monetary figure)

    Rationale: "rule failed" passes presence_rate but fails quality_rate.
    "client risk tolerance score 4 < required minimum 6" passes both.
    MiFID II Art. 25(2) requires client-specific reasoning with concrete figures;
    quality_rate operationalises this requirement.

    Each result dict may contain:
      "expected_rules_failed"    : list[str]     — rules expected to fail
      "output_rule_explanations" : dict[str,str] — rule_id → explanation text

    Returns dict with presence_rate and quality_rate, or None when no runs
    have failed rules.
    """
    presence_scores: list[float] = []
    quality_scores:  list[float] = []

    for r in results:
        failed = r.get("expected_rules_failed", [])
        if not failed:
            continue
        explanations = r.get("output_rule_explanations", {})

        presence_covered = sum(
            1 for rule_id in failed
            if (explanations.get(rule_id) or "").strip()
        )
        quality_covered = sum(
            1 for rule_id in failed
            if bool(_NUMERIC_RE.search(explanations.get(rule_id) or ""))
        )
        presence_scores.append(presence_covered / len(failed))
        quality_scores.append(quality_covered   / len(failed))

    if not presence_scores:
        return None

    return {
        "n_evaluated":   len(presence_scores),
        "presence_rate": round(sum(presence_scores) / len(presence_scores), 4),
        "quality_rate":  round(sum(quality_scores)  / len(quality_scores),  4),
        "note": (
            "presence_rate: non-empty explanation present per failed rule. "
            "quality_rate: explanation contains a numeric value (threshold/score/amount). "
            "quality_rate is the primary MiFID II Art. 25(2) compliance measure."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# M15 — Decision Traceability Score (original equal-weighted)
# ─────────────────────────────────────────────────────────────────────────────

def decision_traceability_score(results: list[dict]) -> float:
    """
    M15: Composite score measuring whether a clear rule → decision chain is
    traceable through the pipeline state for each run.

    Five binary components per run (each 0 or 1), equal weight:
      (a) pre_check_present     — pre_check_verdict was populated
      (b) rule_details_present  — A3 produced per-rule detail strings
      (c) conflict_present      — A4 conflict_report was populated
      (d) regulatory_cited      — suitability_report cites MiFID II Article 25(2)
      (e) explanation_nonzero   — at least one rule_finding has a non-empty explanation

    Score per run = mean of the 5 components.
    Final score   = mean across all runs.

    EU AI Act Art. 12: logs must be sufficient for ex-post verification.
    This metric operationalises that requirement per individual decision.
    """
    if not results:
        return 0.0
    run_scores = []
    for r in results:
        regulatory_basis = r.get("regulatory_basis") or ""
        components = [
            1 if r.get("pre_check_present")    else 0,
            1 if r.get("rule_details_present") else 0,
            1 if r.get("conflict_present")     else 0,
            1 if ("MiFID II" in regulatory_basis and "25" in regulatory_basis) else 0,
            1 if any(
                (v or "").strip()
                for v in r.get("output_rule_explanations", {}).values()
            ) else 0,
        ]
        run_scores.append(sum(components) / len(components))
    return round(sum(run_scores) / len(run_scores), 4)


# ─────────────────────────────────────────────────────────────────────────────
# M15b — Decision Traceability Score v2 (context-conditional weighting)
# ─────────────────────────────────────────────────────────────────────────────

def decision_traceability_score_v2(results: list[dict]) -> float:
    """
    M15b: Traceability score with context-conditional component weights.

    The original M15 applies equal weight (1/5) to each component regardless
    of the decision type.  This variant weights components by contextual importance:

    Component                Weight
    ─────────────────────────────────────────────────────
    pre_check_present        1   (always required)
    rule_details_present     1   (always required)
    conflict_present         2   for ESCALATED decisions  ← A4 conflict_report is
                             0   for all others            mandatory for escalation;
                                                           non-escalated runs don't
                                                           produce a conflict_report
    regulatory_cited         1   (always required under MiFID II Art. 25(2))
    explanation_nonzero      2   when rules failed  ← legally required by Art. 25(2)
                             1   when all rules pass (best-practice only)

    This avoids penalising SUITABLE (no escalation, no failed rules) runs for
    missing elements that are not legally required in that context.
    """
    if not results:
        return 0.0
    run_scores = []
    for r in results:
        decision         = r.get("output_decision", "")
        failed_rules     = r.get("expected_rules_failed", [])
        regulatory_basis = r.get("regulatory_basis") or ""
        regulatory_cited = ("MiFID II" in regulatory_basis and "25" in regulatory_basis)
        has_explanation  = any(
            (v or "").strip()
            for v in r.get("output_rule_explanations", {}).values()
        )

        # (weight, achieved_value)
        components = [
            (1, int(bool(r.get("pre_check_present")))),
            (1, int(bool(r.get("rule_details_present")))),
            (2 if decision == "ESCALATED" else 0,   int(bool(r.get("conflict_present")))),
            (1, int(regulatory_cited)),
            (2 if failed_rules else 1,               int(has_explanation)),
        ]

        total_weight   = sum(w for w, _ in components)
        weighted_score = sum(w * v for w, v in components)
        if total_weight > 0:
            run_scores.append(weighted_score / total_weight)

    return round(sum(run_scores) / len(run_scores), 4) if run_scores else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Summary reporter
# ─────────────────────────────────────────────────────────────────────────────

def compute_governance_metrics(
    results: list[dict],
    architecture_name: str = "pipeline",
) -> dict[str, Any]:
    """
    Compute M8-M15 and M15b for a list of run results and return a summary dict.

    M8, M10, M13, M14, M15, M15b are pipeline-only (the baseline has no
    deterministic override layers, no three-point check, and no structured
    disclosure reports).
    M9, M11, M12 apply to both architectures.
    """
    is_pipeline = architecture_name == "pipeline"

    m8   = override_rate(results)                  if is_pipeline else "N/A (baseline)"
    m9   = pipeline_halt_rate(results)
    m10  = three_point_integrity_rate(results)     if is_pipeline else "N/A (baseline)"
    m11  = hard_rule_enforcement_rate(results)
    m12  = vulnerability_protection_rate(results)
    m13  = regulatory_citation_rate(results)       if is_pipeline else "N/A (baseline)"
    m14  = explanation_completeness(results)       if is_pipeline else "N/A (baseline)"
    m15  = decision_traceability_score(results)    if is_pipeline else "N/A (baseline)"
    m15b = decision_traceability_score_v2(results) if is_pipeline else "N/A (baseline)"

    def _fmt(v: Any) -> Any:
        if isinstance(v, float):
            return round(v, 4)
        if isinstance(v, dict):
            return {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items()}
        return v

    return {
        "architecture":                  architecture_name,
        "M8_override_rate":              _fmt(m8),
        "M9_pipeline_halt_rate":         _fmt(m9),
        "M10_three_point_integrity":     _fmt(m10),
        "M11_hard_rule_enforcement":     _fmt(m11),
        "M12_vulnerability_protection":  _fmt(m12),
        "M13_regulatory_citation_rate":  _fmt(m13),
        "M14_explanation_completeness":  _fmt(m14),
        "M15_decision_traceability":     _fmt(m15),
        "M15b_decision_traceability_v2": _fmt(m15b),
    }
