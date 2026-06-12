"""
evaluation/ablation_study/rule_engine_variants.py
===================================================
Wraps the deterministic rule engine so individual rules can be disabled
and score thresholds can be varied — both needed for the ablation study.

Disabling a rule means it is forced to PASS with zero penalty.
This isolates each rule's individual contribution to the final decision.

Zero LLM calls. Pure function.
"""

from __future__ import annotations

from rule_engine.rule_engine import (
    _r1_knowledge,
    _r2_risk_alignment,
    _r3_horizon,
    _r4_affordability,
    _r5_vulnerability,
    _r6_leverage,
    _r7_complexity,
    _validate_inputs,
    RULE_ENGINE_VERSION,
)

_RULE_FN_MAP = {
    "R1": _r1_knowledge,
    "R2": _r2_risk_alignment,
    "R3": _r3_horizon,
    "R4": _r4_affordability,
    "R5": _r5_vulnerability,
    "R6": _r6_leverage,
    "R7": _r7_complexity,
}

_FORCED_PASS_TEMPLATE = {
    "pass":      True,
    "hard_fail": False,
    "penalty":   0,
    "detail":    "[ABLATION: rule disabled — forced PASS]",
}


def _compute_decision_with_thresholds(
    rules: list[dict],
    suitable_threshold: int,
    conditional_threshold: int,
) -> tuple[int, str, list[str]]:
    hard_failed = [r["rule"] for r in rules if r["hard_fail"] and not r["pass"]]
    soft_penalty = sum(r["penalty"] for r in rules if not r["hard_fail"])
    score = 100 + soft_penalty

    if hard_failed:
        return score, "UNSUITABLE", hard_failed
    if score >= suitable_threshold:
        return score, "SUITABLE", []
    if score >= conditional_threshold:
        return score, "CONDITIONAL", []
    return score, "UNSUITABLE", []


def evaluate_suitability_ablated(
    client: dict,
    product: dict,
    disabled_rules: list[str] | None = None,
    suitable_threshold: int = 70,
    conditional_threshold: int = 40,
) -> dict:
    """
    Evaluate MiFID II suitability with ablation overrides.

    Parameters
    ----------
    client, product       : profile dicts (same schema as the main rule engine)
    disabled_rules        : rule IDs to force-pass, e.g. ["R1", "R5"]
    suitable_threshold    : score >= this → SUITABLE  (default 70)
    conditional_threshold : score >= this → CONDITIONAL (default 40)

    Returns the same schema as evaluate_suitability() plus:
      "ablation_disabled_rules" : list of rules that were forced to pass
      "ablation_suitable_threshold"    : threshold used
      "ablation_conditional_threshold" : threshold used
    """
    _validate_inputs(client, product)
    disabled = set(disabled_rules or [])

    rules: list[dict] = []
    for rule_id, fn in _RULE_FN_MAP.items():
        if rule_id in disabled:
            result = {
                "rule":      rule_id,
                "pass":      True,
                "hard_fail": False,
                "penalty":   0,
                "pillar":    {"R1": 1, "R2": 3, "R3": 2, "R4": 2,
                              "R5": 3, "R6": 3, "R7": 1}[rule_id],
                "detail":    "[ABLATION: rule disabled — forced PASS]",
            }
        else:
            result = fn(client, product)
        rules.append(result)

    score, decision, hard_failed = _compute_decision_with_thresholds(
        rules, suitable_threshold, conditional_threshold
    )

    return {
        "score":               score,
        "decision":            decision,
        "hard_failed_rules":   hard_failed,
        "rules":               rules,
        "rule_engine_version": RULE_ENGINE_VERSION,
        "ablation_disabled_rules":        sorted(disabled),
        "ablation_suitable_threshold":    suitable_threshold,
        "ablation_conditional_threshold": conditional_threshold,
    }
