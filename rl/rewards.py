"""
rl/rewards.py
=============
Verifiable, per-agent reward functions for MARL training.

Each function takes a PipelineState dict and returns a float in [0.0, 1.0].
All rewards are derived purely from existing deterministic infrastructure
(rule engine verdicts, verifier outcomes, validator results) — no LLM calls
are needed to compute any reward at training time.

Reward meanings
---------------
  1.0  agent produced a correct output on the first attempt
  0.6  agent output was wrong first try but the correction loop fixed it (A1/A2 only)
  0.5  state is incomplete — the relevant node did not run; no usable signal
  0.0  agent output is wrong and either uncorrectable or downstream-inconsistent

State key conventions (from schemas/langgraph_state.py)
--------------------------------------------------------
A1 verification trail:
  a1_verification        dict  first AV pass   {"passed": bool, "confidence": float, ...}
  a1_correction          dict  corrector patch  {"original": ..., "corrected": ..., ...}
                               None when first pass was accepted without changes
  a1_final_verification  dict  AV re-check after patch
                               None when no patch was applied

A2 verification trail (same structure as A1):
  a2_verification, a2_correction, a2_final_verification

A3 rule-engine agreement:
  pre_check_verdict  dict  {"decision": str, "rules": list[dict], ...}
                           rules is a list from rule_engine.evaluate_suitability()
                           each entry: {"rule": str, "pass": bool, ...}
  rule_verdict       dict  {"decision": str, "rules": dict, ...}
                           rules is a dict from RuleVerdictModel {"R1": "PASS", ...}

A4 conflict detection:
  conflict_report  dict  {"escalate": bool, "flags": list, "summary": str}

A5 final decision (global reward):
  suitability_report  dict  {"decision": str, ...}
"""

from __future__ import annotations


# ── Individual agent rewards ───────────────────────────────────────────────────

def reward_a1(state: dict) -> float:
    """
    Reward for A1 (Client Profiler) based on verifier (AV) outcome.

    Decision tree:
      a1_verification absent           → 0.5  (node did not run)
      first pass accepted, no patch    → 1.0  (correct on first try)
      first pass rejected, no patch    → 0.0  (failed; no correction attempted)
      patch applied, re-check passed   → 0.6  (needed fixing but fixed correctly)
      patch applied, re-check failed   → 0.0  (still wrong after correction)
      patch applied, no re-check data  → 0.0  (correction unverified — penalise)
    """
    first_pass: dict | None = state.get("a1_verification")
    correction: dict | None = state.get("a1_correction")
    final_pass: dict | None = state.get("a1_final_verification")

    if first_pass is None:
        return 0.5  # A1 node did not run or state snapshot is incomplete

    if correction is None:
        # No patch was applied — the first-pass result is the final result
        return 1.0 if bool(first_pass.get("passed", False)) else 0.0

    # A correction patch was applied — judge by the re-verification result
    if final_pass is None:
        return 0.0  # patch ran but re-check was never recorded
    return 0.6 if bool(final_pass.get("passed", False)) else 0.0


def reward_a2(state: dict) -> float:
    """
    Reward for A2 (Product Classifier) — identical logic to reward_a1.

    Decision tree mirrors reward_a1 exactly but reads a2_* state keys.
    """
    first_pass: dict | None = state.get("a2_verification")
    correction: dict | None = state.get("a2_correction")
    final_pass: dict | None = state.get("a2_final_verification")

    if first_pass is None:
        return 0.5

    if correction is None:
        return 1.0 if bool(first_pass.get("passed", False)) else 0.0

    if final_pass is None:
        return 0.0
    return 0.6 if bool(final_pass.get("passed", False)) else 0.0


def reward_a3(state: dict) -> float:
    """
    Reward for A3 (Rule Engine Agent) based on agreement with the deterministic
    pre-check verdict.

    The pre-check and A3 both call evaluate_suitability() on the same profiles.
    Full credit requires agreement on two dimensions:
      1. decision string  ("SUITABLE" | "CONDITIONAL" | "UNSUITABLE")
      2. set of failed rule IDs  ({"R1", "R4"} etc.)

    The two verdicts store rules in different formats:
      pre_check_verdict["rules"] = list of rule dicts  (engine output)
      rule_verdict["rules"]      = dict {"R1": "PASS", ...}  (RuleVerdictModel)

    Returns:
      1.0  both decision and failed-rule set agree
      0.5  decision agrees but failed-rule sets differ (suspicious — partial credit)
      0.0  decision disagrees, or either verdict is absent
    """
    pre = state.get("pre_check_verdict")
    verdict = state.get("rule_verdict")

    if pre is None or verdict is None:
        return 0.0

    pre_decision = pre.get("decision")
    a3_decision = verdict.get("decision")

    if pre_decision != a3_decision:
        return 0.0

    # ── Extract failed rule IDs from pre_check (engine list format) ────────────
    pre_rules_raw = pre.get("rules", [])
    if isinstance(pre_rules_raw, list):
        # Standard rule_engine output: list of dicts with "rule" and "pass" keys
        pre_failed: set[str] = {
            r["rule"] for r in pre_rules_raw if not r.get("pass", True)
        }
    else:
        # Defensive fallback: dict format {"R1": "PASS", ...}
        pre_failed = {k for k, v in pre_rules_raw.items() if v == "FAIL"}

    # ── Extract failed rule IDs from rule_verdict (RuleVerdictModel dict format)
    a3_rules_raw = verdict.get("rules", {})
    if isinstance(a3_rules_raw, dict):
        # Standard A3 output: dict {"R1": "PASS", "R2": "FAIL", ...}
        a3_failed: set[str] = {k for k, v in a3_rules_raw.items() if v == "FAIL"}
    else:
        # Defensive fallback: list format
        a3_failed = {r["rule"] for r in a3_rules_raw if not r.get("pass", True)}

    return 1.0 if pre_failed == a3_failed else 0.5


def reward_a4(state: dict, expected_escalate: bool) -> float:
    """
    Reward for A4 (Conflict Detector) based on escalation accuracy.

    The ground truth comes from data/scenarios/*.json → "expected_escalate".
    No partial credit: escalation is a binary, legally significant decision.

    Args:
      state:             PipelineState dict after a completed run.
      expected_escalate: Ground-truth bool from the scenario file.

    Returns:
      1.0  conflict_report.escalate matches expected_escalate exactly
      0.0  mismatch, wrong type, or conflict_report absent
    """
    report = state.get("conflict_report")
    if report is None:
        return 0.0
    actual = report.get("escalate")
    if not isinstance(actual, bool):
        return 0.0
    return 1.0 if actual == expected_escalate else 0.0


def reward_a5(state: dict, expected_decision: str) -> float:
    """
    Global reward for A5 (Disclosure Agent) — the primary pipeline accuracy signal.

    The ground truth comes from data/scenarios/*.json → "expected_decision".
    Valid expected values: "SUITABLE" | "CONDITIONAL" | "UNSUITABLE" | "ESCALATED".

    Args:
      state:             PipelineState dict after a completed run.
      expected_decision: Ground-truth decision string from the scenario file.

    Returns:
      1.0  suitability_report.decision matches expected_decision exactly
      0.0  mismatch or suitability_report absent
    """
    report = state.get("suitability_report")
    if report is None:
        return 0.0
    return 1.0 if report.get("decision") == expected_decision else 0.0


# ── Composite reward for cooperative MARL ─────────────────────────────────────

def composite_reward(state: dict, scenario: dict) -> dict[str, float]:
    """
    Compute all per-agent rewards and a weighted composite for cooperative MARL.

    The composite blends local verifiable rewards (which can be computed without
    running downstream agents) with the global outcome reward (A5). Each agent
    receives an immediate local signal plus a shared incentive for final accuracy.

    Weight rationale
    ----------------
    A5 gets the highest weight (0.60) because final decision accuracy is the
    primary training objective. The remaining 0.40 is split equally across the
    four upstream agents so each receives a meaningful local gradient.

      A1 0.10 — profiling quality drives every downstream step
      A2 0.10 — product classification drives every downstream step
      A3 0.10 — deterministic agreement is a system integrity requirement
      A4 0.10 — escalation accuracy is a hard compliance requirement
      A5 0.60 — final decision accuracy is the primary objective

    Composite formula:
      composite = 0.10·r_a1 + 0.10·r_a2 + 0.10·r_a3 + 0.10·r_a4 + 0.60·r_a5

    This keeps composite in [0.0, 1.0] because all individual rewards are in
    [0.0, 1.0] and the weights sum to exactly 1.0.

    Args:
      state:    Final PipelineState dict from a completed pipeline run.
      scenario: Scenario dict from data/scenarios/*.json. Must contain:
                  "expected_decision" (str)   — ground-truth decision
                  "expected_escalate" (bool)  — ground-truth escalation flag

    Returns:
      Dict with keys "a1", "a2", "a3", "a4", "a5", "composite".
      All values are floats in [0.0, 1.0] (composite is rounded to 6 d.p.).
    """
    expected_decision: str = scenario["expected_decision"]
    expected_escalate: bool = scenario["expected_escalate"]

    r_a1 = reward_a1(state)
    r_a2 = reward_a2(state)
    r_a3 = reward_a3(state)
    r_a4 = reward_a4(state, expected_escalate)
    r_a5 = reward_a5(state, expected_decision)

    composite = (
        0.10 * r_a1
        + 0.10 * r_a2
        + 0.10 * r_a3
        + 0.10 * r_a4
        + 0.60 * r_a5
    )

    return {
        "a1": r_a1,
        "a2": r_a2,
        "a3": r_a3,
        "a4": r_a4,
        "a5": r_a5,
        "composite": round(composite, 6),
    }
