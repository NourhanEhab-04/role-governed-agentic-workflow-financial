"""
tests/test_rl_rewards.py
========================
Unit tests for rl/rewards.py.

All tests are pure-function: no LLM calls, no filesystem access, no network.
State dicts are constructed inline to match the exact key shapes used by the
live pipeline (schemas/langgraph_state.py).

Data-structure reference
------------------------
a1_verification / a2_verification:
    {"passed": bool, "confidence": float, "issues": list, "field_checks": dict}

a1_correction / a2_correction:
    {"original": dict, "corrected": dict, "fields_fixed": list}
    (presence means a patch was applied; absence means first pass was accepted)

a1_final_verification / a2_final_verification:
    Same shape as a1_verification. Present only when a correction was applied.

pre_check_verdict (from rule_engine.evaluate_suitability):
    {
        "score": int,
        "decision": str,
        "hard_failed_rules": list[str],
        "rules": [  ← list of dicts (engine format)
            {"rule": "R1", "pass": bool, "hard_fail": bool,
             "penalty": int, "pillar": int, "detail": str},
            ...
        ],
        "rule_engine_version": str,
    }

rule_verdict (from A3 / RuleVerdictModel):
    {
        "score": int,
        "decision": str,
        "hard_failed_rules": list[str],
        "rules": {"R1": "PASS", "R2": "FAIL", ...},  ← dict format
    }

conflict_report (from A4 / ConflictReportModel):
    {"flags": list, "escalate": bool, "summary": str}

suitability_report (from A5 / SuitabilityReportModel):
    {"decision": str, "summary": str, "rule_findings": list, ...}
"""

import pytest
from rl.rewards import (
    reward_a1,
    reward_a2,
    reward_a3,
    reward_a4,
    reward_a5,
    composite_reward,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_verification(passed: bool, confidence: float = 0.95) -> dict:
    """Build a verifier result dict (AV output shape)."""
    return {
        "passed": passed,
        "confidence": confidence,
        "issues": [] if passed else ["field check failed"],
        "field_checks": {},
    }


def _make_correction() -> dict:
    """Build a minimal corrector patch dict (signals a correction was applied)."""
    return {
        "original": {"financial_knowledge": "none"},
        "corrected": {"financial_knowledge": "basic"},
        "fields_fixed": ["financial_knowledge"],
    }


def _make_pre_check_verdict(decision: str, failed_rules: list[str]) -> dict:
    """
    Build a pre_check_verdict matching the output of rule_engine.evaluate_suitability().
    rules is a list of dicts — this is the engine's native format.
    """
    all_rules = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
    rules_list = [
        {
            "rule": r,
            "pass": r not in failed_rules,
            "hard_fail": r in {"R1", "R4", "R5", "R7"},
            "penalty": 0 if r not in failed_rules else (-20 if r == "R2" else -15),
            "pillar": 1,
            "detail": "ok" if r not in failed_rules else "failed",
        }
        for r in all_rules
    ]
    penalty_total = sum(r["penalty"] for r in rules_list)
    score = 100 + penalty_total if not failed_rules else 100 + penalty_total
    return {
        "score": score,
        "decision": decision,
        "hard_failed_rules": [r for r in failed_rules if r in {"R1", "R4", "R5", "R7"}],
        "rules": rules_list,
        "rule_engine_version": "1.0.0",
    }


def _make_rule_verdict(decision: str, failed_rules: list[str]) -> dict:
    """
    Build a rule_verdict matching the output of RuleVerdictModel (A3 format).
    rules is a dict {"R1": "PASS", ...} — this is A3's output format.
    """
    all_rules = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
    rules_dict = {r: ("FAIL" if r in failed_rules else "PASS") for r in all_rules}
    return {
        "score": 100,
        "decision": decision,
        "hard_failed_rules": [r for r in failed_rules if r in {"R1", "R4", "R5", "R7"}],
        "rules": rules_dict,
    }


def _make_conflict_report(escalate: bool) -> dict:
    flags = []
    if escalate:
        flags = [{"rule_id": "CONTRADICTION", "triggered": True,
                  "severity": "HIGH", "message": "test"}]
    return {"flags": flags, "escalate": escalate, "summary": "test summary"}


def _make_suitability_report(decision: str) -> dict:
    findings = [
        {"rule_id": r, "status": "PASS", "explanation": "ok"}
        for r in ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
    ]
    return {
        "decision": decision,
        "summary": "test",
        "rule_findings": findings,
        "flags_addressed": [],
        "regulatory_basis": "MiFID II Article 25(2)",
        "client_facing_summary": "test summary",
    }


def _make_scenario(decision: str, escalate: bool) -> dict:
    return {
        "description": "test scenario",
        "expected_decision": decision,
        "expected_escalate": escalate,
    }


# ── reward_a1 ─────────────────────────────────────────────────────────────────

class TestRewardA1:

    def test_first_pass_no_correction_returns_1(self):
        state = {"a1_verification": _make_verification(passed=True)}
        assert reward_a1(state) == 1.0

    def test_first_pass_failed_no_correction_returns_0(self):
        state = {"a1_verification": _make_verification(passed=False)}
        assert reward_a1(state) == 0.0

    def test_correction_applied_recheck_passes_returns_0_6(self):
        state = {
            "a1_verification": _make_verification(passed=False),
            "a1_correction": _make_correction(),
            "a1_final_verification": _make_verification(passed=True),
        }
        assert reward_a1(state) == 0.6

    def test_correction_applied_recheck_fails_returns_0(self):
        state = {
            "a1_verification": _make_verification(passed=False),
            "a1_correction": _make_correction(),
            "a1_final_verification": _make_verification(passed=False),
        }
        assert reward_a1(state) == 0.0

    def test_correction_applied_no_final_verification_returns_0(self):
        # Patch applied but no re-check recorded — unverified correction
        state = {
            "a1_verification": _make_verification(passed=False),
            "a1_correction": _make_correction(),
            # a1_final_verification absent
        }
        assert reward_a1(state) == 0.0

    def test_no_a1_verification_returns_0_5(self):
        # Node did not run or state is incomplete
        assert reward_a1({}) == 0.5

    def test_a1_verification_none_explicit_returns_0_5(self):
        assert reward_a1({"a1_verification": None}) == 0.5

    def test_first_pass_ok_correction_also_present_uses_correction_path(self):
        # Edge: if correction dict present, we go down the correction path
        # regardless of what first_pass says (first_pass was True but correction
        # was applied anyway — unusual but must not crash).
        state = {
            "a1_verification": _make_verification(passed=True),
            "a1_correction": _make_correction(),
            "a1_final_verification": _make_verification(passed=True),
        }
        # correction path → 0.6 (not 1.0)
        assert reward_a1(state) == 0.6

    def test_return_value_is_float(self):
        state = {"a1_verification": _make_verification(passed=True)}
        assert isinstance(reward_a1(state), float)

    def test_return_value_in_bounds(self):
        for state in [
            {},
            {"a1_verification": _make_verification(True)},
            {"a1_verification": _make_verification(False)},
            {
                "a1_verification": _make_verification(False),
                "a1_correction": _make_correction(),
                "a1_final_verification": _make_verification(True),
            },
        ]:
            r = reward_a1(state)
            assert 0.0 <= r <= 1.0, f"reward out of bounds: {r} for state {state}"


# ── reward_a2 ─────────────────────────────────────────────────────────────────

class TestRewardA2:
    """reward_a2 has identical logic to reward_a1 but reads a2_* keys."""

    def test_first_pass_returns_1(self):
        state = {"a2_verification": _make_verification(passed=True)}
        assert reward_a2(state) == 1.0

    def test_first_pass_failed_returns_0(self):
        state = {"a2_verification": _make_verification(passed=False)}
        assert reward_a2(state) == 0.0

    def test_correction_recheck_passes_returns_0_6(self):
        state = {
            "a2_verification": _make_verification(passed=False),
            "a2_correction": _make_correction(),
            "a2_final_verification": _make_verification(passed=True),
        }
        assert reward_a2(state) == 0.6

    def test_correction_recheck_fails_returns_0(self):
        state = {
            "a2_verification": _make_verification(passed=False),
            "a2_correction": _make_correction(),
            "a2_final_verification": _make_verification(passed=False),
        }
        assert reward_a2(state) == 0.0

    def test_no_verification_returns_0_5(self):
        assert reward_a2({}) == 0.5

    def test_a1_keys_do_not_bleed_into_a2(self):
        # Only a1_* keys present — a2 should see nothing and return 0.5
        state = {
            "a1_verification": _make_verification(passed=True),
        }
        assert reward_a2(state) == 0.5

    def test_return_value_is_float(self):
        assert isinstance(reward_a2({"a2_verification": _make_verification(True)}), float)


# ── reward_a3 ─────────────────────────────────────────────────────────────────

class TestRewardA3:

    def test_full_agreement_returns_1(self):
        state = {
            "pre_check_verdict": _make_pre_check_verdict("SUITABLE", []),
            "rule_verdict": _make_rule_verdict("SUITABLE", []),
        }
        assert reward_a3(state) == 1.0

    def test_decision_mismatch_returns_0(self):
        state = {
            "pre_check_verdict": _make_pre_check_verdict("SUITABLE", []),
            "rule_verdict": _make_rule_verdict("UNSUITABLE", ["R1"]),
        }
        assert reward_a3(state) == 0.0

    def test_decision_agrees_rules_differ_returns_0_5(self):
        # pre_check says R2 failed; A3 says R3 failed — both CONDITIONAL
        state = {
            "pre_check_verdict": _make_pre_check_verdict("CONDITIONAL", ["R2"]),
            "rule_verdict": _make_rule_verdict("CONDITIONAL", ["R3"]),
        }
        assert reward_a3(state) == 0.5

    def test_full_agreement_with_hard_fails(self):
        state = {
            "pre_check_verdict": _make_pre_check_verdict("UNSUITABLE", ["R1", "R7"]),
            "rule_verdict": _make_rule_verdict("UNSUITABLE", ["R1", "R7"]),
        }
        assert reward_a3(state) == 1.0

    def test_pre_check_absent_returns_0(self):
        state = {"rule_verdict": _make_rule_verdict("SUITABLE", [])}
        assert reward_a3(state) == 0.0

    def test_rule_verdict_absent_returns_0(self):
        state = {"pre_check_verdict": _make_pre_check_verdict("SUITABLE", [])}
        assert reward_a3(state) == 0.0

    def test_both_absent_returns_0(self):
        assert reward_a3({}) == 0.0

    def test_conditional_full_agreement_soft_fails(self):
        state = {
            "pre_check_verdict": _make_pre_check_verdict("CONDITIONAL", ["R2", "R3"]),
            "rule_verdict": _make_rule_verdict("CONDITIONAL", ["R2", "R3"]),
        }
        assert reward_a3(state) == 1.0

    def test_escalated_decision_not_in_pre_check_returns_0(self):
        # pre_check never produces ESCALATED (only rule engine decisions)
        # so comparing SUITABLE vs ESCALATED must be 0
        state = {
            "pre_check_verdict": _make_pre_check_verdict("SUITABLE", []),
            "rule_verdict": {
                "score": 100,
                "decision": "ESCALATED",
                "hard_failed_rules": [],
                "rules": {r: "PASS" for r in ["R1","R2","R3","R4","R5","R6","R7"]},
            },
        }
        assert reward_a3(state) == 0.0

    def test_pre_check_dict_format_rules_handled_as_fallback(self):
        # Defensive: if pre_check rules were somehow stored in dict format
        state = {
            "pre_check_verdict": {
                "decision": "SUITABLE",
                "rules": {"R1": "PASS", "R2": "PASS", "R3": "PASS",
                          "R4": "PASS", "R5": "PASS", "R6": "PASS", "R7": "PASS"},
            },
            "rule_verdict": _make_rule_verdict("SUITABLE", []),
        }
        assert reward_a3(state) == 1.0

    def test_return_value_in_bounds(self):
        for state in [
            {},
            {"pre_check_verdict": _make_pre_check_verdict("SUITABLE", []),
             "rule_verdict": _make_rule_verdict("SUITABLE", [])},
            {"pre_check_verdict": _make_pre_check_verdict("CONDITIONAL", ["R2"]),
             "rule_verdict": _make_rule_verdict("CONDITIONAL", ["R3"])},
        ]:
            r = reward_a3(state)
            assert 0.0 <= r <= 1.0, f"reward out of bounds: {r}"


# ── reward_a4 ─────────────────────────────────────────────────────────────────

class TestRewardA4:

    def test_escalate_true_expected_true_returns_1(self):
        state = {"conflict_report": _make_conflict_report(escalate=True)}
        assert reward_a4(state, expected_escalate=True) == 1.0

    def test_escalate_false_expected_false_returns_1(self):
        state = {"conflict_report": _make_conflict_report(escalate=False)}
        assert reward_a4(state, expected_escalate=False) == 1.0

    def test_escalate_true_expected_false_returns_0(self):
        state = {"conflict_report": _make_conflict_report(escalate=True)}
        assert reward_a4(state, expected_escalate=False) == 0.0

    def test_escalate_false_expected_true_returns_0(self):
        state = {"conflict_report": _make_conflict_report(escalate=False)}
        assert reward_a4(state, expected_escalate=True) == 0.0

    def test_conflict_report_absent_returns_0(self):
        assert reward_a4({}, expected_escalate=False) == 0.0

    def test_conflict_report_none_returns_0(self):
        assert reward_a4({"conflict_report": None}, expected_escalate=False) == 0.0

    def test_escalate_wrong_type_string_returns_0(self):
        state = {"conflict_report": {"flags": [], "escalate": "true", "summary": "x"}}
        assert reward_a4(state, expected_escalate=True) == 0.0

    def test_escalate_wrong_type_int_returns_0(self):
        state = {"conflict_report": {"flags": [], "escalate": 1, "summary": "x"}}
        assert reward_a4(state, expected_escalate=True) == 0.0

    def test_escalate_key_absent_returns_0(self):
        state = {"conflict_report": {"flags": [], "summary": "x"}}
        assert reward_a4(state, expected_escalate=False) == 0.0

    def test_return_value_is_float(self):
        state = {"conflict_report": _make_conflict_report(escalate=False)}
        assert isinstance(reward_a4(state, False), float)


# ── reward_a5 ─────────────────────────────────────────────────────────────────

class TestRewardA5:

    def test_suitable_matches_returns_1(self):
        state = {"suitability_report": _make_suitability_report("SUITABLE")}
        assert reward_a5(state, "SUITABLE") == 1.0

    def test_unsuitable_matches_returns_1(self):
        state = {"suitability_report": _make_suitability_report("UNSUITABLE")}
        assert reward_a5(state, "UNSUITABLE") == 1.0

    def test_conditional_matches_returns_1(self):
        state = {"suitability_report": _make_suitability_report("CONDITIONAL")}
        assert reward_a5(state, "CONDITIONAL") == 1.0

    def test_escalated_matches_returns_1(self):
        state = {"suitability_report": _make_suitability_report("ESCALATED")}
        assert reward_a5(state, "ESCALATED") == 1.0

    def test_mismatch_suitable_vs_unsuitable_returns_0(self):
        state = {"suitability_report": _make_suitability_report("SUITABLE")}
        assert reward_a5(state, "UNSUITABLE") == 0.0

    def test_mismatch_escalated_vs_suitable_returns_0(self):
        state = {"suitability_report": _make_suitability_report("ESCALATED")}
        assert reward_a5(state, "SUITABLE") == 0.0

    def test_report_absent_returns_0(self):
        assert reward_a5({}, "SUITABLE") == 0.0

    def test_report_none_returns_0(self):
        assert reward_a5({"suitability_report": None}, "SUITABLE") == 0.0

    def test_decision_key_absent_returns_0(self):
        state = {"suitability_report": {"summary": "no decision key"}}
        assert reward_a5(state, "SUITABLE") == 0.0

    def test_return_value_is_float(self):
        state = {"suitability_report": _make_suitability_report("SUITABLE")}
        assert isinstance(reward_a5(state, "SUITABLE"), float)


# ── composite_reward ──────────────────────────────────────────────────────────

class TestCompositeReward:

    def _perfect_state(self) -> dict:
        """Build a state where every agent output is correct on first try."""
        return {
            # A1: first pass accepted
            "a1_verification": _make_verification(passed=True),
            # A2: first pass accepted
            "a2_verification": _make_verification(passed=True),
            # A3: full agreement
            "pre_check_verdict": _make_pre_check_verdict("SUITABLE", []),
            "rule_verdict": _make_rule_verdict("SUITABLE", []),
            # A4: no escalation
            "conflict_report": _make_conflict_report(escalate=False),
            # A5: correct decision
            "suitability_report": _make_suitability_report("SUITABLE"),
        }

    def test_all_correct_composite_is_1(self):
        state = self._perfect_state()
        scenario = _make_scenario("SUITABLE", False)
        result = composite_reward(state, scenario)
        assert result["composite"] == pytest.approx(1.0)

    def test_result_has_all_agent_keys(self):
        state = self._perfect_state()
        scenario = _make_scenario("SUITABLE", False)
        result = composite_reward(state, scenario)
        assert set(result.keys()) == {"a1", "a2", "a3", "a4", "a5", "composite"}

    def test_all_wrong_composite_is_0(self):
        state = {
            "a1_verification": _make_verification(passed=False),
            "a2_verification": _make_verification(passed=False),
            "pre_check_verdict": _make_pre_check_verdict("SUITABLE", []),
            "rule_verdict": _make_rule_verdict("UNSUITABLE", ["R1"]),  # decision mismatch
            "conflict_report": _make_conflict_report(escalate=True),  # wrong
            "suitability_report": _make_suitability_report("UNSUITABLE"),  # wrong
        }
        scenario = _make_scenario("SUITABLE", False)
        result = composite_reward(state, scenario)
        assert result["composite"] == pytest.approx(0.0)

    def test_correction_needed_but_fixed_composite(self):
        # A1 and A2 needed correction (0.6 each); A3, A4, A5 all correct
        state = {
            "a1_verification": _make_verification(passed=False),
            "a1_correction": _make_correction(),
            "a1_final_verification": _make_verification(passed=True),
            "a2_verification": _make_verification(passed=False),
            "a2_correction": _make_correction(),
            "a2_final_verification": _make_verification(passed=True),
            "pre_check_verdict": _make_pre_check_verdict("SUITABLE", []),
            "rule_verdict": _make_rule_verdict("SUITABLE", []),
            "conflict_report": _make_conflict_report(escalate=False),
            "suitability_report": _make_suitability_report("SUITABLE"),
        }
        scenario = _make_scenario("SUITABLE", False)
        result = composite_reward(state, scenario)
        expected = 0.10 * 0.6 + 0.10 * 0.6 + 0.10 * 1.0 + 0.10 * 1.0 + 0.60 * 1.0
        assert result["composite"] == pytest.approx(expected)
        assert result["a1"] == pytest.approx(0.6)
        assert result["a2"] == pytest.approx(0.6)
        assert result["a3"] == pytest.approx(1.0)
        assert result["a4"] == pytest.approx(1.0)
        assert result["a5"] == pytest.approx(1.0)

    def test_only_a5_wrong_composite(self):
        # All local rewards perfect, only final decision wrong
        state = self._perfect_state()
        state["suitability_report"] = _make_suitability_report("UNSUITABLE")
        scenario = _make_scenario("SUITABLE", False)
        result = composite_reward(state, scenario)
        # a5 = 0.0, all others = 1.0
        expected = 0.10 + 0.10 + 0.10 + 0.10 + 0.60 * 0.0
        assert result["composite"] == pytest.approx(expected)
        assert result["a5"] == pytest.approx(0.0)

    def test_escalated_scenario_all_correct(self):
        escalated_state = {
            "a1_verification": _make_verification(passed=True),
            "a2_verification": _make_verification(passed=True),
            "pre_check_verdict": _make_pre_check_verdict("SUITABLE", []),
            "rule_verdict": _make_rule_verdict("SUITABLE", []),
            "conflict_report": _make_conflict_report(escalate=True),
            "suitability_report": _make_suitability_report("ESCALATED"),
        }
        scenario = _make_scenario("ESCALATED", True)
        result = composite_reward(state=escalated_state, scenario=scenario)
        assert result["composite"] == pytest.approx(1.0)

    def test_composite_is_float_rounded_to_6_decimal_places(self):
        state = self._perfect_state()
        scenario = _make_scenario("SUITABLE", False)
        result = composite_reward(state, scenario)
        # round() to 6 d.p. — check it's a float and has at most 6 d.p.
        assert isinstance(result["composite"], float)
        # Verify rounding: str representation should not exceed 6 d.p. after decimal
        str_val = str(result["composite"])
        if "." in str_val:
            assert len(str_val.split(".")[1]) <= 6

    def test_composite_weights_sum_to_one(self):
        # Verify numerically: composite of all-1.0 rewards must equal exactly 1.0
        state = self._perfect_state()
        scenario = _make_scenario("SUITABLE", False)
        result = composite_reward(state, scenario)
        assert result["composite"] == pytest.approx(1.0), (
            "weights do not sum to 1 — composite of all-1.0 rewards must be 1.0"
        )

    def test_composite_always_in_0_1_range(self):
        scenarios_and_states = [
            ({}, _make_scenario("SUITABLE", False)),
            ({"a1_verification": _make_verification(True),
              "a2_verification": _make_verification(True),
              "pre_check_verdict": _make_pre_check_verdict("CONDITIONAL", ["R2"]),
              "rule_verdict": _make_rule_verdict("CONDITIONAL", ["R3"]),  # 0.5
              "conflict_report": _make_conflict_report(False),
              "suitability_report": _make_suitability_report("CONDITIONAL")},
             _make_scenario("CONDITIONAL", False)),
        ]
        for state, scenario in scenarios_and_states:
            result = composite_reward(state, scenario)
            c = result["composite"]
            assert 0.0 <= c <= 1.0, f"composite {c} out of [0,1] for state {state}"

    def test_scenario_missing_expected_decision_raises(self):
        state = self._perfect_state()
        bad_scenario = {"expected_escalate": False}  # missing "expected_decision"
        with pytest.raises(KeyError):
            composite_reward(state, bad_scenario)

    def test_scenario_missing_expected_escalate_raises(self):
        state = self._perfect_state()
        bad_scenario = {"expected_decision": "SUITABLE"}  # missing "expected_escalate"
        with pytest.raises(KeyError):
            composite_reward(state, bad_scenario)

    def test_a3_partial_credit_flows_into_composite(self):
        # A3 gets 0.5 (decision agrees but rule sets differ)
        state = {
            "a1_verification": _make_verification(passed=True),
            "a2_verification": _make_verification(passed=True),
            "pre_check_verdict": _make_pre_check_verdict("CONDITIONAL", ["R2"]),
            "rule_verdict": _make_rule_verdict("CONDITIONAL", ["R3"]),  # rules differ
            "conflict_report": _make_conflict_report(escalate=False),
            "suitability_report": _make_suitability_report("CONDITIONAL"),
        }
        scenario = _make_scenario("CONDITIONAL", False)
        result = composite_reward(state, scenario)
        assert result["a3"] == pytest.approx(0.5)
        expected = 0.10 * 1.0 + 0.10 * 1.0 + 0.10 * 0.5 + 0.10 * 1.0 + 0.60 * 1.0
        assert result["composite"] == pytest.approx(expected)
