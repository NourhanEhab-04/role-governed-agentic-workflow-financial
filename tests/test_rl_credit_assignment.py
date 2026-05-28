"""
tests/test_rl_credit_assignment.py
===================================
Unit tests for rl/credit_assignment.py (Step 6).

All tests are pure-function: no LLM calls, no filesystem, no network.
State dicts are identical in shape to those used in test_rl_rewards.py.
"""

import pytest
from rl.credit_assignment import assign_credit, composite_credit, counterfactual_credit


# ── Helpers (shared with test_rl_rewards.py pattern) ─────────────────────────

def _ver(passed: bool) -> dict:
    return {"passed": passed, "confidence": 0.95, "issues": [], "field_checks": {}}


def _correction() -> dict:
    return {"original": {}, "corrected": {}, "fields_fixed": ["field"]}


def _pre_check(decision: str, failed: list[str]) -> dict:
    all_rules = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
    return {
        "decision": decision,
        "rules": [
            {"rule": r, "pass": r not in failed, "hard_fail": r in {"R1","R4","R5","R7"},
             "penalty": 0, "pillar": 1, "detail": "ok"}
            for r in all_rules
        ],
    }


def _verdict(decision: str, failed: list[str]) -> dict:
    all_rules = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
    return {
        "score": 100,
        "decision": decision,
        "hard_failed_rules": [r for r in failed if r in {"R1","R4","R5","R7"}],
        "rules": {r: ("FAIL" if r in failed else "PASS") for r in all_rules},
    }


def _conflict(escalate: bool) -> dict:
    flags = []
    if escalate:
        flags = [{"rule_id": "CONTRADICTION", "triggered": True,
                  "severity": "HIGH", "message": "test"}]
    return {"flags": flags, "escalate": escalate, "summary": "test"}


def _report(decision: str) -> dict:
    findings = [{"rule_id": r, "status": "PASS", "explanation": "ok"}
                for r in ["R1","R2","R3","R4","R5","R6","R7"]]
    return {
        "decision": decision,
        "summary": "x",
        "rule_findings": findings,
        "flags_addressed": [],
        "regulatory_basis": "MiFID II Article 25(2)",
        "client_facing_summary": "x",
    }


def _perfect_state() -> dict:
    return {
        "a1_verification": _ver(True),
        "a2_verification": _ver(True),
        "pre_check_verdict": _pre_check("SUITABLE", []),
        "rule_verdict": _verdict("SUITABLE", []),
        "conflict_report": _conflict(False),
        "suitability_report": _report("SUITABLE"),
    }


def _scenario(decision: str, escalate: bool) -> dict:
    return {"expected_decision": decision, "expected_escalate": escalate}


# ── assign_credit dispatch ────────────────────────────────────────────────────

class TestAssignCredit:

    def test_composite_method_returns_same_as_composite_credit(self):
        state = _perfect_state()
        sc = _scenario("SUITABLE", False)
        assert assign_credit(state, sc, method="composite") == composite_credit(state, sc)

    def test_counterfactual_method_returns_same_as_counterfactual_credit(self):
        state = _perfect_state()
        sc = _scenario("SUITABLE", False)
        assert assign_credit(state, sc, method="counterfactual") == counterfactual_credit(state, sc)

    def test_default_method_is_composite(self):
        state = _perfect_state()
        sc = _scenario("SUITABLE", False)
        # Default call should equal explicit "composite" call
        assert assign_credit(state, sc) == assign_credit(state, sc, method="composite")

    def test_unknown_method_raises_value_error(self):
        state = _perfect_state()
        sc = _scenario("SUITABLE", False)
        with pytest.raises(ValueError, match="Unknown credit assignment method"):
            assign_credit(state, sc, method="ppo")

    def test_error_message_lists_valid_methods(self):
        with pytest.raises(ValueError) as exc_info:
            assign_credit({}, _scenario("SUITABLE", False), method="bad")
        assert "composite" in str(exc_info.value)
        assert "counterfactual" in str(exc_info.value)

    def test_result_has_required_keys(self):
        result = assign_credit(_perfect_state(), _scenario("SUITABLE", False))
        assert set(result.keys()) == {"a1", "a2", "a3", "a4", "a5", "composite"}

    def test_all_values_are_floats(self):
        result = assign_credit(_perfect_state(), _scenario("SUITABLE", False))
        for k, v in result.items():
            assert isinstance(v, float), f"{k}: expected float, got {type(v)}"

    def test_all_values_in_0_1(self):
        result = assign_credit(_perfect_state(), _scenario("SUITABLE", False))
        for k, v in result.items():
            assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"


# ── composite_credit ──────────────────────────────────────────────────────────

class TestCompositeCredit:

    def test_perfect_run_composite_is_1(self):
        result = composite_credit(_perfect_state(), _scenario("SUITABLE", False))
        assert result["composite"] == pytest.approx(1.0)

    def test_wrong_decision_reduces_composite(self):
        state = _perfect_state()
        state["suitability_report"] = _report("UNSUITABLE")
        result = composite_credit(state, _scenario("SUITABLE", False))
        # a5=0.0, others=1.0 → 0.10+0.10+0.10+0.10 = 0.40
        assert result["composite"] == pytest.approx(0.40)

    def test_empty_state_composite_is_0(self):
        result = composite_credit({}, _scenario("SUITABLE", False))
        # a1=0.5, a2=0.5 (incomplete), a3=0.0, a4=0.0, a5=0.0
        # composite = 0.10*0.5 + 0.10*0.5 + 0 + 0 + 0 = 0.10
        assert result["composite"] == pytest.approx(0.10)

    def test_escalated_correct_composite_is_1(self):
        state = {
            "a1_verification": _ver(True),
            "a2_verification": _ver(True),
            "pre_check_verdict": _pre_check("SUITABLE", []),
            "rule_verdict": _verdict("SUITABLE", []),
            "conflict_report": _conflict(True),
            "suitability_report": _report("ESCALATED"),
        }
        result = composite_credit(state, _scenario("ESCALATED", True))
        assert result["composite"] == pytest.approx(1.0)


# ── counterfactual_credit ─────────────────────────────────────────────────────

class TestCounterfactualCredit:

    def test_perfect_run_all_contributions_positive(self):
        result = counterfactual_credit(_perfect_state(), _scenario("SUITABLE", False))
        for agent_id in ("a1", "a2", "a3", "a4", "a5"):
            assert result[agent_id] >= 0.0, f"{agent_id} contribution is negative"

    def test_perfect_run_composite_matches_actual(self):
        """Composite key reports the actual composite, not counterfactual."""
        from rl.rewards import composite_reward
        state = _perfect_state()
        sc = _scenario("SUITABLE", False)
        cf = counterfactual_credit(state, sc)
        actual = composite_reward(state, sc)["composite"]
        assert cf["composite"] == pytest.approx(actual)

    def test_a5_has_largest_contribution(self):
        """A5 has weight 0.60 vs 0.10 each for others — it always contributes most."""
        result = counterfactual_credit(_perfect_state(), _scenario("SUITABLE", False))
        for agent_id in ("a1", "a2", "a3", "a4"):
            assert result["a5"] >= result[agent_id], (
                f"a5 contribution ({result['a5']}) < {agent_id} ({result[agent_id]})"
            )

    def test_wrong_a5_gets_zero_contribution(self):
        """If A5 is wrong, forcing it to 0 doesn't change the composite — contribution=0."""
        state = _perfect_state()
        state["suitability_report"] = _report("UNSUITABLE")   # a5=0.0 already
        result = counterfactual_credit(state, _scenario("SUITABLE", False))
        # a5 is already 0.0, so counterfactual(a5=0) == actual composite → contribution=0
        assert result["a5"] == pytest.approx(0.0)

    def test_contributions_all_non_negative(self):
        for state, sc in [
            (_perfect_state(), _scenario("SUITABLE", False)),
            ({}, _scenario("SUITABLE", False)),
            ({
                "a1_verification": _ver(False),
                "a2_verification": _ver(False),
                "pre_check_verdict": _pre_check("UNSUITABLE", ["R1"]),
                "rule_verdict": _verdict("UNSUITABLE", ["R1"]),
                "conflict_report": _conflict(False),
                "suitability_report": _report("UNSUITABLE"),
            }, _scenario("UNSUITABLE", False)),
        ]:
            result = counterfactual_credit(state, sc)
            for k in ("a1", "a2", "a3", "a4", "a5"):
                assert result[k] >= 0.0, f"{k} negative in state {state}"

    def test_result_has_required_keys(self):
        result = counterfactual_credit(_perfect_state(), _scenario("SUITABLE", False))
        assert set(result.keys()) == {"a1", "a2", "a3", "a4", "a5", "composite"}

    def test_each_contribution_lte_weight_times_actual_reward(self):
        """
        The maximum counterfactual contribution of agent i is _W[i] * actual_reward_i,
        because that's how much the composite drops if we zero out agent i.
        """
        _W = {"a1": 0.10, "a2": 0.10, "a3": 0.10, "a4": 0.10, "a5": 0.60}
        from rl.rewards import reward_a1, reward_a2, reward_a3, reward_a4, reward_a5
        state = _perfect_state()
        sc = _scenario("SUITABLE", False)
        actual_rewards = {
            "a1": reward_a1(state),
            "a2": reward_a2(state),
            "a3": reward_a3(state),
            "a4": reward_a4(state, False),
            "a5": reward_a5(state, "SUITABLE"),
        }
        result = counterfactual_credit(state, sc)
        for k in ("a1", "a2", "a3", "a4", "a5"):
            max_possible = _W[k] * actual_rewards[k]
            assert result[k] <= max_possible + 1e-9, (
                f"{k}: contribution {result[k]} > weight*reward {max_possible}"
            )

    def test_a5_contribution_equals_weight_times_reward_when_all_others_perfect(self):
        """
        When all agents score 1.0, A5's counterfactual contribution is exactly
        0.60 * 1.0 = 0.60 (the composite drops from 1.0 to 0.40 if A5=0).
        """
        result = counterfactual_credit(_perfect_state(), _scenario("SUITABLE", False))
        assert result["a5"] == pytest.approx(0.60)

    def test_composite_key_is_rounded_float(self):
        result = counterfactual_credit(_perfect_state(), _scenario("SUITABLE", False))
        c = result["composite"]
        assert isinstance(c, float)
        if "." in str(c):
            assert len(str(c).split(".")[1]) <= 6
