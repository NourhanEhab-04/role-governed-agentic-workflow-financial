"""
tests/test_rl_evaluator.py
===========================
Unit tests for evaluation/evaluator_rl.py (Step 7).

All tests are pure: no LLM, no network, no Ollama.
`run_rl_evaluation` requires a live pipeline and is not tested here.

Tested:
  - extract_rl_result         (required keys, values, RL fields)
  - aggregate_agent_rewards   (stats correctness, edge cases)
  - compute_rl_metrics        (extends standard metrics, RL keys present)
"""

import pytest

from evaluation.evaluator_rl import (
    aggregate_agent_rewards,
    compute_rl_metrics,
    extract_rl_result,
)


# ── Shared fixtures ────────────────────────────────────────────────────────────

def _ver(passed: bool) -> dict:
    return {"passed": passed, "confidence": 0.95, "issues": [], "field_checks": {}}


def _pre_check(decision: str, failed: list[str]) -> dict:
    all_rules = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
    return {
        "decision": decision,
        "rules": [
            {
                "rule": r,
                "pass": r not in failed,
                "hard_fail": r in {"R1", "R4", "R5", "R7"},
                "penalty": 0, "pillar": 1, "detail": "ok",
            }
            for r in all_rules
        ],
    }


def _verdict(decision: str, failed: list[str]) -> dict:
    all_rules = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
    return {
        "score": 100,
        "decision": decision,
        "hard_failed_rules": [r for r in failed if r in {"R1", "R4", "R5", "R7"}],
        "rules": {r: ("FAIL" if r in failed else "PASS") for r in all_rules},
    }


def _conflict(escalate: bool) -> dict:
    flags = [{"rule_id": "C", "triggered": True, "severity": "HIGH", "message": "x"}] if escalate else []
    return {"flags": flags, "escalate": escalate, "summary": "test"}


def _report(decision: str) -> dict:
    findings = [{"rule_id": r, "status": "PASS", "explanation": "ok"}
                for r in ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]]
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


def _scenario(decision: str = "SUITABLE", escalate: bool = False) -> dict:
    return {
        "expected_decision": decision,
        "expected_escalate": escalate,
        "expected_rules_failed": [],
    }


STANDARD_RESULT_KEYS = {
    "scenario_id", "output_decision", "expected_decision",
    "output_escalated", "expected_escalate",
    "output_failed_rules", "expected_rules_failed",
    "output_rules", "a1_verified", "a2_verified",
    "a1_corrected", "a2_corrected", "halted", "halt_reason",
    "architecture",
}
RL_EXTRA_KEYS = {"rewards", "credit"}


# ── extract_rl_result ─────────────────────────────────────────────────────────

class TestExtractRlResult:

    def test_has_all_standard_keys(self):
        result = extract_rl_result(_perfect_state(), _scenario(), "01_test")
        assert STANDARD_RESULT_KEYS.issubset(result.keys())

    def test_has_rl_extra_keys(self):
        result = extract_rl_result(_perfect_state(), _scenario(), "01_test")
        for k in RL_EXTRA_KEYS:
            assert k in result, f"Missing key: {k}"

    def test_architecture_is_pipeline_rl(self):
        result = extract_rl_result(_perfect_state(), _scenario(), "01_test")
        assert result["architecture"] == "pipeline_rl"

    def test_scenario_id_propagated(self):
        result = extract_rl_result(_perfect_state(), _scenario(), "scenario_07")
        assert result["scenario_id"] == "scenario_07"

    def test_output_decision_from_suitability_report(self):
        result = extract_rl_result(_perfect_state(), _scenario(), "test")
        assert result["output_decision"] == "SUITABLE"

    def test_expected_decision_from_scenario(self):
        result = extract_rl_result(_perfect_state(), _scenario("UNSUITABLE"), "test")
        assert result["expected_decision"] == "UNSUITABLE"

    def test_halted_false_in_normal_run(self):
        result = extract_rl_result(_perfect_state(), _scenario(), "test")
        assert result["halted"] is False

    def test_halted_true_when_halt_key_set(self):
        state = {"halt": True}
        result = extract_rl_result(state, _scenario(), "test")
        assert result["halted"] is True

    def test_rewards_has_5_agents_and_composite(self):
        result = extract_rl_result(_perfect_state(), _scenario(), "test")
        rewards = result["rewards"]
        assert set(rewards.keys()) == {"a1", "a2", "a3", "a4", "a5", "composite"}

    def test_credit_has_5_agents_and_composite(self):
        result = extract_rl_result(_perfect_state(), _scenario(), "test")
        credit = result["credit"]
        assert set(credit.keys()) == {"a1", "a2", "a3", "a4", "a5", "composite"}

    def test_perfect_state_composite_reward_is_1(self):
        result = extract_rl_result(_perfect_state(), _scenario(), "test")
        assert result["rewards"]["composite"] == pytest.approx(1.0)

    def test_all_rewards_are_floats(self):
        result = extract_rl_result(_perfect_state(), _scenario(), "test")
        for k, v in result["rewards"].items():
            assert isinstance(v, float), f"rewards[{k}] is not float"

    def test_all_credit_values_are_floats(self):
        result = extract_rl_result(_perfect_state(), _scenario(), "test")
        for k, v in result["credit"].items():
            assert isinstance(v, float), f"credit[{k}] is not float"

    def test_a1_verified_true_when_verification_present(self):
        result = extract_rl_result(_perfect_state(), _scenario(), "test")
        assert result["a1_verified"] is True

    def test_a1_verified_false_when_verification_absent(self):
        state = {k: v for k, v in _perfect_state().items() if k != "a1_verification"}
        result = extract_rl_result(state, _scenario(), "test")
        assert result["a1_verified"] is False

    def test_output_failed_rules_empty_on_passing_verdict(self):
        result = extract_rl_result(_perfect_state(), _scenario(), "test")
        assert result["output_failed_rules"] == []

    def test_output_failed_rules_populated_on_failing_verdict(self):
        state = _perfect_state()
        state["rule_verdict"] = _verdict("UNSUITABLE", ["R1"])
        result = extract_rl_result(state, _scenario(), "test")
        assert "R1" in result["output_failed_rules"]

    def test_empty_state_has_unknown_decision(self):
        result = extract_rl_result({}, _scenario(), "test")
        assert result["output_decision"] == "UNKNOWN"

    def test_counterfactual_credit_method(self):
        result = extract_rl_result(
            _perfect_state(), _scenario(), "test", credit_method="counterfactual"
        )
        assert "a5" in result["credit"]

    def test_expected_escalate_propagated(self):
        result = extract_rl_result(_perfect_state(), _scenario(escalate=True), "test")
        assert result["expected_escalate"] is True

    def test_expected_rules_failed_propagated(self):
        sc = _scenario()
        sc["expected_rules_failed"] = ["R2", "R3"]
        result = extract_rl_result(_perfect_state(), sc, "test")
        assert set(result["expected_rules_failed"]) == {"R2", "R3"}


# ── aggregate_agent_rewards ───────────────────────────────────────────────────

def _make_rl_result(rewards: dict) -> dict:
    """Create a minimal RL result dict with the given rewards."""
    return {
        "scenario_id": "test",
        "output_decision": "SUITABLE",
        "expected_decision": "SUITABLE",
        "output_escalated": False,
        "expected_escalate": False,
        "output_failed_rules": [],
        "expected_rules_failed": [],
        "output_rules": {},
        "a1_verified": True, "a2_verified": True,
        "a1_corrected": False, "a2_corrected": False,
        "halted": False, "halt_reason": None,
        "architecture": "pipeline_rl",
        "rewards": rewards,
        "credit":  {k: v for k, v in rewards.items()},
    }


class TestAggregateAgentRewards:

    def test_empty_list_returns_empty_dict(self):
        assert aggregate_agent_rewards([]) == {}

    def test_keys_match_reward_keys(self):
        rewards = {"a1": 1.0, "a2": 0.8, "a3": 0.9, "a4": 1.0, "a5": 1.0, "composite": 0.98}
        stats = aggregate_agent_rewards([_make_rl_result(rewards)])
        assert set(stats.keys()) == set(rewards.keys())

    def test_single_result_mean_equals_value(self):
        rewards = {"a1": 0.75, "composite": 0.75}
        stats = aggregate_agent_rewards([_make_rl_result(rewards)])
        assert stats["a1"]["mean"] == pytest.approx(0.75)
        assert stats["a1"]["min"]  == pytest.approx(0.75)
        assert stats["a1"]["max"]  == pytest.approx(0.75)

    def test_multiple_results_mean_is_correct(self):
        r1 = _make_rl_result({"a1": 0.5, "composite": 0.5})
        r2 = _make_rl_result({"a1": 1.0, "composite": 1.0})
        stats = aggregate_agent_rewards([r1, r2])
        assert stats["a1"]["mean"] == pytest.approx(0.75)

    def test_min_max_correct(self):
        r1 = _make_rl_result({"a5": 0.2, "composite": 0.2})
        r2 = _make_rl_result({"a5": 0.8, "composite": 0.8})
        r3 = _make_rl_result({"a5": 0.5, "composite": 0.5})
        stats = aggregate_agent_rewards([r1, r2, r3])
        assert stats["a5"]["min"] == pytest.approx(0.2)
        assert stats["a5"]["max"] == pytest.approx(0.8)

    def test_count_is_number_of_results(self):
        results = [_make_rl_result({"a1": 1.0, "composite": 1.0})] * 5
        stats = aggregate_agent_rewards(results)
        assert stats["a1"]["count"] == 5

    def test_std_is_zero_for_identical_rewards(self):
        results = [_make_rl_result({"a1": 0.9, "composite": 0.9})] * 3
        stats = aggregate_agent_rewards(results)
        assert stats["a1"]["std"] == pytest.approx(0.0)

    def test_std_is_nonzero_for_varied_rewards(self):
        r1 = _make_rl_result({"a1": 0.0, "composite": 0.0})
        r2 = _make_rl_result({"a1": 1.0, "composite": 1.0})
        stats = aggregate_agent_rewards([r1, r2])
        assert stats["a1"]["std"] > 0.0

    def test_result_missing_rewards_key_is_skipped(self):
        results = [{"output_decision": "SUITABLE"}]
        stats = aggregate_agent_rewards(results)
        assert stats == {}

    def test_all_stat_values_are_floats(self):
        rewards = {"a1": 1.0, "composite": 1.0}
        stats = aggregate_agent_rewards([_make_rl_result(rewards)])
        for k, v in stats["a1"].items():
            if k != "count":
                assert isinstance(v, float), f"stats['a1'][{k}] is not float"


# ── compute_rl_metrics ────────────────────────────────────────────────────────

def _five_rl_results(decision: str = "SUITABLE", composite_reward: float = 1.0) -> list[dict]:
    rewards = {"a1": 1.0, "a2": 1.0, "a3": 1.0, "a4": 1.0, "a5": 1.0, "composite": composite_reward}
    return [
        _make_rl_result(rewards) | {
            "scenario_id": f"sc_{i}",
            "output_decision": decision,
            "expected_decision": decision,
        }
        for i in range(5)
    ]


class TestComputeRlMetrics:

    def test_has_standard_metric_keys(self):
        results = _five_rl_results()
        metrics = compute_rl_metrics(results)
        assert "M1_decision_accuracy" in metrics

    def test_has_agent_rewards_key(self):
        metrics = compute_rl_metrics(_five_rl_results())
        assert "agent_rewards" in metrics

    def test_has_mean_composite_key(self):
        metrics = compute_rl_metrics(_five_rl_results(composite_reward=1.0))
        assert "mean_composite" in metrics
        assert metrics["mean_composite"] == pytest.approx(1.0)

    def test_has_min_max_composite(self):
        metrics = compute_rl_metrics(_five_rl_results(composite_reward=0.75))
        assert "min_composite" in metrics
        assert "max_composite" in metrics
        assert metrics["min_composite"] == pytest.approx(0.75)
        assert metrics["max_composite"] == pytest.approx(0.75)

    def test_n_rl_results_count(self):
        results = _five_rl_results()
        metrics = compute_rl_metrics(results)
        assert metrics["n_rl_results"] == 5

    def test_m1_accuracy_correct(self):
        results = _five_rl_results(decision="SUITABLE")
        metrics = compute_rl_metrics(results)
        assert metrics["M1_decision_accuracy"] == pytest.approx(1.0)

    def test_agent_rewards_has_all_agents(self):
        metrics = compute_rl_metrics(_five_rl_results())
        reward_stats = metrics["agent_rewards"]
        for aid in ("a1", "a2", "a3", "a4", "a5", "composite"):
            assert aid in reward_stats

    def test_empty_results_returns_safe_defaults(self):
        metrics = compute_rl_metrics([])
        assert metrics["mean_composite"] == pytest.approx(0.0)
        assert metrics["n_rl_results"] == 0

    def test_mixed_rewards_mean_correct(self):
        r1 = _make_rl_result({"a1": 0.0, "a2": 1.0, "a3": 1.0, "a4": 1.0, "a5": 1.0, "composite": 0.4})
        r2 = _make_rl_result({"a1": 1.0, "a2": 1.0, "a3": 1.0, "a4": 1.0, "a5": 1.0, "composite": 1.0})
        r1["output_decision"] = "SUITABLE"; r1["expected_decision"] = "SUITABLE"
        r2["output_decision"] = "SUITABLE"; r2["expected_decision"] = "SUITABLE"
        metrics = compute_rl_metrics([r1, r2])
        assert metrics["mean_composite"] == pytest.approx(0.7)

    def test_architecture_name_in_result(self):
        metrics = compute_rl_metrics(_five_rl_results(), architecture_name="my_arch")
        assert metrics.get("architecture") == "my_arch"

    def test_results_without_rewards_key_handled(self):
        results = [{"output_decision": "SUITABLE", "expected_decision": "SUITABLE"}]
        metrics = compute_rl_metrics(results)
        assert metrics["n_rl_results"] == 0
