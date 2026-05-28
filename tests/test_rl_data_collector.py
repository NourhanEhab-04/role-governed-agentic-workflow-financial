"""
tests/test_rl_data_collector.py
================================
Unit tests for rl/data_collector.py (Step 2).

All tests for the pure functions are isolated — no LLM calls, no filesystem,
no network.  The async collect_traces driver is NOT tested here because it
requires a real pipeline run; that belongs in integration tests.

Tested:
  - _build_user_message   (all 5 agents, missing keys, unknown agent)
  - _build_response       (all 5 agents, missing output, unknown key)
  - extract_traces_from_state  (count, keys, types, reward values, halted state)
"""

import json
import pytest

from rl.data_collector import (
    _build_user_message,
    _build_response,
    extract_traces_from_state,
)
from rl.rewards import composite_reward


# ── Shared fixtures ────────────────────────────────────────────────────────────

AGENT_IDS = ("a1", "a2", "a3", "a4", "a5")

REQUIRED_TRACE_KEYS = {
    "agent_id",
    "scenario_id",
    "run_number",
    "prompt_system",
    "prompt_user",
    "response",
    "reward",
    "composite",
    "all_rewards",
    "pipeline_halted",
    "timestamp",
    "architecture",
}


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
                "penalty": 0,
                "pillar": 1,
                "detail": "ok",
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
    flags = []
    if escalate:
        flags = [
            {
                "rule_id": "CONTRADICTION",
                "triggered": True,
                "severity": "HIGH",
                "message": "test",
            }
        ]
    return {"flags": flags, "escalate": escalate, "summary": "test"}


def _report(decision: str) -> dict:
    findings = [
        {"rule_id": r, "status": "PASS", "explanation": "ok"}
        for r in ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
    ]
    return {
        "decision": decision,
        "summary": "x",
        "rule_findings": findings,
        "flags_addressed": [],
        "regulatory_basis": "MiFID II Article 25(2)",
        "client_facing_summary": "x",
    }


def _full_state() -> dict:
    return {
        "client_input": "I am a cautious investor.",
        "product_input": "Government Bond Fund",
        "client_profile": {"risk_tolerance": "low", "investment_horizon": "long"},
        "product_profile": {"risk_class": 2, "complexity_tier": "non-complex"},
        "a1_verification": _ver(True),
        "a2_verification": _ver(True),
        "pre_check_verdict": _pre_check("SUITABLE", []),
        "rule_verdict": _verdict("SUITABLE", []),
        "conflict_report": _conflict(False),
        "suitability_report": _report("SUITABLE"),
    }


def _scenario(decision: str = "SUITABLE", escalate: bool = False) -> dict:
    return {"expected_decision": decision, "expected_escalate": escalate}


# ── _build_user_message ────────────────────────────────────────────────────────

class TestBuildUserMessage:

    def test_a1_returns_client_input(self):
        state = {"client_input": "test client text"}
        assert _build_user_message("a1", state) == "test client text"

    def test_a2_returns_product_input(self):
        state = {"product_input": "test product text"}
        assert _build_user_message("a2", state) == "test product text"

    def test_a1_missing_key_returns_empty_string(self):
        assert _build_user_message("a1", {}) == ""

    def test_a2_missing_key_returns_empty_string(self):
        assert _build_user_message("a2", {}) == ""

    def test_a3_returns_valid_json(self):
        state = _full_state()
        result = _build_user_message("a3", state)
        parsed = json.loads(result)
        assert "client_profile" in parsed
        assert "product_profile" in parsed

    def test_a3_contains_client_profile_data(self):
        state = _full_state()
        parsed = json.loads(_build_user_message("a3", state))
        assert parsed["client_profile"] == state["client_profile"]

    def test_a3_does_not_contain_rule_verdict(self):
        state = _full_state()
        parsed = json.loads(_build_user_message("a3", state))
        assert "rule_verdict" not in parsed

    def test_a4_returns_valid_json(self):
        state = _full_state()
        result = _build_user_message("a4", state)
        parsed = json.loads(result)
        assert "client_profile" in parsed
        assert "product_profile" in parsed
        assert "rule_verdict" in parsed

    def test_a4_does_not_contain_conflict_report(self):
        state = _full_state()
        parsed = json.loads(_build_user_message("a4", state))
        assert "conflict_report" not in parsed

    def test_a5_returns_valid_json(self):
        state = _full_state()
        result = _build_user_message("a5", state)
        parsed = json.loads(result)
        assert "client_profile" in parsed
        assert "product_profile" in parsed
        assert "rule_verdict" in parsed
        assert "conflict_report" in parsed

    def test_a5_contains_all_four_keys(self):
        state = _full_state()
        parsed = json.loads(_build_user_message("a5", state))
        assert set(parsed.keys()) == {"client_profile", "product_profile", "rule_verdict", "conflict_report"}

    def test_a3_empty_state_returns_empty_dicts(self):
        parsed = json.loads(_build_user_message("a3", {}))
        assert parsed == {"client_profile": {}, "product_profile": {}}

    def test_a4_empty_state_returns_empty_dicts(self):
        parsed = json.loads(_build_user_message("a4", {}))
        assert parsed == {"client_profile": {}, "product_profile": {}, "rule_verdict": {}}

    def test_a5_empty_state_returns_empty_dicts(self):
        parsed = json.loads(_build_user_message("a5", {}))
        assert parsed == {
            "client_profile": {},
            "product_profile": {},
            "rule_verdict": {},
            "conflict_report": {},
        }

    def test_unknown_agent_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown agent_id"):
            _build_user_message("a6", {})

    def test_unknown_agent_error_contains_id(self):
        with pytest.raises(ValueError) as exc_info:
            _build_user_message("bad", {})
        assert "bad" in str(exc_info.value)

    def test_a3_result_is_string(self):
        assert isinstance(_build_user_message("a3", _full_state()), str)

    def test_a4_result_is_string(self):
        assert isinstance(_build_user_message("a4", _full_state()), str)

    def test_a5_result_is_string(self):
        assert isinstance(_build_user_message("a5", _full_state()), str)


# ── _build_response ────────────────────────────────────────────────────────────

class TestBuildResponse:

    def test_a1_returns_client_profile_json(self):
        state = _full_state()
        result = _build_response("a1", state)
        parsed = json.loads(result)
        assert parsed == state["client_profile"]

    def test_a2_returns_product_profile_json(self):
        state = _full_state()
        result = _build_response("a2", state)
        parsed = json.loads(result)
        assert parsed == state["product_profile"]

    def test_a3_returns_rule_verdict_json(self):
        state = _full_state()
        result = _build_response("a3", state)
        parsed = json.loads(result)
        assert parsed == state["rule_verdict"]

    def test_a4_returns_conflict_report_json(self):
        state = _full_state()
        result = _build_response("a4", state)
        parsed = json.loads(result)
        assert parsed == state["conflict_report"]

    def test_a5_returns_suitability_report_json(self):
        state = _full_state()
        result = _build_response("a5", state)
        parsed = json.loads(result)
        assert parsed == state["suitability_report"]

    def test_missing_output_returns_empty_string(self):
        for agent_id in AGENT_IDS:
            assert _build_response(agent_id, {}) == ""

    def test_all_responses_are_strings(self):
        state = _full_state()
        for agent_id in AGENT_IDS:
            assert isinstance(_build_response(agent_id, state), str)

    def test_unknown_agent_returns_empty_string(self):
        # _build_response uses .get() so unknown agent just returns ""
        assert _build_response("a6", _full_state()) == ""

    def test_result_is_valid_json_when_present(self):
        state = _full_state()
        for agent_id in AGENT_IDS:
            result = _build_response(agent_id, state)
            assert result != ""
            json.loads(result)  # must not raise


# ── extract_traces_from_state ─────────────────────────────────────────────────

class TestExtractTracesFromState:

    def test_returns_exactly_5_traces(self):
        traces = extract_traces_from_state(_full_state(), _scenario(), "01_test")
        assert len(traces) == 5

    def test_each_trace_has_all_required_keys(self):
        traces = extract_traces_from_state(_full_state(), _scenario(), "01_test")
        for trace in traces:
            assert set(trace.keys()) >= REQUIRED_TRACE_KEYS, (
                f"trace for {trace.get('agent_id')} is missing keys"
            )

    def test_agent_ids_are_a1_through_a5(self):
        traces = extract_traces_from_state(_full_state(), _scenario(), "01_test")
        ids = [t["agent_id"] for t in traces]
        assert ids == ["a1", "a2", "a3", "a4", "a5"]

    def test_scenario_id_propagated(self):
        traces = extract_traces_from_state(_full_state(), _scenario(), "scenario_42")
        for trace in traces:
            assert trace["scenario_id"] == "scenario_42"

    def test_run_number_default_is_1(self):
        traces = extract_traces_from_state(_full_state(), _scenario(), "test")
        for trace in traces:
            assert trace["run_number"] == 1

    def test_run_number_custom(self):
        traces = extract_traces_from_state(_full_state(), _scenario(), "test", run_number=3)
        for trace in traces:
            assert trace["run_number"] == 3

    def test_architecture_is_pipeline(self):
        traces = extract_traces_from_state(_full_state(), _scenario(), "test")
        for trace in traces:
            assert trace["architecture"] == "pipeline"

    def test_timestamp_is_iso_string(self):
        import re
        traces = extract_traces_from_state(_full_state(), _scenario(), "test")
        iso_pattern = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
        for trace in traces:
            assert isinstance(trace["timestamp"], str)
            assert iso_pattern.search(trace["timestamp"])

    def test_reward_values_match_composite_reward(self):
        state = _full_state()
        sc = _scenario()
        expected = composite_reward(state, sc)
        traces = extract_traces_from_state(state, sc, "test")
        for trace in traces:
            aid = trace["agent_id"]
            assert trace["reward"] == pytest.approx(expected[aid]), (
                f"{aid} reward mismatch"
            )

    def test_composite_matches_composite_reward(self):
        state = _full_state()
        sc = _scenario()
        expected_composite = composite_reward(state, sc)["composite"]
        traces = extract_traces_from_state(state, sc, "test")
        for trace in traces:
            assert trace["composite"] == pytest.approx(expected_composite)

    def test_all_rewards_dict_has_5_agents(self):
        traces = extract_traces_from_state(_full_state(), _scenario(), "test")
        for trace in traces:
            assert set(trace["all_rewards"].keys()) == {"a1", "a2", "a3", "a4", "a5"}

    def test_all_rewards_does_not_contain_composite(self):
        traces = extract_traces_from_state(_full_state(), _scenario(), "test")
        for trace in traces:
            assert "composite" not in trace["all_rewards"]

    def test_reward_is_float(self):
        traces = extract_traces_from_state(_full_state(), _scenario(), "test")
        for trace in traces:
            assert isinstance(trace["reward"], float), (
                f"{trace['agent_id']} reward is {type(trace['reward'])}"
            )

    def test_composite_is_float(self):
        traces = extract_traces_from_state(_full_state(), _scenario(), "test")
        for trace in traces:
            assert isinstance(trace["composite"], float)

    def test_pipeline_halted_false_in_normal_run(self):
        traces = extract_traces_from_state(_full_state(), _scenario(), "test")
        for trace in traces:
            assert trace["pipeline_halted"] is False

    def test_pipeline_halted_true_when_halt_key_set(self):
        state = {"halt": True}
        traces = extract_traces_from_state(state, _scenario(), "test")
        for trace in traces:
            assert trace["pipeline_halted"] is True

    def test_empty_state_produces_5_traces(self):
        traces = extract_traces_from_state({}, _scenario(), "empty")
        assert len(traces) == 5

    def test_empty_state_responses_are_empty_strings(self):
        traces = extract_traces_from_state({}, _scenario(), "empty")
        for trace in traces:
            assert trace["response"] == ""

    def test_prompt_system_is_non_empty_string(self):
        traces = extract_traces_from_state(_full_state(), _scenario(), "test")
        for trace in traces:
            assert isinstance(trace["prompt_system"], str)
            assert len(trace["prompt_system"]) > 0

    def test_prompt_user_a1_is_client_input(self):
        state = _full_state()
        traces = extract_traces_from_state(state, _scenario(), "test")
        a1_trace = next(t for t in traces if t["agent_id"] == "a1")
        assert a1_trace["prompt_user"] == state["client_input"]

    def test_prompt_user_a2_is_product_input(self):
        state = _full_state()
        traces = extract_traces_from_state(state, _scenario(), "test")
        a2_trace = next(t for t in traces if t["agent_id"] == "a2")
        assert a2_trace["prompt_user"] == state["product_input"]

    def test_prompt_user_a3_contains_profiles(self):
        traces = extract_traces_from_state(_full_state(), _scenario(), "test")
        a3_trace = next(t for t in traces if t["agent_id"] == "a3")
        parsed = json.loads(a3_trace["prompt_user"])
        assert "client_profile" in parsed
        assert "product_profile" in parsed

    def test_prompt_user_a5_contains_all_inputs(self):
        traces = extract_traces_from_state(_full_state(), _scenario(), "test")
        a5_trace = next(t for t in traces if t["agent_id"] == "a5")
        parsed = json.loads(a5_trace["prompt_user"])
        assert "client_profile" in parsed
        assert "product_profile" in parsed
        assert "rule_verdict" in parsed
        assert "conflict_report" in parsed

    def test_response_a5_is_suitability_report(self):
        state = _full_state()
        traces = extract_traces_from_state(state, _scenario(), "test")
        a5_trace = next(t for t in traces if t["agent_id"] == "a5")
        parsed = json.loads(a5_trace["response"])
        assert parsed["decision"] == "SUITABLE"

    def test_all_rewards_values_are_floats(self):
        traces = extract_traces_from_state(_full_state(), _scenario(), "test")
        for trace in traces:
            for k, v in trace["all_rewards"].items():
                assert isinstance(v, float), f"{k} in all_rewards is not float"

    def test_all_rewards_in_0_1(self):
        traces = extract_traces_from_state(_full_state(), _scenario(), "test")
        for trace in traces:
            for k, v in trace["all_rewards"].items():
                assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"

    def test_reward_in_0_1(self):
        traces = extract_traces_from_state(_full_state(), _scenario(), "test")
        for trace in traces:
            assert 0.0 <= trace["reward"] <= 1.0

    def test_composite_in_0_1(self):
        traces = extract_traces_from_state(_full_state(), _scenario(), "test")
        for trace in traces:
            assert 0.0 <= trace["composite"] <= 1.0

    def test_perfect_run_all_rewards_1(self):
        traces = extract_traces_from_state(_full_state(), _scenario("SUITABLE", False), "test")
        for trace in traces:
            assert trace["reward"] == pytest.approx(1.0), (
                f"{trace['agent_id']} should be 1.0 in perfect run"
            )

    def test_wrong_decision_a5_reward_is_0(self):
        state = _full_state()
        state["suitability_report"] = _report("UNSUITABLE")
        traces = extract_traces_from_state(state, _scenario("SUITABLE", False), "test")
        a5_trace = next(t for t in traces if t["agent_id"] == "a5")
        assert a5_trace["reward"] == pytest.approx(0.0)

    def test_traces_are_independent_objects(self):
        traces = extract_traces_from_state(_full_state(), _scenario(), "test")
        for i, t1 in enumerate(traces):
            for j, t2 in enumerate(traces):
                if i != j:
                    assert t1 is not t2

    def test_all_same_composite_in_one_run(self):
        traces = extract_traces_from_state(_full_state(), _scenario(), "test")
        composites = {t["composite"] for t in traces}
        assert len(composites) == 1, "All traces in same run should share composite"
