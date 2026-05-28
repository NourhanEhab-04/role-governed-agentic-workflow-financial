"""
tests/test_rl_grpo_config.py
==============================
Unit tests for rl/grpo_config.py (Step 5).

Pure: no LLM, no network, no TRL, no Ollama.

Tested:
  - GRPORunConfig  (defaults, validation, to_dict / from_dict, effective_batch_size)
  - DEFAULT_GRPO_CONFIG
  - parse_json_safe         (valid JSON, invalid, edge cases)
  - clip_reward             (clamping)
  - reward_completion_a1    (all keys, partial, invalid JSON)
  - reward_completion_a2    (all keys, partial, invalid JSON)
  - reward_completion_a3    (RLVR: runs real rule engine, missing profiles fallback)
  - reward_completion_a4    (escalation accuracy)
  - reward_completion_a5    (decision accuracy)
  - build_reward_fn         (dispatch, batched behaviour, kwargs forwarding)
  - VALID_AGENT_IDS
"""

import json
import pytest

from rl.grpo_config import (
    DEFAULT_GRPO_CONFIG,
    VALID_AGENT_IDS,
    GRPORunConfig,
    build_reward_fn,
    clip_reward,
    parse_json_safe,
    reward_completion_a1,
    reward_completion_a2,
    reward_completion_a3,
    reward_completion_a4,
    reward_completion_a5,
)


# ── GRPORunConfig ─────────────────────────────────────────────────────────────

class TestGRPORunConfig:

    def test_default_construction(self):
        cfg = GRPORunConfig()
        assert cfg.num_generations == 8
        assert cfg.learning_rate == pytest.approx(2e-6)
        assert cfg.max_new_tokens == 512

    def test_num_generations_must_be_ge_2(self):
        with pytest.raises(ValueError, match="num_generations"):
            GRPORunConfig(num_generations=1)

    def test_num_generations_exactly_2_ok(self):
        cfg = GRPORunConfig(num_generations=2)
        assert cfg.num_generations == 2

    def test_learning_rate_must_be_positive(self):
        with pytest.raises(ValueError, match="learning_rate"):
            GRPORunConfig(learning_rate=0.0)

    def test_learning_rate_must_be_below_1(self):
        with pytest.raises(ValueError, match="learning_rate"):
            GRPORunConfig(learning_rate=1.0)

    def test_max_new_tokens_minimum_32(self):
        with pytest.raises(ValueError, match="max_new_tokens"):
            GRPORunConfig(max_new_tokens=31)

    def test_max_new_tokens_exactly_32_ok(self):
        cfg = GRPORunConfig(max_new_tokens=32)
        assert cfg.max_new_tokens == 32

    def test_epochs_must_be_positive(self):
        with pytest.raises(ValueError, match="epochs"):
            GRPORunConfig(epochs=0)

    def test_warmup_ratio_range(self):
        with pytest.raises(ValueError, match="warmup_ratio"):
            GRPORunConfig(warmup_ratio=1.1)

    def test_kl_coeff_non_negative(self):
        with pytest.raises(ValueError, match="kl_coeff"):
            GRPORunConfig(kl_coeff=-0.01)

    def test_kl_coeff_zero_allowed(self):
        cfg = GRPORunConfig(kl_coeff=0.0)
        assert cfg.kl_coeff == 0.0

    def test_reward_clip_min_cannot_be_negative(self):
        with pytest.raises(ValueError, match="reward clip"):
            GRPORunConfig(reward_clip_min=-0.1)

    def test_reward_clip_max_cannot_exceed_1(self):
        with pytest.raises(ValueError, match="reward clip"):
            GRPORunConfig(reward_clip_max=1.1)

    def test_reward_clip_min_cannot_exceed_max(self):
        with pytest.raises(ValueError, match="reward_clip_min"):
            GRPORunConfig(reward_clip_min=0.8, reward_clip_max=0.5)

    def test_temperature_must_be_positive(self):
        with pytest.raises(ValueError, match="temperature"):
            GRPORunConfig(temperature=0.0)

    def test_temperature_max_2(self):
        with pytest.raises(ValueError, match="temperature"):
            GRPORunConfig(temperature=2.1)

    def test_effective_batch_size(self):
        cfg = GRPORunConfig(per_device_batch_size=4, gradient_accumulation_steps=8)
        assert cfg.effective_batch_size == 32

    def test_to_dict_has_all_keys(self):
        d = GRPORunConfig().to_dict()
        assert "num_generations" in d
        assert "learning_rate" in d
        assert "effective_batch_size" in d

    def test_from_dict_roundtrip(self):
        cfg = GRPORunConfig(num_generations=4, learning_rate=1e-5)
        restored = GRPORunConfig.from_dict(cfg.to_dict())
        assert restored.num_generations == cfg.num_generations
        assert restored.learning_rate == pytest.approx(cfg.learning_rate)

    def test_from_dict_ignores_unknown_keys(self):
        d = GRPORunConfig().to_dict()
        d["nonexistent_key"] = "should_be_ignored"
        cfg = GRPORunConfig.from_dict(d)
        assert cfg.num_generations == 8

    def test_max_prompt_length_minimum_64(self):
        with pytest.raises(ValueError, match="max_prompt_length"):
            GRPORunConfig(max_prompt_length=63)

    def test_per_device_batch_size_minimum_1(self):
        with pytest.raises(ValueError, match="per_device_batch_size"):
            GRPORunConfig(per_device_batch_size=0)

    def test_gradient_accumulation_steps_minimum_1(self):
        with pytest.raises(ValueError, match="gradient_accumulation_steps"):
            GRPORunConfig(gradient_accumulation_steps=0)


class TestDefaultGRPOConfig:

    def test_is_grpo_run_config(self):
        assert isinstance(DEFAULT_GRPO_CONFIG, GRPORunConfig)

    def test_num_generations_is_8(self):
        assert DEFAULT_GRPO_CONFIG.num_generations == 8

    def test_reward_clip_is_0_to_1(self):
        assert DEFAULT_GRPO_CONFIG.reward_clip_min == 0.0
        assert DEFAULT_GRPO_CONFIG.reward_clip_max == 1.0


# ── parse_json_safe ───────────────────────────────────────────────────────────

class TestParseJsonSafe:

    def test_valid_object(self):
        result = parse_json_safe('{"key": "value"}')
        assert result == {"key": "value"}

    def test_returns_none_for_invalid(self):
        assert parse_json_safe("not json") is None

    def test_returns_none_for_empty_string(self):
        assert parse_json_safe("") is None

    def test_returns_none_for_json_array(self):
        assert parse_json_safe('["a", "b"]') is None

    def test_returns_none_for_json_string(self):
        assert parse_json_safe('"just a string"') is None

    def test_returns_none_for_none_input(self):
        assert parse_json_safe(None) is None  # type: ignore

    def test_whitespace_stripped(self):
        result = parse_json_safe('  {"a": 1}  ')
        assert result == {"a": 1}

    def test_nested_json(self):
        result = parse_json_safe('{"outer": {"inner": 42}}')
        assert result["outer"]["inner"] == 42

    def test_returns_none_for_json_number(self):
        assert parse_json_safe("42") is None


# ── clip_reward ───────────────────────────────────────────────────────────────

class TestClipReward:

    def test_value_in_range_unchanged(self):
        assert clip_reward(0.5) == pytest.approx(0.5)

    def test_clips_below_min(self):
        assert clip_reward(-0.1) == pytest.approx(0.0)

    def test_clips_above_max(self):
        assert clip_reward(1.5) == pytest.approx(1.0)

    def test_boundary_0_kept(self):
        assert clip_reward(0.0) == pytest.approx(0.0)

    def test_boundary_1_kept(self):
        assert clip_reward(1.0) == pytest.approx(1.0)

    def test_custom_min_max(self):
        assert clip_reward(0.3, min_r=0.5, max_r=0.8) == pytest.approx(0.5)
        assert clip_reward(0.9, min_r=0.5, max_r=0.8) == pytest.approx(0.8)


# ── reward_completion_a1 ──────────────────────────────────────────────────────

_A1_ALL_KEYS = {
    "financial_knowledge": "basic",
    "risk_tolerance_score": 5,
    "investment_horizon": 5,
    "liquid_assets": 10000.0,
    "income": 50000.0,
    "investment_amount": 5000.0,
    "can_afford_total_loss": False,
    "financial_vulnerability": "LOW",
}


class TestRewardCompletionA1:

    def test_all_keys_gives_1(self):
        assert reward_completion_a1(json.dumps(_A1_ALL_KEYS)) == pytest.approx(1.0)

    def test_invalid_json_gives_0(self):
        assert reward_completion_a1("not json") == pytest.approx(0.0)

    def test_empty_dict_gives_0(self):
        assert reward_completion_a1("{}") == pytest.approx(0.0)

    def test_partial_keys_gives_between_0_and_1(self):
        partial = {"financial_knowledge": "basic", "risk_tolerance_score": 5}
        r = reward_completion_a1(json.dumps(partial))
        assert 0.0 < r < 1.0

    def test_returns_float(self):
        assert isinstance(reward_completion_a1(json.dumps(_A1_ALL_KEYS)), float)

    def test_extra_kwargs_ignored(self):
        r = reward_completion_a1(json.dumps(_A1_ALL_KEYS), extra_kwarg="ignored")
        assert r == pytest.approx(1.0)


# ── reward_completion_a2 ──────────────────────────────────────────────────────

_A2_ALL_KEYS = {
    "product_name": "Bond Fund",
    "risk_class": 2,
    "complexity_tier": "NON-COMPLEX",
    "requires_knowledge_level": "basic",
    "minimum_horizon": 3,
    "potential_loss": "partial",
    "leverage": False,
}


class TestRewardCompletionA2:

    def test_all_keys_gives_1(self):
        assert reward_completion_a2(json.dumps(_A2_ALL_KEYS)) == pytest.approx(1.0)

    def test_invalid_json_gives_0(self):
        assert reward_completion_a2("not json") == pytest.approx(0.0)

    def test_empty_dict_gives_0(self):
        assert reward_completion_a2("{}") == pytest.approx(0.0)

    def test_partial_keys_between_0_and_1(self):
        partial = {"product_name": "X", "risk_class": 3}
        r = reward_completion_a2(json.dumps(partial))
        assert 0.0 < r < 1.0

    def test_returns_float(self):
        assert isinstance(reward_completion_a2(json.dumps(_A2_ALL_KEYS)), float)


# ── reward_completion_a3 ──────────────────────────────────────────────────────

_CP = {
    "financial_knowledge": "basic",
    "risk_tolerance_score": 3,
    "investment_horizon": 5,
    "liquid_assets": 20000.0,
    "income": 40000.0,
    "investment_amount": 3000.0,
    "can_afford_total_loss": True,
    "financial_vulnerability": "LOW",
}

_PP = {
    "product_name": "Government Bond Fund",
    "risk_class": 2,
    "complexity_tier": "NON-COMPLEX",
    "requires_knowledge_level": "basic",
    "minimum_horizon": 3,
    "potential_loss": "partial",
    "leverage": False,
}


class TestRewardCompletionA3:

    def test_invalid_json_gives_0(self):
        assert reward_completion_a3("not json") == pytest.approx(0.0)

    def test_no_profiles_with_structure_gives_0_5(self):
        completion = json.dumps({"decision": "SUITABLE", "rules": {"R1": "PASS"}})
        r = reward_completion_a3(completion)
        assert r == pytest.approx(0.5)

    def test_no_profiles_missing_keys_gives_0(self):
        r = reward_completion_a3("{}")
        assert r == pytest.approx(0.0)

    def test_correct_verdict_gives_high_reward(self):
        from rule_engine.rule_engine import evaluate_suitability
        gt = evaluate_suitability(_CP, _PP)
        completion = json.dumps({
            "decision": gt["decision"],
            "hard_failed_rules": gt.get("hard_failed_rules", []),
            "rules": {},
            "score": 100,
        })
        r = reward_completion_a3(completion, client_profile=_CP, product_profile=_PP)
        assert r == pytest.approx(1.0)

    def test_wrong_decision_reduces_reward(self):
        from rule_engine.rule_engine import evaluate_suitability
        gt = evaluate_suitability(_CP, _PP)
        wrong_decision = "UNSUITABLE" if gt["decision"] != "UNSUITABLE" else "SUITABLE"
        completion = json.dumps({
            "decision": wrong_decision,
            "hard_failed_rules": gt.get("hard_failed_rules", []),
            "rules": {},
            "score": 0,
        })
        r = reward_completion_a3(completion, client_profile=_CP, product_profile=_PP)
        assert r < 1.0

    def test_returns_float_in_0_1(self):
        from rule_engine.rule_engine import evaluate_suitability
        gt = evaluate_suitability(_CP, _PP)
        completion = json.dumps({"decision": gt["decision"], "hard_failed_rules": [], "rules": {}})
        r = reward_completion_a3(completion, client_profile=_CP, product_profile=_PP)
        assert isinstance(r, float)
        assert 0.0 <= r <= 1.0

    def test_no_profiles_no_structure_gives_0(self):
        r = reward_completion_a3("null")
        assert r == pytest.approx(0.0)


# ── reward_completion_a4 ──────────────────────────────────────────────────────

class TestRewardCompletionA4:

    def test_correct_escalate_false_with_flags_gives_1(self):
        completion = json.dumps({"escalate": False, "flags": [], "summary": "ok"})
        assert reward_completion_a4(completion, expected_escalate=False) == pytest.approx(1.0)

    def test_correct_escalate_true_with_flags_gives_1(self):
        completion = json.dumps({
            "escalate": True,
            "flags": [{"rule_id": "X", "triggered": True}],
            "summary": "escalate",
        })
        assert reward_completion_a4(completion, expected_escalate=True) == pytest.approx(1.0)

    def test_wrong_escalation_gives_0(self):
        completion = json.dumps({"escalate": True, "flags": [], "summary": "x"})
        assert reward_completion_a4(completion, expected_escalate=False) == pytest.approx(0.0)

    def test_correct_escalate_missing_flags_gives_0_5(self):
        completion = json.dumps({"escalate": False, "summary": "ok"})
        assert reward_completion_a4(completion, expected_escalate=False) == pytest.approx(0.5)

    def test_invalid_json_gives_0(self):
        assert reward_completion_a4("bad") == pytest.approx(0.0)

    def test_missing_escalate_key_gives_0(self):
        assert reward_completion_a4("{}") == pytest.approx(0.0)

    def test_default_expected_escalate_is_false(self):
        completion = json.dumps({"escalate": False, "flags": [], "summary": "ok"})
        assert reward_completion_a4(completion) == pytest.approx(1.0)

    def test_returns_float(self):
        r = reward_completion_a4(json.dumps({"escalate": False, "flags": []}))
        assert isinstance(r, float)


# ── reward_completion_a5 ──────────────────────────────────────────────────────

class TestRewardCompletionA5:

    def test_correct_decision_with_summary_gives_1(self):
        completion = json.dumps({"decision": "SUITABLE", "summary": "Client is suitable."})
        assert reward_completion_a5(completion, expected_decision="SUITABLE") == pytest.approx(1.0)

    def test_wrong_decision_gives_0(self):
        completion = json.dumps({"decision": "UNSUITABLE", "summary": "x"})
        assert reward_completion_a5(completion, expected_decision="SUITABLE") == pytest.approx(0.0)

    def test_correct_decision_no_summary_gives_0_5(self):
        completion = json.dumps({"decision": "SUITABLE"})
        assert reward_completion_a5(completion, expected_decision="SUITABLE") == pytest.approx(0.5)

    def test_case_insensitive_decision(self):
        completion = json.dumps({"decision": "suitable", "summary": "x"})
        assert reward_completion_a5(completion, expected_decision="SUITABLE") == pytest.approx(1.0)

    def test_invalid_json_gives_0(self):
        assert reward_completion_a5("not json", expected_decision="SUITABLE") == pytest.approx(0.0)

    def test_escalated_decision(self):
        completion = json.dumps({"decision": "ESCALATED", "summary": "needs review"})
        assert reward_completion_a5(completion, expected_decision="ESCALATED") == pytest.approx(1.0)

    def test_default_expected_empty_string(self):
        completion = json.dumps({"decision": "", "summary": "x"})
        assert reward_completion_a5(completion) == pytest.approx(1.0)

    def test_returns_float(self):
        r = reward_completion_a5(json.dumps({"decision": "SUITABLE", "summary": "x"}), expected_decision="SUITABLE")
        assert isinstance(r, float)


# ── build_reward_fn ───────────────────────────────────────────────────────────

class TestBuildRewardFn:

    def test_valid_agents_return_callable(self):
        for aid in ("a1", "a2", "a3", "a4", "a5"):
            fn = build_reward_fn(aid)
            assert callable(fn)

    def test_unknown_agent_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown agent_id"):
            build_reward_fn("a6")

    def test_error_message_lists_valid_agents(self):
        with pytest.raises(ValueError) as exc_info:
            build_reward_fn("bad")
        for aid in ("a1", "a2", "a3", "a4", "a5"):
            assert aid in str(exc_info.value)

    def test_batched_a1_returns_list(self):
        fn = build_reward_fn("a1")
        completions = [json.dumps({k: "x" for k in [
            "financial_knowledge", "risk_tolerance_score", "investment_horizon",
            "liquid_assets", "income", "investment_amount",
            "can_afford_total_loss", "financial_vulnerability"
        ]})] * 3
        result = fn(completions)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_batched_a5_with_kwargs(self):
        fn = build_reward_fn("a5")
        completions = [
            json.dumps({"decision": "SUITABLE", "summary": "x"}),
            json.dumps({"decision": "UNSUITABLE", "summary": "x"}),
        ]
        expected_decisions = ["SUITABLE", "SUITABLE"]
        result = fn(completions, expected_decision=expected_decisions)
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(0.0)

    def test_batched_a4_with_kwargs(self):
        fn = build_reward_fn("a4")
        completions = [
            json.dumps({"escalate": False, "flags": []}),
            json.dumps({"escalate": True, "flags": [{"rule_id": "X", "triggered": True}]}),
        ]
        expected_escalates = [False, True]
        result = fn(completions, expected_escalate=expected_escalates)
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(1.0)

    def test_all_rewards_in_0_1(self):
        fn = build_reward_fn("a1")
        result = fn([json.dumps(_A1_ALL_KEYS)])
        assert 0.0 <= result[0] <= 1.0

    def test_fn_name_contains_agent_id(self):
        fn = build_reward_fn("a3")
        assert "a3" in fn.__name__

    def test_scalar_kwarg_broadcast_to_all_samples(self):
        fn = build_reward_fn("a5")
        completions = [json.dumps({"decision": "SUITABLE", "summary": "x"})] * 3
        result = fn(completions, expected_decision="SUITABLE")
        assert all(r == pytest.approx(1.0) for r in result)


# ── VALID_AGENT_IDS ───────────────────────────────────────────────────────────

class TestValidAgentIds:

    def test_contains_all_five(self):
        assert VALID_AGENT_IDS == frozenset({"a1", "a2", "a3", "a4", "a5"})

    def test_is_frozenset(self):
        assert isinstance(VALID_AGENT_IDS, frozenset)
