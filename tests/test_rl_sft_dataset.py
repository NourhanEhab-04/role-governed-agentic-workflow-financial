"""
tests/test_rl_sft_dataset.py
==============================
Unit tests for rl/sft_dataset.py (Step 4).

Pure-function tests require no LLM, no network, no Ollama.
I/O tests use tmp_path (pytest fixture) — no permanent filesystem writes.

Tested:
  - trace_to_sft_example      (structure, types, missing keys)
  - filter_high_reward        (threshold, bounds, edge cases)
  - group_by_agent            (grouping, ordering, empty input)
  - traces_to_sft_examples    (integration of filter + convert)
  - load_traces               (reads JSONL, raises on missing file)
  - save_sft_jsonl            (appends, creates dirs)
  - build_sft_dataset         (per_agent=True, per_agent=False)
"""

import json
from pathlib import Path

import pytest

from rl.sft_dataset import (
    DEFAULT_MIN_REWARD,
    build_sft_dataset,
    filter_high_reward,
    group_by_agent,
    load_traces,
    save_sft_jsonl,
    trace_to_sft_example,
    traces_to_sft_examples,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_trace(
    agent_id: str = "a1",
    reward: float = 1.0,
    composite: float = 1.0,
    scenario_id: str = "01_test",
    run_number: int = 1,
    response: str = '{"key": "value"}',
) -> dict:
    return {
        "agent_id":        agent_id,
        "scenario_id":     scenario_id,
        "run_number":      run_number,
        "prompt_system":   f"System prompt for {agent_id}",
        "prompt_user":     f"User message for {agent_id}",
        "response":        response,
        "reward":          reward,
        "composite":       composite,
        "all_rewards":     {"a1": 1.0, "a2": 1.0, "a3": 1.0, "a4": 1.0, "a5": 1.0},
        "pipeline_halted": False,
        "timestamp":       "2026-01-01T00:00:00Z",
        "architecture":    "pipeline",
    }


def _five_traces(reward: float = 1.0) -> list[dict]:
    return [_make_trace(aid, reward) for aid in ("a1", "a2", "a3", "a4", "a5")]


# ── trace_to_sft_example ──────────────────────────────────────────────────────

class TestTraceToSftExample:

    def test_returns_dict(self):
        assert isinstance(trace_to_sft_example(_make_trace()), dict)

    def test_has_messages_key(self):
        ex = trace_to_sft_example(_make_trace())
        assert "messages" in ex

    def test_messages_has_three_entries(self):
        ex = trace_to_sft_example(_make_trace())
        assert len(ex["messages"]) == 3

    def test_roles_are_system_user_assistant(self):
        ex = trace_to_sft_example(_make_trace())
        roles = [m["role"] for m in ex["messages"]]
        assert roles == ["system", "user", "assistant"]

    def test_system_content_matches_prompt_system(self):
        trace = _make_trace()
        ex = trace_to_sft_example(trace)
        assert ex["messages"][0]["content"] == trace["prompt_system"]

    def test_user_content_matches_prompt_user(self):
        trace = _make_trace()
        ex = trace_to_sft_example(trace)
        assert ex["messages"][1]["content"] == trace["prompt_user"]

    def test_assistant_content_matches_response(self):
        trace = _make_trace(response='{"decision": "SUITABLE"}')
        ex = trace_to_sft_example(trace)
        assert ex["messages"][2]["content"] == '{"decision": "SUITABLE"}'

    def test_agent_id_preserved(self):
        ex = trace_to_sft_example(_make_trace(agent_id="a3"))
        assert ex["agent_id"] == "a3"

    def test_scenario_id_preserved(self):
        ex = trace_to_sft_example(_make_trace(scenario_id="07_test"))
        assert ex["scenario_id"] == "07_test"

    def test_reward_is_float(self):
        ex = trace_to_sft_example(_make_trace(reward=0.75))
        assert isinstance(ex["reward"], float)
        assert ex["reward"] == pytest.approx(0.75)

    def test_composite_is_float(self):
        ex = trace_to_sft_example(_make_trace(composite=0.85))
        assert isinstance(ex["composite"], float)
        assert ex["composite"] == pytest.approx(0.85)

    def test_extra_keys_not_included(self):
        ex = trace_to_sft_example(_make_trace())
        assert set(ex.keys()) == {"messages", "agent_id", "scenario_id", "reward", "composite"}

    def test_missing_prompt_system_raises_key_error(self):
        trace = _make_trace()
        del trace["prompt_system"]
        with pytest.raises(KeyError, match="prompt_system"):
            trace_to_sft_example(trace)

    def test_missing_response_raises_key_error(self):
        trace = _make_trace()
        del trace["response"]
        with pytest.raises(KeyError, match="response"):
            trace_to_sft_example(trace)

    def test_missing_agent_id_raises_key_error(self):
        trace = _make_trace()
        del trace["agent_id"]
        with pytest.raises(KeyError, match="agent_id"):
            trace_to_sft_example(trace)

    def test_reward_coerced_to_float_from_int(self):
        trace = _make_trace()
        trace["reward"] = 1  # int
        ex = trace_to_sft_example(trace)
        assert isinstance(ex["reward"], float)

    def test_each_message_has_role_and_content(self):
        ex = trace_to_sft_example(_make_trace())
        for msg in ex["messages"]:
            assert "role" in msg
            assert "content" in msg


# ── filter_high_reward ────────────────────────────────────────────────────────

class TestFilterHighReward:

    def test_all_pass_when_reward_equals_threshold(self):
        traces = [_make_trace("a1", reward=0.8), _make_trace("a2", reward=0.8)]
        result = filter_high_reward(traces, min_reward=0.8)
        assert len(result) == 2

    def test_filters_below_threshold(self):
        traces = [_make_trace("a1", reward=0.5), _make_trace("a2", reward=1.0)]
        result = filter_high_reward(traces, min_reward=0.8)
        assert len(result) == 1
        assert result[0]["agent_id"] == "a2"

    def test_all_filtered_when_all_below_threshold(self):
        traces = _five_traces(reward=0.3)
        result = filter_high_reward(traces, min_reward=0.8)
        assert result == []

    def test_none_filtered_when_threshold_0(self):
        traces = _five_traces(reward=0.0)
        result = filter_high_reward(traces, min_reward=0.0)
        assert len(result) == 5

    def test_only_perfect_when_threshold_1(self):
        traces = [_make_trace("a1", reward=1.0), _make_trace("a2", reward=0.99)]
        result = filter_high_reward(traces, min_reward=1.0)
        assert len(result) == 1

    def test_default_threshold_is_0_8(self):
        traces = [_make_trace("a1", reward=0.79), _make_trace("a2", reward=0.8)]
        result = filter_high_reward(traces)
        assert len(result) == 1
        assert result[0]["agent_id"] == "a2"

    def test_preserves_order(self):
        traces = [_make_trace(f"a{i}", reward=1.0) for i in range(1, 6)]
        result = filter_high_reward(traces, min_reward=1.0)
        assert [t["agent_id"] for t in result] == ["a1", "a2", "a3", "a4", "a5"]

    def test_empty_input_returns_empty(self):
        assert filter_high_reward([], min_reward=0.5) == []

    def test_invalid_threshold_below_0_raises(self):
        with pytest.raises(ValueError, match="min_reward"):
            filter_high_reward([], min_reward=-0.1)

    def test_invalid_threshold_above_1_raises(self):
        with pytest.raises(ValueError, match="min_reward"):
            filter_high_reward([], min_reward=1.1)

    def test_returns_same_objects_not_copies(self):
        traces = _five_traces(reward=1.0)
        result = filter_high_reward(traces, min_reward=0.5)
        for r in result:
            assert r in traces


# ── group_by_agent ────────────────────────────────────────────────────────────

class TestGroupByAgent:

    def test_five_agents_produce_five_groups(self):
        groups = group_by_agent(_five_traces())
        assert set(groups.keys()) == {"a1", "a2", "a3", "a4", "a5"}

    def test_each_group_has_one_trace_in_basic_case(self):
        groups = group_by_agent(_five_traces())
        for aid in ("a1", "a2", "a3", "a4", "a5"):
            assert len(groups[aid]) == 1

    def test_multiple_runs_accumulate_in_group(self):
        traces = [_make_trace("a1", run_number=i) for i in range(1, 4)]
        groups = group_by_agent(traces)
        assert len(groups["a1"]) == 3

    def test_empty_input_returns_empty_dict(self):
        assert group_by_agent([]) == {}

    def test_preserves_order_within_group(self):
        traces = [_make_trace("a1", run_number=i) for i in (3, 1, 2)]
        groups = group_by_agent(traces)
        assert [t["run_number"] for t in groups["a1"]] == [3, 1, 2]

    def test_missing_agent_id_uses_unknown(self):
        trace = _make_trace()
        del trace["agent_id"]
        groups = group_by_agent([trace])
        assert "unknown" in groups


# ── traces_to_sft_examples ────────────────────────────────────────────────────

class TestTracesToSftExamples:

    def test_returns_list(self):
        assert isinstance(traces_to_sft_examples(_five_traces()), list)

    def test_perfect_traces_all_converted(self):
        result = traces_to_sft_examples(_five_traces(reward=1.0), min_reward=0.8)
        assert len(result) == 5

    def test_low_reward_traces_excluded(self):
        traces = [_make_trace("a1", reward=0.3), _make_trace("a2", reward=0.9)]
        result = traces_to_sft_examples(traces, min_reward=0.8)
        assert len(result) == 1
        assert result[0]["agent_id"] == "a2"

    def test_each_result_has_messages_key(self):
        for ex in traces_to_sft_examples(_five_traces()):
            assert "messages" in ex

    def test_empty_traces_returns_empty(self):
        assert traces_to_sft_examples([]) == []


# ── I/O: load_traces ──────────────────────────────────────────────────────────

class TestLoadTraces:

    def test_loads_all_traces(self, tmp_path):
        p = tmp_path / "traces.jsonl"
        traces = _five_traces()
        with open(p, "w") as f:
            for t in traces:
                f.write(json.dumps(t) + "\n")
        loaded = load_traces(p)
        assert len(loaded) == 5

    def test_loaded_traces_match_original(self, tmp_path):
        p = tmp_path / "traces.jsonl"
        trace = _make_trace("a3", reward=0.7)
        with open(p, "w") as f:
            f.write(json.dumps(trace) + "\n")
        loaded = load_traces(p)
        assert loaded[0]["agent_id"] == "a3"
        assert loaded[0]["reward"] == pytest.approx(0.7)

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "traces.jsonl"
        with open(p, "w") as f:
            f.write(json.dumps(_make_trace("a1")) + "\n")
            f.write("\n")
            f.write(json.dumps(_make_trace("a2")) + "\n")
        loaded = load_traces(p)
        assert len(loaded) == 2

    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_traces(tmp_path / "nonexistent.jsonl")


# ── I/O: save_sft_jsonl ───────────────────────────────────────────────────────

class TestSaveSftJsonl:

    def test_creates_file(self, tmp_path):
        p = tmp_path / "out.jsonl"
        save_sft_jsonl([trace_to_sft_example(_make_trace())], p)
        assert p.exists()

    def test_written_lines_are_valid_json(self, tmp_path):
        p = tmp_path / "out.jsonl"
        examples = [trace_to_sft_example(t) for t in _five_traces()]
        save_sft_jsonl(examples, p)
        lines = p.read_text().strip().splitlines()
        for line in lines:
            json.loads(line)  # must not raise

    def test_writes_correct_count(self, tmp_path):
        p = tmp_path / "out.jsonl"
        examples = [trace_to_sft_example(t) for t in _five_traces()]
        save_sft_jsonl(examples, p)
        assert len(p.read_text().strip().splitlines()) == 5

    def test_appends_on_second_call(self, tmp_path):
        p = tmp_path / "out.jsonl"
        ex = [trace_to_sft_example(_make_trace())]
        save_sft_jsonl(ex, p)
        save_sft_jsonl(ex, p)
        assert len(p.read_text().strip().splitlines()) == 2

    def test_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "nested" / "deep" / "out.jsonl"
        save_sft_jsonl([trace_to_sft_example(_make_trace())], p)
        assert p.exists()

    def test_empty_list_creates_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        save_sft_jsonl([], p)
        assert p.exists()
        assert p.read_text() == ""


# ── build_sft_dataset ─────────────────────────────────────────────────────────

class TestBuildSftDataset:

    def _write_traces(self, path: Path, traces: list[dict]) -> None:
        with open(path, "w") as f:
            for t in traces:
                f.write(json.dumps(t) + "\n")

    def test_per_agent_creates_5_files(self, tmp_path):
        traces_path = tmp_path / "traces.jsonl"
        self._write_traces(traces_path, _five_traces(reward=1.0))
        build_sft_dataset(traces_path, tmp_path / "out", per_agent=True)
        for aid in ("a1", "a2", "a3", "a4", "a5"):
            assert (tmp_path / "out" / f"sft_{aid}.jsonl").exists()

    def test_combined_creates_sft_all_file(self, tmp_path):
        traces_path = tmp_path / "traces.jsonl"
        self._write_traces(traces_path, _five_traces(reward=1.0))
        build_sft_dataset(traces_path, tmp_path / "out", per_agent=False)
        assert (tmp_path / "out" / "sft_all.jsonl").exists()

    def test_returns_count_dict_per_agent(self, tmp_path):
        traces_path = tmp_path / "traces.jsonl"
        self._write_traces(traces_path, _five_traces(reward=1.0))
        counts = build_sft_dataset(traces_path, tmp_path / "out", per_agent=True)
        assert all(v == 1 for v in counts.values())

    def test_returns_count_dict_combined(self, tmp_path):
        traces_path = tmp_path / "traces.jsonl"
        self._write_traces(traces_path, _five_traces(reward=1.0))
        counts = build_sft_dataset(
            traces_path, tmp_path / "out", min_reward=0.8, per_agent=False
        )
        assert counts["sft_all"] == 5

    def test_low_reward_traces_excluded(self, tmp_path):
        traces_path = tmp_path / "traces.jsonl"
        traces = _five_traces(reward=0.3)
        self._write_traces(traces_path, traces)
        counts = build_sft_dataset(
            traces_path, tmp_path / "out", min_reward=0.8, per_agent=False
        )
        assert counts["sft_all"] == 0

    def test_mixed_rewards_correct_count(self, tmp_path):
        traces_path = tmp_path / "traces.jsonl"
        traces = (
            [_make_trace("a1", reward=1.0)]
            + [_make_trace("a2", reward=0.5)]
            + [_make_trace("a3", reward=0.9)]
        )
        self._write_traces(traces_path, traces)
        counts = build_sft_dataset(
            traces_path, tmp_path / "out", min_reward=0.8, per_agent=False
        )
        assert counts["sft_all"] == 2

    def test_output_jsonl_contains_messages(self, tmp_path):
        traces_path = tmp_path / "traces.jsonl"
        self._write_traces(traces_path, _five_traces(reward=1.0))
        out_dir = tmp_path / "out"
        build_sft_dataset(traces_path, out_dir, per_agent=False)
        lines = (out_dir / "sft_all.jsonl").read_text().strip().splitlines()
        for line in lines:
            ex = json.loads(line)
            assert "messages" in ex
            assert len(ex["messages"]) == 3
