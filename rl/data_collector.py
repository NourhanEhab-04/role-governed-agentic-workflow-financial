"""
rl/data_collector.py
====================
Rollout trace collector for MARL training data.

What a trace is
---------------
A trace is one (prompt, response, reward) triple for a single agent
from a single pipeline run.  It contains everything needed to train
that agent via GRPO:

  {
    "agent_id":       str,    # "a1" | "a2" | "a3" | "a4" | "a5"
    "scenario_id":    str,    # e.g. "01_suitable_conservative"
    "run_number":     int,    # 1-indexed within this scenario
    "prompt_system":  str,    # agent's system prompt (known constant)
    "prompt_user":    str,    # user message reconstructed from state
    "response":       str,    # agent's output as JSON string
    "reward":         float,  # per-agent reward from rl/rewards.py
    "composite":      float,  # composite reward for this run
    "all_rewards":    dict,   # {"a1": float, ..., "a5": float}
    "pipeline_halted": bool,
    "timestamp":      str,    # ISO 8601
    "architecture":   str,    # "pipeline"
  }

Separating concerns
-------------------
`extract_traces_from_state` is a pure function that takes an already-
completed PipelineState and a scenario dict and returns a list of traces.
It is testable without running any LLM.

`collect_traces` is the async driver that runs the full pipeline for
every scenario file and calls `extract_traces_from_state` on each result.

Storage
-------
Traces are saved as newline-delimited JSON (JSONL) at:
  data/rl_training/traces.jsonl

Each line is one trace dict.  The file is appended — multiple collection
runs accumulate traces in the same file.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

from rl.rewards import (
    reward_a1,
    reward_a2,
    reward_a3,
    reward_a4,
    reward_a5,
    composite_reward,
)


# ── Agent system prompts (imported once at module load) ────────────────────────
# These are the canonical prompts used by the live pipeline.
# We import them so trace reconstruction is byte-compatible with training.

def _load_system_prompts() -> dict[str, str]:
    """Return a dict mapping agent_id → system prompt string."""
    from agents.client_profiler import CLIENT_PROFILER_SYSTEM_PROMPT
    from agents.product_classifier import PRODUCT_CLASSIFIER_SYSTEM_PROMPT
    from agents.rule_engine_agent import RULE_ENGINE_AGENT_SYSTEM_PROMPT
    from agents.conflict_detector import CONFLICT_DETECTOR_SYSTEM_PROMPT
    from agents.disclosure_agent import DISCLOSURE_AGENT_SYSTEM_PROMPT
    return {
        "a1": CLIENT_PROFILER_SYSTEM_PROMPT.strip(),
        "a2": PRODUCT_CLASSIFIER_SYSTEM_PROMPT.strip(),
        "a3": RULE_ENGINE_AGENT_SYSTEM_PROMPT.strip(),
        "a4": CONFLICT_DETECTOR_SYSTEM_PROMPT.strip(),
        "a5": DISCLOSURE_AGENT_SYSTEM_PROMPT.strip(),
    }


# ── User message reconstruction ────────────────────────────────────────────────

def _build_user_message(agent_id: str, state: dict) -> str:
    """
    Reconstruct the user message that was sent to `agent_id` during the run.

    The pipeline sends structured JSON to A3–A5; A1 and A2 receive the raw
    text inputs directly.

    A1 → raw client_input string
    A2 → raw product_input string
    A3 → JSON {"client_profile": ..., "product_profile": ...}
    A4 → JSON {"client_profile": ..., "product_profile": ..., "rule_verdict": ...}
    A5 → JSON {"client_profile": ..., "product_profile": ...,
               "rule_verdict": ..., "conflict_report": ...}
    """
    if agent_id == "a1":
        return state.get("client_input", "")
    if agent_id == "a2":
        return state.get("product_input", "")
    if agent_id == "a3":
        return json.dumps(
            {
                "client_profile":  state.get("client_profile", {}),
                "product_profile": state.get("product_profile", {}),
            },
            indent=2,
        )
    if agent_id == "a4":
        return json.dumps(
            {
                "client_profile":  state.get("client_profile", {}),
                "product_profile": state.get("product_profile", {}),
                "rule_verdict":    state.get("rule_verdict", {}),
            },
            indent=2,
        )
    if agent_id == "a5":
        return json.dumps(
            {
                "client_profile":  state.get("client_profile", {}),
                "product_profile": state.get("product_profile", {}),
                "rule_verdict":    state.get("rule_verdict", {}),
                "conflict_report": state.get("conflict_report", {}),
            },
            indent=2,
        )
    raise ValueError(f"Unknown agent_id: '{agent_id}'")


def _build_response(agent_id: str, state: dict) -> str:
    """
    Return the agent's output from the pipeline state as a JSON string.

    Returns an empty string if the output key is absent (pipeline halted
    before this agent ran).
    """
    _key_map = {
        "a1": "client_profile",
        "a2": "product_profile",
        "a3": "rule_verdict",
        "a4": "conflict_report",
        "a5": "suitability_report",
    }
    output_key = _key_map.get(agent_id)
    if output_key is None:
        return ""
    value: Any = state.get(output_key)
    if value is None:
        return ""
    return json.dumps(value, indent=2)


# ── Core extractor (pure function — fully testable) ────────────────────────────

def extract_traces_from_state(
    state: dict,
    scenario: dict,
    scenario_id: str,
    run_number: int = 1,
) -> list[dict]:
    """
    Extract per-agent training traces from a completed PipelineState.

    This is a pure function: it reads `state` and `scenario` and returns
    a list of 5 trace dicts (one per agent A1–A5).  Agents that did not
    produce output because the pipeline halted early still get a trace
    with an empty response and 0.0 reward.

    Args:
      state:       Final PipelineState dict (keys defined in schemas/langgraph_state.py).
      scenario:    Scenario dict from data/scenarios/*.json.
                   Must contain "expected_decision" and "expected_escalate".
      scenario_id: Stem of the scenario filename (e.g. "01_suitable_conservative").
      run_number:  1-indexed run counter for this scenario.

    Returns:
      List of 5 trace dicts, one per agent A1–A5.
    """
    system_prompts = _load_system_prompts()
    rewards = composite_reward(state, scenario)
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    halted = bool(state.get("halt", False))

    traces: list[dict] = []
    for agent_id in ("a1", "a2", "a3", "a4", "a5"):
        traces.append(
            {
                "agent_id":        agent_id,
                "scenario_id":     scenario_id,
                "run_number":      run_number,
                "prompt_system":   system_prompts[agent_id],
                "prompt_user":     _build_user_message(agent_id, state),
                "response":        _build_response(agent_id, state),
                "reward":          rewards[agent_id],
                "composite":       rewards["composite"],
                "all_rewards":     {k: v for k, v in rewards.items() if k != "composite"},
                "pipeline_halted": halted,
                "timestamp":       ts,
                "architecture":    "pipeline",
            }
        )
    return traces


# ── Async collection driver ────────────────────────────────────────────────────

async def collect_traces(
    scenarios_dir: Path,
    products_dir: Path,
    output_dir: Path,
    model_client,
    runs_per_scenario: int = 1,
) -> list[dict]:
    """
    Run the full pipeline on every scenario file and collect training traces.

    For each scenario × run:
      1. Load the scenario JSON and its product JSON.
      2. Run orchestrator/graph.run_pipeline() with the given model_client.
      3. Call extract_traces_from_state() on the result.
      4. Append traces to output_dir/traces.jsonl.

    Args:
      scenarios_dir:      Path to data/scenarios/
      products_dir:       Path to data/products/
      output_dir:         Path to data/rl_training/ (created if absent)
      model_client:       AutoGen model client (Groq or local)
      runs_per_scenario:  How many times to run each scenario (default 1).
                          For GRPO group sampling, set to >= 4.

    Returns:
      Flat list of all trace dicts collected in this call.

    Notes:
      - The output JSONL file is APPENDED, not overwritten.
        Call `output_dir / "traces.jsonl"` and delete it before a fresh run.
      - Pipeline errors (halts) are captured: the trace records them with
        pipeline_halted=True and 0.0 rewards for missing outputs.
    """
    from orchestrator.graph import run_pipeline

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "traces.jsonl"

    scenario_files = sorted(scenarios_dir.glob("*.json"))
    if not scenario_files:
        raise FileNotFoundError(f"No scenario files found in {scenarios_dir}")

    all_traces: list[dict] = []

    for sc_path in scenario_files:
        scenario = json.loads(sc_path.read_text())
        product_path = products_dir / scenario["product_file"]
        product = json.loads(product_path.read_text())
        scenario_id = sc_path.stem

        client_text  = json.dumps(scenario["client"])
        product_text = json.dumps(product)

        for run_n in range(1, runs_per_scenario + 1):
            try:
                state, _audit = await run_pipeline(
                    client_input=client_text,
                    product_input=product_text,
                    model_client=model_client,
                )
            except Exception as exc:
                state = {
                    "client_input":  client_text,
                    "product_input": product_text,
                    "halt":          True,
                    "halt_reason":   f"Collection error: {exc}",
                }

            traces = extract_traces_from_state(
                state, scenario, scenario_id, run_number=run_n
            )
            all_traces.extend(traces)

            with open(out_path, "a", encoding="utf-8") as fh:
                for trace in traces:
                    fh.write(json.dumps(trace) + "\n")

    return all_traces
