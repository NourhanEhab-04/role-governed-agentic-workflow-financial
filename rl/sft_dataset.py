"""
rl/sft_dataset.py
=================
SFT (Supervised Fine-Tuning) dataset builder for cold-start MARL training.

Why SFT first?
--------------
GRPO requires the base model to already produce *structured* outputs (JSON).
A randomly-initialised model won't do this, so we first run SFT on "gold"
traces — runs where the agent's per-agent reward was high (>= `min_reward`).
SFT teaches the base JSON format before GRPO refines decision quality.

What this module produces
-------------------------
A JSONL file where every line is one SFT example in HuggingFace chat format:

  {
    "messages": [
      {"role": "system",    "content": "<agent system prompt>"},
      {"role": "user",      "content": "<reconstructed user message>"},
      {"role": "assistant", "content": "<agent JSON response>"}
    ],
    "agent_id":    str,   # "a1" | ... | "a5"
    "scenario_id": str,
    "reward":      float, # per-agent reward (why this example was selected)
    "composite":   float,
  }

This format is directly consumable by TRL's SFTTrainer with `apply_chat_template`.

Separating concerns
-------------------
All non-I/O logic lives in pure functions:
  - `trace_to_sft_example`   — converts one trace dict → one SFT example dict
  - `filter_high_reward`     — filter a list of traces by per-agent reward
  - `group_by_agent`         — split traces into per-agent lists

I/O helpers:
  - `load_traces`            — read JSONL traces file
  - `save_sft_jsonl`         — write SFT examples to JSONL
  - `build_sft_dataset`      — pipeline: load → filter → convert → save

Storage
-------
Default output: data/rl_training/sft_<agent_id>.jsonl  (one file per agent)
or:             data/rl_training/sft_all.jsonl          (all agents combined)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


# ── Default threshold ──────────────────────────────────────────────────────────
DEFAULT_MIN_REWARD: float = 0.8


# ── Pure functions (fully testable without filesystem) ─────────────────────────

def trace_to_sft_example(trace: dict) -> dict:
    """
    Convert one rollout trace dict into an SFT training example.

    Args:
      trace: A trace dict as produced by `extract_traces_from_state`.
             Required keys: prompt_system, prompt_user, response,
                            agent_id, scenario_id, reward, composite.

    Returns:
      An SFT example dict:
        {
          "messages": [system, user, assistant],
          "agent_id": str,
          "scenario_id": str,
          "reward": float,
          "composite": float,
        }

    Raises:
      KeyError: if any required key is absent from `trace`.
    """
    required = ("prompt_system", "prompt_user", "response",
                "agent_id", "scenario_id", "reward", "composite")
    for key in required:
        if key not in trace:
            raise KeyError(f"Trace is missing required key: '{key}'")

    return {
        "messages": [
            {"role": "system",    "content": trace["prompt_system"]},
            {"role": "user",      "content": trace["prompt_user"]},
            {"role": "assistant", "content": trace["response"]},
        ],
        "agent_id":    trace["agent_id"],
        "scenario_id": trace["scenario_id"],
        "reward":      float(trace["reward"]),
        "composite":   float(trace["composite"]),
    }


def filter_high_reward(
    traces: list[dict],
    min_reward: float = DEFAULT_MIN_REWARD,
) -> list[dict]:
    """
    Return only traces whose per-agent `reward` >= `min_reward`.

    Args:
      traces:     List of trace dicts (as from load_traces or extract_traces_from_state).
      min_reward: Minimum per-agent reward to include (default 0.8).
                  Use 0.0 to include all traces; 1.0 for only perfect traces.

    Returns:
      Filtered list, preserving original order.

    Raises:
      ValueError: if min_reward is outside [0.0, 1.0].
    """
    if not (0.0 <= min_reward <= 1.0):
        raise ValueError(
            f"min_reward must be in [0.0, 1.0], got {min_reward}"
        )
    return [t for t in traces if float(t.get("reward", 0.0)) >= min_reward]


def group_by_agent(traces: list[dict]) -> dict[str, list[dict]]:
    """
    Partition a list of traces into per-agent sublists.

    Returns:
      Dict mapping agent_id → list of traces for that agent.
      Keys are only present if at least one trace exists for that agent.
      Order within each list matches the input order.
    """
    groups: dict[str, list[dict]] = {}
    for trace in traces:
        aid = trace.get("agent_id", "unknown")
        groups.setdefault(aid, []).append(trace)
    return groups


def traces_to_sft_examples(
    traces: list[dict],
    min_reward: float = DEFAULT_MIN_REWARD,
) -> list[dict]:
    """
    Filter traces by reward and convert each to an SFT example.

    Args:
      traces:     Raw trace dicts.
      min_reward: Per-agent reward threshold.

    Returns:
      List of SFT example dicts, one per qualifying trace.
    """
    filtered = filter_high_reward(traces, min_reward)
    return [trace_to_sft_example(t) for t in filtered]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield parsed dicts from a JSONL file, skipping blank lines."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_traces(traces_path: Path) -> list[dict]:
    """
    Load all traces from a JSONL file.

    Args:
      traces_path: Path to traces.jsonl (produced by collect_traces).

    Returns:
      List of trace dicts.

    Raises:
      FileNotFoundError: if traces_path does not exist.
    """
    if not traces_path.exists():
        raise FileNotFoundError(f"Traces file not found: {traces_path}")
    return list(_iter_jsonl(traces_path))


def save_sft_jsonl(examples: list[dict], output_path: Path) -> None:
    """
    Append SFT examples to a JSONL file.

    Creates the file (and parent dirs) if they do not exist.
    APPENDS rather than overwrites so multiple calls accumulate.

    Args:
      examples:    List of SFT example dicts.
      output_path: Destination path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex) + "\n")


def build_sft_dataset(
    traces_path: Path,
    output_dir: Path,
    min_reward: float = DEFAULT_MIN_REWARD,
    per_agent: bool = True,
) -> dict[str, int]:
    """
    End-to-end pipeline: load traces → filter → convert → save.

    Args:
      traces_path: Path to traces.jsonl.
      output_dir:  Directory where SFT JSONL files are written.
      min_reward:  Per-agent reward threshold (default 0.8).
      per_agent:   If True, write one sft_<agent_id>.jsonl per agent.
                   If False, write a single sft_all.jsonl.

    Returns:
      Dict mapping output filename (stem) → number of examples written.
      E.g. {"sft_a1": 12, "sft_a2": 11, ...} or {"sft_all": 55}

    Notes:
      Output files are APPENDED, not overwritten.
    """
    traces = load_traces(traces_path)
    counts: dict[str, int] = {}

    if per_agent:
        groups = group_by_agent(traces)
        for agent_id, agent_traces in groups.items():
            examples = traces_to_sft_examples(agent_traces, min_reward)
            out_path = output_dir / f"sft_{agent_id}.jsonl"
            save_sft_jsonl(examples, out_path)
            counts[f"sft_{agent_id}"] = len(examples)
    else:
        examples = traces_to_sft_examples(traces, min_reward)
        out_path = output_dir / "sft_all.jsonl"
        save_sft_jsonl(examples, out_path)
        counts["sft_all"] = len(examples)

    return counts
