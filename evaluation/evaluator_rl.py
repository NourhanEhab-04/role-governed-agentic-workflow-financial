"""
evaluation/evaluator_rl.py
===========================
RL-augmented evaluation pipeline.

This module extends `evaluation/evaluator.py` with MARL reward tracking.
Every pipeline run additionally records:
  - Per-agent rewards  (from rl/rewards.py via composite_reward)
  - Per-agent credit   (from rl/credit_assignment.py)

Why a separate module?
-----------------------
`evaluation/evaluator.py` is the thesis comparison driver — it runs both
baseline and pipeline architectures and must stay identical to what is
reported in the paper.  The RL evaluator is a separate layer that only
runs the pipeline architecture and adds RL-specific diagnostics on top.
The two modules share the standard result-dict format so metrics.py works
unchanged on either.

Pure functions
--------------
`extract_rl_result`       — Build a combined eval+RL result dict from a
                            completed PipelineState.  Pure, testable.
`aggregate_agent_rewards` — Compute per-agent reward descriptive stats
                            from a list of result dicts.  Pure, testable.
`compute_rl_metrics`      — Compute all M1-M6 metrics plus RL-specific
                            metrics from a list of RL result dicts.
                            Pure, testable.

Async driver
------------
`run_rl_evaluation`       — Run all scenarios and return RL result dicts.
                            Requires a live model client; not unit-tested.

Result dict format
------------------
Each result dict has all standard fields from evaluator.py PLUS:
  "rewards"   : {"a1": float, ..., "a5": float, "composite": float}
  "credit"    : {"a1": float, ..., "a5": float, "composite": float}
  "architecture" : "pipeline_rl"
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from evaluation.metrics import compute_all_metrics
from rl.credit_assignment import assign_credit
from rl.rewards import composite_reward


# ── Pure: build RL result dict from pipeline state ────────────────────────────

def extract_rl_result(
    state: dict,
    scenario: dict,
    scenario_id: str,
    credit_method: str = "composite",
) -> dict:
    """
    Build a combined evaluation + RL result dict from a completed PipelineState.

    This is a pure function: it reads `state` and `scenario` and returns a
    dict suitable for both `evaluation/metrics.py` (standard fields) and the
    RL-specific aggregation functions in this module.

    Args:
      state:          Final PipelineState dict.
      scenario:       Scenario dict with "expected_decision" / "expected_escalate".
      scenario_id:    Stem of the scenario filename.
      credit_method:  "composite" or "counterfactual" (passed to assign_credit).

    Returns:
      Dict with standard evaluator fields + "rewards", "credit", "architecture".
    """
    rv = state.get("rule_verdict", {})
    sr = state.get("suitability_report", {})

    output_decision = (sr.get("decision") or rv.get("decision") or "UNKNOWN")
    output_escalated = bool(state.get("escalated", False))

    output_rules: dict = rv.get("rules", {}) if isinstance(rv.get("rules"), dict) else {}
    output_failed = [k for k, v in output_rules.items() if v == "FAIL"]

    rewards = composite_reward(state, scenario)
    credit  = assign_credit(state, scenario, method=credit_method)

    return {
        # ── Standard evaluator fields ─────────────────────────────────────
        "scenario_id":            scenario_id,
        "output_decision":        output_decision,
        "expected_decision":      scenario.get("expected_decision", ""),
        "output_escalated":       output_escalated,
        "expected_escalate":      scenario.get("expected_escalate", False),
        "output_failed_rules":    output_failed,
        "expected_rules_failed":  scenario.get("expected_rules_failed", []),
        "output_rules":           output_rules,
        "a1_verified":  bool(state.get("a1_verification")),
        "a2_verified":  bool(state.get("a2_verification")),
        "a1_corrected": bool((state.get("a1_correction") or {}).get("corrected")),
        "a2_corrected": bool((state.get("a2_correction") or {}).get("corrected")),
        "halted":       bool(state.get("halt", False)),
        "halt_reason":  state.get("halt_reason"),
        "architecture": "pipeline_rl",
        # ── RL-specific fields ────────────────────────────────────────────
        "rewards": rewards,
        "credit":  credit,
    }


# ── Pure: aggregate per-agent reward stats ────────────────────────────────────

def aggregate_agent_rewards(results: list[dict]) -> dict[str, dict[str, float]]:
    """
    Compute per-agent reward descriptive statistics from a list of RL results.

    Each result dict must contain a "rewards" key mapping agent_id → float.

    Args:
      results: List of RL result dicts (as from extract_rl_result).

    Returns:
      Dict mapping agent_id → {"mean": float, "min": float, "max": float,
                                "std": float, "count": int}.
      Also includes "composite" key with the same structure.

    Returns an empty dict if results is empty or no rewards are present.
    """
    if not results:
        return {}

    buckets: dict[str, list[float]] = {}
    for result in results:
        rewards = result.get("rewards", {})
        for agent_id, r in rewards.items():
            buckets.setdefault(agent_id, []).append(float(r))

    stats: dict[str, dict[str, float]] = {}
    for agent_id, values in buckets.items():
        n = len(values)
        mean = sum(values) / n
        std  = statistics.pstdev(values) if n > 1 else 0.0
        stats[agent_id] = {
            "mean":  round(mean, 6),
            "min":   round(min(values), 6),
            "max":   round(max(values), 6),
            "std":   round(std, 6),
            "count": n,
        }
    return stats


# ── Pure: compute standard + RL metrics ──────────────────────────────────────

def compute_rl_metrics(
    results: list[dict],
    architecture_name: str = "pipeline_rl",
) -> dict[str, Any]:
    """
    Compute all M1-M6 standard metrics PLUS per-agent RL reward statistics.

    Extends `evaluation.metrics.compute_all_metrics` with:
      "agent_rewards":    {agent_id: {mean, min, max, std, count}, ...}
      "mean_composite":   float — mean composite reward across all runs
      "min_composite":    float — minimum composite reward
      "max_composite":    float — maximum composite reward
      "n_rl_results":     int   — number of results with reward data

    Args:
      results:           List of RL result dicts (must have "rewards" key).
      architecture_name: Label embedded in the returned dict.

    Returns:
      Metrics dict with all standard keys + RL-specific keys.
    """
    base_metrics = compute_all_metrics(results, architecture_name)

    reward_stats = aggregate_agent_rewards(results)

    composite_values = [
        float(r["rewards"]["composite"])
        for r in results
        if isinstance(r.get("rewards"), dict) and "composite" in r["rewards"]
    ]

    if composite_values:
        mean_comp = sum(composite_values) / len(composite_values)
        min_comp  = min(composite_values)
        max_comp  = max(composite_values)
    else:
        mean_comp = min_comp = max_comp = 0.0

    base_metrics["agent_rewards"]   = reward_stats
    base_metrics["mean_composite"]  = round(mean_comp, 6)
    base_metrics["min_composite"]   = round(min_comp, 6)
    base_metrics["max_composite"]   = round(max_comp, 6)
    base_metrics["n_rl_results"]    = len(composite_values)

    return base_metrics


# ── Async driver (requires live model client — not unit-tested) ───────────────

async def run_rl_evaluation(
    scenarios_dir: Path,
    products_dir: Path,
    model_client,
    runs_per_scenario: int = 1,
    credit_method: str = "composite",
) -> list[dict]:
    """
    Run the full pipeline on every scenario and return RL result dicts.

    For each scenario × run:
      1. Load scenario and product JSON.
      2. Run orchestrator/graph.run_pipeline().
      3. Call extract_rl_result() on the final state.

    Args:
      scenarios_dir:      Path to data/scenarios/.
      products_dir:       Path to data/products/.
      model_client:       AutoGen model client (from config.model_selector).
      runs_per_scenario:  Repeats per scenario for consistency measurement.
      credit_method:      "composite" or "counterfactual".

    Returns:
      Flat list of RL result dicts.
    """
    from orchestrator.graph import run_pipeline

    scenario_files = sorted(scenarios_dir.glob("*.json"))
    if not scenario_files:
        raise FileNotFoundError(f"No scenario files found in {scenarios_dir}")

    all_results: list[dict] = []

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
                    "halt_reason":   f"Evaluation error: {exc}",
                }

            result = extract_rl_result(
                state, scenario, scenario_id, credit_method=credit_method
            )
            result["run_number"] = run_n
            all_results.append(result)

    return all_results
