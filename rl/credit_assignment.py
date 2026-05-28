"""
rl/credit_assignment.py
=======================
Credit assignment for cooperative MARL.

Why credit assignment?
----------------------
When multiple agents collectively produce a wrong final answer, a naive
"share global reward equally" approach sends the same gradient to every
agent — including agents whose outputs were already correct.  This slows
convergence and can degrade previously-correct behaviour.

Credit assignment gives each agent a share proportional to how much its
output contributed to the final outcome.

Two methods are implemented:

1. `composite_credit`  (simple, default)
   Re-uses `composite_reward` from rewards.py with local-signal blending.
   Local rewards distinguish agents whose outputs were already wrong BEFORE
   the final failure, giving them higher penalty.

2. `counterfactual_credit`  (more precise)
   For each agent, asks: "if this agent had failed (reward=0), how much
   would the composite have dropped?"  The delta is that agent's
   counterfactual contribution.  Agents whose failure would cause the
   largest composite drop get the largest share of the global signal.

The returned dicts always contain the same keys: "a1", "a2", "a3", "a4",
"a5", and "composite".  All values are floats in [0.0, 1.0].

These dicts are stored in rollout traces and used during GRPO training as
per-sample reward signals.
"""

from __future__ import annotations

from rl.rewards import (
    reward_a1,
    reward_a2,
    reward_a3,
    reward_a4,
    reward_a5,
    composite_reward,
)


# ── Local weights (must sum to 1.0, must match composite_reward in rewards.py)
_W = {"a1": 0.10, "a2": 0.10, "a3": 0.10, "a4": 0.10, "a5": 0.60}


def composite_credit(state: dict, scenario: dict) -> dict[str, float]:
    """
    Per-agent credit using the composite reward from rewards.py.

    This is the default credit signal for cooperative MARL training.
    It blends local verifiable rewards with the global outcome reward so
    each agent receives both an immediate signal and a shared incentive.

    Returns the same structure as `composite_reward`:
      {"a1": float, "a2": float, "a3": float, "a4": float, "a5": float,
       "composite": float}
    """
    return composite_reward(state, scenario)


def counterfactual_credit(state: dict, scenario: dict) -> dict[str, float]:
    """
    Per-agent counterfactual credit.

    For each agent i, compute:
      contribution_i = composite(actual) − composite(with agent_i forced to 0.0)

    A high contribution means: if this agent had failed, the composite would
    have dropped a lot, so this agent is load-bearing for the current result.

    Contributions are normalised so they sum to `composite(actual)` and each
    value stays in [0.0, 1.0].

    This gives a more targeted gradient than the simple composite:
      - Agents that are already at 0.0 get 0.0 contribution (they already
        received their penalty from the local reward).
      - Agents with high local rewards that happen to be in a globally-failing
        run get a proportionally lower counterfactual penalty.

    Returns:
      {
        "a1": float,  # counterfactual contribution of A1
        "a2": float,
        "a3": float,
        "a4": float,
        "a5": float,
        "composite": float,  # actual composite (unchanged)
      }
    """
    expected_decision: str = scenario["expected_decision"]
    expected_escalate: bool = scenario["expected_escalate"]

    # ── Actual per-agent rewards ──────────────────────────────────────────────
    actual = {
        "a1": reward_a1(state),
        "a2": reward_a2(state),
        "a3": reward_a3(state),
        "a4": reward_a4(state, expected_escalate),
        "a5": reward_a5(state, expected_decision),
    }
    actual_composite = sum(_W[k] * v for k, v in actual.items())

    # ── Counterfactual: force each agent to 0.0 in turn ──────────────────────
    contributions: dict[str, float] = {}
    for agent_id in ("a1", "a2", "a3", "a4", "a5"):
        # Replace this agent's reward with 0.0; keep all others at actual
        counterfactual = {k: (0.0 if k == agent_id else v) for k, v in actual.items()}
        cf_composite = sum(_W[k] * v for k, v in counterfactual.items())
        # How much did this agent contribute beyond the baseline (agent absent)?
        raw_contribution = actual_composite - cf_composite
        # Clamp to [0, 1] — negative contribution means the agent's removal
        # would *increase* the composite (shouldn't happen with non-negative
        # rewards, but guard against float rounding)
        contributions[agent_id] = max(0.0, round(raw_contribution, 6))

    contributions["composite"] = round(actual_composite, 6)
    return contributions


def assign_credit(
    state: dict,
    scenario: dict,
    method: str = "composite",
) -> dict[str, float]:
    """
    Compute per-agent credit using the specified method.

    Args:
      state:    PipelineState dict from a completed run.
      scenario: Scenario dict with "expected_decision" and "expected_escalate".
      method:   "composite" (default) or "counterfactual".

    Returns:
      Dict with keys "a1"–"a5" and "composite", all floats in [0.0, 1.0].

    Raises:
      ValueError: if `method` is not recognised.
    """
    if method == "composite":
        return composite_credit(state, scenario)
    if method == "counterfactual":
        return counterfactual_credit(state, scenario)
    raise ValueError(
        f"Unknown credit assignment method '{method}'. "
        f"Valid values: 'composite', 'counterfactual'."
    )
