"""
rl/grpo_config.py
=================
GRPO (Group Relative Policy Optimization) configuration and per-agent reward
wrappers for cooperative MARL fine-tuning.

Architecture
------------
GRPO trains each agent independently using a "frozen context" approach:
  - A1/A2 inputs come from raw scenario text (same as live inference).
  - A3-A5 inputs come from gold-standard SFT data (not live other agents).
    This decouples training stability from upstream agent quality.

Each agent has a "completion reward function" — a pure Python callable that
takes a single generated completion (JSON string) plus structured context kwargs,
and returns a float reward in [0.0, 1.0].

These single-sample functions compose into a batched reward function compatible
with TRL GRPOTrainer's expected signature:
  def reward_fn(completions: list[str], **kwargs) -> list[float]

RLVR (RL with Verifiable Rewards)
----------------------------------
A3 implements a fully-deterministic RLVR reward: it re-runs the rule engine
on the client/product profiles and checks whether the agent's verdict matches
the ground truth.  No LLM judge — the reward is always correct.

GRPORunConfig
-------------
Dataclass of all GRPO hyperparameters.  Validated at construction time.
The only allowed entry points to training are via this config — nothing is
hard-coded in training scripts.

Note on TRL dependency
----------------------
This module is intentionally TRL-free.  It defines the config and reward
logic as plain Python so it can be imported and tested without HuggingFace
TRL installed.  The actual training script (not part of this file) will
import GRPORunConfig and pass build_reward_fn() to TRL's GRPOTrainer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from schemas.client_profile import REQUIRED_CLIENT_KEYS
from schemas.product_profile import REQUIRED_PRODUCT_KEYS


# ── Dataclass: training hyperparameters ───────────────────────────────────────

@dataclass
class GRPORunConfig:
    """
    All tunable hyperparameters for a single GRPO training run.

    Attributes:
      num_generations:              Group size G (>=2).  Larger G → better
                                    baseline estimate, higher memory cost.
                                    Typical: 4–16.
      learning_rate:                AdamW learning rate.  Range (0, 1).
      max_new_tokens:               Max tokens the model may generate per
                                    completion.  Must be >= 32.
      max_prompt_length:            Tokens per prompt before truncation.
      epochs:                       Training epochs over the dataset.
      per_device_batch_size:        Samples per GPU per step.
      gradient_accumulation_steps:  Effective batch = per_device * accum.
      warmup_ratio:                 Fraction of steps used for LR warmup.
      kl_coeff:                     β in the KL-penalty term.  0.0 = no KL.
      reward_clip_min:              Floor for clipping reward values.
      reward_clip_max:              Ceiling for clipping reward values.
      temperature:                  Sampling temperature for group generation.
      seed:                         Reproducibility seed.
    """
    num_generations:             int   = 8
    learning_rate:               float = 2e-6
    max_new_tokens:              int   = 512
    max_prompt_length:           int   = 2048
    epochs:                      int   = 1
    per_device_batch_size:       int   = 2
    gradient_accumulation_steps: int   = 4
    warmup_ratio:                float = 0.1
    kl_coeff:                    float = 0.04
    reward_clip_min:             float = 0.0
    reward_clip_max:             float = 1.0
    temperature:                 float = 0.9
    seed:                        int   = 42

    def __post_init__(self) -> None:
        if self.num_generations < 2:
            raise ValueError(
                f"num_generations must be >= 2, got {self.num_generations}"
            )
        if not (0.0 < self.learning_rate < 1.0):
            raise ValueError(
                f"learning_rate must be in (0, 1), got {self.learning_rate}"
            )
        if self.max_new_tokens < 32:
            raise ValueError(
                f"max_new_tokens must be >= 32, got {self.max_new_tokens}"
            )
        if self.max_prompt_length < 64:
            raise ValueError(
                f"max_prompt_length must be >= 64, got {self.max_prompt_length}"
            )
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.per_device_batch_size < 1:
            raise ValueError(
                f"per_device_batch_size must be >= 1, got {self.per_device_batch_size}"
            )
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}"
            )
        if not (0.0 <= self.warmup_ratio <= 1.0):
            raise ValueError(
                f"warmup_ratio must be in [0, 1], got {self.warmup_ratio}"
            )
        if self.kl_coeff < 0.0:
            raise ValueError(
                f"kl_coeff must be >= 0.0, got {self.kl_coeff}"
            )
        if self.reward_clip_min < 0.0 or self.reward_clip_max > 1.0:
            raise ValueError(
                f"reward clip range must be within [0.0, 1.0], "
                f"got [{self.reward_clip_min}, {self.reward_clip_max}]"
            )
        if self.reward_clip_min > self.reward_clip_max:
            raise ValueError(
                f"reward_clip_min ({self.reward_clip_min}) > "
                f"reward_clip_max ({self.reward_clip_max})"
            )
        if not (0.0 < self.temperature <= 2.0):
            raise ValueError(
                f"temperature must be in (0, 2], got {self.temperature}"
            )

    @property
    def effective_batch_size(self) -> int:
        """Total samples processed per optimiser step."""
        return self.per_device_batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> dict[str, Any]:
        """Serialise config for logging / saving."""
        return {
            "num_generations":             self.num_generations,
            "learning_rate":               self.learning_rate,
            "max_new_tokens":              self.max_new_tokens,
            "max_prompt_length":           self.max_prompt_length,
            "epochs":                      self.epochs,
            "per_device_batch_size":       self.per_device_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_ratio":                self.warmup_ratio,
            "kl_coeff":                    self.kl_coeff,
            "reward_clip_min":             self.reward_clip_min,
            "reward_clip_max":             self.reward_clip_max,
            "temperature":                 self.temperature,
            "seed":                        self.seed,
            "effective_batch_size":        self.effective_batch_size,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GRPORunConfig":
        """Reconstruct a config from a previously serialised dict."""
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in allowed}
        return cls(**filtered)


# ── Default config (sensible for local 8B model) ──────────────────────────────

DEFAULT_GRPO_CONFIG = GRPORunConfig()


# ── Utility ───────────────────────────────────────────────────────────────────

def parse_json_safe(text: str) -> dict | None:
    """
    Attempt to parse `text` as a JSON object.

    Returns the parsed dict, or None if parsing fails or result is not a dict.
    Never raises.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        result = json.loads(text.strip())
        return result if isinstance(result, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def clip_reward(
    reward: float,
    min_r: float = 0.0,
    max_r: float = 1.0,
) -> float:
    """Clamp reward to [min_r, max_r]."""
    return max(min_r, min(max_r, float(reward)))


# ── Per-agent standalone reward functions (pure — no LLM, no I/O) ────────────

def reward_completion_a1(completion: str, **kwargs) -> float:
    """
    Reward for A1 (client profiler) completion.

    Checks that the JSON output contains all required client profile keys.
    Returns 1.0 if all keys present, 0.5 if some present, 0.0 if no valid JSON.

    Args:
      completion: Raw string output from A1 (expected to be JSON).
      **kwargs:   Ignored (for batched-function compatibility).
    """
    parsed = parse_json_safe(completion)
    if parsed is None:
        return 0.0
    present = sum(1 for k in REQUIRED_CLIENT_KEYS if k in parsed)
    total = len(REQUIRED_CLIENT_KEYS)
    if present == total:
        return 1.0
    if present > 0:
        return round(0.5 * present / total + 0.25, 6)
    return 0.0


def reward_completion_a2(completion: str, **kwargs) -> float:
    """
    Reward for A2 (product classifier) completion.

    Checks that the JSON output contains all required product profile keys.
    Returns 1.0 if all keys present, partial credit if some present, 0.0 otherwise.

    Args:
      completion: Raw string output from A2 (expected to be JSON).
      **kwargs:   Ignored.
    """
    parsed = parse_json_safe(completion)
    if parsed is None:
        return 0.0
    present = sum(1 for k in REQUIRED_PRODUCT_KEYS if k in parsed)
    total = len(REQUIRED_PRODUCT_KEYS)
    if present == total:
        return 1.0
    if present > 0:
        return round(0.5 * present / total + 0.25, 6)
    return 0.0


def reward_completion_a3(
    completion: str,
    client_profile: dict | None = None,
    product_profile: dict | None = None,
    **kwargs,
) -> float:
    """
    RLVR reward for A3 (rule engine agent) completion.

    Runs the deterministic rule engine on the provided gold-standard profiles
    and compares the agent's verdict to the ground truth.

    Reward breakdown:
      +0.50  if agent decision matches ground truth decision
      +0.50  if agent hard_failed_rules matches ground truth exactly
      = 1.0  for a perfect verdict

    Falls back to 0.0 if:
      - completion is not valid JSON
      - client_profile or product_profile are None / invalid
      - rule engine fails

    Args:
      completion:      A3 output JSON string (rule_verdict format).
      client_profile:  Gold-standard client profile dict.
      product_profile: Gold-standard product profile dict.
      **kwargs:        Ignored.
    """
    parsed = parse_json_safe(completion)
    if parsed is None:
        return 0.0
    if not client_profile or not product_profile:
        # Without profiles we can only check structure
        has_decision = "decision" in parsed
        has_rules = "rules" in parsed
        return 0.5 if (has_decision and has_rules) else 0.0

    try:
        from rule_engine.rule_engine import evaluate_suitability
        ground_truth = evaluate_suitability(client_profile, product_profile)
    except Exception:
        return 0.0

    reward = 0.0
    # Decision match
    agent_decision = parsed.get("decision", "")
    if agent_decision == ground_truth.get("decision", ""):
        reward += 0.50
    # Hard-failed rules match
    gt_hard_fails = set(ground_truth.get("hard_failed_rules", []))
    agent_hard_fails = set(parsed.get("hard_failed_rules", []))
    if agent_hard_fails == gt_hard_fails:
        reward += 0.50
    return reward


def reward_completion_a4(
    completion: str,
    expected_escalate: bool = False,
    **kwargs,
) -> float:
    """
    Reward for A4 (conflict detector) completion.

    Returns 1.0 if the agent's `escalate` flag matches `expected_escalate`
    and the output contains a `flags` list.  Returns 0.5 for correct escalate
    but missing/invalid flags.  Returns 0.0 for wrong escalate or invalid JSON.

    Args:
      completion:        A4 output JSON string (conflict_report format).
      expected_escalate: Ground truth escalation flag from scenario.
      **kwargs:          Ignored.
    """
    parsed = parse_json_safe(completion)
    if parsed is None:
        return 0.0
    agent_escalate = parsed.get("escalate")
    if agent_escalate is None:
        return 0.0
    if bool(agent_escalate) != bool(expected_escalate):
        return 0.0
    # Correct escalate flag
    has_flags = isinstance(parsed.get("flags"), list)
    return 1.0 if has_flags else 0.5


def reward_completion_a5(
    completion: str,
    expected_decision: str = "",
    **kwargs,
) -> float:
    """
    Reward for A5 (disclosure agent) completion.

    Returns 1.0 if the agent's `decision` matches `expected_decision` (case-
    insensitive) and the output contains a `summary` field.
    Returns 0.5 for correct decision but missing summary.
    Returns 0.0 for wrong decision or invalid JSON.

    Args:
      completion:        A5 output JSON string (suitability_report format).
      expected_decision: Ground truth decision from scenario
                         (e.g. "SUITABLE", "UNSUITABLE", "ESCALATED").
      **kwargs:          Ignored.
    """
    parsed = parse_json_safe(completion)
    if parsed is None:
        return 0.0
    agent_decision = str(parsed.get("decision", "")).upper()
    if agent_decision != expected_decision.upper():
        return 0.0
    has_summary = bool(parsed.get("summary"))
    return 1.0 if has_summary else 0.5


# ── Reward function registry ──────────────────────────────────────────────────

_SINGLE_SAMPLE_FNS: dict[str, Callable] = {
    "a1": reward_completion_a1,
    "a2": reward_completion_a2,
    "a3": reward_completion_a3,
    "a4": reward_completion_a4,
    "a5": reward_completion_a5,
}

VALID_AGENT_IDS = frozenset(_SINGLE_SAMPLE_FNS.keys())


def build_reward_fn(agent_id: str) -> Callable[[list[str]], list[float]]:
    """
    Build a batched reward function for the given agent.

    The returned callable has the TRL GRPOTrainer-compatible signature:
      fn(completions: list[str], **kwargs) -> list[float]

    Extra context (e.g. `expected_decision`, `client_profile`) should be
    passed via **kwargs; they are forwarded to the per-completion function.

    Args:
      agent_id: One of "a1", "a2", "a3", "a4", "a5".

    Returns:
      Batched reward function.

    Raises:
      ValueError: if agent_id is not recognised.
    """
    if agent_id not in VALID_AGENT_IDS:
        raise ValueError(
            f"Unknown agent_id '{agent_id}'. "
            f"Valid values: {sorted(VALID_AGENT_IDS)}"
        )
    single_fn = _SINGLE_SAMPLE_FNS[agent_id]

    def batched_reward(completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for i, completion in enumerate(completions):
            # Extract per-sample kwargs if they are lists (TRL passes batched columns)
            sample_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, (list, tuple)) and len(v) == len(completions):
                    sample_kwargs[k] = v[i]
                else:
                    sample_kwargs[k] = v
            rewards.append(single_fn(completion, **sample_kwargs))
        return rewards

    batched_reward.__name__ = f"reward_fn_{agent_id}"
    return batched_reward
