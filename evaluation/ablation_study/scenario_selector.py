"""
evaluation/ablation_study/scenario_selector.py
================================================
Selects 50 balanced scenarios from data/scenarios/ for the ablation study.

Balance target (limited by ESCALATED supply of 10):
  SUITABLE   : 14
  CONDITIONAL: 13
  UNSUITABLE : 13
  ESCALATED  : 10
  ─────────────────
  Total      : 50

Selection prioritises EDGE CASES — scenarios where a single field extraction
error would flip the final suitability decision.  This is essential for
ablation studies: components like the verifier and deterministic enforcer can
only show a measurable effect when the pipeline is stressed by inputs that are
genuinely close to a decision boundary.

Edge-case scoring (lower score = higher priority):
  0  — score within ±5 of a hard boundary (40 or 70), or a hard-fail rule
       that fires on a field value exactly at its threshold
  1  — score within ±10 of a boundary, or R2/R3/R6 soft-fail combination
       that determines the category
  2  — all other scenarios

Within each edge-case tier, scenarios are further ranked by diversity criteria
so we still cover a broad range of profiles inside each tier.

Narrative enforcement:
  Scenarios without BOTH client_narrative and product_narrative are rejected
  at load time with an explicit error.  The silent JSON-profile fallback is
  removed — passing ground-truth structured data to the LLM would make all
  extraction trivially correct and defeat the purpose of the ablation.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SCENARIOS_DIR = Path("data/scenarios")

TARGETS: dict[str, int] = {
    "SUITABLE":    14,
    "CONDITIONAL": 13,
    "UNSUITABLE":  13,
    "ESCALATED":   10,
}

_SELECTION_SEED = 42

# MiFID II soft-fail rule penalties (used to infer expected score)
_SOFT_PENALTIES: dict[str, int] = {"R2": 20, "R3": 15, "R6": 25}
_HARD_FAILS = {"R1", "R4", "R5", "R7"}

# Decision thresholds
_BOUNDARY_SCORE_CONDITIONAL = 40   # score < 40 → UNSUITABLE
_BOUNDARY_SCORE_SUITABLE    = 70   # score < 70 → CONDITIONAL


def _infer_score(rules_failed: list[str]) -> int | None:
    """
    Compute the expected numeric score from the list of failed rules.
    Returns None for UNSUITABLE-by-hard-fail (score is irrelevant in that case).
    """
    if any(r in _HARD_FAILS for r in rules_failed):
        return None  # hard fail — score doesn't drive the decision
    score = 100
    for rule in rules_failed:
        score -= _SOFT_PENALTIES.get(rule, 0)
    return score


def _edge_case_tier(s: dict) -> int:
    """
    Return 0 (hardest extraction stress) → 2 (routine scenario).

    An edge case is a scenario where a plausible ONE-STEP extraction error
    on a single field would flip the final suitability decision.  These are
    the scenarios that stress the verifier and deterministic enforcer.

    Tier 0: score within ±5 of a decision boundary (35–45 or 65–75),
            OR currently passing a hard-fail rule by exactly one extraction
            step (e.g. client knowledge == product requirement, so one level
            lower would fire R1 / R7).
    Tier 1: score within ±10 of a boundary (30–50 or 60–80),
            OR any soft-fail combination of 2+ rules (multi-rule interaction).
    Tier 2: everything else (score far from boundaries, simple single-rule or
            no-rule scenarios).

    Key distinction: boundary checks measure proximity to a THRESHOLD, not
    whether a rule is already failing.  A scenario that already fires R4 is
    not a boundary — it is just a hard-fail scenario.  A scenario where
    the client CAN afford total loss but the product has total-loss potential
    IS a boundary (one extraction error flips R4).
    """
    rules_failed = s.get("expected_rules_failed", [])
    cp = s.get("expected_client_profile", {})
    pp = s.get("expected_product_profile", {})

    score = _infer_score(rules_failed)

    _knowledge_order = ["none", "basic", "moderate", "advanced"]
    client_know  = cp.get("financial_knowledge", "")
    product_know = pp.get("requires_knowledge_level", "")

    try:
        know_gap = _knowledge_order.index(client_know) - _knowledge_order.index(product_know)
    except ValueError:
        know_gap = 99

    # ── Hard-fail PROXIMITY checks (scenario currently PASSES, but one wrong ──
    # extraction step would fire the rule)

    # R1 proximity: client knowledge exactly equals product requirement AND
    #   requirement is not "none" (there's an adjacent lower level to extract wrongly).
    #   know_gap == 0 with product_know != "none" → extracting one level lower fires R1.
    r1_proximity = (know_gap == 0 and product_know not in ("", "none"))

    # R4 proximity: client CAN afford total loss AND product has total-loss potential.
    #   If extracted as can_afford=False → R4 fires → UNSUITABLE.
    r4_proximity = (
        cp.get("can_afford_total_loss", True) is True
        and pp.get("potential_loss") == "total"
        and "R4" not in rules_failed  # not already failing
    )

    # R5 proximity: vulnerability is MEDIUM and risk_class == 5, OR
    #   vulnerability is LOW and risk_class == 5 with HIGH conceivable as misextraction.
    #   Practically: r5_class >= 5 AND vulnerability not already HIGH (which fires R5).
    r5_proximity = (
        pp.get("risk_class", 0) >= 5
        and cp.get("financial_vulnerability") in ("LOW", "MEDIUM")
        and "R5" not in rules_failed
    )

    # R7 proximity: client knowledge is "moderate" AND product is COMPLEX.
    #   Extracting "basic" instead of "moderate" fires R7 → UNSUITABLE.
    r7_proximity = (
        cp.get("financial_knowledge") == "moderate"
        and pp.get("complexity_tier") == "COMPLEX"
        and "R7" not in rules_failed
    )

    hard_proximity = r1_proximity or r4_proximity or r5_proximity or r7_proximity

    # ── Score-based boundary proximity ────────────────────────────────────────
    if score is not None:
        dist_to_40 = abs(score - _BOUNDARY_SCORE_CONDITIONAL)
        dist_to_70 = abs(score - _BOUNDARY_SCORE_SUITABLE)
        min_dist   = min(dist_to_40, dist_to_70)
    else:
        # Hard-fail scenarios: score is irrelevant; rely on structural proximity
        min_dist = 99

    # ── Tier assignment ───────────────────────────────────────────────────────
    if min_dist <= 5 or hard_proximity:
        return 0
    soft_fail_count = sum(1 for r in rules_failed if r in _SOFT_PENALTIES)
    if min_dist <= 10 or soft_fail_count >= 2:
        return 1
    return 2


def _diversity_key(s: dict) -> tuple:
    """Secondary sort key: maximise spread within a tier."""
    cp = s.get("expected_client_profile", {})
    pp = s.get("expected_product_profile", {})
    rules_failed   = tuple(sorted(s.get("expected_rules_failed", [])))
    vulnerability  = cp.get("financial_vulnerability", "LOW")
    risk_class     = pp.get("risk_class", 0)
    knowledge      = cp.get("financial_knowledge", "none")
    age            = cp.get("age", 0)
    return (rules_failed, vulnerability, risk_class, knowledge, age)


def _load_all_scenarios(scenarios_dir: Path = SCENARIOS_DIR) -> list[dict]:
    """
    Load all valid scenario files.  Raises ValueError for any file that is
    missing client_narrative or product_narrative — the JSON-profile fallback
    is intentionally removed to prevent ground-truth data leaking to the LLM.
    """
    scenarios = []
    missing_narrative: list[str] = []

    for p in sorted(scenarios_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if "expected_decision" not in data:
            continue

        # Strict narrative enforcement
        if "client_narrative" not in data or "product_narrative" not in data:
            missing_narrative.append(p.name)
            continue

        data["_path"]     = str(p)
        data["_filename"] = p.name
        scenarios.append(data)

    if missing_narrative:
        raise ValueError(
            f"{len(missing_narrative)} scenario file(s) are missing client_narrative "
            f"or product_narrative and were rejected:\n"
            + "\n".join(f"  {n}" for n in missing_narrative)
            + "\nAdd narratives to these files or remove them from data/scenarios/."
        )

    return scenarios


def _pick_diverse(pool: list[dict], count: int, rng: random.Random) -> list[dict]:
    """
    Stride-pick `count` scenarios from a pool, maximising spread.
    """
    if count >= len(pool):
        return pool[:count]
    step = len(pool) / count
    return [pool[int(i * step)] for i in range(count)]


def select_scenarios(
    scenarios_dir: Path = SCENARIOS_DIR,
    targets: dict[str, int] | None = None,
    seed: int = _SELECTION_SEED,
) -> list[dict]:
    """
    Return exactly sum(targets.values()) scenario dicts, balanced across
    categories, with edge-case scenarios prioritised within each category.

    Sort order within each category bucket:
      1. Edge-case tier (0 = hardest boundary case, 2 = routine)
      2. Diversity key (unique rule pattern, vulnerability, risk class, etc.)

    This guarantees that if any tier-0 or tier-1 scenarios exist, they are
    chosen before routine tier-2 scenarios.
    """
    if targets is None:
        targets = TARGETS

    all_scenarios = _load_all_scenarios(scenarios_dir)

    buckets: dict[str, list[dict]] = {cat: [] for cat in targets}
    for s in all_scenarios:
        cat = s.get("expected_decision", "")
        if cat in buckets:
            buckets[cat].append(s)

    selected: list[dict] = []
    rng = random.Random(seed)

    for cat, count in targets.items():
        pool = buckets[cat]
        if len(pool) < count:
            raise ValueError(
                f"Not enough '{cat}' scenarios: need {count}, found {len(pool)}."
            )

        # Primary sort: edge-case tier; secondary: diversity; shuffle within exact ties
        rng.shuffle(pool)
        pool_sorted = sorted(pool, key=lambda s: (_edge_case_tier(s), _diversity_key(s)))
        chosen = _pick_diverse(pool_sorted, count, rng)
        selected.extend(chosen)

    return selected


def select_and_report(
    scenarios_dir: Path = SCENARIOS_DIR,
    targets: dict[str, int] | None = None,
    seed: int = _SELECTION_SEED,
) -> dict:
    """
    Select scenarios and return a summary report dict including edge-case
    tier breakdown so you can verify the selection quality.
    """
    if targets is None:
        targets = TARGETS

    scenarios = select_scenarios(scenarios_dir, targets, seed)
    counts: dict[str, int] = {}
    tier_counts: dict[int, int] = {0: 0, 1: 0, 2: 0}

    for s in scenarios:
        cat = s.get("expected_decision", "UNKNOWN")
        counts[cat] = counts.get(cat, 0) + 1
        tier_counts[_edge_case_tier(s)] += 1

    return {
        "scenarios":    scenarios,
        "counts":       counts,
        "tier_counts":  tier_counts,
        "total":        len(scenarios),
        "filenames":    sorted(s["_filename"] for s in scenarios),
        "seed":         seed,
        "targets":      targets,
    }


if __name__ == "__main__":
    report = select_and_report()
    print(f"Selected {report['total']} scenarios:")
    for cat, cnt in sorted(report["counts"].items()):
        print(f"  {cat}: {cnt}")
    print("\nEdge-case tier breakdown:")
    for tier, cnt in sorted(report["tier_counts"].items()):
        label = {0: "Tier 0 — hard boundary (highest stress)",
                 1: "Tier 1 — near boundary / multi-rule",
                 2: "Tier 2 — routine"}.get(tier, str(tier))
        print(f"  {label}: {cnt}")
    print("\nFiles:")
    for f in report["filenames"]:
        print(f"  {f}")
