"""
evaluation/ablation_study/configs.py
======================================
Defines all 16 ablation configurations across three independent axes.

Axis A — Architecture  (5 configs): which conflict-detection flags are active
Axis R — Rule          (8 configs): which MiFID II rules the rule engine enforces
Axis T — Threshold     (3 configs): decision-score thresholds

All configs share the same schema so the runner can iterate them uniformly.
Each config is a plain dict — no LLM, no I/O.

Config schema
-------------
{
  "id":          str   — short identifier used in filenames (e.g. "A0", "R1_off")
  "axis":        str   — "architecture" | "rule" | "threshold"
  "label":       str   — human-readable name for tables/reports
  "description": str   — what is ablated
  "disabled_rules":   list[str]  — rule IDs forced to pass  (e.g. ["R1"])
  "disabled_flags":   list[str]  — conflict flags disabled  (e.g. ["BORDERLINE"])
  "suitable_threshold":    int   — default 70
  "conditional_threshold": int   — default 40
  "is_control":   bool  — True for the reference configuration
}
"""

from __future__ import annotations

# ── Defaults ───────────────────────────────────────────────────────────────────
_DEFAULT_SUITABLE    = 70
_DEFAULT_CONDITIONAL = 40

# ── Architecture axis ──────────────────────────────────────────────────────────
# Tests which conflict-detection flags contribute to escalation accuracy.
# Rule engine is always full; thresholds are always default.

ARCH_CONFIGS: list[dict] = [
    {
        "id":          "A0",
        "axis":        "architecture",
        "label":       "Full Pipeline (control)",
        "description": "All components active — rule engine + all four conflict flags.",
        "disabled_rules":            [],
        "disabled_flags":            [],
        "suitable_threshold":    _DEFAULT_SUITABLE,
        "conditional_threshold": _DEFAULT_CONDITIONAL,
        "is_control":            True,
    },
    {
        "id":          "A1",
        "axis":        "architecture",
        "label":       "No Conflict Detector",
        "description": "Conflict detection entirely disabled. Final decision = rule engine verdict. "
                       "No escalation is ever produced.",
        "disabled_rules":            [],
        "disabled_flags":            ["BORDERLINE", "CONCENTRATION", "CONTRADICTION", "ESCALATION"],
        "suitable_threshold":    _DEFAULT_SUITABLE,
        "conditional_threshold": _DEFAULT_CONDITIONAL,
        "is_control":            False,
    },
    {
        "id":          "A2",
        "axis":        "architecture",
        "label":       "No BORDERLINE Flag",
        "description": "BORDERLINE (LOW-severity) flag disabled. "
                       "Only HIGH-severity flags (CONCENTRATION, CONTRADICTION) can escalate.",
        "disabled_rules":            [],
        "disabled_flags":            ["BORDERLINE"],
        "suitable_threshold":    _DEFAULT_SUITABLE,
        "conditional_threshold": _DEFAULT_CONDITIONAL,
        "is_control":            False,
    },
    {
        "id":          "A3",
        "axis":        "architecture",
        "label":       "No CONCENTRATION Flag",
        "description": "CONCENTRATION risk flag disabled. "
                       "Only BORDERLINE and CONTRADICTION can trigger escalation.",
        "disabled_rules":            [],
        "disabled_flags":            ["CONCENTRATION"],
        "suitable_threshold":    _DEFAULT_SUITABLE,
        "conditional_threshold": _DEFAULT_CONDITIONAL,
        "is_control":            False,
    },
    {
        "id":          "A4",
        "axis":        "architecture",
        "label":       "No CONTRADICTION Flag",
        "description": "CONTRADICTION flag disabled (vulnerability × suitable-verdict check). "
                       "Only BORDERLINE and CONCENTRATION can trigger escalation.",
        "disabled_rules":            [],
        "disabled_flags":            ["CONTRADICTION"],
        "suitable_threshold":    _DEFAULT_SUITABLE,
        "conditional_threshold": _DEFAULT_CONDITIONAL,
        "is_control":            False,
    },
]

# ── Rule axis ──────────────────────────────────────────────────────────────────
# Tests each MiFID II rule's individual contribution.
# Disabling a rule means it is forced to PASS (no penalty, no hard-fail).
# Architecture flags and thresholds remain at defaults.

RULE_CONFIGS: list[dict] = [
    {
        "id":          "R0",
        "axis":        "rule",
        "label":       "All Rules (control)",
        "description": "All 7 MiFID II rules active. Equivalent to A0.",
        "disabled_rules":            [],
        "disabled_flags":            [],
        "suitable_threshold":    _DEFAULT_SUITABLE,
        "conditional_threshold": _DEFAULT_CONDITIONAL,
        "is_control":            True,
    },
    {
        "id":          "R1_off",
        "axis":        "rule",
        "label":       "R1 Disabled (Knowledge)",
        "description": "R1 knowledge hard-fail suppressed — client knowledge gap never blocks recommendation.",
        "disabled_rules":            ["R1"],
        "disabled_flags":            [],
        "suitable_threshold":    _DEFAULT_SUITABLE,
        "conditional_threshold": _DEFAULT_CONDITIONAL,
        "is_control":            False,
    },
    {
        "id":          "R4_off",
        "axis":        "rule",
        "label":       "R4 Disabled (Affordability)",
        "description": "R4 affordability hard-fail suppressed — total-loss products allowed for any client.",
        "disabled_rules":            ["R4"],
        "disabled_flags":            [],
        "suitable_threshold":    _DEFAULT_SUITABLE,
        "conditional_threshold": _DEFAULT_CONDITIONAL,
        "is_control":            False,
    },
    {
        "id":          "R5_off",
        "axis":        "rule",
        "label":       "R5 Disabled (Vulnerability)",
        "description": "R5 vulnerability hard-fail suppressed — HIGH-vulnerability clients not protected from RC≥5.",
        "disabled_rules":            ["R5"],
        "disabled_flags":            [],
        "suitable_threshold":    _DEFAULT_SUITABLE,
        "conditional_threshold": _DEFAULT_CONDITIONAL,
        "is_control":            False,
    },
    {
        "id":          "R7_off",
        "axis":        "rule",
        "label":       "R7 Disabled (Complexity)",
        "description": "R7 complexity hard-fail suppressed — COMPLEX products allowed for any knowledge level.",
        "disabled_rules":            ["R7"],
        "disabled_flags":            [],
        "suitable_threshold":    _DEFAULT_SUITABLE,
        "conditional_threshold": _DEFAULT_CONDITIONAL,
        "is_control":            False,
    },
    {
        "id":          "R2_off",
        "axis":        "rule",
        "label":       "R2 Disabled (Risk Alignment)",
        "description": "R2 risk-alignment soft-fail suppressed — risk-class/tolerance mismatch ignored (no −20 penalty).",
        "disabled_rules":            ["R2"],
        "disabled_flags":            [],
        "suitable_threshold":    _DEFAULT_SUITABLE,
        "conditional_threshold": _DEFAULT_CONDITIONAL,
        "is_control":            False,
    },
    {
        "id":          "R3_off",
        "axis":        "rule",
        "label":       "R3 Disabled (Horizon)",
        "description": "R3 horizon soft-fail suppressed — short investment horizon ignored (no −15 penalty).",
        "disabled_rules":            ["R3"],
        "disabled_flags":            [],
        "suitable_threshold":    _DEFAULT_SUITABLE,
        "conditional_threshold": _DEFAULT_CONDITIONAL,
        "is_control":            False,
    },
    {
        "id":          "R6_off",
        "axis":        "rule",
        "label":       "R6 Disabled (Leverage)",
        "description": "R6 leverage soft-fail suppressed — leverage tolerance mismatch ignored (no −25 penalty).",
        "disabled_rules":            ["R6"],
        "disabled_flags":            [],
        "suitable_threshold":    _DEFAULT_SUITABLE,
        "conditional_threshold": _DEFAULT_CONDITIONAL,
        "is_control":            False,
    },
]

# ── Threshold axis ─────────────────────────────────────────────────────────────
# Tests sensitivity of the score → decision mapping.
# All rules and conflict flags remain at defaults.

THRESHOLD_CONFIGS: list[dict] = [
    {
        "id":          "T0",
        "axis":        "threshold",
        "label":       "Default Thresholds (control)",
        "description": "SUITABLE≥70, CONDITIONAL≥40. Equivalent to A0.",
        "disabled_rules":            [],
        "disabled_flags":            [],
        "suitable_threshold":    70,
        "conditional_threshold": 40,
        "is_control":            True,
    },
    {
        "id":          "T1",
        "axis":        "threshold",
        "label":       "Strict Thresholds",
        "description": "SUITABLE≥80, CONDITIONAL≥50. Raises bar — more decisions pushed to CONDITIONAL or UNSUITABLE.",
        "disabled_rules":            [],
        "disabled_flags":            [],
        "suitable_threshold":    80,
        "conditional_threshold": 50,
        "is_control":            False,
    },
    {
        "id":          "T2",
        "axis":        "threshold",
        "label":       "Lenient Thresholds",
        "description": "SUITABLE≥60, CONDITIONAL≥30. Lowers bar — more decisions pushed to SUITABLE.",
        "disabled_rules":            [],
        "disabled_flags":            [],
        "suitable_threshold":    60,
        "conditional_threshold": 30,
        "is_control":            False,
    },
]

# ── Master list ────────────────────────────────────────────────────────────────
ALL_CONFIGS: list[dict] = ARCH_CONFIGS + RULE_CONFIGS + THRESHOLD_CONFIGS

# Deduplicate controls so the runner doesn't run the same config three times.
# A0, R0, T0 are all functionally identical — keep only A0 as THE control.
_seen_ids: set[str] = set()
UNIQUE_CONFIGS: list[dict] = []
for _cfg in ALL_CONFIGS:
    # Mark R0 and T0 as non-control duplicates and skip them from UNIQUE list
    if _cfg["id"] in ("R0", "T0"):
        continue
    UNIQUE_CONFIGS.append(_cfg)

CONFIG_BY_ID: dict[str, dict] = {c["id"]: c for c in ALL_CONFIGS}


def get_config(config_id: str) -> dict:
    if config_id not in CONFIG_BY_ID:
        raise KeyError(f"Unknown ablation config id: '{config_id}'. "
                       f"Valid ids: {sorted(CONFIG_BY_ID)}")
    return CONFIG_BY_ID[config_id]


def list_config_ids() -> list[str]:
    return [c["id"] for c in UNIQUE_CONFIGS]
