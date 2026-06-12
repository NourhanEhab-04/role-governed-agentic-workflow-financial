"""
evaluation/ablation_study/live_configs.py
==========================================
Defines the 8 live ablation configurations for the MiFID II pipeline study.

Each config disables exactly ONE architectural component so its individual
contribution can be measured cleanly.

| ID | Variant              | What changes                         | What it measures                    |
|----|----------------------|--------------------------------------|-------------------------------------|
| P0 | Full Pipeline        | Nothing removed                      | Reference ceiling                   |
| P1 | No Verifier          | A1/A2 run without AV check/correct   | Extraction reliability              |
| P2 | No pre_check         | First deterministic rule call removed| Early deterministic checkpoint      |
| P3 | No audit             | Third deterministic rule call removed| Deterministic backstop value        |
| P4 | LLM-only A3          | A3 reasons without tool call         | Prohibition-by-design value         |
| P5 | No A4                | Conflict detector removed            | Safety / escalation value           |
| P6 | Single LLM Baseline  | Entire pipeline replaced             | Overall architecture contribution   |
| P7 | No Rule Engine       | No deterministic Python calls at all | Full LLM reasoning vs. hybrid arch  |

pipeline_variant keys map to graph builders in orchestrator/graph.py _GRAPH_BUILDERS.
"""

from __future__ import annotations

LIVE_CONFIGS: list[dict] = [
    {
        "id":               "P0",
        "label":            "Full Pipeline",
        "description":      (
            "All components active: A1 → AV → A2 → AV → pre_check → A3(tool) → audit → A4 → A5. "
            "Reference configuration. All deterministic guards and LLM agents on."
        ),
        "pipeline_variant": "full",
        "is_control":       True,
        "disabled":         [],
    },
    {
        "id":               "P1",
        "label":            "No Verifier",
        "description":      (
            "AV (verifier) feedback loop disabled for both A1 and A2. "
            "Agents extract profiles with no cross-check and no correction loop. "
            "Deterministic consistency overrides still apply afterward. "
            "Measures: how much does LLM-as-judge verification improve extraction accuracy?"
        ),
        "pipeline_variant": "no_verifier",
        "is_control":       False,
        "disabled":         ["AV verify A1", "AV verify A2", "A1 correction loop", "A2 correction loop"],
    },
    {
        "id":               "P2",
        "label":            "No pre_check",
        "description":      (
            "pre_check node skipped — the first deterministic call to evaluate_suitability() "
            "does not run before A3. The three-point integrity architecture is incomplete "
            "(only A3 tool + audit remain). A3 still calls evaluate_suitability_tool. "
            "Measures: does the early deterministic checkpoint add value?"
        ),
        "pipeline_variant": "no_precheck",
        "is_control":       False,
        "disabled":         ["pre_check node (first deterministic rule call)"],
    },
    {
        "id":               "P3",
        "label":            "No audit",
        "description":      (
            "audit node skipped — the third deterministic call to evaluate_suitability() "
            "does not run before A4. A4 receives no audit_verdict in state. "
            "pre_check and A3 tool still run. "
            "Measures: does the post-A3 deterministic backstop add value?"
        ),
        "pipeline_variant": "no_audit",
        "is_control":       False,
        "disabled":         ["audit node (third deterministic rule call)"],
    },
    {
        "id":               "P4",
        "label":            "LLM-only A3",
        "description":      (
            "A3 receives client_profile and product_profile but has NO evaluate_suitability_tool "
            "registered. The LLM must reason about all 7 MiFID II rules from first principles "
            "using the rule definitions in its system prompt. "
            "pre_check and audit still run so we can measure disagreement rate. "
            "Disagreement is LOGGED but LLM verdict is KEPT (not replaced by pre_check). "
            "Measures: value of the tool-call prohibition-by-design constraint."
        ),
        "pipeline_variant": "llm_only_a3",
        "is_control":       False,
        "disabled":         ["evaluate_suitability_tool (A3 forced to reason itself)"],
    },
    {
        "id":               "P5",
        "label":            "No A4",
        "description":      (
            "A4 replaced with a passthrough node that returns an empty conflict report "
            "with no flags and escalate=False. The final decision is always the raw rule "
            "engine verdict — ESCALATED is never produced. "
            "Measures: contribution of conflict detection and escalation safety layer."
        ),
        "pipeline_variant": "no_conflict_detector",
        "is_control":       False,
        "disabled":         ["A4 conflict detector", "all conflict flags", "escalation"],
    },
    {
        "id":               "P6",
        "label":            "Single LLM Baseline",
        "description":      (
            "The entire multi-agent pipeline replaced by a single LLM call. "
            "The model receives raw client + product text and applies all 7 MiFID II "
            "rules from its own reasoning with no tool calls, no verifier, no rule engine, "
            "and no conflict detector. Same model as pipeline agents. "
            "Measures: overall architecture contribution vs monolithic LLM."
        ),
        "pipeline_variant": "single_llm",
        "is_control":       False,
        "disabled":         ["A1", "AV", "A2", "pre_check", "A3 tool", "audit", "A4", "A5",
                             "rule engine", "conflict detector"],
    },
    {
        "id":               "P7",
        "label":            "No Rule Engine",
        "description":      (
            "Zero deterministic Python evaluate_suitability() calls anywhere in the pipeline. "
            "pre_check skipped, audit skipped, A3 uses LLM-only reasoning (no tool). "
            "A1, A2, AV, A4, A5 all still run. "
            "Measures: contribution of ALL deterministic rule-engine checkpoints combined."
        ),
        "pipeline_variant": "no_rule_engine",
        "is_control":       False,
        "disabled":         ["pre_check", "audit", "evaluate_suitability_tool in A3",
                             "all deterministic Python rule calls"],
    },
]

# Index for fast lookup
LIVE_CONFIG_BY_ID: dict[str, dict] = {c["id"]: c for c in LIVE_CONFIGS}

CONTROL_CONFIG_ID = "P0"


def get_live_config(config_id: str) -> dict:
    if config_id not in LIVE_CONFIG_BY_ID:
        raise KeyError(
            f"Unknown live ablation config: '{config_id}'. "
            f"Valid ids: {[c['id'] for c in LIVE_CONFIGS]}"
        )
    return LIVE_CONFIG_BY_ID[config_id]


def list_live_config_ids() -> list[str]:
    return [c["id"] for c in LIVE_CONFIGS]
