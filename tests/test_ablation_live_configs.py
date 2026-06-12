"""
tests/test_ablation_live_configs.py
=====================================
Static tests for the live ablation configuration definitions.
Zero API calls. Zero LLM calls.

Run with:
  pytest tests/test_ablation_live_configs.py -v
"""

import pytest

from evaluation.ablation_study.live_configs import (
    LIVE_CONFIGS, LIVE_CONFIG_BY_ID,
    get_live_config, list_live_config_ids, CONTROL_CONFIG_ID,
)

EXPECTED_IDS = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7"]

VALID_VARIANTS = {
    "full", "no_verifier", "no_precheck", "no_audit",
    "llm_only_a3", "no_conflict_detector", "single_llm", "no_rule_engine",
}

REQUIRED_KEYS = {"id", "label", "description", "pipeline_variant", "is_control", "disabled"}


# ── Count / structure ──────────────────────────────────────────────────────────

def test_exactly_8_configs():
    assert len(LIVE_CONFIGS) == 8

def test_ids_are_p0_through_p7():
    assert [c["id"] for c in LIVE_CONFIGS] == EXPECTED_IDS

def test_exactly_one_control():
    controls = [c for c in LIVE_CONFIGS if c["is_control"]]
    assert len(controls) == 1
    assert controls[0]["id"] == "P0"

def test_control_constant_is_p0():
    assert CONTROL_CONFIG_ID == "P0"

@pytest.mark.parametrize("config", LIVE_CONFIGS, ids=EXPECTED_IDS)
def test_required_keys_present(config):
    assert REQUIRED_KEYS <= config.keys()

@pytest.mark.parametrize("config", LIVE_CONFIGS, ids=EXPECTED_IDS)
def test_pipeline_variant_valid(config):
    assert config["pipeline_variant"] in VALID_VARIANTS

def test_all_variants_used():
    used = {c["pipeline_variant"] for c in LIVE_CONFIGS}
    assert used == VALID_VARIANTS

def test_variants_unique():
    variants = [c["pipeline_variant"] for c in LIVE_CONFIGS]
    assert len(variants) == len(set(variants))

def test_p0_is_full_and_no_disabled():
    p0 = LIVE_CONFIG_BY_ID["P0"]
    assert p0["pipeline_variant"] == "full"
    assert p0["disabled"] == []

def test_p4_is_llm_only_a3():
    assert LIVE_CONFIG_BY_ID["P4"]["pipeline_variant"] == "llm_only_a3"

def test_p5_is_no_conflict_detector():
    assert LIVE_CONFIG_BY_ID["P5"]["pipeline_variant"] == "no_conflict_detector"

def test_p6_is_single_llm():
    assert LIVE_CONFIG_BY_ID["P6"]["pipeline_variant"] == "single_llm"

def test_p7_is_no_rule_engine():
    assert LIVE_CONFIG_BY_ID["P7"]["pipeline_variant"] == "no_rule_engine"

def test_get_live_config_works():
    for cid in EXPECTED_IDS:
        assert get_live_config(cid)["id"] == cid

def test_get_live_config_raises_on_unknown():
    with pytest.raises(KeyError):
        get_live_config("C0")   # old ID no longer valid

def test_list_ids():
    assert list_live_config_ids() == EXPECTED_IDS


# ── Graph builders exist in graph.py ──────────────────────────────────────────

def test_run_pipeline_ablated_importable():
    from orchestrator.graph import run_pipeline_ablated
    assert callable(run_pipeline_ablated)

def test_all_graph_builders_exist():
    from orchestrator.graph import (
        build_graph,
        build_graph_c1_no_verifier,
        build_graph_c2_no_precheck,
        build_graph_c3_no_audit,
        build_graph_p4_llm_only_a3,
        build_graph_c4_no_conflict_detector,
        build_graph_p7_no_rule_engine,
    )
    for fn in [build_graph, build_graph_c1_no_verifier, build_graph_c2_no_precheck,
               build_graph_c3_no_audit, build_graph_p4_llm_only_a3,
               build_graph_c4_no_conflict_detector, build_graph_p7_no_rule_engine]:
        assert callable(fn)

def test_graph_dispatcher_has_all_variants():
    from orchestrator.graph import _GRAPH_BUILDERS
    for variant in VALID_VARIANTS - {"single_llm"}:
        assert variant in _GRAPH_BUILDERS, f"Missing variant in _GRAPH_BUILDERS: {variant}"

def test_llm_only_a3_node_importable():
    from orchestrator.graph import _node_a3_llm_only
    assert callable(_node_a3_llm_only)

def test_llm_only_a3_system_prompt_exists():
    from orchestrator.graph import _LLM_ONLY_A3_SYSTEM_PROMPT
    assert "R1" in _LLM_ONLY_A3_SYSTEM_PROMPT
    assert "R7" in _LLM_ONLY_A3_SYSTEM_PROMPT
    assert "UNSUITABLE" in _LLM_ONLY_A3_SYSTEM_PROMPT
    assert "SUITABLE" in _LLM_ONLY_A3_SYSTEM_PROMPT
    assert "score" in _LLM_ONLY_A3_SYSTEM_PROMPT


# ── Runner importable ──────────────────────────────────────────────────────────

def test_live_runner_functions_importable():
    from evaluation.ablation_study.live_runner import (
        run_cell_live, run_all_live, load_live_results,
        compute_live_metrics, compute_verifier_impact, compute_llm_a3_impact,
    )
    for fn in [run_cell_live, run_all_live, load_live_results,
               compute_live_metrics, compute_verifier_impact, compute_llm_a3_impact]:
        assert callable(fn)

def test_compute_ablation_results_importable():
    import compute_ablation_results
    assert callable(compute_ablation_results.main)
    assert callable(compute_ablation_results.load_results)
    assert callable(compute_ablation_results.compute_all_metrics)
