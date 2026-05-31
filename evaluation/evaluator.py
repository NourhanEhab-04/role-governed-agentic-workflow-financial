"""
evaluation/evaluator.py
========================
Main evaluation driver for the thesis comparison.

Runs every scenario in data/scenarios/ through:
  Architecture A — baseline single LLM (evaluation/baseline_single_llm.py)
  Architecture B — full multi-agent pipeline  (orchestrator/graph.py)

Each scenario is run EVAL_RUNS_PER_SCENARIO times (default 3) to measure
decision consistency (M3) under LLM non-determinism.

Results are saved to:
  data/eval/results_baseline.json
  data/eval/results_pipeline.json
  data/eval/metrics_comparison.json   ← the table for the thesis

Usage
-----
  python -m evaluation.evaluator
  python -m evaluation.evaluator --runs 5          # override run count
  python -m evaluation.evaluator --scenarios 01 03  # run only specific scenarios
"""

import asyncio
import json
import argparse
import datetime
import sys
from pathlib import Path

from config.llm_config import get_model_client
from config.settings import EVAL_RUNS_PER_SCENARIO
from evaluation.baseline_single_llm import run_baseline
from evaluation.metrics import compute_all_metrics
from evaluation.governance_metrics import compute_governance_metrics
from rule_engine.rule_engine import evaluate_suitability


class EvalHaltError(Exception):
    """Raised when the evaluation must stop due to a rate/token limit."""


_LIMIT_KEYWORDS = (
    "rate limit", "rate_limit", "ratelimit",
    "429", "too many requests",
    "tokens per minute", "token limit",
    "context length exceeded", "maximum context",
    "quota exceeded", "insufficient_quota",
)


def _is_limit_error(exc: Exception) -> bool:
    """Return True if the exception signals a rate-limit or token-limit."""
    msg = str(exc).lower()
    if any(kw in msg for kw in _LIMIT_KEYWORDS):
        return True
    # Also check by exception class name (works without importing openai directly)
    cls = type(exc).__name__.lower()
    return "ratelimit" in cls or "tokenlimit" in cls or "quotaexceeded" in cls


SCENARIOS_DIR = Path("data/scenarios")
PRODUCTS_DIR  = Path("data/products")
EVAL_OUT_DIR  = Path("data/eval")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_scenario(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_product(product_file: str) -> dict:
    return json.loads((PRODUCTS_DIR / product_file).read_text(encoding="utf-8"))


def expected_rules_for_scenario(scenario: dict, product: dict) -> dict:
    """
    Derive expected per-rule PASS/FAIL from expected_rules_failed.
    All rules not in expected_rules_failed are PASS.
    """
    RULE_IDS = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
    failed   = set(scenario.get("expected_rules_failed", []))
    return {r: ("FAIL" if r in failed else "PASS") for r in RULE_IDS}


def _extract_failed_rules_from_pipeline(state: dict) -> list[str]:
    rv = state.get("rule_verdict", {})
    if isinstance(rv.get("rules"), dict):
        return [k for k, v in rv["rules"].items() if v == "FAIL"]
    return []


def _extract_pipeline_result(state: dict, scenario: dict, product: dict) -> dict:
    rv = state.get("rule_verdict", {}) or {}
    sr = state.get("suitability_report", {}) or {}
    cf = state.get("conflict_report", {}) or {}

    pipeline_halted  = bool(state.get("halt", False))
    output_decision  = (
        "HALT"
        if pipeline_halted
        else (sr.get("decision") or rv.get("decision") or "UNKNOWN")
    )
    output_escalated = bool(state.get("escalated", False))

    # A1/A2 verification tracking (M5, existing)
    a1_ver = state.get("a1_verification") or {}
    a2_ver = state.get("a2_verification") or {}
    a1_cor = state.get("a1_correction") or {}
    a2_cor = state.get("a2_correction") or {}

    # ── M8: Override tracking ────────────────────────────────────────────────
    a1_prov = state.get("client_profile_provenance") or {}
    a2_prov = state.get("product_profile_provenance") or {}
    a1_overrides_applied = bool(a1_prov.get("deterministic_overrides"))
    a2_overrides_applied = bool(a2_prov.get("deterministic_overrides"))
    # A4 overrides are logged to error_chain by the graph node
    error_chain = state.get("_error_chain") or []
    a4_overrides_applied = any(
        e.get("stage") == "A4" and e.get("event") == "deterministic_override"
        for e in error_chain
    )

    # ── M10: Three-point integrity ───────────────────────────────────────────
    audit_verdict  = state.get("audit_verdict") or {}
    three_point_agreed = bool(audit_verdict.get("agreed", False))

    # ── M11: Hard rule enforcement ────────────────────────────────────────────
    output_hard_failed_rules = rv.get("hard_failed_rules", [])

    # ── M12: Vulnerability protection ─────────────────────────────────────────
    client_profile = state.get("client_profile") or {}
    client_vulnerability = client_profile.get("financial_vulnerability")

    # ── M13: Regulatory citation ──────────────────────────────────────────────
    regulatory_basis = sr.get("regulatory_basis")

    # ── M14: Explanation completeness ─────────────────────────────────────────
    rule_findings = sr.get("rule_findings") or []
    output_rule_explanations = {
        f["rule_id"]: f.get("explanation", "")
        for f in rule_findings
        if isinstance(f, dict) and "rule_id" in f
    }

    # ── M15: Decision traceability components ─────────────────────────────────
    pre_check_present    = bool(state.get("pre_check_verdict"))
    rule_details_present = bool(rv.get("rule_details"))
    conflict_present     = bool(state.get("conflict_report"))

    return {
        # Existing M1-M7 fields
        "output_decision":    output_decision,
        "expected_decision":  scenario.get("expected_decision"),
        "output_escalated":   output_escalated,
        "expected_escalate":  scenario.get("expected_escalate", False),
        "output_failed_rules":    _extract_failed_rules_from_pipeline(state),
        "expected_rules_failed":  scenario.get("expected_rules_failed", []),
        "output_rules":           rv.get("rules", {}),
        "expected_rules":         expected_rules_for_scenario(scenario, product),
        "a1_verified":   bool(a1_ver),
        "a2_verified":   bool(a2_ver),
        "a1_corrected":  bool(a1_cor.get("corrected")),
        "a2_corrected":  bool(a2_cor.get("corrected")),
        # Full correction dicts for M5b correction_precision (original + corrected profiles)
        "a1_correction": a1_cor if a1_cor else None,
        "a2_correction": a2_cor if a2_cor else None,
        "halted":        pipeline_halted,
        "halt_reason":   state.get("halt_reason"),
        # A1/A2 extraction output (used for M_A1 / M_A2 accuracy when scenario has expected profiles)
        "output_client_profile":  state.get("client_profile") or {},
        "output_product_profile": state.get("product_profile") or {},
        # New M8-M15 fields
        "a1_overrides_applied":    a1_overrides_applied,
        "a2_overrides_applied":    a2_overrides_applied,
        "a4_overrides_applied":    a4_overrides_applied,
        "three_point_agreed":      three_point_agreed,
        "output_hard_failed_rules": output_hard_failed_rules,
        "client_vulnerability":    client_vulnerability,
        "regulatory_basis":        regulatory_basis,
        "output_rule_explanations": output_rule_explanations,
        "pre_check_present":       pre_check_present,
        "rule_details_present":    rule_details_present,
        "conflict_present":        conflict_present,
    }


def _extract_baseline_result(baseline: dict, scenario: dict, product: dict) -> dict:
    output_rules = baseline.get("rules", {})
    failed_rules = [k for k, v in output_rules.items() if v == "FAIL"]
    return {
        "output_decision":        baseline.get("decision", "UNKNOWN"),
        "expected_decision":      scenario.get("expected_decision"),
        "output_escalated":       False,   # baseline has no escalation logic
        "expected_escalate":      scenario.get("expected_escalate", False),
        "output_failed_rules":    failed_rules,
        "expected_rules_failed":  scenario.get("expected_rules_failed", []),
        "output_rules":           output_rules,
        "expected_rules":         expected_rules_for_scenario(scenario, product),
        "a1_verified":  False,
        "a2_verified":  False,
        "a1_corrected": False,
        "a2_corrected": False,
        "halted":       baseline.get("decision") == "PARSE_ERROR",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single-scenario runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_scenario_once(
    scenario: dict,
    product: dict,
    model_client,
    architecture: str,      # "pipeline" or "baseline"
) -> dict:
    """Run one scenario once through the chosen architecture."""
    # Use narrative text when present so A1/A2 face real NLP extraction work.
    # Fall back to json.dumps of the structured dict for non-narrative scenarios.
    client_text  = scenario.get("client_narrative") or json.dumps(scenario["client"])
    product_text = scenario.get("product_narrative") or json.dumps(product)

    if architecture == "baseline":
        result = await run_baseline(client_text, product_text, model_client)
        result = _extract_baseline_result(result, scenario, product)
    else:
        # Pipeline
        from orchestrator.graph import run_pipeline
        state, _audit = await run_pipeline(
            client_input=client_text,
            product_input=product_text,
            model_client=model_client,
        )
        result = _extract_pipeline_result(state, scenario, product)

    # Attach expected extraction targets when the scenario defines them
    result["expected_client_profile"]  = scenario.get("expected_client_profile")
    result["expected_product_profile"] = scenario.get("expected_product_profile")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

async def evaluate_architecture(
    architecture: str,
    scenario_files: list[Path],
    runs_per_scenario: int,
    model_client,
) -> list[dict]:
    """Run all scenarios × runs_per_scenario and return a flat results list."""
    all_results: list[dict] = []
    total = len(scenario_files) * runs_per_scenario

    print(f"\n{'='*60}")
    print(f"  Architecture: {architecture.upper()}")
    print(f"  Scenarios: {len(scenario_files)}  ×  Runs: {runs_per_scenario} = {total}")
    print(f"{'='*60}")

    for i, sc_path in enumerate(sorted(scenario_files), 1):
        scenario = load_scenario(sc_path)
        product  = load_product(scenario["product_file"])
        sc_id    = sc_path.stem

        print(f"  [{i}/{len(scenario_files)}] {sc_id}: {scenario.get('description', '')[:60]}")

        for run_n in range(1, runs_per_scenario + 1):
            try:
                result = await run_scenario_once(
                    scenario, product, model_client, architecture
                )
                result["scenario_id"]  = sc_id
                result["run_number"]   = run_n
                result["architecture"] = architecture
                if "pair_id" in scenario:
                    result["pair_id"] = scenario["pair_id"]
                if "demographic_variant" in scenario:
                    result["demographic_variant"] = scenario["demographic_variant"]
                symbol = "✓" if result["output_decision"] == result["expected_decision"] else "✗"
                print(f"      run {run_n}: {result['output_decision']} {symbol} "
                      f"(expected {result['expected_decision']})")
                all_results.append(result)
            except Exception as exc:
                if _is_limit_error(exc):
                    banner = "!" * 60
                    print(f"\n{banner}")
                    print(f"  HALT — API rate/token limit reached")
                    print(f"  Architecture : {architecture.upper()}")
                    print(f"  Scenario     : {sc_id}  (#{i}/{len(scenario_files)})")
                    print(f"  Run          : {run_n}/{runs_per_scenario}")
                    print(f"  Reason       : {exc}")
                    print(f"  Results so far: {len(all_results)} collected — stopping now.")
                    print(f"{banner}\n")
                    raise EvalHaltError(str(exc)) from exc

                print(f"      run {run_n}: ERROR — {exc}")
                err_result: dict = {
                    "scenario_id":        sc_id,
                    "run_number":         run_n,
                    "architecture":       architecture,
                    "output_decision":    "ERROR",
                    "expected_decision":  scenario.get("expected_decision"),
                    "output_escalated":   False,
                    "expected_escalate":  scenario.get("expected_escalate", False),
                    "output_failed_rules":   [],
                    "expected_rules_failed": scenario.get("expected_rules_failed", []),
                    "output_rules":       {},
                    "expected_rules":     expected_rules_for_scenario(scenario, product),
                    "error":              str(exc),
                }
                if "pair_id" in scenario:
                    err_result["pair_id"] = scenario["pair_id"]
                if "demographic_variant" in scenario:
                    err_result["demographic_variant"] = scenario["demographic_variant"]
                all_results.append(err_result)

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main(runs: int, scenario_filter: list[str] | None, pipeline_only: str | None) -> None:
    EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_client = get_model_client()

    # Run DB migrations if DATABASE_URL is configured (idempotent — safe every startup)
    import os
    if os.environ.get("DATABASE_URL"):
        from orchestrator.db import apply_migrations
        try:
            await apply_migrations()
            print("PostgreSQL: migrations applied.")
        except Exception as exc:
            print(f"WARNING: PostgreSQL migration failed — audit will not be persisted. ({exc})")

    # Collect scenario files
    all_files = sorted(SCENARIOS_DIR.glob("*.json"))
    if scenario_filter:
        all_files = [f for f in all_files
                     if any(f.stem.startswith(s) for s in scenario_filter)]
    if not all_files:
        print("No scenario files found.")
        return

    # Run both architectures — stop immediately if a rate/token limit is hit
    baseline_results: list[dict] = []
    pipeline_results: list[dict] = []

    if pipeline_only:
        baseline_path = Path(pipeline_only)
        if not baseline_path.exists():
            print(f"ERROR: baseline file not found: {baseline_path}")
            sys.exit(1)
        baseline_results = json.loads(baseline_path.read_text(encoding="utf-8"))
        print(f"Loaded {len(baseline_results)} baseline results from {baseline_path}")

    try:
        if not pipeline_only:
            baseline_results = await evaluate_architecture(
                "baseline", all_files, runs, model_client
            )
        pipeline_results = await evaluate_architecture(
            "pipeline", all_files, runs, model_client
        )
    except EvalHaltError as halt:
        print(f"Evaluation halted: {halt}")
        print("Saving partial results and exiting.")
        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        if baseline_results:
            (EVAL_OUT_DIR / f"results_baseline_partial_{ts}.json").write_text(
                json.dumps(baseline_results, indent=2)
            )
        if pipeline_results:
            (EVAL_OUT_DIR / f"results_pipeline_partial_{ts}.json").write_text(
                json.dumps(pipeline_results, indent=2)
            )
        sys.exit(1)

    # Save raw results
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    (EVAL_OUT_DIR / f"results_baseline_{ts}.json").write_text(
        json.dumps(baseline_results, indent=2)
    )
    (EVAL_OUT_DIR / f"results_pipeline_{ts}.json").write_text(
        json.dumps(pipeline_results, indent=2)
    )

    # Compute and save metrics (M1-M7 + M8-M15)
    baseline_metrics    = compute_all_metrics(baseline_results, "baseline")
    pipeline_metrics    = compute_all_metrics(pipeline_results, "pipeline")
    baseline_governance = compute_governance_metrics(baseline_results, "baseline")
    pipeline_governance = compute_governance_metrics(pipeline_results, "pipeline")

    comparison = {
        "timestamp":        ts,
        "runs_per_scenario": runs,
        "n_scenarios":      len(all_files),
        "baseline":         {**baseline_metrics, **baseline_governance},
        "pipeline":         {**pipeline_metrics, **pipeline_governance},
    }

    metrics_path = EVAL_OUT_DIR / f"metrics_comparison_{ts}.json"
    metrics_path.write_text(json.dumps(comparison, indent=2))

    # Print comparison table
    print(f"\n{'='*60}")
    print("  METRICS COMPARISON (M1-M7: accuracy / M8-M15: governance)")
    print(f"{'='*60}")
    labels = {
        "M1_decision_accuracy":    "M1 Decision Accuracy",
        "M2_rule_compliance_rate": "M2 Rule Compliance",
        "M3_decision_consistency": "M3 Consistency",
    }
    print(f"  {'Metric':<35} {'Baseline':>10} {'Pipeline':>10}")
    print(f"  {'-'*57}")
    for key, label in labels.items():
        b = baseline_metrics.get(key, "—")
        p = pipeline_metrics.get(key, "—")
        print(f"  {label:<35} {b:>10} {p:>10}")
    # M4 is now a dict — print its sub-metrics separately
    print(f"\n  M4 Escalation Accuracy (completed runs only):")
    for arch, metrics in [("Baseline", baseline_metrics), ("Pipeline", pipeline_metrics)]:
        m4 = metrics.get("M4_escalation_accuracy", {})
        if isinstance(m4, dict):
            print(f"    {arch}: overall={m4.get('overall_accuracy','—')}  "
                  f"sensitivity={m4.get('sensitivity','—')}  "
                  f"specificity={m4.get('specificity','—')}  "
                  f"(n_pos={m4.get('n_positive','—')}, n_neg={m4.get('n_negative','—')})")
        else:
            print(f"    {arch}: {m4}")
    print(f"\n  M5 Verification Correction Rate (pipeline only):")
    m5 = pipeline_metrics.get("M5_verification", {})
    if isinstance(m5, dict):
        for k, v in m5.items():
            print(f"    {k}: {v:.4f}")
    print(f"\n  M_A1 Client Extraction Accuracy (narrative scenarios only):")
    ma1 = pipeline_metrics.get("M_A1_extraction", {})
    if isinstance(ma1, dict) and ma1.get("n_evaluated", 0) > 0:
        print(f"    n_evaluated={ma1['n_evaluated']}  exact_match={ma1.get('overall_exact_match_rate', '—'):.4f}")
        for field, acc in ma1.get("per_field_accuracy", {}).items():
            print(f"      {field}: {acc:.4f}" if acc is not None else f"      {field}: —")
    else:
        print(f"    no narrative scenarios with expected_client_profile found")
    print(f"\n  M_A2 Product Extraction Accuracy (narrative scenarios only):")
    ma2 = pipeline_metrics.get("M_A2_extraction", {})
    if isinstance(ma2, dict) and ma2.get("n_evaluated", 0) > 0:
        print(f"    n_evaluated={ma2['n_evaluated']}  exact_match={ma2.get('overall_exact_match_rate', '—'):.4f}")
        for field, acc in ma2.get("per_field_accuracy", {}).items():
            print(f"      {field}: {acc:.4f}" if acc is not None else f"      {field}: —")
    else:
        print(f"    no narrative scenarios with expected_product_profile found")
    print(f"\n  Governance / Compliance / Explainability:")
    gov_labels = {
        "M8_override_rate":             "M8 Override Rate",
        "M9_pipeline_halt_rate":        "M9 Halt Rate",
        "M10_three_point_integrity":    "M10 Three-Point Integrity",
        "M11_hard_rule_enforcement":    "M11 Hard Rule Enforcement",
        "M12_vulnerability_protection": "M12 Vulnerability Protection",
        "M13_regulatory_citation_rate": "M13 Regulatory Citation",
        "M14_explanation_completeness": "M14 Explanation Completeness",
        "M15_decision_traceability":    "M15 Decision Traceability",
    }
    for key, label in gov_labels.items():
        b = baseline_governance.get(key, "—")
        p = pipeline_governance.get(key, "—")
        b_s = str(b) if not isinstance(b, float) else f"{b:.4f}"
        p_s = str(p) if not isinstance(p, float) else f"{p:.4f}"
        print(f"  {label:<35} {b_s:>10} {p_s:>10}")
    print(f"\n  Metrics saved → {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiFID II evaluation runner")
    parser.add_argument("--runs",      type=int,  default=EVAL_RUNS_PER_SCENARIO)
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="Scenario ID prefixes to run (e.g. 01 03 07)")
    parser.add_argument("--pipeline-only", metavar="BASELINE_FILE", default=None,
                        help="Skip baseline; load results from BASELINE_FILE and run pipeline only")
    args = parser.parse_args()
    asyncio.run(main(args.runs, args.scenarios, args.pipeline_only))
