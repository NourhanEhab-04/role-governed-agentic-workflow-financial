"""
evaluation/ablation_study/live_runner.py
=========================================
Async runner for the LIVE ablation study.

PERSISTENCE BEHAVIOUR
---------------------
Files are NEVER overwritten by default.
Each cell result is written to:
  data/ablation_results/live/{scenario_id}__{config_id}.json

If that file already exists the cell is SKIPPED — this means you can run
the study in multiple partial batches (e.g. 10 scenarios at a time) and
results accumulate safely across runs.

Pass overwrite=True (or --overwrite on the CLI) to re-run existing cells.

RESULT FORMAT per cell
----------------------
{
  "scenario_id":        str
  "scenario_file":      str
  "category":           str   — expected_decision (SUITABLE/CONDITIONAL/UNSUITABLE/ESCALATED)
  "config_id":          str
  "config_label":       str
  "pipeline_variant":   str
  "final_decision":     str
  "expected_decision":  str
  "expected_escalate":  bool
  "output_escalate":    bool
  "decision_correct":   bool
  "escalation_correct": bool
  "score":              int
  "rules_failed":       list[str]
  "rule_decision":      str
  "conflict_flags":     list[str]
  "verifier":           dict
  "consistency":        dict
  "llm_only_a3":        dict   — only present for P4/P7
  "halted":             bool
  "halt_reason":        str | None
  "warnings":           list
  "audit_summary":      dict
  "error":              str | None
}
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from config.llm_config import get_model_client
from evaluation.ablation_study.live_configs import LIVE_CONFIGS, get_live_config
from orchestrator.graph import run_pipeline_ablated

LIVE_RESULTS_DIR = Path("data/ablation_results/live")


# ── Input formatting ───────────────────────────────────────────────────────────

def _scenario_to_inputs(scenario: dict) -> tuple[str, str, int | None]:
    """
    Convert a scenario dict to (client_input, product_input, portfolio_concentration_pct).

    Raises ValueError if either narrative is missing — the JSON-profile fallback
    is intentionally absent.  Passing ground-truth structured data to the LLM
    makes extraction trivially correct and defeats the ablation study.

    portfolio_concentration_pct is the one exception: it comes from the structured
    client block because the narratives do not always quantify it.  It is injected
    into pipeline *state* directly, not into the LLM prompt.
    """
    scenario_id = Path(scenario.get("_filename", "unknown")).stem

    client_input = scenario.get("client_narrative")
    if not client_input:
        raise ValueError(
            f"Scenario '{scenario_id}' is missing client_narrative. "
            "Add a narrative or remove the scenario from data/scenarios/."
        )

    product_input = scenario.get("product_narrative")
    if not product_input:
        raise ValueError(
            f"Scenario '{scenario_id}' is missing product_narrative. "
            "Add a narrative or remove the scenario from data/scenarios/."
        )

    # portfolio_concentration_pct: injected into state, NOT into the LLM prompt
    portfolio_concentration_pct: int | None = scenario.get("client", {}).get("portfolio_concentration_pct")

    return client_input, product_input, portfolio_concentration_pct


# ── Result extraction helpers ──────────────────────────────────────────────────

def _extract_final_decision(final_state: dict) -> str:
    if final_state.get("halt"):
        return "ERROR"
    report = final_state.get("suitability_report") or {}
    if report.get("decision"):
        return report["decision"]
    if final_state.get("escalated"):
        return "ESCALATED"
    verdict = final_state.get("rule_verdict") or {}
    if verdict.get("decision"):
        return verdict["decision"]
    return "ERROR"


def _extract_escalated(final_state: dict) -> bool:
    if final_state.get("halt"):
        return False
    report = final_state.get("suitability_report") or {}
    if report.get("decision") == "ESCALATED":
        return True
    if final_state.get("escalated"):
        return True
    conflict = final_state.get("conflict_report") or {}
    return bool(conflict.get("escalate"))


def _extract_verifier_telemetry(final_state: dict) -> dict:
    a1_ver  = final_state.get("a1_verification") or {}
    a2_ver  = final_state.get("a2_verification") or {}
    a1_corr = final_state.get("a1_correction")
    a2_corr = final_state.get("a2_correction")
    a1_correction_applied = bool(a1_corr and a1_corr.get("corrected") is not None)
    a2_correction_applied = bool(a2_corr and a2_corr.get("corrected") is not None)
    return {
        "a1_verifier_ran":       a1_ver.get("passed") is not None,
        "a1_verifier_passed":    a1_ver.get("passed"),
        "a1_correction_applied": a1_correction_applied,
        "a1_fields_fixed":       (a1_corr or {}).get("fields_fixed", []),
        "a2_verifier_ran":       a2_ver.get("passed") is not None,
        "a2_verifier_passed":    a2_ver.get("passed"),
        "a2_correction_applied": a2_correction_applied,
        "a2_fields_fixed":       (a2_corr or {}).get("fields_fixed", []),
        "any_correction":        a1_correction_applied or a2_correction_applied,
    }


def _extract_consistency_telemetry(final_state: dict) -> dict:
    prov_a1 = final_state.get("client_profile_provenance") or {}
    prov_a2 = final_state.get("product_profile_provenance") or {}
    return {
        "a1_overrides": prov_a1.get("deterministic_overrides", []),
        "a2_overrides": prov_a2.get("deterministic_overrides", []),
        "any_override": bool(
            prov_a1.get("deterministic_overrides") or
            prov_a2.get("deterministic_overrides")
        ),
    }


def _extract_llm_only_a3_telemetry(final_state: dict) -> dict:
    """Extra telemetry only relevant for P4/P7 (LLM-only A3)."""
    a3_reasoning = final_state.get("a3_reasoning") or {}
    if a3_reasoning.get("ablation") != "llm_only_a3":
        return {}
    # Look for disagreement warnings
    warnings = final_state.get("_warnings") or []
    disagreements = [
        w for w in warnings
        if w.get("event") == "llm_only_a3_disagrees_with_precheck"
    ]
    return {
        "agreed_with_precheck": a3_reasoning.get("agreed_with_precheck", True),
        "disagreements": disagreements,
    }


# ── Single cell ────────────────────────────────────────────────────────────────

async def run_cell_live(
    scenario: dict,
    config: dict,
    model_client,
    semaphore: asyncio.Semaphore | None = None,
    overwrite: bool = False,
    output_dir: Path = LIVE_RESULTS_DIR,
) -> dict | None:
    """
    Run one (scenario, config) cell with real API calls.

    Returns the result dict, or None if the cell was skipped (file exists).
    Writes result to output_dir/{scenario_id}__{config_id}.json.
    Never overwrites unless overwrite=True.
    """
    scenario_id  = Path(scenario["_filename"]).stem
    config_id    = config["id"]
    cell_path    = output_dir / f"{scenario_id}__{config_id}.json"

    # Skip if already done
    if cell_path.exists() and not overwrite:
        return None

    expected_dec = scenario.get("expected_decision", "")
    expected_esc = bool(scenario.get("expected_escalate", expected_dec == "ESCALATED"))

    async def _run():
        client_input, product_input, concentration_pct = _scenario_to_inputs(scenario)
        final_state, audit_log = await run_pipeline_ablated(
            client_input=client_input,
            product_input=product_input,
            model_client=model_client,
            pipeline_variant=config["pipeline_variant"],
            portfolio_concentration_pct=concentration_pct,  # passed separately, not in input
        )
        return final_state, audit_log

    try:
        if semaphore:
            async with semaphore:
                final_state, audit_log = await _run()
        else:
            final_state, audit_log = await _run()

        final_decision  = _extract_final_decision(final_state)
        output_escalate = _extract_escalated(final_state)
        verifier_tel    = _extract_verifier_telemetry(final_state)
        consistency_tel = _extract_consistency_telemetry(final_state)
        llm_a3_tel      = _extract_llm_only_a3_telemetry(final_state)

        rule_verdict    = final_state.get("rule_verdict") or {}
        conflict_report = final_state.get("conflict_report") or {}

        result = {
            # ── Identity ──────────────────────────────────────────────────────
            "scenario_id":        scenario_id,
            "scenario_file":      scenario.get("_path", scenario["_filename"]),
            "category":           expected_dec,
            "config_id":          config_id,
            "config_label":       config["label"],
            "pipeline_variant":   config["pipeline_variant"],

            # ── Decision outcome ──────────────────────────────────────────────
            "final_decision":     final_decision,
            "expected_decision":  expected_dec,
            "expected_escalate":  expected_esc,
            "output_escalate":    output_escalate,
            "decision_correct":   (final_decision == expected_dec),
            "escalation_correct": (output_escalate == expected_esc),

            # ── Rule engine data ──────────────────────────────────────────────
            "score":           rule_verdict.get("score", -1),
            "rules_failed":    rule_verdict.get("hard_failed_rules", []),
            "rule_decision":   rule_verdict.get("decision", ""),

            # ── Conflict detector data ────────────────────────────────────────
            "conflict_flags": [
                f.get("rule_id") for f in conflict_report.get("flags", [])
                if f.get("triggered")
            ],

            # ── Component telemetry ───────────────────────────────────────────
            "verifier":    verifier_tel,
            "consistency": consistency_tel,
            "llm_only_a3": llm_a3_tel,   # empty dict for P0-P3, P5-P6

            # ── Pipeline health ───────────────────────────────────────────────
            "halted":     bool(final_state.get("halt")),
            "halt_reason": final_state.get("halt_reason"),
            "warnings":   final_state.get("_warnings", []),

            "audit_summary": {
                "assessment_id":      audit_log.get("assessment_id"),
                "model_version":      audit_log.get("model_version"),
                "rule_engine_version": audit_log.get("rule_engine_version"),
            },

            "error": None,
        }

    except Exception as exc:
        result = {
            "scenario_id":        scenario_id,
            "scenario_file":      scenario.get("_path", scenario["_filename"]),
            "category":           expected_dec,
            "config_id":          config_id,
            "config_label":       config["label"],
            "pipeline_variant":   config["pipeline_variant"],
            "final_decision":     "ERROR",
            "expected_decision":  expected_dec,
            "expected_escalate":  expected_esc,
            "output_escalate":    False,
            "decision_correct":   False,
            "escalation_correct": False,
            "score":              -1,
            "rules_failed":       [],
            "rule_decision":      "",
            "conflict_flags":     [],
            "verifier":           {},
            "consistency":        {},
            "llm_only_a3":        {},
            "halted":             True,
            "halt_reason":        str(exc),
            "warnings":           [],
            "audit_summary":      {},
            "error":              str(exc),
        }

    # Write result (safe — file did not exist or overwrite=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


# ── Batch runner ───────────────────────────────────────────────────────────────

async def run_all_live(
    scenarios: list[dict],
    configs: list[dict],
    model_client,
    output_dir: Path = LIVE_RESULTS_DIR,
    max_concurrent: int = 3,
    verbose: bool = True,
    dry_run: bool = False,
    overwrite: bool = False,
) -> list[dict]:
    """
    Run all (scenario, config) combinations with real API calls.

    Files already in output_dir are SKIPPED (not overwritten) unless overwrite=True.
    This means you can run in batches and accumulate results safely.

    Returns list of result dicts for cells that were actually run (not skipped).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    total_cells = len(scenarios) * len(configs)

    # Count already-done cells
    already_done = sum(
        1 for cfg in configs for s in scenarios
        if (output_dir / f"{Path(s['_filename']).stem}__{cfg['id']}.json").exists()
    )
    remaining = total_cells - already_done

    if dry_run:
        print(f"[DRY RUN] {total_cells} total cells  |  "
              f"{already_done} already done  |  {remaining} would run")
        for cfg in configs:
            done = sum(
                1 for s in scenarios
                if (output_dir / f"{Path(s['_filename']).stem}__{cfg['id']}.json").exists()
            )
            print(f"  {cfg['id']:4s}  {cfg['label']:<40}  "
                  f"{done}/{len(scenarios)} done")
        return []

    if verbose:
        print(f"LIVE ablation: {total_cells} total cells  |  "
              f"{already_done} already done  |  {remaining} to run")
        if already_done and not overwrite:
            print("  (existing files are skipped — use --overwrite to re-run them)")

    semaphore = asyncio.Semaphore(max_concurrent)
    t0 = time.time()
    all_results: list[dict] = []

    for cfg in configs:
        tasks = [
            run_cell_live(s, cfg, model_client, semaphore, overwrite, output_dir)
            for s in scenarios
        ]
        cfg_results_raw = await asyncio.gather(*tasks)

        ran     = [r for r in cfg_results_raw if r is not None]
        skipped = sum(1 for r in cfg_results_raw if r is None)
        all_results.extend(ran)

        if verbose:
            correct = sum(1 for r in ran if r.get("decision_correct"))
            errors  = sum(1 for r in ran if r.get("error"))
            elapsed = time.time() - t0
            print(f"  {cfg['id']:4s}  {cfg['label']:<38}  "
                  f"ran={len(ran):2d}  correct={correct:2d}  "
                  f"errors={errors}  skipped={skipped}  ({elapsed:.0f}s)")

    # Write / update manifest
    _write_manifest(output_dir, scenarios, configs)

    elapsed = time.time() - t0
    if verbose and all_results:
        total_correct = sum(1 for r in all_results if r.get("decision_correct"))
        print(f"\nFinished in {elapsed:.0f}s.  "
              f"Ran {len(all_results)} cells  |  "
              f"{total_correct} correct  |  "
              f"Results → {output_dir}/")

    return all_results


def _write_manifest(output_dir: Path, scenarios: list[dict], configs: list[dict]) -> None:
    """Write/update the manifest from all files currently in output_dir."""
    all_results: list[dict] = []
    for p in sorted(output_dir.glob("*.json")):
        if "manifest" in p.name:
            continue
        try:
            all_results.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass

    category_counts: dict[str, int] = {}
    for s in scenarios:
        cat = s.get("expected_decision", "UNKNOWN")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    per_config: dict[str, dict] = {}
    for cfg in configs:
        cfg_res = [r for r in all_results if r.get("config_id") == cfg["id"]]
        per_config[cfg["id"]] = {
            "label":   cfg["label"],
            "variant": cfg["pipeline_variant"],
            "total":   len(cfg_res),
            "correct": sum(1 for r in cfg_res if r.get("decision_correct")),
            "errors":  sum(1 for r in cfg_res if r.get("error")),
        }

    manifest = {
        "study_type":      "live_ablation",
        "total_files":     len(all_results),
        "total_scenarios": len(scenarios),
        "total_configs":   len(configs),
        "config_ids":      [c["id"] for c in configs],
        "scenario_ids":    sorted({r["scenario_id"] for r in all_results}),
        "category_counts": category_counts,
        "per_config":      per_config,
    }
    (output_dir / "live_ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )


# ── Load + metrics ─────────────────────────────────────────────────────────────

def load_live_results(results_dir: Path = LIVE_RESULTS_DIR) -> dict[str, list[dict]]:
    """
    Load all live cell result files from results_dir.
    Returns {config_id: [result, ...]} grouped by config.
    Skips manifest files.
    """
    from collections import defaultdict
    by_config: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(results_dir.glob("*.json")):
        if "manifest" in p.name:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        cid = data.get("config_id")
        if cid:
            by_config[cid].append(data)
    return dict(by_config)


def compute_live_metrics(results_by_config: dict[str, list[dict]]) -> dict:
    """Compute M1–M6 metrics. Delegates to ablation_study/metrics.py."""
    from evaluation.ablation_study.metrics import compute_all_metrics
    return compute_all_metrics(results_by_config, control_id="P0")


def compute_verifier_impact(results_by_config: dict[str, list[dict]]) -> dict:
    """Per-config correction and override rates (live-only telemetry)."""
    out: dict[str, dict] = {}
    for config_id, results in results_by_config.items():
        valid = [r for r in results if not r.get("error")]
        n = len(valid)
        if n == 0:
            out[config_id] = {"n": 0}
            continue
        out[config_id] = {
            "n": n,
            "correction_rate":    sum(1 for r in valid if r.get("verifier", {}).get("any_correction")) / n,
            "a1_correction_rate": sum(1 for r in valid if r.get("verifier", {}).get("a1_correction_applied")) / n,
            "a2_correction_rate": sum(1 for r in valid if r.get("verifier", {}).get("a2_correction_applied")) / n,
            "override_rate":      sum(1 for r in valid if r.get("consistency", {}).get("any_override")) / n,
        }
    return out


def compute_llm_a3_impact(results_by_config: dict[str, list[dict]]) -> dict:
    """
    P4/P7 specific: how often did LLM-only A3 disagree with the Python pre_check?
    """
    out: dict[str, dict] = {}
    for config_id, results in results_by_config.items():
        a3_results = [
            r for r in results
            if r.get("pipeline_variant") in ("llm_only_a3", "no_rule_engine")
            and not r.get("error")
        ]
        if not a3_results:
            continue
        n = len(a3_results)
        disagreements = sum(
            1 for r in a3_results
            if not r.get("llm_only_a3", {}).get("agreed_with_precheck", True)
        )
        out[config_id] = {
            "n": n,
            "llm_precheck_disagreement_rate": disagreements / n,
            "disagreements": disagreements,
        }
    return out
