"""
evaluation/ablation_study/runner.py
=====================================
Executes the 50 × 14 = 700 ablation cells.

For each (scenario, config) pair:
  1. Load ground-truth profiles from scenario file
  2. Run ablated rule engine (deterministic, no LLM)
  3. Run ablated conflict detection (deterministic, no LLM)
  4. Derive final decision
  5. Compare against expected_decision
  6. Write result JSON to data/ablation_results/

Zero LLM API calls. Fully offline. Idempotent (re-running overwrites same files).

Usage
-----
  python -m evaluation.ablation_study.runner
  python -m evaluation.ablation_study.runner --dry-run        # count cells only
  python -m evaluation.ablation_study.runner --configs A0 A1  # run subset
  python -m evaluation.ablation_study.runner --scenarios data/scenarios/01_suitable_conservative.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from evaluation.ablation_study.configs import UNIQUE_CONFIGS, get_config, list_config_ids
from evaluation.ablation_study.rule_engine_variants import evaluate_suitability_ablated
from evaluation.ablation_study.conflict_logic import run_conflict_detection, derive_final_decision
from evaluation.ablation_study.scenario_selector import select_and_report, SCENARIOS_DIR

RESULTS_DIR = Path("data/ablation_results")


# ── Profile extraction ─────────────────────────────────────────────────────────

def _extract_profiles(scenario: dict) -> tuple[dict, dict]:
    """
    Extract client and product profiles from a scenario dict.

    Uses expected_client_profile / expected_product_profile as ground truth.
    Merges portfolio_concentration_pct from the 'client' key when present
    (needed for the CONCENTRATION conflict check).
    """
    client  = dict(scenario["expected_client_profile"])
    product = dict(scenario["expected_product_profile"])

    # Merge concentration field if available in the richer 'client' key
    client_extra = scenario.get("client", {})
    if "portfolio_concentration_pct" in client_extra and \
       "portfolio_concentration_pct" not in client:
        client["portfolio_concentration_pct"] = client_extra["portfolio_concentration_pct"]

    return client, product


def _expected_escalate(scenario: dict) -> bool:
    """Derive expected escalation from scenario metadata."""
    # explicit field wins
    if "expected_escalate" in scenario:
        return bool(scenario["expected_escalate"])
    # infer from decision
    return scenario.get("expected_decision", "") == "ESCALATED"


# ── Single cell ────────────────────────────────────────────────────────────────

def run_cell(scenario: dict, config: dict) -> dict:
    """
    Run one (scenario, config) cell. Returns a result dict.

    The result dict schema:
    {
      "scenario_id":        str   — filename without .json
      "scenario_file":      str   — full path
      "category":           str   — SUITABLE | CONDITIONAL | UNSUITABLE | ESCALATED
      "config_id":          str
      "config_label":       str
      "config_axis":        str
      "rule_verdict":       dict  — raw rule engine output
      "conflict_report":    dict  — conflict detection output (or None if A1)
      "final_decision":     str   — SUITABLE | CONDITIONAL | UNSUITABLE | ESCALATED
      "expected_decision":  str
      "expected_escalate":  bool
      "output_escalate":    bool
      "decision_correct":   bool
      "escalation_correct": bool
      "score":              int
      "rules_failed":       list[str]
      "error":              str | None  — populated if an exception occurred
    }
    """
    scenario_id   = Path(scenario["_filename"]).stem
    scenario_file = scenario["_path"]
    category      = scenario.get("expected_decision", "UNKNOWN")
    expected_dec  = scenario.get("expected_decision", "")
    expected_esc  = _expected_escalate(scenario)

    try:
        client, product = _extract_profiles(scenario)

        # ── Rule engine (ablated) ─────────────────────────────────────────────
        rule_verdict = evaluate_suitability_ablated(
            client,
            product,
            disabled_rules=config["disabled_rules"],
            suitable_threshold=config["suitable_threshold"],
            conditional_threshold=config["conditional_threshold"],
        )

        # ── Conflict detection (ablated) ──────────────────────────────────────
        conflict_report = run_conflict_detection(
            client,
            product,
            rule_verdict,
            disabled_flags=config["disabled_flags"],
        )

        # ── Final decision ────────────────────────────────────────────────────
        final_decision = derive_final_decision(rule_verdict, conflict_report)
        output_escalate = conflict_report.get("escalate", False)

        decision_correct   = (final_decision == expected_dec)
        escalation_correct = (output_escalate == expected_esc)

        return {
            "scenario_id":        scenario_id,
            "scenario_file":      scenario_file,
            "category":           category,
            "config_id":          config["id"],
            "config_label":       config["label"],
            "config_axis":        config["axis"],
            "rule_verdict":       rule_verdict,
            "conflict_report":    conflict_report,
            "final_decision":     final_decision,
            "expected_decision":  expected_dec,
            "expected_escalate":  expected_esc,
            "output_escalate":    output_escalate,
            "decision_correct":   decision_correct,
            "escalation_correct": escalation_correct,
            "score":              rule_verdict["score"],
            "rules_failed":       rule_verdict.get("hard_failed_rules", []),
            "error":              None,
        }

    except Exception as exc:
        return {
            "scenario_id":        scenario_id,
            "scenario_file":      scenario_file,
            "category":           category,
            "config_id":          config["id"],
            "config_label":       config["label"],
            "config_axis":        config["axis"],
            "rule_verdict":       None,
            "conflict_report":    None,
            "final_decision":     "ERROR",
            "expected_decision":  expected_dec,
            "expected_escalate":  expected_esc,
            "output_escalate":    False,
            "decision_correct":   False,
            "escalation_correct": False,
            "score":              -1,
            "rules_failed":       [],
            "error":              str(exc),
        }


# ── Batch runner ───────────────────────────────────────────────────────────────

def run_all(
    scenarios: list[dict],
    configs: list[dict],
    output_dir: Path = RESULTS_DIR,
    dry_run: bool = False,
    verbose: bool = True,
) -> list[dict]:
    """
    Run all (scenario, config) combinations. Returns list of all result dicts.

    Writes one JSON file per cell: {output_dir}/{scenario_id}__{config_id}.json
    Also writes a combined manifest: {output_dir}/ablation_manifest.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    total_cells = len(scenarios) * len(configs)

    if dry_run:
        print(f"[DRY RUN] Would run {len(scenarios)} scenarios × {len(configs)} configs "
              f"= {total_cells} cells. No files written.")
        return []

    if verbose:
        print(f"Running {total_cells} ablation cells "
              f"({len(scenarios)} scenarios × {len(configs)} configs)...")

    all_results: list[dict] = []
    t0 = time.time()

    for i, config in enumerate(configs):
        for j, scenario in enumerate(scenarios):
            result = run_cell(scenario, config)
            all_results.append(result)

            # Write individual cell result
            cell_path = output_dir / f"{result['scenario_id']}__{config['id']}.json"
            cell_path.write_text(
                json.dumps(result, indent=2, default=str), encoding="utf-8"
            )

            if result["error"]:
                print(f"  [ERROR] {result['scenario_id']} × {config['id']}: {result['error']}")

        if verbose:
            pct = (i + 1) / len(configs) * 100
            elapsed = time.time() - t0
            print(f"  Config {config['id']:12s} done  ({pct:5.1f}%  {elapsed:.1f}s)")

    # Write manifest (lightweight — no rule/conflict detail, just decision outcomes)
    manifest = _build_manifest(all_results, scenarios, configs)
    manifest_path = output_dir / "ablation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    elapsed = time.time() - t0
    if verbose:
        correct = sum(1 for r in all_results if r["decision_correct"])
        print(f"\nDone in {elapsed:.1f}s. "
              f"{correct}/{total_cells} cells correct "
              f"({correct/total_cells*100:.1f}%).")
        print(f"Results written to: {output_dir}/")

    return all_results


def _build_manifest(
    results: list[dict],
    scenarios: list[dict],
    configs: list[dict],
) -> dict:
    return {
        "total_cells":      len(results),
        "total_scenarios":  len(scenarios),
        "total_configs":    len(configs),
        "config_ids":       [c["id"] for c in configs],
        "scenario_ids":     sorted({r["scenario_id"] for r in results}),
        "category_counts":  _count_categories(scenarios),
        "errors":           [r for r in results if r["error"]],
        "summary": {
            r["config_id"]: {
                "total":   sum(1 for x in results if x["config_id"] == r["config_id"]),
                "correct": sum(1 for x in results if x["config_id"] == r["config_id"]
                               and x["decision_correct"]),
            }
            for r in results
        },
    }


def _count_categories(scenarios: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in scenarios:
        cat = s.get("expected_decision", "UNKNOWN")
        counts[cat] = counts.get(cat, 0) + 1
    return counts


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the MiFID II ablation study (no LLM calls)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Count cells only; write nothing."
    )
    parser.add_argument(
        "--configs", nargs="+", metavar="ID",
        help=f"Run only these config IDs. Available: {list_config_ids()}"
    )
    parser.add_argument(
        "--scenarios", nargs="+", metavar="FILE",
        help="Paths to specific scenario JSON files to use instead of auto-selection."
    )
    parser.add_argument(
        "--output-dir", default=str(RESULTS_DIR), metavar="DIR",
        help=f"Output directory (default: {RESULTS_DIR})"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for scenario selection (default: 42)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output."
    )
    args = parser.parse_args()

    # ── Resolve configs ───────────────────────────────────────────────────────
    if args.configs:
        try:
            configs = [get_config(cid) for cid in args.configs]
        except KeyError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        configs = UNIQUE_CONFIGS

    # ── Resolve scenarios ─────────────────────────────────────────────────────
    if args.scenarios:
        scenarios = []
        for f in args.scenarios:
            p = Path(f)
            if not p.exists():
                print(f"Error: scenario file not found: {f}", file=sys.stderr)
                sys.exit(1)
            data = json.loads(p.read_text(encoding="utf-8"))
            data["_path"]     = str(p)
            data["_filename"] = p.name
            scenarios.append(data)
    else:
        report = select_and_report(seed=args.seed)
        scenarios = report["scenarios"]
        if not args.dry_run and not args.quiet:
            print(f"Selected {report['total']} scenarios: "
                  + ", ".join(f"{k}={v}" for k, v in sorted(report["counts"].items())))

    run_all(
        scenarios=scenarios,
        configs=configs,
        output_dir=Path(args.output_dir),
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
