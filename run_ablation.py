"""
run_ablation.py
================
Top-level entry point for the full ablation study pipeline.

Runs in three stages:
  Stage 1 — Select 50 balanced scenarios from data/scenarios/
  Stage 2 — Execute 700 ablation cells (50 × 14 configs)  [NO LLM CALLS]
  Stage 3 — Compute metrics and generate comparison reports

Output directories
------------------
  data/ablation_results/   — 700 individual cell JSON files + manifest
  data/ablation_reports/   — comparison table (txt), metrics (json), spreadsheet (csv)

Usage
-----
  # Full run (recommended first time):
  python run_ablation.py

  # Dry run — verify setup, count cells, write nothing:
  python run_ablation.py --dry-run

  # Run only specific configs:
  python run_ablation.py --configs A0 A1 A2 A3 A4

  # Run only specific axes:
  python run_ablation.py --axis architecture
  python run_ablation.py --axis rule
  python run_ablation.py --axis threshold

  # Skip stage 2 (re-use existing results) and regenerate report only:
  python run_ablation.py --report-only

  # Change random seed for scenario selection:
  python run_ablation.py --seed 99
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from evaluation.ablation_study.configs import UNIQUE_CONFIGS, get_config, list_config_ids
from evaluation.ablation_study.scenario_selector import select_and_report
from evaluation.ablation_study.runner import run_all
from evaluation.ablation_study.report import load_results, main as generate_report_main

RESULTS_DIR = Path("data/ablation_results")
REPORTS_DIR = Path("data/ablation_reports")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the MiFID II ablation study — zero LLM calls."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Count cells only; write nothing to disk."
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Skip scenario selection and runner; load existing results and regenerate report."
    )
    parser.add_argument(
        "--configs", nargs="+", metavar="ID",
        help=f"Run only these config IDs. Available: {list_config_ids()}"
    )
    parser.add_argument(
        "--axis", choices=["architecture", "rule", "threshold"],
        help="Run only configs belonging to this axis."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for scenario selection (default: 42)."
    )
    parser.add_argument(
        "--results-dir", default=str(RESULTS_DIR),
        help=f"Where to write / read cell results (default: {RESULTS_DIR})"
    )
    parser.add_argument(
        "--reports-dir", default=str(REPORTS_DIR),
        help=f"Where to write final reports (default: {REPORTS_DIR})"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    reports_dir = Path(args.reports_dir)

    # ── Resolve config list ────────────────────────────────────────────────────
    if args.configs:
        try:
            configs = [get_config(cid) for cid in args.configs]
        except KeyError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.axis:
        configs = [c for c in UNIQUE_CONFIGS if c["axis"] == args.axis]
        if not configs:
            print(f"No configs found for axis '{args.axis}'", file=sys.stderr)
            sys.exit(1)
    else:
        configs = UNIQUE_CONFIGS

    # ── Report-only mode ───────────────────────────────────────────────────────
    if args.report_only:
        print("=== Stage 3: Generating report from existing results ===")
        sys.argv = [
            "report",
            "--results-dir", str(results_dir),
            "--output-dir",  str(reports_dir),
        ]
        generate_report_main()
        return

    # ── Stage 1: Select scenarios ──────────────────────────────────────────────
    print("=" * 60)
    print("MiFID II Ablation Study — Zero LLM Calls")
    print("=" * 60)
    print(f"\nStage 1: Selecting 50 balanced scenarios (seed={args.seed})...")
    t0 = time.time()

    try:
        report = select_and_report(seed=args.seed)
    except ValueError as e:
        print(f"Error selecting scenarios: {e}", file=sys.stderr)
        sys.exit(1)

    scenarios = report["scenarios"]
    print(f"  Selected {report['total']} scenarios:")
    for cat, cnt in sorted(report["counts"].items()):
        print(f"    {cat:12s}: {cnt}")

    # ── Stage 2: Run ablation ──────────────────────────────────────────────────
    total_cells = len(scenarios) * len(configs)
    print(f"\nStage 2: Running {total_cells} cells "
          f"({len(scenarios)} scenarios × {len(configs)} configs)...")

    all_results = run_all(
        scenarios=scenarios,
        configs=configs,
        output_dir=results_dir,
        dry_run=args.dry_run,
        verbose=True,
    )

    if args.dry_run:
        print("\n[DRY RUN] No files written. Remove --dry-run to execute.")
        return

    elapsed = time.time() - t0
    errors = [r for r in all_results if r.get("error")]
    if errors:
        print(f"\nWARNING: {len(errors)} cells errored:")
        for e in errors[:5]:
            print(f"  {e['scenario_id']} × {e['config_id']}: {e['error']}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more (see manifest for full list)")

    # ── Stage 3: Report ────────────────────────────────────────────────────────
    print(f"\nStage 3: Computing metrics and generating reports...")
    by_config = {}
    for r in all_results:
        by_config.setdefault(r["config_id"], []).append(r)

    from evaluation.ablation_study.metrics import compute_all_metrics
    from evaluation.ablation_study.report import (
        build_comparison_table, build_axis_summary, write_csv
    )
    import json

    metrics = compute_all_metrics(by_config, control_id="A0")

    table  = build_comparison_table(metrics)
    axis_s = build_axis_summary(metrics)
    print("\n" + table)
    print(axis_s)

    reports_dir.mkdir(parents=True, exist_ok=True)

    # Text table
    (reports_dir / "ablation_comparison_table.txt").write_text(
        table + "\n\n" + axis_s, encoding="utf-8"
    )
    # JSON
    (reports_dir / "ablation_metrics_summary.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    # CSV
    write_csv(metrics, reports_dir / "ablation_metrics_table.csv")

    print(f"\nReports written to: {reports_dir}/")
    print(f"  ablation_comparison_table.txt")
    print(f"  ablation_metrics_summary.json")
    print(f"  ablation_metrics_table.csv")
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
