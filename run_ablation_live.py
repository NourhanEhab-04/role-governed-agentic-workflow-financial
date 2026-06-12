"""
run_ablation_live.py
=====================
CLI entry point for the LIVE MiFID II ablation study.

HOW SCENARIO SELECTION WORKS
-----------------------------
By default (no --scenarios flag):
  50 balanced scenarios are auto-selected from data/scenarios/ using a seeded
  diversity algorithm:  SUITABLE=14, CONDITIONAL=13, UNSUITABLE=13, ESCALATED=10
  The same 50 are always selected (seed=42 is fixed).

With --scenarios flag:
  You specify exact filenames (without .json) or prefixes to load DIRECTLY
  from data/scenarios/. The auto-selector is bypassed entirely.
  Example:  --scenarios 01 07 10
  Loads:    01_suitable_conservative.json, 07_escalated_contradiction.json,
            10_unsuitable_vulnerable.json

FILE PERSISTENCE (no overwrites)
---------------------------------
Each cell result is saved as:
  data/ablation_results/live/{scenario_id}__{config_id}.json

If that file already exists the cell is SKIPPED.
This lets you run in batches safely. Use --overwrite to redo.

The 8 Configurations
---------------------
  P0  Full Pipeline       — reference, all components active
  P1  No Verifier         — A1/A2 without AV check/correction
  P2  No pre_check        — first deterministic rule call removed
  P3  No audit            — third deterministic rule call removed
  P4  LLM-only A3         — A3 reasons without evaluate_suitability_tool
  P5  No A4               — conflict detector replaced with passthrough
  P6  Single LLM Baseline — entire pipeline replaced by one LLM call
  P7  No Rule Engine      — no deterministic Python evaluate_suitability() anywhere

Typical Usage
-------------
  # See exactly which 50 scenarios are selected (no API calls):
  python run_ablation_live.py --status

  # Dry run — full picture of what would run vs already done:
  python run_ablation_live.py --dry-run

  # Test with 3 specific scenarios (loads them directly from data/scenarios/):
  python run_ablation_live.py --scenarios 01 07 10 --configs P0 P6

  # Run the auto-selected 50 for configs P0 and P1:
  python run_ablation_live.py --configs P0 P1

  # Run everything (400 cells total, in any number of batches):
  python run_ablation_live.py

  # After all runs complete, compute results:
  python compute_ablation_results.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from evaluation.ablation_study.live_configs import (
    LIVE_CONFIGS, get_live_config, list_live_config_ids,
)
from evaluation.ablation_study.live_runner import run_all_live, load_live_results
from evaluation.ablation_study.scenario_selector import select_and_report

SCENARIOS_DIR    = Path("data/scenarios")
LIVE_RESULTS_DIR = Path("data/ablation_results/live")
LIVE_REPORTS_DIR = Path("data/ablation_reports/live")


# ── Scenario loading ───────────────────────────────────────────────────────────

def _load_specific_scenarios(names: list[str]) -> list[dict]:
    """
    Load specific scenarios DIRECTLY from data/scenarios/ by filename prefix.

    Each name in `names` is matched as a filename prefix (without .json).
    Example: names=["01", "07", "10"]
      → loads 01_suitable_conservative.json, 07_escalated_contradiction.json,
               10_unsuitable_vulnerable.json

    Raises ValueError if any name matches zero files.
    """
    loaded: list[dict] = []
    for name in names:
        # Try exact match first (e.g. "01_suitable_conservative")
        exact = SCENARIOS_DIR / f"{name}.json"
        if exact.exists():
            data = json.loads(exact.read_text(encoding="utf-8"))
            data["_path"]     = str(exact)
            data["_filename"] = exact.name
            loaded.append(data)
            continue

        # Prefix match (e.g. "01" → "01_suitable_conservative.json")
        matches = sorted(SCENARIOS_DIR.glob(f"{name}*.json"))
        if not matches:
            raise ValueError(
                f"No scenario file found matching prefix '{name}' in {SCENARIOS_DIR}/\n"
                f"Run:  python run_ablation_live.py --status  to see available files."
            )
        if len(matches) > 1:
            print(f"  [NOTE] Prefix '{name}' matches {len(matches)} files — "
                  f"using all: {[m.name for m in matches]}")
        for p in matches:
            data = json.loads(p.read_text(encoding="utf-8"))
            data["_path"]     = str(p)
            data["_filename"] = p.name
            loaded.append(data)

    # Deduplicate by filename (in case of overlap)
    seen: set[str] = set()
    deduped: list[dict] = []
    for s in loaded:
        if s["_filename"] not in seen:
            seen.add(s["_filename"])
            deduped.append(s)
    return deduped


def _load_selected_50(seed: int) -> tuple[list[dict], dict]:
    """Load the auto-selected 50 balanced scenarios."""
    report = select_and_report(seed=seed)
    return report["scenarios"], report


# ── Status command ─────────────────────────────────────────────────────────────

def _print_status(seed: int, results_dir: Path, configs: list[dict]) -> None:
    """
    Show exactly which 50 scenarios are selected and which cells are done/pending.
    Zero API calls.
    """
    scenarios, report = _load_selected_50(seed)

    print("=" * 72)
    print("ABLATION STUDY STATUS")
    print("=" * 72)
    print(f"\nAuto-selected {report['total']} scenarios (seed={seed}):")
    print(f"  {report['counts']}")
    print()

    # Load existing results
    existing: set[str] = set()
    if results_dir.exists():
        existing = {p.stem for p in results_dir.glob("*.json") if "manifest" not in p.name}

    # Table: scenarios (rows) × configs (cols)
    config_ids = [c["id"] for c in configs]
    col_w = 7

    # Header
    print(f"  {'Scenario':<52} {'Category':<12}", end="")
    for cid in config_ids:
        print(f"  {cid:>{col_w - 2}}", end="")
    print()
    print("  " + "-" * (52 + 12 + len(config_ids) * col_w))

    total_done    = 0
    total_pending = 0

    for s in scenarios:
        sid = Path(s["_filename"]).stem
        cat = s.get("expected_decision", "?")
        print(f"  {sid:<52} {cat:<12}", end="")
        for cid in config_ids:
            cell_key = f"{sid}__{cid}"
            if cell_key in existing:
                done_data = json.loads((results_dir / f"{cell_key}.json").read_text())
                mark = "✓" if done_data.get("decision_correct") else "✗"
                total_done += 1
            else:
                mark = "·"
                total_pending += 1
            print(f"  {mark:>{col_w - 2}}", end="")
        print()

    print()
    print(f"  ✓ = correct   ✗ = wrong   · = not yet run")
    print()
    print(f"  Done:    {total_done}")
    print(f"  Pending: {total_pending}")
    print(f"  Total:   {total_done + total_pending}")
    print()
    print("To run pending cells:")
    print("  python run_ablation_live.py")
    print()
    print("To run specific scenarios by prefix:")
    pending_scenarios = []
    for s in scenarios:
        sid = Path(s["_filename"]).stem
        cids_pending = [c["id"] for c in configs if f"{sid}__{c['id']}" not in existing]
        if cids_pending:
            pending_scenarios.append(sid[:2])
    if pending_scenarios:
        example = " ".join(sorted(set(pending_scenarios))[:5])
        print(f"  python run_ablation_live.py --scenarios {example}")


# ── Main ───────────────────────────────────────────────────────────────────────

async def _run(args: argparse.Namespace) -> None:

    # Resolve configs
    if args.configs:
        try:
            configs = [get_live_config(cid) for cid in args.configs]
        except KeyError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        configs = LIVE_CONFIGS

    # --status: show selection table and exit
    if args.status:
        _print_status(args.seed, Path(args.output_dir), configs)
        return

    # Resolve scenarios
    if args.scenarios:
        # User specified explicit scenario names/prefixes — load directly from disk
        # The auto-selector is bypassed entirely
        try:
            scenarios = _load_specific_scenarios(args.scenarios)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        counts = {}
        for s in scenarios:
            cat = s.get("expected_decision", "UNKNOWN")
            counts[cat] = counts.get(cat, 0) + 1
        print(f"Loaded {len(scenarios)} specified scenarios: {counts}")
    else:
        # Auto-select the balanced 50
        scenarios, sel_report = _load_selected_50(args.seed)
        print(f"Auto-selected {sel_report['total']} scenarios (seed={args.seed}): "
              + ", ".join(f"{k}={v}" for k, v in sorted(sel_report["counts"].items())))

    print(f"Configs: {[c['id'] for c in configs]}")
    print(f"Results: {args.output_dir}")

    if args.dry_run:
        output_dir = Path(args.output_dir)
        existing = {p.stem for p in output_dir.glob("*.json")
                    if output_dir.exists() and "manifest" not in p.name}
        total    = len(scenarios) * len(configs)
        done     = sum(
            1 for s in scenarios for c in configs
            if f"{Path(s['_filename']).stem}__{c['id']}" in existing
        )
        print(f"\n[DRY RUN]  {total} cells total  |  {done} already done  |  {total - done} would run")
        print("\nSelected scenarios:")
        for s in scenarios:
            sid = Path(s["_filename"]).stem
            cat = s.get("expected_decision", "?")
            marks = ""
            for c in configs:
                marks += "✓" if f"{sid}__{c['id']}" in existing else "·"
            print(f"  [{marks}]  {sid:<52}  {cat}")
        print(f"\n  Legend:  ✓ = done   · = pending")
        return

    from config.llm_config import get_model_client
    model_client = get_model_client()

    await run_all_live(
        scenarios=scenarios,
        configs=configs,
        model_client=model_client,
        output_dir=Path(args.output_dir),
        max_concurrent=args.max_concurrent,
        verbose=True,
        dry_run=False,
        overwrite=args.overwrite,
    )

    print("\nTo compute results across all completed cells:")
    print("  python compute_ablation_results.py")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the live MiFID II ablation study.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--configs", nargs="+", metavar="ID",
        help=f"Config IDs to run. Valid: {list_live_config_ids()} (default: all 8)",
    )
    parser.add_argument(
        "--scenarios", nargs="+", metavar="NAME",
        help=(
            "Scenario filename prefixes to load DIRECTLY from data/scenarios/. "
            "Example: --scenarios 01 07 10  loads 01_*.json, 07_*.json, 10_*.json. "
            "If omitted, the auto-selected 50 balanced scenarios are used."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Scenario auto-selection seed (default: 42, only used when --scenarios is omitted).",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=3,
        help="Max simultaneous pipeline runs (default: 3).",
    )
    parser.add_argument(
        "--output-dir", default=str(LIVE_RESULTS_DIR),
        help=f"Cell results directory (default: {LIVE_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--status", action="store_true",
        help=(
            "Show which 50 scenarios are selected and which cells are done/pending. "
            "Zero API calls. Use this to understand exactly what will run."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would run and which files already exist. Zero API calls.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-run cells even if result file already exists.",
    )
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
