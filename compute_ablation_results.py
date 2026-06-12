"""
compute_ablation_results.py
============================
Aggregates all completed ablation cells from data/ablation_results/live/
and computes the full metric comparison table.

Run this at any point — after 10 scenarios, after 50, or anywhere in between.
It reads whatever files are currently in the results directory.

Usage
-----
  python compute_ablation_results.py

  # Show per-scenario breakdown too:
  python compute_ablation_results.py --verbose

  # Use a different results directory:
  python compute_ablation_results.py --results-dir data/ablation_results/live

Output
------
  Prints the comparison table to stdout.
  Saves to data/ablation_reports/live/:
    live_metrics.json          — full metrics dict
    live_comparison_table.txt  — human-readable table
    live_comparison_table.csv  — machine-readable
    live_verifier_impact.json  — correction/override rates
    live_llm_a3_impact.json    — P4/P7 LLM-reasoning accuracy
    live_progress.json         — how many cells done per config
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

LIVE_RESULTS_DIR = Path("data/ablation_results/live")
LIVE_REPORTS_DIR = Path("data/ablation_reports/live")

# Config display order and labels (matches live_configs.py)
CONFIG_ORDER = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7"]
CONFIG_LABELS = {
    "P0": "Full Pipeline",
    "P1": "No Verifier",
    "P2": "No pre_check",
    "P3": "No audit",
    "P4": "LLM-only A3",
    "P5": "No A4",
    "P6": "Single LLM",
    "P7": "No Rule Engine",
}
CATEGORIES = ("SUITABLE", "CONDITIONAL", "UNSUITABLE", "ESCALATED")
CONTROL_ID  = "P0"


# ── Load ───────────────────────────────────────────────────────────────────────

def load_results(results_dir: Path) -> dict[str, list[dict]]:
    """Load all cell JSON files. Returns {config_id: [result, ...]}."""
    by_config: dict[str, list[dict]] = defaultdict(list)
    loaded = 0
    for p in sorted(results_dir.glob("*.json")):
        if "manifest" in p.name:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        cid = data.get("config_id")
        if cid:
            by_config[cid].append(data)
            loaded += 1
    return dict(by_config)


# ── Metrics ────────────────────────────────────────────────────────────────────

def _safe_div(n: float, d: float) -> float:
    return n / d if d else float("nan")


def decision_accuracy(results: list[dict]) -> float:
    valid = [r for r in results if not r.get("error")]
    return _safe_div(sum(1 for r in valid if r.get("decision_correct")), len(valid))


def per_category_accuracy(results: list[dict]) -> dict[str, float]:
    buckets: dict[str, list] = defaultdict(list)
    for r in results:
        if not r.get("error"):
            buckets[r.get("category", "UNKNOWN")].append(r)
    return {
        cat: _safe_div(sum(1 for r in buckets[cat] if r.get("decision_correct")), len(buckets[cat]))
        for cat in CATEGORIES
    }


def unsuitable_recall(results: list[dict]) -> float:
    truly = [r for r in results if r.get("expected_decision") == "UNSUITABLE" and not r.get("error")]
    return _safe_div(sum(1 for r in truly if r.get("final_decision") == "UNSUITABLE"), len(truly))


def unsuitable_precision(results: list[dict]) -> float:
    output = [r for r in results if r.get("final_decision") == "UNSUITABLE" and not r.get("error")]
    return _safe_div(sum(1 for r in output if r.get("expected_decision") == "UNSUITABLE"), len(output))


def escalation_recall(results: list[dict]) -> float:
    truly = [r for r in results if r.get("expected_decision") == "ESCALATED" and not r.get("error")]
    return _safe_div(sum(1 for r in truly if r.get("final_decision") == "ESCALATED"), len(truly))


def escalation_precision(results: list[dict]) -> float:
    output = [r for r in results if r.get("final_decision") == "ESCALATED" and not r.get("error")]
    return _safe_div(sum(1 for r in output if r.get("expected_decision") == "ESCALATED"), len(output))


def escalation_f1(results: list[dict]) -> float:
    r = escalation_recall(results)
    p = escalation_precision(results)
    return _safe_div(2 * r * p, r + p)


def parity_gap(results: list[dict]) -> float:
    cat_acc = per_category_accuracy(results)
    vals = [v for v in cat_acc.values() if v == v]
    return max(vals) - min(vals) if len(vals) >= 2 else float("nan")


def compute_all_metrics(by_config: dict[str, list[dict]]) -> dict:
    per_cfg = {
        cid: {
            "M1_accuracy":          decision_accuracy(r),
            "M1c_per_category":     per_category_accuracy(r),
            "M2_unsuitable_recall": unsuitable_recall(r),
            "M2p_precision":        unsuitable_precision(r),
            "M3_esc_recall":        escalation_recall(r),
            "M3p_esc_precision":    escalation_precision(r),
            "M3f_esc_f1":           escalation_f1(r),
            "M6_parity_gap":        parity_gap(r),
            "n_correct":            sum(1 for x in r if x.get("decision_correct")),
            "n_total":              len([x for x in r if not x.get("error")]),
            "n_errors":             sum(1 for x in r if x.get("error")),
        }
        for cid, r in by_config.items()
    }
    # Deltas vs control
    ctrl_acc = per_cfg.get(CONTROL_ID, {}).get("M1_accuracy", float("nan"))
    deltas = {
        cid: (per_cfg[cid]["M1_accuracy"] - ctrl_acc)
        for cid in per_cfg
        if cid != CONTROL_ID
    }
    return {"per_config": per_cfg, "deltas": deltas, "control_id": CONTROL_ID}


def compute_verifier_impact(by_config: dict[str, list[dict]]) -> dict:
    out = {}
    for cid, results in by_config.items():
        valid = [r for r in results if not r.get("error")]
        n = len(valid)
        if n == 0:
            out[cid] = {"n": 0}
            continue
        out[cid] = {
            "n": n,
            "correction_rate": _safe_div(
                sum(1 for r in valid if r.get("verifier", {}).get("any_correction")), n),
            "override_rate": _safe_div(
                sum(1 for r in valid if r.get("consistency", {}).get("any_override")), n),
        }
    return out


def compute_llm_a3_impact(by_config: dict[str, list[dict]]) -> dict:
    out = {}
    for cid, results in by_config.items():
        relevant = [
            r for r in results
            if r.get("pipeline_variant") in ("llm_only_a3", "no_rule_engine")
            and not r.get("error")
        ]
        if not relevant:
            continue
        n = len(relevant)
        disagree = sum(
            1 for r in relevant
            if not r.get("llm_only_a3", {}).get("agreed_with_precheck", True)
        )
        out[cid] = {
            "n": n,
            "llm_precheck_disagreement_rate": _safe_div(disagree, n),
            "disagreements": disagree,
        }
    return out


def compute_progress(by_config: dict[str, list[dict]], target: int = 50) -> dict:
    return {
        cid: {
            "done":    len(r),
            "target":  target,
            "pct":     f"{len(r) / target * 100:.0f}%",
            "missing": target - len(r),
        }
        for cid, r in by_config.items()
    }


# ── Formatting ─────────────────────────────────────────────────────────────────

def _p(v: float, decimals: int = 1) -> str:
    """Format float as percentage string."""
    if v != v:  # NaN
        return "  N/A "
    return f"{v * 100:{5 + decimals}.{decimals}f}%"


def _d(v: float) -> str:
    """Format delta."""
    if v != v:
        return "   N/A"
    s = "+" if v >= 0 else ""
    return f"{s}{v * 100:.1f}%"


def build_table(metrics: dict, progress: dict, verifier: dict, llm_a3: dict) -> str:
    per_cfg = metrics["per_config"]
    deltas  = metrics["deltas"]

    # Only show configs that have at least one result, in canonical order
    cols = [c for c in CONFIG_ORDER if c in per_cfg]

    W = 11  # column width
    lines = []
    lines.append("=" * (36 + W * len(cols)))
    lines.append("LIVE ABLATION STUDY — MiFID II Multi-Agent Pipeline")
    lines.append("=" * (36 + W * len(cols)))

    # Header row
    hdr = f"{'Metric':<36}" + "".join(f"{c:>{W}}" for c in cols)
    lines.append(hdr)

    # Config label sub-header
    sub = f"{'':36}" + "".join(
        f"{CONFIG_LABELS.get(c, c)[:W-1]:>{W}}" for c in cols
    )
    lines.append(sub)
    lines.append("-" * (36 + W * len(cols)))

    def row(label, vals):
        return f"{label:<36}" + "".join(f"{v:>{W}}" for v in vals)

    # Progress
    prog_vals = [f"{progress.get(c, {}).get('done', 0)}/{progress.get(c, {}).get('target', 50)}"
                 for c in cols]
    lines.append(row("Progress (done/50)", prog_vals))
    lines.append("-" * (36 + W * len(cols)))

    # M1 overall
    lines.append(row("M1  Decision Accuracy",
                      [_p(per_cfg.get(c, {}).get("M1_accuracy", float("nan"))) for c in cols]))

    # M1c per category
    for cat in CATEGORIES:
        lines.append(row(f"  ↳ {cat}",
                         [_p(per_cfg.get(c, {}).get("M1c_per_category", {}).get(cat, float("nan")))
                          for c in cols]))

    lines.append("-" * (36 + W * len(cols)))

    # M2
    lines.append(row("M2  Unsuitable Recall",
                      [_p(per_cfg.get(c, {}).get("M2_unsuitable_recall", float("nan"))) for c in cols]))
    lines.append(row("M2p Unsuitable Precision",
                      [_p(per_cfg.get(c, {}).get("M2p_precision", float("nan"))) for c in cols]))

    # M3
    lines.append(row("M3f Escalation F1",
                      [_p(per_cfg.get(c, {}).get("M3f_esc_f1", float("nan"))) for c in cols]))

    lines.append("-" * (36 + W * len(cols)))

    # M5 delta
    delta_vals = []
    for c in cols:
        if c == CONTROL_ID:
            delta_vals.append("(ctrl)")
        else:
            delta_vals.append(_d(deltas.get(c, float("nan"))))
    lines.append(row("M5  Δ Accuracy vs P0", delta_vals))

    # M6 parity gap
    lines.append(row("M6  Parity Gap (↓ = fairer)",
                      [_p(per_cfg.get(c, {}).get("M6_parity_gap", float("nan"))) for c in cols]))

    lines.append("-" * (36 + W * len(cols)))
    lines.append("Component Telemetry")
    lines.append("-" * (36 + W * len(cols)))

    lines.append(row("  AV Correction Rate",
                      [_p(verifier.get(c, {}).get("correction_rate", float("nan"))) for c in cols]))
    lines.append(row("  Override Rate",
                      [_p(verifier.get(c, {}).get("override_rate", float("nan"))) for c in cols]))

    # LLM-only A3 disagreement (P4/P7 only)
    llm_vals = []
    for c in cols:
        if c in llm_a3:
            llm_vals.append(_p(llm_a3[c].get("llm_precheck_disagreement_rate", float("nan"))))
        else:
            llm_vals.append("  N/A ")
    lines.append(row("  LLM-A3 ≠ pre_check rate", llm_vals))

    lines.append(row("  Errors / Halts",
                      [str(per_cfg.get(c, {}).get("n_errors", "?")).rjust(W) for c in cols]))

    lines.append("=" * (36 + W * len(cols)))

    # Legend
    lines.append("\nConfig legend:")
    for c in cols:
        from evaluation.ablation_study.live_configs import LIVE_CONFIG_BY_ID
        cfg = LIVE_CONFIG_BY_ID.get(c, {})
        dis = cfg.get("disabled", [])
        dis_str = (", ".join(str(d) for d in dis[:2]) + ("…" if len(dis) > 2 else "")) if dis else "—"
        lines.append(f"  {c}  {CONFIG_LABELS.get(c, c):<30}  disabled: {dis_str}")

    return "\n".join(lines)


def build_csv_rows(metrics: dict) -> list[list[str]]:
    per_cfg = metrics["per_config"]
    deltas  = metrics["deltas"]
    cols    = [c for c in CONFIG_ORDER if c in per_cfg]
    rows    = [["metric"] + cols]

    def add(label, getter):
        rows.append([label] + [str(getter(c)) for c in cols])

    add("M1_accuracy",          lambda c: per_cfg.get(c, {}).get("M1_accuracy", ""))
    for cat in CATEGORIES:
        add(f"M1c_{cat}",       lambda c, cat=cat: per_cfg.get(c, {}).get("M1c_per_category", {}).get(cat, ""))
    add("M2_unsuitable_recall", lambda c: per_cfg.get(c, {}).get("M2_unsuitable_recall", ""))
    add("M2p_precision",        lambda c: per_cfg.get(c, {}).get("M2p_precision", ""))
    add("M3f_escalation_f1",    lambda c: per_cfg.get(c, {}).get("M3f_esc_f1", ""))
    add("M5_delta_vs_P0",       lambda c: deltas.get(c, "control") if c != CONTROL_ID else "control")
    add("M6_parity_gap",        lambda c: per_cfg.get(c, {}).get("M6_parity_gap", ""))
    add("n_correct",            lambda c: per_cfg.get(c, {}).get("n_correct", ""))
    add("n_total",              lambda c: per_cfg.get(c, {}).get("n_total", ""))
    add("n_errors",             lambda c: per_cfg.get(c, {}).get("n_errors", ""))
    return rows


# ── Main ───────────────────────────────────────────────────────────────────────

def main(results_dir: Path, reports_dir: Path, verbose: bool = False) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_dir} ...")
    by_config = load_results(results_dir)

    if not by_config:
        print("No result files found. Run some ablation cells first:")
        print("  python run_ablation_live.py --configs P0 P1 --scenarios 01 07 10")
        sys.exit(0)

    total_files = sum(len(v) for v in by_config.values())
    print(f"Loaded {total_files} cell results across "
          f"{len(by_config)} configs: {sorted(by_config.keys())}\n")

    # Per-config progress
    progress     = compute_progress(by_config, target=50)
    metrics      = compute_all_metrics(by_config)
    verifier     = compute_verifier_impact(by_config)
    llm_a3       = compute_llm_a3_impact(by_config)

    # Progress summary
    print("Progress:")
    for cid in CONFIG_ORDER:
        if cid in progress:
            p = progress[cid]
            bar = "█" * int(p["done"] / 50 * 20) + "░" * (20 - int(p["done"] / 50 * 20))
            print(f"  {cid}  {CONFIG_LABELS.get(cid, cid):<30}  "
                  f"[{bar}]  {p['done']:2d}/50")
    print()

    # Build and print table
    table = build_table(metrics, progress, verifier, llm_a3)
    print(table)

    # Save outputs
    metrics_path = reports_dir / "live_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    verifier_path = reports_dir / "live_verifier_impact.json"
    verifier_path.write_text(json.dumps(verifier, indent=2, default=str), encoding="utf-8")

    llm_a3_path = reports_dir / "live_llm_a3_impact.json"
    llm_a3_path.write_text(json.dumps(llm_a3, indent=2, default=str), encoding="utf-8")

    progress_path = reports_dir / "live_progress.json"
    progress_path.write_text(json.dumps(progress, indent=2, default=str), encoding="utf-8")

    table_path = reports_dir / "live_comparison_table.txt"
    table_path.write_text(table, encoding="utf-8")

    csv_path = reports_dir / "live_comparison_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in build_csv_rows(metrics):
            writer.writerow(row)

    if verbose:
        print("\nPer-scenario breakdown:")
        all_results = [r for v in by_config.values() for r in v]
        scenarios = sorted({r["scenario_id"] for r in all_results})
        for sid in scenarios:
            s_results = [r for r in all_results if r["scenario_id"] == sid]
            cat = s_results[0].get("category", "?") if s_results else "?"
            line = f"  {sid:<50} [{cat:<11}]  "
            for cid in CONFIG_ORDER:
                r = next((x for x in s_results if x.get("config_id") == cid), None)
                if r is None:
                    line += "  .  "
                elif r.get("error"):
                    line += "  E  "
                elif r.get("decision_correct"):
                    line += "  ✓  "
                else:
                    line += "  ✗  "
            print(line)

    print(f"\nSaved to {reports_dir}/")
    print(f"  live_metrics.json")
    print(f"  live_comparison_table.txt")
    print(f"  live_comparison_table.csv")
    print(f"  live_verifier_impact.json")
    print(f"  live_llm_a3_impact.json")
    print(f"  live_progress.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute ablation study results from completed cell files."
    )
    parser.add_argument(
        "--results-dir", default=str(LIVE_RESULTS_DIR),
        help=f"Directory containing cell JSON files (default: {LIVE_RESULTS_DIR})"
    )
    parser.add_argument(
        "--reports-dir", default=str(LIVE_REPORTS_DIR),
        help=f"Directory for output reports (default: {LIVE_REPORTS_DIR})"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Also print per-scenario breakdown table."
    )
    args = parser.parse_args()
    main(Path(args.results_dir), Path(args.reports_dir), args.verbose)
