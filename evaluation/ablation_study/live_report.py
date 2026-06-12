"""
evaluation/ablation_study/live_report.py
=========================================
Generates the comparison table and metric summaries from live ablation results.

Reads from  : data/ablation_results/live/
Writes to   : data/ablation_reports/live/
  - live_metrics.json        — full metrics dict
  - live_comparison_table.txt — human-readable table
  - live_comparison_table.csv — machine-readable table
  - live_verifier_impact.json — verifier/correction impact analysis

Usage
-----
  python -m evaluation.ablation_study.live_report
  python -m evaluation.ablation_study.live_report --results-dir data/ablation_results/live
  python -m evaluation.ablation_study.live_report --output-dir  data/ablation_reports/live
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evaluation.ablation_study.live_configs import LIVE_CONFIGS, LIVE_CONFIG_BY_ID
from evaluation.ablation_study.live_runner import (
    load_live_results,
    compute_live_metrics,
    compute_verifier_impact,
)

LIVE_RESULTS_DIR = Path("data/ablation_results/live")
LIVE_REPORTS_DIR = Path("data/ablation_reports/live")

CATEGORIES = ("SUITABLE", "CONDITIONAL", "UNSUITABLE", "ESCALATED")


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _fmt(v: float | None, pct: bool = True, decimals: int = 1) -> str:
    if v is None or v != v:  # None or NaN
        return " N/A"
    if pct:
        return f"{v * 100:{decimals + 4}.{decimals}f}%"
    return f"{v:{decimals + 4}.{decimals}f}"


def _fmt_delta(v: float | None) -> str:
    if v is None or v != v:
        return "   N/A"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.1f}%"


# ── Table builder ──────────────────────────────────────────────────────────────

def build_comparison_table(
    metrics: dict,
    results_by_config: dict[str, list[dict]],
    verifier_impact: dict,
) -> str:
    """
    Build the human-readable comparison table.

    Rows: metrics (M1, M1c per category, M2, M3f, M5 delta, correction rate, override rate)
    Columns: C0 (control) then C1–C7
    """
    per_cfg     = metrics.get("per_config", {})
    deltas      = metrics.get("deltas", {})
    config_ids  = [c["id"] for c in LIVE_CONFIGS]

    lines = []
    lines.append("=" * 120)
    lines.append("LIVE ABLATION STUDY — MiFID II Multi-Agent Pipeline")
    lines.append("=" * 120)

    # Header
    hdr_labels = [f"{c['id']:>10}" for c in LIVE_CONFIGS]
    lines.append(f"{'Metric':<38}" + "".join(hdr_labels))
    lines.append("-" * 120)

    def row(label, values):
        return f"{label:<38}" + "".join(f"{v:>10}" for v in values)

    # M1 — overall accuracy
    vals = [_fmt(per_cfg.get(cid, {}).get("M1_decision_accuracy")) for cid in config_ids]
    lines.append(row("M1  Decision Accuracy", vals))

    # M1c per category
    for cat in CATEGORIES:
        vals = [
            _fmt(per_cfg.get(cid, {}).get("M1c_per_category", {}).get(cat))
            for cid in config_ids
        ]
        lines.append(row(f"  M1c  {cat}", vals))

    lines.append("-" * 120)

    # M2 — hard-fail recall
    vals = [_fmt(per_cfg.get(cid, {}).get("M2_unsuitable_recall")) for cid in config_ids]
    lines.append(row("M2  Unsuitable Recall", vals))

    vals = [_fmt(per_cfg.get(cid, {}).get("M2p_unsuitable_precision")) for cid in config_ids]
    lines.append(row("M2p Unsuitable Precision", vals))

    # M3 — escalation
    vals = [_fmt(per_cfg.get(cid, {}).get("M3_escalation_recall")) for cid in config_ids]
    lines.append(row("M3  Escalation Recall", vals))

    vals = [_fmt(per_cfg.get(cid, {}).get("M3p_escalation_precision")) for cid in config_ids]
    lines.append(row("M3p Escalation Precision", vals))

    vals = [_fmt(per_cfg.get(cid, {}).get("M3f_escalation_f1")) for cid in config_ids]
    lines.append(row("M3f Escalation F1", vals))

    lines.append("-" * 120)

    # M5 — delta vs C0
    vals = []
    for cid in config_ids:
        if cid == "C0":
            vals.append("  (ctrl)")
        else:
            vals.append(_fmt_delta(deltas.get(cid)))
    lines.append(row("M5  Δ vs C0 (accuracy)", vals))

    # M6 — parity gap
    vals = [_fmt(per_cfg.get(cid, {}).get("M6_parity_gap")) for cid in config_ids]
    lines.append(row("M6  Parity Gap (lower=fairer)", vals))

    lines.append("-" * 120)
    lines.append("Pipeline Component Telemetry")
    lines.append("-" * 120)

    # Correction rate (live only)
    vals = [
        _fmt(verifier_impact.get(cid, {}).get("correction_rate"), pct=True)
        for cid in config_ids
    ]
    lines.append(row("  Correction Rate", vals))

    # Override rate (live only)
    vals = [
        _fmt(verifier_impact.get(cid, {}).get("override_rate"), pct=True)
        for cid in config_ids
    ]
    lines.append(row("  Override Rate", vals))

    # Error count
    vals = [str(per_cfg.get(cid, {}).get("n_errors", "?")).rjust(9) for cid in config_ids]
    lines.append(row("  Errors / Halts", vals))

    lines.append("=" * 120)

    # Config legend
    lines.append("\nConfig legend:")
    for cfg in LIVE_CONFIGS:
        disabled = (", ".join(cfg["disabled"][:3])
                    + ("…" if len(cfg["disabled"]) > 3 else ""))
        lines.append(f"  {cfg['id']:4s}  {cfg['label']:<40}  disabled: [{disabled}]")

    return "\n".join(lines)


def build_csv(metrics: dict, verifier_impact: dict) -> list[list[str]]:
    """Build CSV rows for the comparison table."""
    per_cfg    = metrics.get("per_config", {})
    deltas     = metrics.get("deltas", {})
    config_ids = [c["id"] for c in LIVE_CONFIGS]

    header = ["metric"] + config_ids
    rows   = [header]

    def add_row(label, getter):
        rows.append([label] + [str(getter(cid)) for cid in config_ids])

    add_row("M1_decision_accuracy",
            lambda cid: per_cfg.get(cid, {}).get("M1_decision_accuracy", ""))

    for cat in CATEGORIES:
        add_row(f"M1c_{cat}",
                lambda cid, c=cat: per_cfg.get(cid, {}).get("M1c_per_category", {}).get(c, ""))

    add_row("M2_unsuitable_recall",
            lambda cid: per_cfg.get(cid, {}).get("M2_unsuitable_recall", ""))
    add_row("M2p_unsuitable_precision",
            lambda cid: per_cfg.get(cid, {}).get("M2p_unsuitable_precision", ""))
    add_row("M3_escalation_recall",
            lambda cid: per_cfg.get(cid, {}).get("M3_escalation_recall", ""))
    add_row("M3p_escalation_precision",
            lambda cid: per_cfg.get(cid, {}).get("M3p_escalation_precision", ""))
    add_row("M3f_escalation_f1",
            lambda cid: per_cfg.get(cid, {}).get("M3f_escalation_f1", ""))
    add_row("M5_delta_vs_C0",
            lambda cid: deltas.get(cid, "") if cid != "C0" else "control")
    add_row("M6_parity_gap",
            lambda cid: per_cfg.get(cid, {}).get("M6_parity_gap", ""))
    add_row("correction_rate",
            lambda cid: verifier_impact.get(cid, {}).get("correction_rate", ""))
    add_row("override_rate",
            lambda cid: verifier_impact.get(cid, {}).get("override_rate", ""))
    add_row("n_errors",
            lambda cid: per_cfg.get(cid, {}).get("n_errors", ""))

    return rows


# ── Main ───────────────────────────────────────────────────────────────────────

def main(
    results_dir: Path = LIVE_RESULTS_DIR,
    output_dir:  Path = LIVE_REPORTS_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading live results from {results_dir}...")
    results_by_config = load_live_results(results_dir)
    if not results_by_config:
        print("No results found. Run run_ablation_live.py first.")
        return

    n_configs = len(results_by_config)
    n_cells   = sum(len(v) for v in results_by_config.values())
    print(f"  Loaded {n_cells} cells across {n_configs} configs: "
          + ", ".join(sorted(results_by_config.keys())))

    print("Computing metrics...")
    metrics         = compute_live_metrics(results_by_config)
    verifier_impact = compute_verifier_impact(results_by_config)

    # Write full metrics JSON
    metrics_path = output_dir / "live_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(f"  → {metrics_path}")

    # Write verifier impact JSON
    vi_path = output_dir / "live_verifier_impact.json"
    vi_path.write_text(json.dumps(verifier_impact, indent=2, default=str), encoding="utf-8")
    print(f"  → {vi_path}")

    # Write comparison table (text)
    table = build_comparison_table(metrics, results_by_config, verifier_impact)
    table_path = output_dir / "live_comparison_table.txt"
    table_path.write_text(table, encoding="utf-8")
    print(f"  → {table_path}")
    print()
    print(table)

    # Write CSV
    csv_path = output_dir / "live_comparison_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in build_csv(metrics, verifier_impact):
            writer.writerow(row)
    print(f"  → {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate live ablation report.")
    parser.add_argument("--results-dir", default=str(LIVE_RESULTS_DIR))
    parser.add_argument("--output-dir",  default=str(LIVE_REPORTS_DIR))
    args = parser.parse_args()
    main(Path(args.results_dir), Path(args.output_dir))
