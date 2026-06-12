"""
evaluation/ablation_study/report.py
=====================================
Loads ablation results from data/ablation_results/ and produces:
  1. Per-config metrics summary (JSON)
  2. Human-readable comparison table (plain text + CSV)
  3. Axis-level summary tables (architecture / rule / threshold)

No LLM calls. Pure I/O + computation.

Usage
-----
  python -m evaluation.ablation_study.report
  python -m evaluation.ablation_study.report --results-dir data/ablation_results
  python -m evaluation.ablation_study.report --output-dir  data/ablation_reports
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

from evaluation.ablation_study.configs import UNIQUE_CONFIGS, CONFIG_BY_ID
from evaluation.ablation_study.metrics import compute_all_metrics, CATEGORIES

RESULTS_DIR = Path("data/ablation_results")
REPORTS_DIR = Path("data/ablation_reports")


# ── Load results ───────────────────────────────────────────────────────────────

def load_results(results_dir: Path = RESULTS_DIR) -> dict[str, list[dict]]:
    """
    Load all cell result JSON files from results_dir.
    Returns {config_id: [result, ...]} grouped by config.
    """
    by_config: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(results_dir.glob("*.json")):
        if p.name == "ablation_manifest.json":
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        config_id = data.get("config_id")
        if config_id:
            by_config[config_id].append(data)
    return dict(by_config)


# ── Table rendering ────────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    if v != v:   # NaN
        return "  N/A"
    return f"{v * 100:5.1f}%"


def _delta(v: float) -> str:
    if v != v:
        return "   N/A"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:5.1f}%"


def build_comparison_table(metrics: dict) -> str:
    """Render a plain-text comparison table for all configs."""
    per_config = metrics["per_config"]
    deltas     = metrics["deltas"]
    control_id = metrics["control_id"]

    # Order: control first, then by config id
    config_order = [control_id] + [
        c["id"] for c in UNIQUE_CONFIGS if c["id"] != control_id
        and c["id"] in per_config
    ]
    # Remove duplicates
    seen = set()
    ordered = []
    for cid in config_order:
        if cid not in seen and cid in per_config:
            ordered.append(cid)
            seen.add(cid)

    col_w = 13
    id_w  = 12

    def col(s: str) -> str:
        return str(s).center(col_w)

    header_ids = [c[:id_w].ljust(id_w) for c in ordered]

    lines = []
    lines.append("=" * (id_w + 2 + col_w * len(ordered) + len(ordered)))

    # Config IDs row
    lines.append(
        "Metric".ljust(id_w + 2) +
        "".join(col(c) for c in ordered)
    )
    lines.append("-" * (id_w + 2 + col_w * len(ordered) + len(ordered)))

    def row(label: str, values: list[str]) -> str:
        return label[:id_w + 1].ljust(id_w + 2) + "".join(v.center(col_w) for v in values)

    # M1 Decision Accuracy
    lines.append(row(
        "M1 Accuracy",
        [_pct(per_config[c]["M1_decision_accuracy"]) for c in ordered]
    ))

    # M1c per category
    for cat in CATEGORIES:
        lines.append(row(
            f"  {cat[:10]}",
            [_pct(per_config[c]["M1c_per_category"].get(cat, float("nan"))) for c in ordered]
        ))

    lines.append("-" * (id_w + 2 + col_w * len(ordered) + len(ordered)))

    # M2 Unsuitable recall / precision
    lines.append(row(
        "M2 Unsut.Rec",
        [_pct(per_config[c]["M2_unsuitable_recall"]) for c in ordered]
    ))
    lines.append(row(
        "M2p Unsut.Pre",
        [_pct(per_config[c]["M2p_unsuitable_precision"]) for c in ordered]
    ))

    lines.append("-" * (id_w + 2 + col_w * len(ordered) + len(ordered)))

    # M3 Escalation
    lines.append(row(
        "M3 Escl.Rec",
        [_pct(per_config[c]["M3_escalation_recall"]) for c in ordered]
    ))
    lines.append(row(
        "M3p Escl.Pre",
        [_pct(per_config[c]["M3p_escalation_precision"]) for c in ordered]
    ))
    lines.append(row(
        "M3f Escl.F1",
        [_pct(per_config[c]["M3f_escalation_f1"]) for c in ordered]
    ))

    lines.append("-" * (id_w + 2 + col_w * len(ordered) + len(ordered)))

    # M5 Component delta
    lines.append(row(
        "M5 Delta vs A0",
        [
            ("  —" if c == control_id else _delta(deltas.get(c, float("nan"))))
            for c in ordered
        ]
    ))

    # M6 Parity gap
    lines.append(row(
        "M6 Parity Gap",
        [_pct(per_config[c]["M6_parity_gap"]) for c in ordered]
    ))

    lines.append("=" * (id_w + 2 + col_w * len(ordered) + len(ordered)))
    return "\n".join(lines)


def build_axis_summary(metrics: dict) -> str:
    """Render axis-level summary: best and worst config per axis."""
    per_config = metrics["per_config"]
    lines      = []

    axes = {"architecture": [], "rule": [], "threshold": []}
    for c in UNIQUE_CONFIGS:
        if c["id"] in per_config:
            axes.get(c["axis"], []).append(c["id"])

    for axis, config_ids in axes.items():
        if not config_ids:
            continue
        lines.append(f"\n── {axis.upper()} AXIS ──")
        sorted_by_acc = sorted(
            config_ids,
            key=lambda cid: per_config[cid]["M1_decision_accuracy"],
            reverse=True,
        )
        for cid in sorted_by_acc:
            m  = per_config[cid]
            cfg = CONFIG_BY_ID.get(cid, {})
            delta = metrics["deltas"].get(cid, 0.0)
            lines.append(
                f"  {cid:12s}  M1={_pct(m['M1_decision_accuracy'])}  "
                f"Δ={_delta(delta)}  F1={_pct(m['M3f_escalation_f1'])}  "
                f"  {cfg.get('label', '')}"
            )
    return "\n".join(lines)


# ── CSV export ─────────────────────────────────────────────────────────────────

def write_csv(metrics: dict, output_path: Path) -> None:
    """Write metrics to CSV for spreadsheet analysis."""
    per_config = metrics["per_config"]
    deltas     = metrics["deltas"]
    control_id = metrics["control_id"]

    fieldnames = [
        "config_id", "axis", "label", "is_control",
        "M1_accuracy",
        "M1c_SUITABLE", "M1c_CONDITIONAL", "M1c_UNSUITABLE", "M1c_ESCALATED",
        "M2_unsuitable_recall", "M2p_unsuitable_precision",
        "M3_escalation_recall", "M3p_escalation_precision", "M3f_escalation_f1",
        "M5_delta_vs_control",
        "M6_parity_gap",
        "n_cells", "n_errors",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in UNIQUE_CONFIGS:
            cid = c["id"]
            if cid not in per_config:
                continue
            m   = per_config[cid]
            cat = m["M1c_per_category"]
            writer.writerow({
                "config_id":    cid,
                "axis":         c["axis"],
                "label":        c["label"],
                "is_control":   c["is_control"],
                "M1_accuracy":  round(m["M1_decision_accuracy"], 4),
                "M1c_SUITABLE":    round(cat.get("SUITABLE",    float("nan")), 4),
                "M1c_CONDITIONAL": round(cat.get("CONDITIONAL", float("nan")), 4),
                "M1c_UNSUITABLE":  round(cat.get("UNSUITABLE",  float("nan")), 4),
                "M1c_ESCALATED":   round(cat.get("ESCALATED",   float("nan")), 4),
                "M2_unsuitable_recall":     round(m["M2_unsuitable_recall"],     4),
                "M2p_unsuitable_precision": round(m["M2p_unsuitable_precision"], 4),
                "M3_escalation_recall":     round(m["M3_escalation_recall"],     4),
                "M3p_escalation_precision": round(m["M3p_escalation_precision"], 4),
                "M3f_escalation_f1":        round(m["M3f_escalation_f1"],        4),
                "M5_delta_vs_control":  round(deltas.get(cid, float("nan")), 4),
                "M6_parity_gap":        round(m["M6_parity_gap"], 4),
                "n_cells":  m["n_cells"],
                "n_errors": m["n_errors"],
            })


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ablation study report from saved results."
    )
    parser.add_argument(
        "--results-dir", default=str(RESULTS_DIR),
        help=f"Directory containing cell result JSONs (default: {RESULTS_DIR})"
    )
    parser.add_argument(
        "--output-dir", default=str(REPORTS_DIR),
        help=f"Where to write reports (default: {REPORTS_DIR})"
    )
    parser.add_argument(
        "--control", default="A0",
        help="Config ID to use as reference for delta computation (default: A0)"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_dir}...")
    by_config = load_results(results_dir)

    if not by_config:
        print("No results found. Run the ablation study first:")
        print("  python -m evaluation.ablation_study.runner")
        return

    print(f"Loaded results for {len(by_config)} configs, "
          f"{sum(len(v) for v in by_config.values())} cells total.")

    metrics = compute_all_metrics(by_config, control_id=args.control)

    # ── Text table ──────────────────────────────────────────────────────────
    table   = build_comparison_table(metrics)
    axis_s  = build_axis_summary(metrics)
    print("\n" + table)
    print(axis_s)

    table_path = output_dir / "ablation_comparison_table.txt"
    table_path.write_text(table + "\n\n" + axis_s, encoding="utf-8")
    print(f"\nTable written to: {table_path}")

    # ── JSON summary ─────────────────────────────────────────────────────────
    json_path = output_dir / "ablation_metrics_summary.json"
    json_path.write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    print(f"JSON written to:  {json_path}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = output_dir / "ablation_metrics_table.csv"
    write_csv(metrics, csv_path)
    print(f"CSV written to:   {csv_path}")


if __name__ == "__main__":
    main()
