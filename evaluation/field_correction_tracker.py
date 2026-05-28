"""
evaluation/field_correction_tracker.py
========================================
Computes per-field correction rates from evaluation results or audit logs.

Purpose: identify which A1 (client profile) and A2 (product profile) fields
are most frequently corrected by the verifier.  This information is used to:

  1. Focus verifier scrutiny on high-error-rate fields (adaptive verification).
  2. Identify which fields to prioritise in supervised fine-tuning of A1/A2.
  3. Track drift in field error rates across model versions.

Usage
-----
  # From evaluation result files:
  python -m evaluation.field_correction_tracker

  # Programmatically:
  from evaluation.field_correction_tracker import compute_field_correction_rates
  rates = compute_field_correction_rates(pipeline_results)
  print(rates["a1"]["by_field"])

Input
-----
Pipeline result dicts must contain `a1_correction` and/or `a2_correction` keys
(as produced by orchestrator.graph and stored in audit/eval output files).

Each correction dict has:
  {
    "original":     {field: value, ...},
    "corrected":    {field: value, ...},
    "fields_fixed": ["field_name_1", ...]
  }

If "fields_fixed" is not present, the tracker falls back to comparing
"original" and "corrected" dicts directly.

Output
------
{
  "a1": {
    "total_runs":          int,
    "runs_with_correction": int,
    "correction_rate":      float,        # runs_with_correction / total_runs
    "by_field": {
      "financial_knowledge":    {"corrections": int, "rate": float},
      "risk_tolerance_score":   {"corrections": int, "rate": float},
      "investment_horizon":     {"corrections": int, "rate": float},
      "liquid_assets":          {"corrections": int, "rate": float},
      "income":                 {"corrections": int, "rate": float},
      "investment_amount":      {"corrections": int, "rate": float},
      "can_afford_total_loss":  {"corrections": int, "rate": float},
      "financial_vulnerability":{"corrections": int, "rate": float},
      "age":                    {"corrections": int, "rate": float},
    },
    "highest_error_fields": ["field1", "field2", ...]  # sorted by rate desc
  },
  "a2": { ... same structure for product fields ... }
}
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


# ── Known field sets ───────────────────────────────────────────────────────────

A1_FIELDS = [
    "financial_knowledge",
    "risk_tolerance_score",
    "investment_horizon",
    "liquid_assets",
    "income",
    "investment_amount",
    "can_afford_total_loss",
    "financial_vulnerability",
    "age",
]

A2_FIELDS = [
    "product_name",
    "risk_class",
    "complexity_tier",
    "requires_knowledge_level",
    "minimum_horizon",
    "potential_loss",
    "leverage",
]


# ── Core computation ────────────────────────────────────────────────────────────

def _extract_corrected_fields(correction: dict, known_fields: list[str]) -> list[str]:
    """
    Return the list of field names that were changed by the verifier correction.

    Primary source: correction["fields_fixed"] (populated by graph.py).
    Fallback: compare original vs corrected dicts directly.
    """
    if correction.get("corrected") is None:
        return []

    if "fields_fixed" in correction and correction["fields_fixed"]:
        return [f for f in correction["fields_fixed"] if f in known_fields]

    original  = correction.get("original", {})
    corrected = correction.get("corrected", {})
    return [f for f in known_fields if original.get(f) != corrected.get(f)]


def compute_field_correction_rates(
    pipeline_results: list[dict],
) -> dict:
    """
    Compute per-field correction rates for A1 and A2 from a list of pipeline
    result dicts (as produced by evaluation/evaluator.py or read from audit logs).

    Each result dict may contain:
      - state["a1_correction"] or result["a1_correction"]
      - state["a2_correction"] or result["a2_correction"]
    """
    a1_counts: dict[str, int] = defaultdict(int)
    a2_counts: dict[str, int] = defaultdict(int)
    a1_runs = 0
    a2_runs = 0
    a1_corrected_runs = 0
    a2_corrected_runs = 0

    for result in pipeline_results:
        # Results may be direct state dicts or evaluator result dicts that
        # contain nested state — handle both.
        a1_cor = result.get("a1_correction")
        a2_cor = result.get("a2_correction")

        if a1_cor is not None:
            a1_runs += 1
            fields = _extract_corrected_fields(a1_cor, A1_FIELDS)
            if fields:
                a1_corrected_runs += 1
                for f in fields:
                    a1_counts[f] += 1

        if a2_cor is not None:
            a2_runs += 1
            fields = _extract_corrected_fields(a2_cor, A2_FIELDS)
            if fields:
                a2_corrected_runs += 1
                for f in fields:
                    a2_counts[f] += 1

    def _build_section(runs, corrected_runs, counts, field_list):
        by_field = {
            f: {
                "corrections": counts.get(f, 0),
                "rate": counts.get(f, 0) / runs if runs else 0.0,
            }
            for f in field_list
        }
        sorted_fields = sorted(
            field_list,
            key=lambda f: by_field[f]["rate"],
            reverse=True,
        )
        return {
            "total_runs":            runs,
            "runs_with_correction":  corrected_runs,
            "correction_rate":       corrected_runs / runs if runs else 0.0,
            "by_field":              by_field,
            "highest_error_fields":  [f for f in sorted_fields if by_field[f]["rate"] > 0],
        }

    return {
        "a1": _build_section(a1_runs, a1_corrected_runs, a1_counts, A1_FIELDS),
        "a2": _build_section(a2_runs, a2_corrected_runs, a2_counts, A2_FIELDS),
    }


def load_pipeline_results_from_eval_dir(eval_dir: Path) -> list[dict]:
    """
    Load all pipeline result JSON files from the evaluation output directory.
    Expects files matching pattern results_pipeline_*.json.
    """
    results = []
    for path in sorted(eval_dir.glob("results_pipeline_*.json")):
        data = json.loads(path.read_text())
        if isinstance(data, list):
            results.extend(data)
    return results


def load_pipeline_results_from_audit_dir(audit_dir: Path) -> list[dict]:
    """
    Load audit records and extract a1_correction/a2_correction from state snapshots.
    Expects files matching pattern run_*.json or audit_*.json.
    """
    results = []
    for path in sorted(audit_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text())
            entry = {
                "a1_correction": record.get("a1_correction"),
                "a2_correction": record.get("a2_correction"),
            }
            results.append(entry)
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def print_report(rates: dict) -> None:
    """Print a human-readable correction rate report."""
    for agent, label in [("a1", "A1 — Client Profiler"), ("a2", "A2 — Product Classifier")]:
        section = rates[agent]
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        print(f"  Runs with verifier:     {section['total_runs']}")
        print(f"  Runs with correction:   {section['runs_with_correction']}")
        print(f"  Overall correction rate: {section['correction_rate']:.1%}")
        print()
        print(f"  {'Field':<30} {'Corrections':>12} {'Rate':>8}")
        print(f"  {'-'*52}")
        for field in (section["highest_error_fields"] or
                      [f for f in (A1_FIELDS if agent == "a1" else A2_FIELDS)]):
            info = section["by_field"][field]
            bar = "█" * int(info["rate"] * 20)
            print(f"  {field:<30} {info['corrections']:>12}   {info['rate']:.1%}  {bar}")


# ── CLI entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute per-field correction rates from evaluation results."
    )
    parser.add_argument(
        "--eval-dir",
        default="data/eval",
        help="Directory containing results_pipeline_*.json files",
    )
    parser.add_argument(
        "--audit-dir",
        default=None,
        help="Directory containing audit log JSON files (optional, in addition to eval-dir)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write JSON report to this path (default: print only)",
    )
    args = parser.parse_args()

    all_results: list[dict] = []

    eval_path = Path(args.eval_dir)
    if eval_path.exists():
        all_results.extend(load_pipeline_results_from_eval_dir(eval_path))
        print(f"Loaded {len(all_results)} results from {eval_path}")

    if args.audit_dir:
        audit_path = Path(args.audit_dir)
        if audit_path.exists():
            audit_results = load_pipeline_results_from_audit_dir(audit_path)
            all_results.extend(audit_results)
            print(f"Loaded {len(audit_results)} audit records from {audit_path}")

    if not all_results:
        print("No results found. Run the evaluator first or specify --audit-dir.")
    else:
        rates = compute_field_correction_rates(all_results)
        print_report(rates)

        if args.output:
            output_path = Path(args.output)
            output_path.write_text(json.dumps(rates, indent=2))
            print(f"\nReport saved to {output_path}")
