"""
evaluation/fine_tuning_prep.py
================================
Export labeled training examples for supervised fine-tuning (SFT) of A1 and A2.

Rationale
---------
A1 (Client Profiler) and A2 (Product Classifier) are the only agents that
make free-form extraction decisions.  The verifier corrects them when they err,
but a better-tuned specialist model would need fewer corrections — directly
improving M5 (verification correction rate) and, transitively, M1/M2.

This script turns existing scenario fixtures into JSONL training files suitable
for fine-tuning via the Groq/OpenAI fine-tuning API or any chat-format SFT
framework.  It does NOT touch rules, thresholds, or any decision-layer agent.

Output format
-------------
Each line is a JSON object with the chat-format used by most fine-tuning APIs:
{
  "messages": [
    {"role": "system", "content": "<system prompt>"},
    {"role": "user",   "content": "<raw input text>"},
    {"role": "assistant", "content": "<expected structured JSON output>"}
  ],
  "metadata": {
    "scenario_id":        "01_suitable_conservative",
    "agent":              "A1",
    "source":             "scenario_fixture",
    "expected_decision":  "SUITABLE"
  }
}

Usage
-----
  python -m evaluation.fine_tuning_prep
  python -m evaluation.fine_tuning_prep --output data/fine_tuning/
  python -m evaluation.fine_tuning_prep --agent A1      # A1 examples only
  python -m evaluation.fine_tuning_prep --agent A2      # A2 examples only

Limitations
-----------
- Scenario fixtures provide ground-truth client/product profiles.  They do NOT
  include raw free-text descriptions, so this script generates a canonical text
  representation from the structured JSON.  Real fine-tuning benefits most from
  diverse natural language inputs — augment with paraphrased versions before use.
- A1 examples use the client dict; A2 examples use the product JSON.
- This script is idempotent — re-running with the same scenarios produces the
  same output.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from agents.client_profiler import CLIENT_PROFILER_SYSTEM_PROMPT
from agents.product_classifier import PRODUCT_CLASSIFIER_SYSTEM_PROMPT


SCENARIOS_DIR = Path("data/scenarios")
PRODUCTS_DIR  = Path("data/products")
DEFAULT_OUT   = Path("data/fine_tuning")

# Fields included in A1 target output (matches ClientProfileModel)
A1_OUTPUT_FIELDS = [
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

# Fields included in A2 target output (matches ProductProfileModel)
A2_OUTPUT_FIELDS = [
    "product_name",
    "risk_class",
    "complexity_tier",
    "requires_knowledge_level",
    "minimum_horizon",
    "potential_loss",
    "leverage",
]


# ── Text generation ─────────────────────────────────────────────────────────────

def _client_dict_to_text(client: dict) -> str:
    """
    Convert a structured client dict to a natural-language text description
    suitable as the A1 user input.

    This produces a canonical representation.  For fine-tuning diversity,
    generate paraphrases of this text before using for SFT.
    """
    name   = client.get("name", "The client")
    age    = client.get("age")
    age_str = f", aged {age}," if age else ""

    knowledge = client.get("financial_knowledge", "unknown")
    risk      = client.get("risk_tolerance_score", 5)
    horizon   = client.get("investment_horizon", 5)
    liquid    = client.get("liquid_assets", 0.0)
    income    = client.get("income", 0.0)
    invest    = client.get("investment_amount", 0.0)
    afford    = client.get("can_afford_total_loss", False)
    vuln      = client.get("financial_vulnerability", "LOW")
    conc      = client.get("portfolio_concentration_pct")
    obj       = client.get("investment_objective", "growth")

    afford_str = (
        "I am comfortable with the possibility of losing my entire investment."
        if afford else
        "I cannot afford to lose the entire investment amount."
    )
    vuln_str = {
        "HIGH":   "I have significant financial obligations and consider myself financially vulnerable.",
        "MEDIUM": "I have some upcoming financial commitments.",
        "LOW":    "My financial situation is stable.",
    }.get(vuln, "")
    conc_str = (
        f" Approximately {conc}% of my portfolio is in a single asset class."
        if conc is not None else ""
    )

    return (
        f"{name}{age_str} is seeking investment advice. "
        f"My investment objective is {obj}. "
        f"I have {knowledge} financial knowledge and experience. "
        f"My risk tolerance is {risk} out of 10. "
        f"I plan to invest for {horizon} year(s). "
        f"I have EUR {liquid:,.0f} in liquid assets and earn EUR {income:,.0f} per year. "
        f"I would like to invest EUR {invest:,.0f}. "
        f"{afford_str} "
        f"{vuln_str}"
        f"{conc_str}"
    ).strip()


def _product_dict_to_text(product: dict) -> str:
    """
    Convert a structured product dict to a description suitable as A2 user input.
    """
    name       = product.get("name", product.get("product_name", "Unknown Product"))
    desc       = product.get("description", "")
    risk_class = product.get("risk_class", "unknown")
    complexity = product.get("complexity_tier", "NON-COMPLEX")
    leverage   = product.get("leverage", False)
    loss       = product.get("potential_loss", "partial")
    horizon    = product.get("minimum_horizon", 1)
    knowledge  = product.get("requires_knowledge_level", "basic")

    lev_str   = " It uses leverage." if leverage else ""
    loss_str  = f"The potential loss is {loss}."
    compl_str = f"This is a {complexity.lower().replace('-', ' ')} product."

    return (
        f"Product: {name}. "
        + (f"{desc} " if desc else "")
        + f"Risk class: {risk_class} on the ESMA PRIIP 1-7 scale. "
        f"{compl_str}"
        f"{lev_str} "
        f"{loss_str} "
        f"Minimum recommended investment horizon: {horizon} year(s). "
        f"Required client knowledge level: {knowledge}."
    ).strip()


# ── Example builders ────────────────────────────────────────────────────────────

def _build_a1_example(scenario: dict, scenario_id: str) -> dict:
    client = scenario["client"]
    user_text = _client_dict_to_text(client)

    # Ground truth: only the fields A1 is expected to output.
    target = {k: client[k] for k in A1_OUTPUT_FIELDS if k in client}

    return {
        "messages": [
            {"role": "system",    "content": CLIENT_PROFILER_SYSTEM_PROMPT},
            {"role": "user",      "content": user_text},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ],
        "metadata": {
            "scenario_id":       scenario_id,
            "agent":             "A1",
            "source":            "scenario_fixture",
            "expected_decision": scenario.get("expected_decision"),
        },
    }


def _build_a2_example(scenario: dict, product: dict, scenario_id: str) -> dict:
    user_text = _product_dict_to_text(product)

    # Ground truth: only the fields A2 is expected to output.
    target = {k: product[k] for k in A2_OUTPUT_FIELDS if k in product}
    # product_name falls back to "name" key in product JSON
    if "product_name" not in target and "name" in product:
        target["product_name"] = product["name"]

    return {
        "messages": [
            {"role": "system",    "content": PRODUCT_CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user",      "content": user_text},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ],
        "metadata": {
            "scenario_id":       scenario_id,
            "agent":             "A2",
            "source":            "scenario_fixture",
            "expected_decision": scenario.get("expected_decision"),
        },
    }


# ── Main export ─────────────────────────────────────────────────────────────────

def export_fine_tuning_data(
    scenarios_dir: Path = SCENARIOS_DIR,
    products_dir:  Path = PRODUCTS_DIR,
    output_dir:    Path = DEFAULT_OUT,
    agents:        list[str] | None = None,
) -> dict[str, int]:
    """
    Export JSONL fine-tuning files for A1 and/or A2.

    Returns dict with counts: {"A1": n_examples, "A2": n_examples}.
    """
    if agents is None:
        agents = ["A1", "A2"]

    output_dir.mkdir(parents=True, exist_ok=True)

    a1_examples: list[dict] = []
    a2_examples: list[dict] = []

    scenario_files = sorted(scenarios_dir.glob("*.json"))
    for sc_path in scenario_files:
        try:
            scenario    = json.loads(sc_path.read_text())
            product_file = scenario.get("product_file")
            if not product_file:
                continue
            product = json.loads((products_dir / product_file).read_text())
        except (json.JSONDecodeError, FileNotFoundError) as exc:
            print(f"  Skipping {sc_path.name}: {exc}")
            continue

        sc_id = sc_path.stem

        if "A1" in agents and "client" in scenario:
            a1_examples.append(_build_a1_example(scenario, sc_id))

        if "A2" in agents:
            a2_examples.append(_build_a2_example(scenario, product, sc_id))

    counts = {}

    if "A1" in agents:
        a1_path = output_dir / "a1_client_profiler.jsonl"
        a1_path.write_text(
            "\n".join(json.dumps(ex, ensure_ascii=False) for ex in a1_examples),
            encoding="utf-8",
        )
        counts["A1"] = len(a1_examples)
        print(f"  A1: {len(a1_examples)} examples -> {a1_path}")

    if "A2" in agents:
        a2_path = output_dir / "a2_product_classifier.jsonl"
        a2_path.write_text(
            "\n".join(json.dumps(ex, ensure_ascii=False) for ex in a2_examples),
            encoding="utf-8",
        )
        counts["A2"] = len(a2_examples)
        print(f"  A2: {len(a2_examples)} examples -> {a2_path}")

    # Write a manifest so the output is self-documenting.
    manifest = {
        "description": (
            "Supervised fine-tuning examples for A1 (client profiler) and "
            "A2 (product classifier) in the MiFID II suitability pipeline."
        ),
        "warning": (
            "These examples use canonical text generated from structured JSON. "
            "Augment with paraphrased inputs before using for production SFT."
        ),
        "format":    "JSONL — one JSON object per line, chat-message format",
        "agents":    agents,
        "counts":    counts,
        "source_scenarios": [p.name for p in scenario_files],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return counts


# ── CLI entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export SFT training data for A1 and A2 from scenario fixtures."
    )
    parser.add_argument(
        "--scenarios", default=str(SCENARIOS_DIR),
        help="Directory containing scenario JSON files",
    )
    parser.add_argument(
        "--products", default=str(PRODUCTS_DIR),
        help="Directory containing product JSON files",
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUT),
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--agent", choices=["A1", "A2"], default=None,
        help="Export only A1 or A2 examples (default: both)",
    )
    args = parser.parse_args()

    agents = [args.agent] if args.agent else ["A1", "A2"]
    print(f"\nExporting fine-tuning data for {agents}...")
    counts = export_fine_tuning_data(
        scenarios_dir=Path(args.scenarios),
        products_dir=Path(args.products),
        output_dir=Path(args.output),
        agents=agents,
    )
    print(f"\nDone. {sum(counts.values())} total examples exported.")
