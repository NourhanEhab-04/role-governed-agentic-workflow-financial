# main.py

import asyncio
import json
import datetime
from pathlib import Path
import argparse

from config.llm_config import get_model_client
from orchestrator.orchestrator import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="MiFID II Suitability Pipeline")
    parser.add_argument("--client",  required=True, help="Path to client JSON file")
    parser.add_argument("--product", required=True, help="Path to product JSON file")
    args = parser.parse_args()

    client_path  = Path(args.client)
    product_path = Path(args.product)

    for label, path in [("--client", client_path), ("--product", product_path)]:
        if not path.exists():
            parser.error(f"{label}: file not found: {path}")
        if path.stat().st_size == 0:
            parser.error(f"{label}: file is empty: {path}")

    client_input  = client_path.read_text()
    product_input = product_path.read_text()

    model_client = get_model_client()

    state, audit_log = asyncio.run(
        run_pipeline(client_input, product_input, model_client)
    )

    report = state.get("suitability_report", {})
    print("\n" + "=" * 60)
    print(f"DECISION : {report.get('decision', 'UNKNOWN')}")
    print("=" * 60)
    print(report.get("client_facing_summary", "No summary available."))
    if state.get("escalated"):
        print(f"\nESCALATED: {state.get('halt_reason', '')}")
    if state.get("halt"):
        print(f"\nHALT: {state['halt_reason']}")
    print("=" * 60 + "\n")

    audit_dir = Path("data/audit")
    audit_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = audit_dir / f"run_{ts}.json"
    out.write_text(json.dumps(audit_log, indent=2))
    print(f"Audit log → {out}")


if __name__ == "__main__":
    main()