# tests/test_fixture_integrity.py
#
# Loads every scenario fixture and runs ONLY the rule engine (no LLM)
# to verify expected_rules_failed and expected_decision align with
# what the deterministic rule engine actually produces.
#
# If a scenario fails here it means the fixture is wrong, not the pipeline.

import json
import pytest
from pathlib import Path
from rule_engine.rule_engine import evaluate_suitability


SCENARIOS_DIR = Path("data/scenarios")
PRODUCTS_DIR  = Path("data/products")


def load_scenarios():
    files = sorted(SCENARIOS_DIR.glob("*.json"))
    assert len(files) == 10, f"Expected 10 scenarios, found {len(files)}"
    return files


@pytest.mark.parametrize("scenario_file", load_scenarios(),
                         ids=lambda f: f.stem)
def test_fixture_matches_rule_engine(scenario_file):
    scenario = json.loads(scenario_file.read_text())
    product  = json.loads(
        (PRODUCTS_DIR / scenario["product_file"]).read_text()
    )

    verdict =evaluate_suitability(scenario["client"], product)

    expected_decision = scenario["expected_decision"]
    expected_failed   = set(scenario["expected_rules_failed"])

    # For escalated scenarios the rule engine itself won't say ESCALATED —
    # escalation is decided by A4 on top of the verdict.
    # Map ESCALATED → check the underlying verdict is not UNSUITABLE
    # (some escalations sit on a SUITABLE base verdict).
    actual_failed = {r["rule"] for r in verdict["rules"] if not r["pass"]}

    if expected_decision == "ESCALATED":
        # Just verify the failed rules match — decision check is skipped
        assert expected_failed == actual_failed, (
            f"{scenario_file.name}: expected failed rules {expected_failed}, "
            f"got {actual_failed}"
        )
    else:
        assert verdict["decision"] == expected_decision, (
            f"{scenario_file.name}: expected {expected_decision}, "
            f"got {verdict['decision']} (score={verdict['score']})"
        )
        assert expected_failed == actual_failed, (
            f"{scenario_file.name}: expected failed rules {expected_failed}, "
            f"got {actual_failed}"
        )