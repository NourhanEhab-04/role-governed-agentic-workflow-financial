# tests/test_pipeline_e2e.py

import json
import pytest
import asyncio
from pathlib import Path

from config.llm_config import get_model_client
from orchestrator.orchestrator import run_pipeline

SCENARIOS_DIR = Path("data/scenarios")
PRODUCTS_DIR  = Path("data/products")


def load_scenarios():
    files = sorted(SCENARIOS_DIR.glob("*.json"))
    assert len(files) == 10, f"Expected 10 scenarios, found {len(files)}"
    return files


def pytest_collect_scenario(scenario_file):
    scenario = json.loads(scenario_file.read_text())
    product  = json.loads(
        (PRODUCTS_DIR / scenario["product_file"]).read_text()
    )
    return scenario, product


@pytest.fixture(scope="module")
def model_client():
    return get_model_client()
@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.parametrize("scenario_file", load_scenarios(),
                         ids=lambda f: f.stem)
async def test_e2e_decision(scenario_file, model_client):
    scenario, product = pytest_collect_scenario(scenario_file)

    client_input  = json.dumps(scenario["client"])
    product_input = json.dumps(product)

    state, log = await run_pipeline(client_input, product_input, model_client)

    report = state.get("suitability_report", {})
    actual_decision = report.get("decision", "MISSING")

    assert actual_decision == scenario["expected_decision"], (
        f"\nScenario : {scenario_file.stem}"
        f"\nExpected : {scenario['expected_decision']}"
        f"\nActual   : {actual_decision}"
        f"\nScore    : {state.get('rule_verdict', {}).get('score', 'N/A')}"
        f"\nFailed   : {state.get('rule_verdict', {}).get('failed_rules', [])}"
        f"\nHalt     : {state.get('halt_reason')}"
    )
@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.parametrize("scenario_file", load_scenarios(),
                         ids=lambda f: f.stem)
async def test_e2e_escalation_flag(scenario_file, model_client):
    scenario, product = pytest_collect_scenario(scenario_file)

    client_input  = json.dumps(scenario["client"])
    product_input = json.dumps(product)

    state, log = await run_pipeline(client_input, product_input, model_client)

    actual_escalate = state.get("escalated", False)

    assert actual_escalate == scenario["expected_escalate"], (
        f"\nScenario    : {scenario_file.stem}"
        f"\nExpected escalate : {scenario['expected_escalate']}"
        f"\nActual escalate   : {actual_escalate}"
        f"\nConflict report   : {state.get('conflict_report', {})}"
    )

@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.parametrize("scenario_file", load_scenarios(),
                         ids=lambda f: f.stem)
async def test_e2e_rule_findings_completeness(scenario_file, model_client):
    scenario, product = pytest_collect_scenario(scenario_file)

    client_input  = json.dumps(scenario["client"])
    product_input = json.dumps(product)

    state, log = await run_pipeline(client_input, product_input, model_client)

    report = state.get("suitability_report", {})
    findings = report.get("rule_findings", [])

    assert len(findings) == 7, (
        f"\nScenario : {scenario_file.stem}"
        f"\nExpected : 7 rule findings"
        f"\nActual   : {len(findings)}"
    )

    actual_ids = {f["rule_id"] for f in findings}
    expected_ids = {"R1", "R2", "R3", "R4", "R5", "R6", "R7"}
    assert actual_ids == expected_ids, (
        f"\nScenario : {scenario_file.stem}"
        f"\nMissing rule IDs : {expected_ids - actual_ids}"
        f"\nExtra rule IDs   : {actual_ids - expected_ids}"
    )
@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.parametrize("scenario_file", load_scenarios(),
                         ids=lambda f: f.stem)
async def test_e2e_regulatory_citation(scenario_file, model_client):
    scenario, product = pytest_collect_scenario(scenario_file)

    client_input  = json.dumps(scenario["client"])
    product_input = json.dumps(product)

    state, log = await run_pipeline(client_input, product_input, model_client)

    report   = state.get("suitability_report", {})
    reg_text = report.get("regulatory_basis", "")

    assert "Article 25" in reg_text, (
        f"\nScenario : {scenario_file.stem}"
        f"\nExpected 'Article 25' in regulatory_basis"
        f"\nActual   : '{reg_text}'"
    )
@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.parametrize("scenario_file", load_scenarios(),
                         ids=lambda f: f.stem)
async def test_e2e_audit_log_completeness(scenario_file, model_client):
    scenario, product = pytest_collect_scenario(scenario_file)

    client_input  = json.dumps(scenario["client"])
    product_input = json.dumps(product)

    state, log = await run_pipeline(client_input, product_input, model_client)

    # Timestamp is a valid ISO 8601 string
    ts = log.get("timestamp", "")
    assert ts.endswith("Z"), f"timestamp must end with Z, got: '{ts}'"
    assert "T" in ts,        f"timestamp must be ISO 8601, got: '{ts}'"

    # All 5 stage keys are present
    assert set(log["stages"].keys()) == {"A1", "A2", "A3", "A4", "A5"}, (
        f"Stages present: {list(log['stages'].keys())}"
    )

    # Every stage has retry_count ≥ 0
    for stage, info in log["stages"].items():
        assert isinstance(info["retry_count"], int), (
            f"{stage}: retry_count must be int"
        )
        assert info["retry_count"] >= 0

    # final_decision must be one of the valid values
    valid = {"SUITABLE", "CONDITIONAL", "UNSUITABLE", "ESCALATED", "UNKNOWN"}
    assert log["final_decision"] in valid, (
        f"final_decision '{log['final_decision']}' is not valid"
    )
# Three representative scenarios: one per decision class
DETERMINISM_SCENARIOS = [
    "01_suitable_conservative",    # SUITABLE
    "03_conditional_borderline",   # CONDITIONAL
    "06_unsuitable_multi_rule",    # UNSUITABLE
]


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.parametrize("stem", DETERMINISM_SCENARIOS)
async def test_e2e_determinism(stem, model_client):
    """
    Run the same scenario twice and assert the rule engine output
    is identical across both runs.

    The LLM agents (A1, A2) may produce slightly different natural
    language but the structured outputs fed into A3 must be stable
    enough that the rule engine always reaches the same verdict.
    """
    scenario_file = next(Path("data/scenarios").glob(f"{stem}.json"))
    scenario = json.loads(scenario_file.read_text())
    product  = json.loads(
        (PRODUCTS_DIR / scenario["product_file"]).read_text()
    )

    client_input  = json.dumps(scenario["client"])
    product_input = json.dumps(product)

    state_1, _ = await run_pipeline(client_input, product_input, model_client)
    state_2, _ = await run_pipeline(client_input, product_input, model_client)

    verdict_1 = state_1.get("rule_verdict", {})
    verdict_2 = state_2.get("rule_verdict", {})

    assert verdict_1.get("decision") == verdict_2.get("decision"), (
        f"\nScenario : {stem}"
        f"\nRun 1 decision : {verdict_1.get('decision')} (score={verdict_1.get('score')})"
        f"\nRun 2 decision : {verdict_2.get('decision')} (score={verdict_2.get('score')})"
        f"\nDecision diverged — A1 or A2 produced inconsistent structured output."
    )

    assert verdict_1.get("score") == verdict_2.get("score"), (
        f"\nScenario : {stem}"
        f"\nRun 1 score : {verdict_1.get('score')}"
        f"\nRun 2 score : {verdict_2.get('score')}"
        f"\nScore diverged — rule engine received different inputs across runs."
    )

    assert verdict_1.get("failed_rules") == verdict_2.get("failed_rules"), (
        f"\nScenario : {stem}"
        f"\nRun 1 failed : {verdict_1.get('failed_rules')}"
        f"\nRun 2 failed : {verdict_2.get('failed_rules')}"
    )