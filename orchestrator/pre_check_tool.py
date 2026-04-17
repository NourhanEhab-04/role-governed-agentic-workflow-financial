# orchestrator/pre_check_tool.py
"""
Pre-check tool: lets A0 (OrchestratorAgent) call the rule engine directly
after A1 and A2 have populated pipeline_state.
This is the FIRST of three independent rule engine calls in the pipeline.
"""
from rule_engine.rule_engine import evaluate_suitability


def run_pre_check(client_profile: dict, product_profile: dict) -> dict:
    """
    Call the rule engine with the profiles produced by A1 and A2.
    Returns the full verdict dict: {score, decision, rules}.
    Raises ValueError if either profile is missing required keys.
    """
    return evaluate_suitability(client_profile, product_profile)