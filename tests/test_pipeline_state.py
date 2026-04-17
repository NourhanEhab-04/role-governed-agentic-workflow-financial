# tests/test_pipeline_state.py
"""Zero-LLM tests for the updated pipeline state schema."""
from schemas.pipeline_state import make_empty_pipeline_state, PIPELINE_STATE_KEYS


def test_all_keys_present():
    state = make_empty_pipeline_state()
    assert set(state.keys()) == PIPELINE_STATE_KEYS


def test_all_values_none():
    state = make_empty_pipeline_state()
    assert all(v is None for v in state.values())


def test_has_three_verdict_slots():
    assert {"pre_check_verdict", "rule_verdict", "audit_verdict"} <= PIPELINE_STATE_KEYS


def test_has_control_keys():
    assert {"escalated", "halted", "halt_reason"} <= PIPELINE_STATE_KEYS