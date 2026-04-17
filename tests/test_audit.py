# tests/test_audit.py

from orchestrator.audit import build_audit_log


def test_clean_run_audit_log():
    state = {
        "suitability_report": {"decision": "SUITABLE"},
        "escalated": False,
        "halt": False,
        "halt_reason": None,
    }
    log = build_audit_log(
        pipeline_state=state,
        retry_counts={"A1": 0, "A2": 0},
        agent_outputs={"A1": '{"knowledge_level":"moderate"}', "A2": '{"risk_class":4}'},
        validation_results={
            "A1": (True, ""),
            "A2": (True, ""),
            "A3": (True, ""),
            "A4": (True, ""),
            "A5": (True, ""),
        },
    )
    assert log["final_decision"] == "SUITABLE"
    assert log["escalated"] is False
    assert log["halted"] is False
    assert set(log["stages"].keys()) == {"A1", "A2", "A3", "A4", "A5"}
    assert log["stages"]["A1"]["validation_passed"] is True
    assert log["stages"]["A1"]["retry_count"] == 0
    assert "timestamp" in log


def test_escalated_run_audit_log():
    state = {
        "suitability_report": {"decision": "ESCALATED"},
        "escalated": True,
        "halt": False,
        "halt_reason": "Escalation flagged by conflict detector.",
    }
    log = build_audit_log(state, {}, {}, {})
    assert log["final_decision"] == "ESCALATED"
    assert log["escalated"] is True
    assert log["halt_reason"] == "Escalation flagged by conflict detector."


def test_halted_run_has_unknown_decision():
    state = {"halt": True, "halt_reason": "A1 failed twice."}
    log = build_audit_log(state, {"A1": 1}, {}, {"A1": (False, "missing keys")})
    assert log["final_decision"] == "UNKNOWN"
    assert log["halted"] is True
    assert log["stages"]["A1"]["retry_count"] == 1
    assert log["stages"]["A1"]["validation_passed"] is False
    assert log["stages"]["A1"]["validation_error"] == "missing keys"


def test_missing_stages_default_to_none():
    log = build_audit_log({}, {}, {}, {})
    for stage in ["A1", "A2", "A3", "A4", "A5"]:
        assert log["stages"][stage]["raw_output"] is None
        assert log["stages"][stage]["validation_passed"] is None
        assert log["stages"][stage]["retry_count"] == 0