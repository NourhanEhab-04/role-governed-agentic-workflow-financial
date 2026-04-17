# orchestrator/audit.py

import datetime


def build_audit_log(
    pipeline_state: dict,
    retry_counts: dict,
    agent_outputs: dict,
    validation_results: dict,
) -> dict:
    """
    Assemble a fully serialisable audit log from a pipeline run.

    Parameters
    ----------
    pipeline_state     : final state dict after the run
    retry_counts       : {stage_id: int}
    agent_outputs      : {stage_id: str}  raw text output per stage
    validation_results : {stage_id: (bool, str)}
    """
    return {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "final_decision": (
            pipeline_state
            .get("suitability_report", {})
            .get("decision", "UNKNOWN")
        ),
        "escalated": pipeline_state.get("escalated", False),
        "halted": pipeline_state.get("halt", False),
        "halt_reason": pipeline_state.get("halt_reason"),
        "stages": {
            stage: {
                "raw_output":        agent_outputs.get(stage),
                "validation_passed": validation_results.get(stage, (None, ""))[0],
                "validation_error":  validation_results.get(stage, (None, ""))[1],
                "retry_count":       retry_counts.get(stage, 0),
            }
            for stage in ["A1", "A2", "A3", "A4", "A5"]
        },
    }