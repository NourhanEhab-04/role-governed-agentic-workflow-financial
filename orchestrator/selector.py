# orchestrator/selector.py

from orchestrator.validators import (
    validate_after_a1,
    validate_after_a2,
    validate_after_a3,
    validate_after_a4,
    validate_after_a5,
)

_AGENT_TO_STAGE = {
    "client_profiler":    "A1",
    "product_classifier": "A2",
    "rule_engine_agent":  "A3",
    "conflict_detector":  "A4",
    "disclosure_agent":   "A5",
}

_STAGE_TO_NEXT_AGENT = {
    "A1": "product_classifier",
    "A2": "rule_engine_agent",
    "A3": "conflict_detector",
    "A4": "disclosure_agent",
    "A5": None,
}

_VALIDATORS = {
    "A1": validate_after_a1,
    "A2": validate_after_a2,
    "A3": validate_after_a3,
    "A4": validate_after_a4,
    "A5": validate_after_a5,
}


def make_selector(pipeline_state: dict, retry_counts: dict):
    """
    Returns a selector function that drives the pipeline stage order.

    pipeline_state and retry_counts are dicts mutated in place.
    The returned function matches the signature AutoGen's SelectorGroupChat
    expects: (last_speaker, messages) -> str.
    """

    def selector(last_speaker, messages) -> str:
        speaker_name = getattr(last_speaker, "name", None)
        completed_stage = _AGENT_TO_STAGE.get(speaker_name)

        # No agent has spoken yet — start the pipeline
        if completed_stage is None:
            return "client_profiler"

        # Validate the just-completed stage
        validator = _VALIDATORS[completed_stage]
        ok, error_msg = validator(pipeline_state)

        if not ok:
            retries = retry_counts.get(completed_stage, 0)
            if retries < 1:
                retry_counts[completed_stage] = retries + 1
                return speaker_name          # one retry
            else:
                pipeline_state["halt"] = True
                pipeline_state["halt_reason"] = (
                    f"{speaker_name} failed validation twice: {error_msg}"
                )
                return "PIPELINE_HALT"

        # After A4: check escalation flag before routing forward
        if completed_stage == "A4":
            if pipeline_state.get("conflict_report", {}).get("escalate") is True:
                pipeline_state["escalated"] = True
                pipeline_state["halt_reason"] = (
                    "Escalation flagged by conflict detector."
                )
                # Still route to A5 so it can write the ESCALATED report
                return "disclosure_agent"

        # Normal forward routing
        next_agent = _STAGE_TO_NEXT_AGENT[completed_stage]
        if next_agent is None:
            return "TERMINATE"
        return next_agent

    return selector