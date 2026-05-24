# orchestrator/orchestrator.py
"""
MiFID II suitability pipeline — Chunk 9.3 refactor.

Three-point rule engine architecture:
  state["pre_check_verdict"]  ← run_pre_check()               between A2 and A3
  state["rule_verdict"]       ← run_rule_engine_agent()        A3 stage
  state["audit_verdict"]      ← check_rule_engine_agreement()  inside A4 stage

Validation layer (after A1 and A2):
  state["a1_consistency_issues"]   ← check_a1_consistency()    after A1
  state["a1_verification"]         ← run_verifier_on_a1()      after A1 (if flagged or sampled)
  state["a2_consistency_issues"]   ← check_a2_consistency()    after A2
  state["a2_verification"]         ← run_verifier_on_a2()      after A2 (if flagged or sampled)
  state["cross_consistency_issues"]← check_a1_a2_cross()       before pre_check

All other behaviour (run_stage, retry logic, rate-limit guard,
portfolio_concentration_pct injection, return tuple) is unchanged.
"""

import asyncio
import datetime
import json
import openai

from orchestrator.validators import (
    validate_after_a1,
    validate_after_a2,
    validate_after_a3,
    validate_after_a4,
    validate_after_a5,
)
from orchestrator.audit import build_full_audit_record
from orchestrator.pre_check_tool import run_pre_check
from orchestrator.consistency_checks import (
    check_a1_consistency,
    check_a2_consistency,
    check_a1_a2_cross,
    enforce_a1_consistency,
    enforce_a2_consistency,
)
from orchestrator.feedback_loops import run_a1_feedback_loop, run_a2_feedback_loop

# Verifier always runs on every A1/A2 output using a stronger independent model.
# The feedback loop exits early as soon as the verifier passes, so the common
# case (correct output on the first try) costs only one extra API call.


async def run_pipeline(
    client_input: str,
    product_input: str,
    model_client,
    on_event=None,
) -> tuple:
    """
    Run the full MiFID II suitability pipeline.

    Returns (pipeline_state, audit_log).

    Three-point rule engine architecture is enforced automatically:
      - pre_check_verdict is written between A2 and A3
      - validate_after_a3 cross-checks it against rule_verdict
      - audit_verdict is written inside the A4 stage
      - validate_after_a4 enforces that disagreement implies escalation

    on_event: optional async callable(event_type: str, state_snapshot: dict)
              called after each major pipeline stage with a shallow copy of state.
    """
    # Agent imports deferred inside function to prevent circular imports
    from agents.client_profiler import run_client_profiler
    from agents.product_classifier import run_product_classifier
    from agents.rule_engine_agent import run_rule_engine_agent
    from agents.conflict_detector import run_conflict_detector, check_rule_engine_agreement
    from agents.disclosure_agent import run_disclosure_agent
    from config.llm_config import get_verifier_client, GROQ_MODEL, GROQ_VERIFIER_MODEL

    _model_version = {
        "agent_model":    GROQ_MODEL,
        "verifier_model": GROQ_VERIFIER_MODEL,
        "provider":       "groq",
    }
    # Separate stronger model for the verifier — created once per pipeline run.
    verifier_client = get_verifier_client()
    state: dict = {}
    retries: dict = {}
    outputs: dict = {}
    validations: dict = {}

    def halt(reason: str):
        state["halt"] = True
        state["halt_reason"] = reason

    async def emit(event_type: str):
        if on_event:
            await on_event(event_type, dict(state))

    async def run_stage(stage_id, coro, validator, state_key):
        """
        Run one agent stage with one retry on validation failure.
        Rate-limit errors (HTTP 429) are not retried.
        Returns True if the stage completed successfully, False if halted.
        """
        for attempt in range(2):
            try:
                result = await coro()
                state[state_key] = result
                outputs[stage_id] = str(result)
                ok, err = validator(state)
                validations[stage_id] = (ok, err)
                if ok:
                    return True
                retries[stage_id] = retries.get(stage_id, 0) + 1
            except openai.RateLimitError as exc:
                msg = f"Rate limit reached at stage {stage_id}: {exc}"
                outputs[stage_id] = msg
                validations[stage_id] = (False, msg)
                halt(msg)
                return False
            except Exception as exc:
                outputs[stage_id] = str(exc)
                validations[stage_id] = (False, str(exc))
                retries[stage_id] = retries.get(stage_id, 0) + 1

        halt(
            f"Stage {stage_id} failed after 2 attempts: "
            f"{validations.get(stage_id, (None, ''))[1]}"
        )
        return False

    # ── A1 + A2 run concurrently ─────────────────────────────────────────────
    # Both agents and their verification loops are fully independent, so we
    # run them in parallel to halve wall-clock time for the profiling phase.

    async def _run_a1() -> bool:
        ok = await run_stage(
            "A1",
            lambda: run_client_profiler(client_input, model_client=model_client),
            validate_after_a1,
            "client_profile",
        )
        if not ok:
            return False

        a1_issues = check_a1_consistency(state["client_profile"])
        state["a1_consistency_issues"] = a1_issues
        await emit("a1_profiled")

        # Verification always runs — the loop exits immediately when the verifier
        # passes, so a correct first-pass costs only one extra API call.
        state["a1_initial_profile"] = dict(state["client_profile"])
        try:
            final_profile, loop_state = await run_a1_feedback_loop(
                client_input,
                state["client_profile"],
                model_client=model_client,
                verifier_client=verifier_client,
                consistency_issues=a1_issues or None,
            )
            state["client_profile"] = final_profile
            if "verification" in loop_state:
                state["a1_verification"] = loop_state["verification"]
            if "correction" in loop_state:
                state["a1_correction"] = loop_state["correction"]
            if "final_verification" in loop_state:
                state["a1_final_verification"] = loop_state["final_verification"]
        except Exception as exc:
            state["a1_verification"] = {
                "passed": None,
                "confidence": None,
                "issues": [f"Verifier error: {exc}"],
                "field_checks": {},
            }

        # Deterministic overrides — applied after all LLM attempts.
        # These rules are unambiguous (age > 70 → HIGH, class 7 → leverage, etc.)
        # and must hold regardless of what the LLM produced.
        state["client_profile"] = enforce_a1_consistency(state["client_profile"])

        # Inject portfolio_concentration_pct — not in REQUIRED_CLIENT_KEYS
        # so A1 strips it, but the conflict detector needs it downstream.
        try:
            raw_client = json.loads(client_input)
            if "portfolio_concentration_pct" in raw_client:
                state["client_profile"]["portfolio_concentration_pct"] = (
                    raw_client["portfolio_concentration_pct"]
                )
        except (json.JSONDecodeError, KeyError):
            pass

        await emit("a1_validated")
        return True

    async def _run_a2() -> bool:
        ok = await run_stage(
            "A2",
            lambda: run_product_classifier(product_input, model_client=model_client),
            validate_after_a2,
            "product_profile",
        )
        if not ok:
            return False

        a2_issues = check_a2_consistency(state["product_profile"])
        state["a2_consistency_issues"] = a2_issues
        await emit("a2_profiled")

        # Verification always runs — same rationale as A1 above.
        state["a2_initial_profile"] = dict(state["product_profile"])
        try:
            final_profile, loop_state = await run_a2_feedback_loop(
                product_input,
                state["product_profile"],
                model_client=model_client,
                verifier_client=verifier_client,
                consistency_issues=a2_issues or None,
            )
            state["product_profile"] = final_profile
            if "verification" in loop_state:
                state["a2_verification"] = loop_state["verification"]
            if "correction" in loop_state:
                state["a2_correction"] = loop_state["correction"]
            if "final_verification" in loop_state:
                state["a2_final_verification"] = loop_state["final_verification"]
        except Exception as exc:
            state["a2_verification"] = {
                "passed": None,
                "confidence": None,
                "issues": [f"Verifier error: {exc}"],
                "field_checks": {},
            }

        # Deterministic overrides for ESMA rules that must always hold.
        state["product_profile"] = enforce_a2_consistency(state["product_profile"])

        await emit("a2_validated")
        return True

    a1_ok, a2_ok = await asyncio.gather(_run_a1(), _run_a2())
    if not a1_ok or not a2_ok:
        await emit("halt")
        return state, build_full_audit_record(
            state, retries, outputs, validations,
            client_text=client_input,
            product_text=product_input,
            model_version=_model_version,
        )

    # ── A1 × A2 cross-consistency check ─────────────────────────────────────
    cross_issues = check_a1_a2_cross(state["client_profile"], state["product_profile"])
    state["cross_consistency_issues"] = cross_issues
    await emit("cross_checked")  # cross_consistency_issues now in state

    # ── Pre-check (A0 role) ──────────────────────────────────────────────────
    # First of three independent rule engine contacts.
    # Runs after A1 + A2 complete, before A3, so validate_after_a3 can
    # cross-check A3's verdict against this deterministic baseline.
    try:
        state["pre_check_verdict"] = run_pre_check(
            state["client_profile"],
            state["product_profile"],
        )
        await emit("precheck_done")  # pre_check_verdict now in state
    except Exception as exc:
        halt(f"pre_check failed: {exc}")
        await emit("halt")
        return state, build_full_audit_record(
            state, retries, outputs, validations,
            client_text=client_input,
            product_text=product_input,
            model_version=_model_version,
        )

    # ── A3 ───────────────────────────────────────────────────────────────────
    # validate_after_a3 will now cross-check rule_verdict against pre_check_verdict.
    ok = await run_stage(
        "A3",
        lambda: run_rule_engine_agent(
            state["client_profile"],
            state["product_profile"],
            model_client=model_client,
        ),
        validate_after_a3,
        "rule_verdict",
    )
    if not ok:
        await emit("halt")
        return state, build_full_audit_record(
            state, retries, outputs, validations,
            client_text=client_input,
            product_text=product_input,
            model_version=_model_version,
        )
    await emit("a3_done")  # rule_verdict now in state

    # ── A4 ───────────────────────────────────────────────────────────────────
    # Before calling the LLM agent, run the third independent rule engine
    # contact and store audit_verdict so validate_after_a4 can enforce
    # the disagreement → escalation invariant.
    try:
        state["audit_verdict"] = check_rule_engine_agreement(
            state["client_profile"],
            state["product_profile"],
            state["rule_verdict"],
        )
        await emit("audit_done")  # audit_verdict now in state
    except Exception as exc:
        halt(f"audit pre-check failed: {exc}")
        await emit("halt")
        return state, build_full_audit_record(
            state, retries, outputs, validations,
            client_text=client_input,
            product_text=product_input,
            model_version=_model_version,
        )

    ok = await run_stage(
        "A4",
        lambda: run_conflict_detector(
            state["client_profile"],
            state["product_profile"],
            state["rule_verdict"],
            model_client=model_client,
        ),
        validate_after_a4,
        "conflict_report",
    )
    if not ok:
        await emit("halt")
        return state, build_full_audit_record(
            state, retries, outputs, validations,
            client_text=client_input,
            product_text=product_input,
            model_version=_model_version,
        )

    # Escalation flag — set before A5 so A5 writes the correct report type.
    if state.get("conflict_report", {}).get("escalate") is True:
        state["escalated"] = True
        state["halt_reason"] = "Escalation flagged by conflict detector."
    await emit("a4_done")  # conflict_report + escalated now in state

    # ── A5 — always runs, even on escalation ─────────────────────────────────
    await run_stage(
        "A5",
        lambda: run_disclosure_agent(
            state["client_profile"],
            state["product_profile"],
            state["rule_verdict"],
            state["conflict_report"],
            model_client=model_client,
        ),
        validate_after_a5,
        "suitability_report",
    )
    await emit("a5_done")  # suitability_report now in state

    return state, build_full_audit_record(
            state, retries, outputs, validations,
            client_text=client_input,
            product_text=product_input,
            model_version=_model_version,
        )
