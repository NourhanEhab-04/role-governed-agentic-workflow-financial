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

import datetime
import json
import random
import openai

from orchestrator.validators import (
    validate_after_a1,
    validate_after_a2,
    validate_after_a3,
    validate_after_a4,
    validate_after_a5,
)
from orchestrator.audit import build_audit_log
from orchestrator.pre_check_tool import run_pre_check
from orchestrator.consistency_checks import (
    check_a1_consistency,
    check_a2_consistency,
    check_a1_a2_cross,
)

# The verifier always runs after A1 and A2 (rate = 1.0).
# Set to 0.0 to disable (verifier only runs when consistency issues are found).
_VERIFIER_SAMPLE_RATE = 1.0


async def run_pipeline(
    client_input: str,
    product_input: str,
    model_client,
) -> tuple:
    """
    Run the full MiFID II suitability pipeline.

    Returns (pipeline_state, audit_log).

    Three-point rule engine architecture is enforced automatically:
      - pre_check_verdict is written between A2 and A3
      - validate_after_a3 cross-checks it against rule_verdict
      - audit_verdict is written inside the A4 stage
      - validate_after_a4 enforces that disagreement implies escalation
    """
    # Agent imports deferred inside function to prevent circular imports
    from agents.client_profiler import run_client_profiler
    from agents.product_classifier import run_product_classifier
    from agents.rule_engine_agent import run_rule_engine_agent
    from agents.conflict_detector import run_conflict_detector, check_rule_engine_agreement
    from agents.disclosure_agent import run_disclosure_agent
    from agents.verifier_agent import (
        run_verifier_on_a1, run_verifier_on_a2,
        run_corrector_on_a1, run_corrector_on_a2,
    )

    state: dict = {}
    retries: dict = {}
    outputs: dict = {}
    validations: dict = {}

    def halt(reason: str):
        state["halt"] = True
        state["halt_reason"] = reason

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

    # ── A1 ───────────────────────────────────────────────────────────────────
    ok = await run_stage(
        "A1",
        lambda: run_client_profiler(client_input, model_client=model_client),
        validate_after_a1,
        "client_profile",
    )
    if not ok:
        return state, build_audit_log(state, retries, outputs, validations)

    # ── A1 validation layer ──────────────────────────────────────────────────
    # Layer 1: deterministic consistency check (always runs, zero cost)
    a1_issues = check_a1_consistency(state["client_profile"])
    state["a1_consistency_issues"] = a1_issues

    # Layer 2: LLM verifier (runs based on sample rate or consistency issues)
    if a1_issues or random.random() < _VERIFIER_SAMPLE_RATE:
        try:
            a1_verification = await run_verifier_on_a1(
                client_input,
                state["client_profile"],
                model_client=model_client,
            )
            state["a1_verification"] = a1_verification

            # Layer 3: Corrector — runs once if verifier failed
            if a1_verification.get("passed") is False:
                try:
                    corrected_profile = await run_corrector_on_a1(
                        client_input,
                        state["client_profile"],
                        a1_verification,
                        model_client=model_client,
                    )
                    state["a1_correction"] = {
                        "original": state["client_profile"],
                        "corrected": corrected_profile,
                        "fields_fixed": [
                            f for f, c in a1_verification.get("field_checks", {}).items()
                            if c.get("supported") is False
                        ],
                    }
                    # Replace profile with corrected version
                    state["client_profile"] = corrected_profile
                    # Re-verify the corrected output
                    try:
                        state["a1_final_verification"] = await run_verifier_on_a1(
                            client_input,
                            corrected_profile,
                            model_client=model_client,
                        )
                    except Exception as exc:
                        state["a1_final_verification"] = {
                            "passed": None,
                            "confidence": None,
                            "issues": [f"Re-verify error: {exc}"],
                            "field_checks": {},
                        }
                except Exception as exc:
                    # Corrector failure is non-fatal — keep original profile
                    state["a1_correction"] = {
                        "original": state["client_profile"],
                        "corrected": None,
                        "fields_fixed": [],
                        "error": str(exc),
                    }

        except Exception as exc:
            # Verifier failure is non-fatal: record it and continue
            state["a1_verification"] = {
                "passed": None,
                "confidence": None,
                "issues": [f"Verifier error: {exc}"],
                "field_checks": {},
            }

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

    # ── A2 ───────────────────────────────────────────────────────────────────
    ok = await run_stage(
        "A2",
        lambda: run_product_classifier(product_input, model_client=model_client),
        validate_after_a2,
        "product_profile",
    )
    if not ok:
        return state, build_audit_log(state, retries, outputs, validations)

    # ── A2 validation layer ──────────────────────────────────────────────────
    # Layer 1: deterministic consistency check
    a2_issues = check_a2_consistency(state["product_profile"])
    state["a2_consistency_issues"] = a2_issues

    # Layer 2: LLM verifier
    if a2_issues or random.random() < _VERIFIER_SAMPLE_RATE:
        try:
            a2_verification = await run_verifier_on_a2(
                product_input,
                state["product_profile"],
                model_client=model_client,
            )
            state["a2_verification"] = a2_verification

            # Layer 3: Corrector — runs once if verifier failed
            if a2_verification.get("passed") is False:
                try:
                    corrected_product = await run_corrector_on_a2(
                        product_input,
                        state["product_profile"],
                        a2_verification,
                        model_client=model_client,
                    )
                    state["a2_correction"] = {
                        "original": state["product_profile"],
                        "corrected": corrected_product,
                        "fields_fixed": [
                            f for f, c in a2_verification.get("field_checks", {}).items()
                            if c.get("supported") is False
                        ],
                    }
                    # Replace profile with corrected version
                    state["product_profile"] = corrected_product
                    # Re-verify the corrected output
                    try:
                        state["a2_final_verification"] = await run_verifier_on_a2(
                            product_input,
                            corrected_product,
                            model_client=model_client,
                        )
                    except Exception as exc:
                        state["a2_final_verification"] = {
                            "passed": None,
                            "confidence": None,
                            "issues": [f"Re-verify error: {exc}"],
                            "field_checks": {},
                        }
                except Exception as exc:
                    # Corrector failure is non-fatal — keep original profile
                    state["a2_correction"] = {
                        "original": state["product_profile"],
                        "corrected": None,
                        "fields_fixed": [],
                        "error": str(exc),
                    }

        except Exception as exc:
            state["a2_verification"] = {
                "passed": None,
                "confidence": None,
                "issues": [f"Verifier error: {exc}"],
                "field_checks": {},
            }

    # ── A1 × A2 cross-consistency check ─────────────────────────────────────
    cross_issues = check_a1_a2_cross(state["client_profile"], state["product_profile"])
    state["cross_consistency_issues"] = cross_issues

    # ── Pre-check (A0 role) ──────────────────────────────────────────────────
    # First of three independent rule engine contacts.
    # Runs after A1 + A2 complete, before A3, so validate_after_a3 can
    # cross-check A3's verdict against this deterministic baseline.
    try:
        state["pre_check_verdict"] = run_pre_check(
            state["client_profile"],
            state["product_profile"],
        )
    except Exception as exc:
        halt(f"pre_check failed: {exc}")
        return state, build_audit_log(state, retries, outputs, validations)

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
        return state, build_audit_log(state, retries, outputs, validations)

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
    except Exception as exc:
        halt(f"audit pre-check failed: {exc}")
        return state, build_audit_log(state, retries, outputs, validations)

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
        return state, build_audit_log(state, retries, outputs, validations)

    # Escalation flag — set before A5 so A5 writes the correct report type.
    if state.get("conflict_report", {}).get("escalate") is True:
        state["escalated"] = True
        state["halt_reason"] = "Escalation flagged by conflict detector."

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

    return state, build_audit_log(state, retries, outputs, validations)