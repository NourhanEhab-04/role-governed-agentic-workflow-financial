# orchestrator/feedback_loops.py
"""
Verifier-feedback loops for A1 (Client Profiler) and A2 (Product Classifier).

Each loop:
  1. Runs the verifier (using a stronger, independent model) on the current output.
  2. If verification fails and iterations remain, re-runs the specialist agent
     with targeted verifier feedback AND the full prior-attempt history so it
     cannot go in circles.
  3. Repeats up to max_iterations times, then accepts the best result.

Returns (final_profile, loop_state) where loop_state contains:
  "verification"       — first verification result (always present)
  "correction"         — {original, corrected, fields_fixed, attempts}
                         present only when at least one retry ran
  "final_verification" — last verification result after retries
                         present only when at least one retry ran
"""

_MAX_ITERATIONS = 3


async def run_a1_feedback_loop(
    raw_input: str,
    initial_profile: dict,
    model_client,
    verifier_client=None,
    max_iterations: int = _MAX_ITERATIONS,
    consistency_issues: list | None = None,
) -> tuple:
    """
    Verifier → specialist-feedback loop for A1 (Client Profiler).

    verifier_client: the stronger model client used for verification; falls back
                     to model_client when not provided (e.g. in tests).
    consistency_issues: pre-confirmed errors from the deterministic check layer,
                        forwarded to the verifier so it can focus on semantic fields.
    Returns (final_profile, loop_state).
    """
    from agents.verifier_agent import run_verifier_on_a1
    from agents.client_profiler import run_client_profiler_with_feedback
    from orchestrator.consistency_checks import check_a1_consistency

    _verifier_client = verifier_client if verifier_client is not None else model_client

    current_profile = initial_profile
    # Always reflects the consistency state of current_profile, not the original.
    current_consistency_issues = consistency_issues
    loop_state: dict = {}
    last_verification: dict | None = None
    # Full history of every attempt + its verifier verdict, passed into each
    # correction call so the agent sees what it already tried and why it failed.
    attempt_history: list = []

    for attempt in range(1, max_iterations + 1):
        try:
            verification = await run_verifier_on_a1(
                raw_input, current_profile, _verifier_client,
                consistency_issues=current_consistency_issues,
                prior_verification=last_verification,
            )
        except Exception as exc:
            if attempt == 1:
                raise  # first-verify errors propagate to the orchestrator's outer try/except
            loop_state["final_verification"] = {
                "passed": None, "confidence": None,
                "issues": [f"Re-verify error: {exc}"], "field_checks": {},
            }
            break

        if attempt == 1:
            loop_state["verification"] = verification
        else:
            loop_state["final_verification"] = verification

        last_verification = verification

        # Stop when passed, or on the last attempt (accept whatever we have).
        if verification.get("passed") or attempt == max_iterations:
            break

        # Record this round's outcome so subsequent correction calls see the
        # full conversation history and don't repeat the same mistakes.
        failed_fields = {
            f: c for f, c in verification.get("field_checks", {}).items()
            if not c.get("supported")
        }
        attempt_history.append({
            "attempt": attempt,
            "output": dict(current_profile),
            "verifier_issues": verification.get("issues", []),
            "failed_fields": failed_fields,
        })

        if "correction" not in loop_state:
            loop_state["correction"] = {
                "original": initial_profile,
                "corrected": None,
                "fields_fixed": list(failed_fields.keys()),
                "attempts": 0,
            }

        try:
            corrected = await run_client_profiler_with_feedback(
                raw_input, current_profile, verification, model_client,
                attempt_history=attempt_history,
            )
            # Apply deterministic overrides before the next verifier pass so the
            # verifier never sees fixable rule violations (e.g. age>70 + MEDIUM).
            from orchestrator.consistency_checks import enforce_a1_consistency
            corrected = enforce_a1_consistency(corrected)
            loop_state["correction"]["corrected"] = corrected
            loop_state["correction"]["fields_fixed"] = list(failed_fields.keys())
            loop_state["correction"]["attempts"] += 1
            current_profile = corrected
            # Recompute consistency issues against the corrected profile so the
            # next verifier call reflects the current state, not the initial one.
            fresh = check_a1_consistency(corrected)
            current_consistency_issues = fresh or None
        except Exception as exc:
            loop_state["correction"]["error"] = str(exc)
            break  # non-fatal: keep current_profile, stop loop

    return current_profile, loop_state


async def run_a2_feedback_loop(
    raw_input: str,
    initial_profile: dict,
    model_client,
    verifier_client=None,
    max_iterations: int = _MAX_ITERATIONS,
    consistency_issues: list | None = None,
) -> tuple:
    """
    Verifier → specialist-feedback loop for A2 (Product Classifier).

    verifier_client: the stronger model client used for verification; falls back
                     to model_client when not provided (e.g. in tests).
    consistency_issues: pre-confirmed errors from the deterministic check layer,
                        forwarded to the verifier so it can focus on semantic fields.
    Returns (final_profile, loop_state).
    """
    from agents.verifier_agent import run_verifier_on_a2
    from agents.product_classifier import run_product_classifier_with_feedback
    from orchestrator.consistency_checks import check_a2_consistency

    _verifier_client = verifier_client if verifier_client is not None else model_client

    current_profile = initial_profile
    current_consistency_issues = consistency_issues
    loop_state: dict = {}
    last_verification: dict | None = None
    attempt_history: list = []

    for attempt in range(1, max_iterations + 1):
        try:
            verification = await run_verifier_on_a2(
                raw_input, current_profile, _verifier_client,
                consistency_issues=current_consistency_issues,
                prior_verification=last_verification,
            )
        except Exception as exc:
            if attempt == 1:
                raise
            loop_state["final_verification"] = {
                "passed": None, "confidence": None,
                "issues": [f"Re-verify error: {exc}"], "field_checks": {},
            }
            break

        if attempt == 1:
            loop_state["verification"] = verification
        else:
            loop_state["final_verification"] = verification

        last_verification = verification

        if verification.get("passed") or attempt == max_iterations:
            break

        failed_fields = {
            f: c for f, c in verification.get("field_checks", {}).items()
            if not c.get("supported")
        }
        attempt_history.append({
            "attempt": attempt,
            "output": dict(current_profile),
            "verifier_issues": verification.get("issues", []),
            "failed_fields": failed_fields,
        })

        if "correction" not in loop_state:
            loop_state["correction"] = {
                "original": initial_profile,
                "corrected": None,
                "fields_fixed": list(failed_fields.keys()),
                "attempts": 0,
            }

        try:
            corrected = await run_product_classifier_with_feedback(
                raw_input, current_profile, verification, model_client,
                attempt_history=attempt_history,
            )
            # Apply deterministic overrides before the next verifier pass so the
            # verifier never sees fixable ESMA rule violations.
            from orchestrator.consistency_checks import enforce_a2_consistency
            corrected = enforce_a2_consistency(corrected)
            loop_state["correction"]["corrected"] = corrected
            loop_state["correction"]["fields_fixed"] = list(failed_fields.keys())
            loop_state["correction"]["attempts"] += 1
            current_profile = corrected
            fresh = check_a2_consistency(corrected)
            current_consistency_issues = fresh or None
        except Exception as exc:
            loop_state["correction"]["error"] = str(exc)
            break

    return current_profile, loop_state
