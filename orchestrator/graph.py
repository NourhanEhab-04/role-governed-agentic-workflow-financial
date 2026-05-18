# orchestrator/graph.py
"""
LangGraph-based pipeline.

Architecture
------------
Each pipeline stage is a LangGraph node that receives the full PipelineState
and returns only the keys it mutates.  Conditional edges route to END on halt.

A1 and A2 emit intermediate on_event calls so the frontend can stream the
verification loop in real time:

  a1_profiled   → initial parse done; a1_initial_profile written
  a1_verified   → AV first check done; a1_verification written
  a1_corrected  → agent corrected; a1_correction + updated client_profile written
  a1_reverified → AV re-check done; a1_final_verification written
  A1            → node complete (LangGraph fires this automatically)

Same pattern for A2 (a2_profiled, a2_verified, a2_corrected, a2_reverified, A2).

Verifier uses a separate, stronger client (OpenRouter / Gemini) so its failure
modes are independent from the Groq/Llama agents it checks.
"""

from __future__ import annotations

import json
from typing import Any

import openai
from langgraph.graph import StateGraph, END
from langgraph.types import RetryPolicy

from orchestrator.audit import build_audit_log
from orchestrator.consistency_checks import (
    check_a1_consistency,
    check_a2_consistency,
    check_a1_a2_cross,
    enforce_a1_consistency,
    enforce_a2_consistency,
)
from orchestrator.pre_check_tool import run_pre_check
from orchestrator.validators import (
    validate_after_a1,
    validate_after_a2,
    validate_after_a3,
    validate_after_a4,
    validate_after_a5,
)
from schemas.langgraph_state import PipelineState


# ── Routing helpers ────────────────────────────────────────────────────────────

def _route(state: PipelineState) -> str:
    return END if state.get("halt") else "continue"


# ── A1 node ───────────────────────────────────────────────────────────────────

def _node_a1(model_client, verifier_client, on_event=None):
    from agents.client_profiler import run_client_profiler, run_client_profiler_with_feedback
    from agents.verifier_agent import run_verifier_on_a1

    async def node(state: PipelineState) -> dict:
        retries     = dict(state.get("_retries")     or {})
        outputs     = dict(state.get("_outputs")     or {})
        validations = dict(state.get("_validations") or {})

        # ── 1. Run profiler ──────────────────────────────────────────────────
        try:
            profile = await run_client_profiler(
                state["client_input"], model_client=model_client
            )
        except openai.RateLimitError as exc:
            msg = f"Rate limit at A1: {exc}"
            return {"halt": True, "halt_reason": msg,
                    "_retries": retries, "_outputs": outputs, "_validations": validations}

        probe = {**state, "client_profile": profile}
        ok, err = validate_after_a1(probe)
        validations["A1"] = (ok, err)
        outputs["A1"] = str(profile)

        if not ok:
            return {"halt": True, "halt_reason": f"A1 failed: {err}",
                    "client_profile": profile,
                    "_retries": retries, "_outputs": outputs, "_validations": validations}

        a1_issues = check_a1_consistency(profile)
        updates: dict[str, Any] = {
            "client_profile":        profile,
            "a1_initial_profile":    profile,   # snapshot for frontend diff
            "a1_consistency_issues": a1_issues,
        }

        # ── Emit: initial parse done ─────────────────────────────────────────
        if on_event:
            await on_event("a1_profiled", {**state, **updates})

        # ── 2. Verify — always runs, using the stronger verifier_client ──────
        try:
            verification = await run_verifier_on_a1(
                state["client_input"], profile,
                model_client=verifier_client,
                consistency_issues=a1_issues or None,
            )
            updates["a1_verification"] = verification
        except Exception as exc:
            updates["a1_verification"] = {
                "passed": None, "confidence": None,
                "issues": [f"Verifier error: {exc}"], "field_checks": {},
            }

        # ── Emit: AV first check done (always — even on error) ──────────────
        if on_event:
            await on_event("a1_verified", {**state, **updates})

        # ── 3. Correction (only when AV explicitly failed, not errored) ──────
        if updates.get("a1_verification", {}).get("passed") is False:
            fields_fixed = [
                f for f, c in updates["a1_verification"].get("field_checks", {}).items()
                if c.get("supported") is False
            ]
            try:
                corrected = await run_client_profiler_with_feedback(
                    state["client_input"], profile, updates["a1_verification"], model_client
                )
                fresh_issues = check_a1_consistency(corrected)
                updates["a1_correction"]  = {
                    "original":     profile,
                    "corrected":    corrected,
                    "fields_fixed": fields_fixed,
                }
                updates["client_profile"] = corrected
            except Exception as exc:
                updates["a1_correction"] = {
                    "original": profile, "corrected": None,
                    "fields_fixed": [], "error": str(exc),
                }

            # ── Emit: A1 corrected (always after correction attempt) ─────────
            if on_event:
                await on_event("a1_corrected", {**state, **updates})

            # ── 4. Re-verify after correction ────────────────────────────────
            corrected_profile = updates.get("a1_correction", {}).get("corrected")
            if corrected_profile:
                try:
                    re_ver = await run_verifier_on_a1(
                        state["client_input"], corrected_profile,
                        model_client=verifier_client,
                        consistency_issues=check_a1_consistency(corrected_profile) or None,
                        prior_verification=updates["a1_verification"],
                    )
                    updates["a1_final_verification"] = re_ver
                except Exception as exc:
                    updates["a1_final_verification"] = {
                        "passed": None, "confidence": None,
                        "issues": [f"Re-verify error: {exc}"], "field_checks": {},
                    }

                # ── Emit: AV re-check done (always) ──────────────────────────
                if on_event:
                    await on_event("a1_reverified", {**state, **updates})

        # ── 5. Deterministic override — always applied last ──────────────────
        updates["client_profile"] = enforce_a1_consistency(updates["client_profile"])

        try:
            raw = json.loads(state["client_input"])
            if "portfolio_concentration_pct" in raw:
                updates["client_profile"] = {
                    **updates["client_profile"],
                    "portfolio_concentration_pct": raw["portfolio_concentration_pct"],
                }
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        updates["_retries"]     = retries
        updates["_outputs"]     = outputs
        updates["_validations"] = validations
        return updates

    node.__name__ = "node_A1"
    return node


# ── A2 node ───────────────────────────────────────────────────────────────────

def _node_a2(model_client, verifier_client, on_event=None):
    from agents.product_classifier import (
        run_product_classifier,
        run_product_classifier_with_feedback,
    )
    from agents.verifier_agent import run_verifier_on_a2

    async def node(state: PipelineState) -> dict:
        retries     = dict(state.get("_retries")     or {})
        outputs     = dict(state.get("_outputs")     or {})
        validations = dict(state.get("_validations") or {})

        # ── 1. Run classifier ────────────────────────────────────────────────
        try:
            profile = await run_product_classifier(
                state["product_input"], model_client=model_client
            )
        except openai.RateLimitError as exc:
            msg = f"Rate limit at A2: {exc}"
            return {"halt": True, "halt_reason": msg,
                    "_retries": retries, "_outputs": outputs, "_validations": validations}

        probe = {**state, "product_profile": profile}
        ok, err = validate_after_a2(probe)
        validations["A2"] = (ok, err)
        outputs["A2"] = str(profile)

        if not ok:
            return {"halt": True, "halt_reason": f"A2 failed: {err}",
                    "product_profile": profile,
                    "_retries": retries, "_outputs": outputs, "_validations": validations}

        a2_issues    = check_a2_consistency(profile)
        cross_issues = check_a1_a2_cross(state.get("client_profile", {}), profile)

        updates: dict[str, Any] = {
            "product_profile":        profile,
            "a2_initial_profile":     profile,
            "a2_consistency_issues":  a2_issues,
            "cross_consistency_issues": cross_issues,
        }

        # ── Emit: initial classification done ───────────────────────────────
        if on_event:
            await on_event("a2_profiled", {**state, **updates})

        # ── 2. Verify ────────────────────────────────────────────────────────
        try:
            verification = await run_verifier_on_a2(
                state["product_input"], profile,
                model_client=verifier_client,
                consistency_issues=a2_issues or None,
            )
            updates["a2_verification"] = verification
        except Exception as exc:
            updates["a2_verification"] = {
                "passed": None, "confidence": None,
                "issues": [f"Verifier error: {exc}"], "field_checks": {},
            }

        # ── Emit: AV first check done (always) ──────────────────────────────
        if on_event:
            await on_event("a2_verified", {**state, **updates})

        # ── 3. Correction (only when AV explicitly failed, not errored) ──────
        if updates.get("a2_verification", {}).get("passed") is False:
            fields_fixed = [
                f for f, c in updates["a2_verification"].get("field_checks", {}).items()
                if c.get("supported") is False
            ]
            try:
                corrected = await run_product_classifier_with_feedback(
                    state["product_input"], profile, updates["a2_verification"], model_client
                )
                fresh_issues = check_a2_consistency(corrected)
                updates["a2_correction"]   = {
                    "original":     profile,
                    "corrected":    corrected,
                    "fields_fixed": fields_fixed,
                }
                updates["product_profile"] = corrected
            except Exception as exc:
                updates["a2_correction"] = {
                    "original": profile, "corrected": None,
                    "fields_fixed": [], "error": str(exc),
                }

            # ── Emit: A2 corrected (always after correction attempt) ─────────
            if on_event:
                await on_event("a2_corrected", {**state, **updates})

            # ── 4. Re-verify ─────────────────────────────────────────────────
            corrected_profile = updates.get("a2_correction", {}).get("corrected")
            if corrected_profile:
                try:
                    re_ver = await run_verifier_on_a2(
                        state["product_input"], corrected_profile,
                        model_client=verifier_client,
                        consistency_issues=check_a2_consistency(corrected_profile) or None,
                        prior_verification=updates["a2_verification"],
                    )
                    updates["a2_final_verification"] = re_ver
                except Exception as exc:
                    updates["a2_final_verification"] = {
                        "passed": None, "confidence": None,
                        "issues": [f"Re-verify error: {exc}"], "field_checks": {},
                    }

                # ── Emit: AV re-check done (always) ──────────────────────────
                if on_event:
                    await on_event("a2_reverified", {**state, **updates})

        # ── 5. Deterministic ESMA overrides ─────────────────────────────────
        updates["product_profile"] = enforce_a2_consistency(updates["product_profile"])

        updates["_retries"]     = retries
        updates["_outputs"]     = outputs
        updates["_validations"] = validations
        return updates

    node.__name__ = "node_A2"
    return node


# ── Remaining nodes (unchanged logic) ─────────────────────────────────────────

def _node_pre_check():
    async def node(state: PipelineState) -> dict:
        try:
            verdict = run_pre_check(state["client_profile"], state["product_profile"])
            return {"pre_check_verdict": verdict}
        except Exception as exc:
            return {"halt": True, "halt_reason": f"pre_check failed: {exc}"}
    node.__name__ = "node_pre_check"
    return node


def _node_a3(model_client):
    from agents.rule_engine_agent import run_rule_engine_agent

    async def node(state: PipelineState) -> dict:
        retries     = dict(state.get("_retries")     or {})
        outputs     = dict(state.get("_outputs")     or {})
        validations = dict(state.get("_validations") or {})

        try:
            verdict = await run_rule_engine_agent(
                state["client_profile"], state["product_profile"], model_client
            )
        except openai.RateLimitError as exc:
            msg = f"Rate limit at A3: {exc}"
            return {"halt": True, "halt_reason": msg,
                    "_retries": retries, "_outputs": outputs, "_validations": validations}

        probe = {**state, "rule_verdict": verdict}
        ok, err = validate_after_a3(probe)
        validations["A3"] = (ok, err)
        outputs["A3"] = str(verdict)

        if not ok:
            return {"halt": True, "halt_reason": f"A3 failed: {err}",
                    "rule_verdict": verdict,
                    "_retries": retries, "_outputs": outputs, "_validations": validations}

        return {"rule_verdict": verdict,
                "_retries": retries, "_outputs": outputs, "_validations": validations}

    node.__name__ = "node_A3"
    return node


def _node_audit():
    from agents.conflict_detector import check_rule_engine_agreement

    async def node(state: PipelineState) -> dict:
        try:
            verdict = check_rule_engine_agreement(
                state["client_profile"], state["product_profile"], state["rule_verdict"]
            )
            return {"audit_verdict": verdict}
        except Exception as exc:
            return {"halt": True, "halt_reason": f"audit pre-check failed: {exc}"}

    node.__name__ = "node_audit"
    return node


def _node_a4(model_client):
    from agents.conflict_detector import run_conflict_detector

    async def node(state: PipelineState) -> dict:
        retries     = dict(state.get("_retries")     or {})
        outputs     = dict(state.get("_outputs")     or {})
        validations = dict(state.get("_validations") or {})

        try:
            report = await run_conflict_detector(
                state["client_profile"], state["product_profile"],
                state["rule_verdict"], model_client
            )
        except openai.RateLimitError as exc:
            msg = f"Rate limit at A4: {exc}"
            return {"halt": True, "halt_reason": msg,
                    "_retries": retries, "_outputs": outputs, "_validations": validations}

        probe = {**state, "conflict_report": report}
        ok, err = validate_after_a4(probe)
        validations["A4"] = (ok, err)
        outputs["A4"] = str(report)

        updates: dict[str, Any] = {
            "conflict_report": report,
            "_retries": retries, "_outputs": outputs, "_validations": validations,
        }

        if not ok:
            updates["halt"] = True
            updates["halt_reason"] = f"A4 failed: {err}"
            return updates

        if report.get("escalate") is True:
            updates["escalated"]   = True
            updates["halt_reason"] = "Escalation flagged by conflict detector."

        return updates

    node.__name__ = "node_A4"
    return node


def _node_a5(model_client):
    from agents.disclosure_agent import run_disclosure_agent

    async def node(state: PipelineState) -> dict:
        retries     = dict(state.get("_retries")     or {})
        outputs     = dict(state.get("_outputs")     or {})
        validations = dict(state.get("_validations") or {})

        try:
            report = await run_disclosure_agent(
                state["client_profile"], state["product_profile"],
                state["rule_verdict"], state["conflict_report"],
                model_client=model_client,
            )
        except openai.RateLimitError as exc:
            msg = f"Rate limit at A5: {exc}"
            return {"halt": True, "halt_reason": msg,
                    "_retries": retries, "_outputs": outputs, "_validations": validations}

        probe = {**state, "suitability_report": report}
        ok, err = validate_after_a5(probe)
        validations["A5"] = (ok, err)
        outputs["A5"] = str(report)

        if not ok:
            return {"halt": True, "halt_reason": f"A5 failed: {err}",
                    "suitability_report": report,
                    "_retries": retries, "_outputs": outputs, "_validations": validations}

        return {"suitability_report": report,
                "_retries": retries, "_outputs": outputs, "_validations": validations}

    node.__name__ = "node_A5"
    return node


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph(model_client, verifier_client, on_event=None):
    """
    Build and compile the MiFID II suitability pipeline as a LangGraph StateGraph.

    verifier_client: stronger model (OpenRouter/Gemini) used by AV nodes only.
    on_event: forwarded into A1/A2 nodes so they can stream intermediate steps.
    """
    _retry = RetryPolicy(max_attempts=2, retry_on=Exception)

    g = StateGraph(PipelineState)

    g.add_node("A1",        _node_a1(model_client, verifier_client, on_event), retry=_retry)
    g.add_node("A2",        _node_a2(model_client, verifier_client, on_event), retry=_retry)
    g.add_node("pre_check", _node_pre_check())
    g.add_node("A3",        _node_a3(model_client), retry=_retry)
    g.add_node("audit",     _node_audit())
    g.add_node("A4",        _node_a4(model_client), retry=_retry)
    g.add_node("A5",        _node_a5(model_client), retry=_retry)

    g.set_entry_point("A1")

    g.add_conditional_edges("A1",        _route, {"continue": "A2",        END: END})
    g.add_conditional_edges("A2",        _route, {"continue": "pre_check", END: END})
    g.add_conditional_edges("pre_check", _route, {"continue": "A3",        END: END})
    g.add_conditional_edges("A3",        _route, {"continue": "audit",     END: END})
    g.add_conditional_edges("audit",     _route, {"continue": "A4",        END: END})
    g.add_conditional_edges("A4",        _route, {"continue": "A5",        END: END})
    g.add_edge("A5", END)

    return g.compile()


# ── Public interface ───────────────────────────────────────────────────────────

async def run_pipeline(
    client_input: str,
    product_input: str,
    model_client,
    on_event=None,
) -> tuple:
    """
    Run the full MiFID II suitability pipeline.
    Returns (pipeline_state, audit_log).

    on_event: async callable(event_type: str, state_snapshot: dict)
              called after every major step — including intermediate steps
              inside A1 and A2 (a1_profiled, a1_verified, a1_corrected,
              a1_reverified, and the same for A2).
    """
    from config.llm_config import get_verifier_client
    verifier_client = get_verifier_client()

    graph = build_graph(model_client, verifier_client, on_event)

    initial: PipelineState = {
        "client_input":  client_input,
        "product_input": product_input,
        "_retries":      {},
        "_outputs":      {},
        "_validations":  {},
    }

    accumulated = dict(initial)

    try:
        async for chunk in graph.astream(initial, stream_mode="updates"):
            for node_name, update in chunk.items():
                accumulated.update(update)
                # Emit the node-complete event so the frontend sees the final
                # accumulated state after each LangGraph node finishes.
                if on_event:
                    await on_event(node_name, dict(accumulated))
        final_state = accumulated
    except Exception as exc:
        final_state = {
            **accumulated,
            "halt":        True,
            "halt_reason": f"Pipeline error: {exc}",
        }
        if on_event:
            await on_event("halt", dict(final_state))

    retries     = final_state.pop("_retries",     {})
    outputs     = final_state.pop("_outputs",     {})
    validations = final_state.pop("_validations", {})

    audit_log = build_audit_log(final_state, retries, outputs, validations)
    return final_state, audit_log
