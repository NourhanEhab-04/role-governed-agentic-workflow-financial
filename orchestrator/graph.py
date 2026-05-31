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

Verifier uses a separate Groq client (llama-3.3-70b-versatile, Meta Llama family)
so its failure modes are statistically independent from the OpenAI gpt-4.1-nano
specialist agents it checks — different provider, different model architecture.
"""

from __future__ import annotations

import json
from typing import Any

import openai
from langgraph.graph import StateGraph, END
from langgraph.types import RetryPolicy

from config.llm_config import OPENAI_MODEL, OPENAI_VERIFIER_MODEL


def _detect_field_overrides(before: dict, after: dict) -> list[dict]:
    """
    Return one entry per field that changed between before and after.
    Used to log deterministic override applications to the error chain.
    """
    return [
        {"field": k, "from": before[k], "to": after[k]}
        for k in after
        if k in before and before[k] != after[k]
    ]

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
    check_a3_score_drift,
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
        warnings    = list(state.get("_warnings")    or [])
        error_chain = list(state.get("_error_chain") or [])

        # ── 1. Run profiler ──────────────────────────────────────────────────
        try:
            profile = await run_client_profiler(
                state["client_input"], model_client=model_client
            )
        except openai.RateLimitError as exc:
            msg = f"Rate limit at A1: {exc}"
            return {"halt": True, "halt_reason": msg,
                    "_retries": retries, "_outputs": outputs, "_validations": validations,
                    "_warnings": warnings, "_error_chain": error_chain}

        probe = {**state, "client_profile": profile}
        ok, err = validate_after_a1(probe)
        validations["A1"] = (ok, err)
        outputs["A1"] = str(profile)

        if not ok:
            return {"halt": True, "halt_reason": f"A1 failed: {err}",
                    "client_profile": profile,
                    "_retries": retries, "_outputs": outputs, "_validations": validations,
                    "_warnings": warnings, "_error_chain": error_chain}

        a1_issues = check_a1_consistency(profile)
        updates: dict[str, Any] = {
            "client_profile":        profile,
            "a1_initial_profile":    profile,
            "a1_consistency_issues": a1_issues,
        }

        if on_event:
            await on_event("a1_profiled", {**state, **updates})

        # ── 2. Verify ────────────────────────────────────────────────────────
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

        if on_event:
            await on_event("a1_verified", {**state, **updates})

        # ── 3. Correction ────────────────────────────────────────────────────
        correction_applied = False
        fields_fixed: list[str] = []
        if updates.get("a1_verification", {}).get("passed") is False:
            fields_fixed = [
                f for f, c in updates["a1_verification"].get("field_checks", {}).items()
                if c.get("supported") is False
            ]
            try:
                corrected = await run_client_profiler_with_feedback(
                    state["client_input"], profile, updates["a1_verification"], model_client
                )
                updates["a1_correction"]  = {
                    "original":     profile,
                    "corrected":    corrected,
                    "fields_fixed": fields_fixed,
                }
                updates["client_profile"] = corrected
                correction_applied = True
                error_chain.append({
                    "stage":        "A1",
                    "event":        "verifier_correction_applied",
                    "fields_fixed": fields_fixed,
                    "specialist_model": OPENAI_MODEL,
                    "verifier_model":   OPENAI_VERIFIER_MODEL,
                })
            except Exception as exc:
                updates["a1_correction"] = {
                    "original": profile, "corrected": None,
                    "fields_fixed": [], "error": str(exc),
                }

            if on_event:
                await on_event("a1_corrected", {**state, **updates})

            # ── 4. Re-verify ─────────────────────────────────────────────────
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

                if on_event:
                    await on_event("a1_reverified", {**state, **updates})

        # ── 5. Deterministic override — always applied last ──────────────────
        profile_before_enforce = dict(updates["client_profile"])
        updates["client_profile"] = enforce_a1_consistency(updates["client_profile"])
        overrides = _detect_field_overrides(profile_before_enforce, updates["client_profile"])
        for override in overrides:
            error_chain.append({"stage": "A1", "event": "deterministic_override", **override})

        try:
            raw = json.loads(state["client_input"])
            if "portfolio_concentration_pct" in raw:
                updates["client_profile"] = {
                    **updates["client_profile"],
                    "portfolio_concentration_pct": raw["portfolio_concentration_pct"],
                }
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        # ── 6. Provenance record ─────────────────────────────────────────────
        override_fields = [o["field"] for o in overrides]
        updates["client_profile_provenance"] = {
            "source":               "verifier_corrected" if correction_applied else "original_a1",
            "specialist_model":     OPENAI_MODEL,
            "verifier_model":       OPENAI_VERIFIER_MODEL,
            "correction_applied":   correction_applied,
            "correction_fields":    fields_fixed,
            "deterministic_overrides": override_fields,
        }

        # ── 7. Reasoning trace (explainability) ──────────────────────────────
        final_profile = updates["client_profile"]
        updates["a1_reasoning"] = {
            "agent": "A1",
            "role":  "Client Profiler",
            "method": "llm_extraction_with_verifier_feedback",
            "verification": {
                "run":     bool(updates.get("a1_verification")),
                "passed":  (updates.get("a1_verification") or {}).get("passed"),
                "confidence": (updates.get("a1_verification") or {}).get("confidence"),
                "issues":  (updates.get("a1_verification") or {}).get("issues", []),
            },
            "correction": {
                "applied":    correction_applied,
                "fields_fixed": fields_fixed,
            },
            "deterministic_overrides": override_fields,
            "decision_factors": {
                field: {
                    "value":  final_profile.get(field),
                    "source": (
                        "deterministic_override" if field in override_fields
                        else "verifier_corrected" if (correction_applied and field in fields_fixed)
                        else "llm_extracted"
                    ),
                }
                for field in final_profile
            },
        }

        updates["_retries"]     = retries
        updates["_outputs"]     = outputs
        updates["_validations"] = validations
        updates["_warnings"]    = warnings
        updates["_error_chain"] = error_chain
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
        warnings    = list(state.get("_warnings")    or [])
        error_chain = list(state.get("_error_chain") or [])

        # ── 1. Run classifier ────────────────────────────────────────────────
        try:
            profile = await run_product_classifier(
                state["product_input"], model_client=model_client
            )
        except openai.RateLimitError as exc:
            msg = f"Rate limit at A2: {exc}"
            return {"halt": True, "halt_reason": msg,
                    "_retries": retries, "_outputs": outputs, "_validations": validations,
                    "_warnings": warnings, "_error_chain": error_chain}

        probe = {**state, "product_profile": profile}
        ok, err = validate_after_a2(probe)
        validations["A2"] = (ok, err)
        outputs["A2"] = str(profile)

        if not ok:
            return {"halt": True, "halt_reason": f"A2 failed: {err}",
                    "product_profile": profile,
                    "_retries": retries, "_outputs": outputs, "_validations": validations,
                    "_warnings": warnings, "_error_chain": error_chain}

        a2_issues    = check_a2_consistency(profile)
        cross_issues = check_a1_a2_cross(state.get("client_profile", {}), profile)

        updates: dict[str, Any] = {
            "product_profile":          profile,
            "a2_initial_profile":       profile,
            "a2_consistency_issues":    a2_issues,
            "cross_consistency_issues": cross_issues,
        }

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

        if on_event:
            await on_event("a2_verified", {**state, **updates})

        # ── 3. Correction ────────────────────────────────────────────────────
        correction_applied = False
        fields_fixed: list[str] = []
        if updates.get("a2_verification", {}).get("passed") is False:
            fields_fixed = [
                f for f, c in updates["a2_verification"].get("field_checks", {}).items()
                if c.get("supported") is False
            ]
            try:
                corrected = await run_product_classifier_with_feedback(
                    state["product_input"], profile, updates["a2_verification"], model_client
                )
                updates["a2_correction"]   = {
                    "original":     profile,
                    "corrected":    corrected,
                    "fields_fixed": fields_fixed,
                }
                updates["product_profile"] = corrected
                correction_applied = True
                error_chain.append({
                    "stage":        "A2",
                    "event":        "verifier_correction_applied",
                    "fields_fixed": fields_fixed,
                    "specialist_model": OPENAI_MODEL,
                    "verifier_model":   OPENAI_VERIFIER_MODEL,
                })
            except Exception as exc:
                updates["a2_correction"] = {
                    "original": profile, "corrected": None,
                    "fields_fixed": [], "error": str(exc),
                }

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

                if on_event:
                    await on_event("a2_reverified", {**state, **updates})

        # ── 5. Deterministic ESMA overrides — always applied last ────────────
        profile_before_enforce = dict(updates["product_profile"])
        updates["product_profile"] = enforce_a2_consistency(updates["product_profile"])
        overrides = _detect_field_overrides(profile_before_enforce, updates["product_profile"])
        for override in overrides:
            error_chain.append({"stage": "A2", "event": "deterministic_override", **override})

        # ── 6. Provenance record ─────────────────────────────────────────────
        override_fields = [o["field"] for o in overrides]
        updates["product_profile_provenance"] = {
            "source":               "verifier_corrected" if correction_applied else "original_a2",
            "specialist_model":     OPENAI_MODEL,
            "verifier_model":       OPENAI_VERIFIER_MODEL,
            "correction_applied":   correction_applied,
            "correction_fields":    fields_fixed,
            "deterministic_overrides": override_fields,
        }

        # ── 7. Reasoning trace (explainability) ──────────────────────────────
        final_profile = updates["product_profile"]
        updates["a2_reasoning"] = {
            "agent": "A2",
            "role":  "Product Classifier",
            "method": "llm_extraction_with_verifier_feedback",
            "verification": {
                "run":        bool(updates.get("a2_verification")),
                "passed":     (updates.get("a2_verification") or {}).get("passed"),
                "confidence": (updates.get("a2_verification") or {}).get("confidence"),
                "issues":     (updates.get("a2_verification") or {}).get("issues", []),
            },
            "correction": {
                "applied":    correction_applied,
                "fields_fixed": fields_fixed,
            },
            "deterministic_overrides": override_fields,
            "decision_factors": {
                field: {
                    "value":  final_profile.get(field),
                    "source": (
                        "deterministic_override" if field in override_fields
                        else "verifier_corrected" if (correction_applied and field in fields_fixed)
                        else "llm_extracted"
                    ),
                }
                for field in final_profile
            },
        }

        updates["_retries"]     = retries
        updates["_outputs"]     = outputs
        updates["_validations"] = validations
        updates["_warnings"]    = warnings
        updates["_error_chain"] = error_chain
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
        warnings    = list(state.get("_warnings")    or [])
        error_chain = list(state.get("_error_chain") or [])

        try:
            verdict = await run_rule_engine_agent(
                state["client_profile"], state["product_profile"], model_client
            )
        except openai.RateLimitError as exc:
            msg = f"Rate limit at A3: {exc}"
            return {"halt": True, "halt_reason": msg,
                    "_retries": retries, "_outputs": outputs, "_validations": validations,
                    "_warnings": warnings, "_error_chain": error_chain}

        probe = {**state, "rule_verdict": verdict}
        ok, err = validate_after_a3(probe)
        validations["A3"] = (ok, err)
        outputs["A3"] = str(verdict)

        if not ok:
            # Three-point integrity failure: A3 LLM disagrees with pre_check.
            # Recover using pre_check_verdict (deterministic, already validated).
            pre = state.get("pre_check_verdict")
            if pre is not None:
                pre_rules = {r["rule"]: "PASS" if r["pass"] else "FAIL"
                             for r in pre.get("rules", [])}
                from rule_engine.rule_engine import evaluate_suitability as _eval_suit
                _raw = _eval_suit(state["client_profile"], state["product_profile"])
                verdict = {
                    "score":             pre["score"],
                    "decision":          pre["decision"],
                    "hard_failed_rules": pre.get("hard_failed_rules", []),
                    "rules":             pre_rules,
                    "rule_details":      {r["rule"]: r["detail"] for r in _raw["rules"]},
                }
                warnings.append({
                    "stage": "A3", "event": "integrity_failure_recovered",
                    "a3_decision": outputs["A3"],
                    "pre_check_decision": pre.get("decision"),
                    "message": f"A3 integrity check failed ({err}). Recovered from pre_check.",
                })
                error_chain.append({
                    "stage": "A3", "event": "integrity_failure_recovered", "error": err,
                })
                validations["A3"] = (True, "pre_check_fallback")
                outputs["A3"] = str(verdict)
            else:
                return {"halt": True, "halt_reason": f"A3 failed: {err}",
                        "rule_verdict": verdict,
                        "_retries": retries, "_outputs": outputs, "_validations": validations,
                        "_warnings": warnings, "_error_chain": error_chain}

        # Score drift: same decision + failed rules but different numeric score.
        # Non-halting: integrity is intact, but log it so auditors can investigate.
        drift = check_a3_score_drift(probe)
        if drift is not None:
            warnings.append(drift)
            error_chain.append({**drift, "event": "score_drift_warning"})

        # Reasoning trace: the rule engine IS the chain of thought for A3.
        # Every rule evaluation has a deterministic detail string from the rule engine.
        rule_details = verdict.get("rule_details", {})
        a3_reasoning = {
            "agent":  "A3",
            "role":   "Rule Engine Agent",
            "method": "deterministic_tool_call",
            "tool_used": "evaluate_suitability_tool",
            "governance": "LLM must call tool exactly once; cannot reason about suitability itself",
            "rule_evaluations": {
                rule_id: {
                    "result": verdict.get("rules", {}).get(rule_id),
                    "detail": rule_details.get(rule_id, ""),
                }
                for rule_id in ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
            },
            "score":            verdict.get("score"),
            "decision":         verdict.get("decision"),
            "hard_failed_rules": verdict.get("hard_failed_rules", []),
            "three_point_check": {
                "pre_check_decision":     (state.get("pre_check_verdict") or {}).get("decision"),
                "a3_decision":            verdict.get("decision"),
                "audit_agreement":        (state.get("audit_verdict") or {}).get("agreed"),
            },
        }

        return {"rule_verdict": verdict,
                "a3_reasoning": a3_reasoning,
                "_retries": retries, "_outputs": outputs, "_validations": validations,
                "_warnings": warnings, "_error_chain": error_chain}

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
        warnings    = list(state.get("_warnings")    or [])
        error_chain = list(state.get("_error_chain") or [])

        try:
            report, a4_overrides = await run_conflict_detector(
                state["client_profile"], state["product_profile"],
                state["rule_verdict"], model_client
            )
        except openai.RateLimitError as exc:
            msg = f"Rate limit at A4: {exc}"
            return {"halt": True, "halt_reason": msg,
                    "_retries": retries, "_outputs": outputs, "_validations": validations,
                    "_warnings": warnings, "_error_chain": error_chain}

        # Log A4 deterministic overrides to the error chain for auditors
        for override in a4_overrides:
            error_chain.append({
                "stage": "A4",
                "event": "deterministic_override",
                **override,
            })

        probe = {**state, "conflict_report": report}
        ok, err = validate_after_a4(probe)
        validations["A4"] = (ok, err)
        outputs["A4"] = str(report)

        # Reasoning trace: document each flag's evaluation and any corrections made
        a4_reasoning = {
            "agent":  "A4",
            "role":   "Conflict Detector",
            "method": "llm_with_deterministic_override",
            "governance": "LLM flags checked against four deterministic rules; hard invariants enforced",
            "flags_evaluated": [
                {
                    "rule_id":   f.get("rule_id"),
                    "triggered": f.get("triggered"),
                    "severity":  f.get("severity"),
                    "message":   f.get("message"),
                }
                for f in report.get("flags", [])
            ],
            "escalation_triggered": report.get("escalate", False),
            "deterministic_overrides_applied": bool(a4_overrides),
            "overrides_detail": a4_overrides,
            "reasoning_chain": " | ".join(
                f"{f.get('rule_id')}: {'TRIGGERED' if f.get('triggered') else 'CLEAR'}"
                for f in report.get("flags", [])
            ),
            "audit_triple_check_agreed": (state.get("audit_verdict") or {}).get("agreed"),
        }

        updates: dict[str, Any] = {
            "conflict_report": report,
            "a4_reasoning":    a4_reasoning,
            "_retries": retries, "_outputs": outputs, "_validations": validations,
            "_warnings": warnings, "_error_chain": error_chain,
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


def _node_a5(disclosure_client):
    from agents.disclosure_agent import run_disclosure_agent

    async def node(state: PipelineState) -> dict:
        retries     = dict(state.get("_retries")     or {})
        outputs     = dict(state.get("_outputs")     or {})
        validations = dict(state.get("_validations") or {})
        warnings    = list(state.get("_warnings")    or [])
        error_chain = list(state.get("_error_chain") or [])

        try:
            report = await run_disclosure_agent(
                state["client_profile"], state["product_profile"],
                state["rule_verdict"], state["conflict_report"],
                model_client=disclosure_client,
            )
        except openai.RateLimitError as exc:
            msg = f"Rate limit at A5: {exc}"
            return {"halt": True, "halt_reason": msg,
                    "_retries": retries, "_outputs": outputs, "_validations": validations,
                    "_warnings": warnings, "_error_chain": error_chain}

        probe = {**state, "suitability_report": report}
        ok, err = validate_after_a5(probe)
        validations["A5"] = (ok, err)
        outputs["A5"] = str(report)

        if not ok:
            return {"halt": True, "halt_reason": f"A5 failed: {err}",
                    "suitability_report": report,
                    "_retries": retries, "_outputs": outputs, "_validations": validations,
                    "_warnings": warnings, "_error_chain": error_chain}

        # Reasoning trace: A5 decision path is fully deterministic given its inputs
        conflict_report = state.get("conflict_report") or {}
        rule_verdict    = state.get("rule_verdict") or {}
        escalate        = conflict_report.get("escalate", False)
        a5_reasoning = {
            "agent":  "A5",
            "role":   "Disclosure Agent",
            "method": "llm_disclosure_with_structured_skeleton",
            "governance": "Decision determined by conflict escalation flag; A5 writes explanations only",
            "decision_path": (
                "conflict_escalate=true → decision=ESCALATED"
                if escalate
                else f"conflict_escalate=false → mirror rule_verdict.decision={rule_verdict.get('decision')}"
            ),
            "final_decision": report.get("decision"),
            "rules_failed": [
                f["rule_id"] for f in report.get("rule_findings", [])
                if f.get("status") == "FAIL"
            ],
            "rules_passed": [
                f["rule_id"] for f in report.get("rule_findings", [])
                if f.get("status") == "PASS"
            ],
            "regulatory_basis":       report.get("regulatory_basis"),
            "explanation_provided":   bool(report.get("client_facing_summary")),
            "flags_addressed_count":  len(report.get("flags_addressed") or []),
        }

        return {"suitability_report": report,
                "a5_reasoning": a5_reasoning,
                "_retries": retries, "_outputs": outputs, "_validations": validations,
                "_warnings": warnings, "_error_chain": error_chain}

    node.__name__ = "node_A5"
    return node


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph(model_client, verifier_client, disclosure_client, on_event=None):
    """
    Build and compile the MiFID II suitability pipeline as a LangGraph StateGraph.

    model_client:      gpt-4.1-nano (OpenAI)                   — used by A1, A2, A3, A4.
    verifier_client:   llama-3.3-70b-versatile (Groq/Meta)     — used by the verifier (AV) nodes only.
    disclosure_client: gpt-4o-mini (OpenAI)                    — used exclusively by A5 (disclosure agent).
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
    g.add_node("A5",        _node_a5(disclosure_client), retry=_retry)

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

    If DATABASE_URL is set in the environment, the full EU AI Act audit record
    is persisted to PostgreSQL automatically.  If DATABASE_URL is absent or the
    insert fails, the pipeline result is still returned normally — DB persistence
    is non-blocking and never causes the pipeline to fail.

    on_event: async callable(event_type: str, state_snapshot: dict)
              called after every major step — including intermediate steps
              inside A1 and A2 (a1_profiled, a1_verified, a1_corrected,
              a1_reverified, and the same for A2).
    """
    import os
    from config.llm_config import (
        get_verifier_client, get_disclosure_client,
        OPENAI_MODEL, OPENAI_VERIFIER_MODEL,
    )
    verifier_client   = get_verifier_client()
    disclosure_client = get_disclosure_client()

    graph = build_graph(model_client, verifier_client, disclosure_client, on_event)

    initial: PipelineState = {
        "client_input":  client_input,
        "product_input": product_input,
        "_retries":      {},
        "_outputs":      {},
        "_validations":  {},
        "_warnings":     [],
        "_error_chain":  [],
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
    # _warnings and _error_chain stay in final_state so build_audit_log can read them.

    audit_log = build_audit_log(final_state, retries, outputs, validations)

    # ── PostgreSQL persistence (non-blocking) ──────────────────────────────────
    # Only runs when DATABASE_URL is configured.  Failures are logged to warnings
    # but never propagate — the pipeline result is always returned to the caller.
    if os.environ.get("DATABASE_URL"):
        from orchestrator.audit import build_full_audit_record, persist_audit_log
        try:
            full_record = build_full_audit_record(
                final_state, retries, outputs, validations,
                client_text=client_input,
                product_text=product_input,
                model_version={
                    "specialist":  OPENAI_MODEL,
                    "verifier":    OPENAI_VERIFIER_MODEL,
                    "disclosure":  "gpt-4o-mini",
                },
            )
            assessment_id = await persist_audit_log(full_record)
            audit_log["assessment_id"] = assessment_id
            audit_log["db_persisted"]  = True
        except Exception as exc:
            audit_log["db_persisted"]  = False
            audit_log["db_error"]      = str(exc)
            final_state.setdefault("_warnings", []).append({
                "stage": "PERSIST",
                "event": "db_persist_failed",
                "error": str(exc),
            })

    return final_state, audit_log
