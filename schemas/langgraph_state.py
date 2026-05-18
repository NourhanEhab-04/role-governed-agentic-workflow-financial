# schemas/langgraph_state.py
"""
Typed state for the LangGraph pipeline.

Every key is Optional / has a default so nodes can be added incrementally.
LangGraph passes this dict through all nodes; each node returns only the
keys it wants to update (a partial dict).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class PipelineState(TypedDict, total=False):
    # ── Inputs ──────────────────────────────────────────────────────────────
    client_input: str
    product_input: str

    # ── A1 — Client Profiler ────────────────────────────────────────────────
    client_profile: Dict[str, Any]
    a1_initial_profile: Optional[Dict[str, Any]]      # snapshot before any correction
    a1_consistency_issues: List[str]
    a1_verification: Optional[Dict[str, Any]]          # first AV pass
    a1_correction: Optional[Dict[str, Any]]            # corrector output {original, corrected, fields_fixed}
    a1_final_verification: Optional[Dict[str, Any]]    # AV re-check after correction

    # ── A2 — Product Classifier ─────────────────────────────────────────────
    product_profile: Dict[str, Any]
    a2_initial_profile: Optional[Dict[str, Any]]
    a2_consistency_issues: List[str]
    a2_verification: Optional[Dict[str, Any]]
    a2_correction: Optional[Dict[str, Any]]
    a2_final_verification: Optional[Dict[str, Any]]
    cross_consistency_issues: List[str]

    # ── Pre-check (between A2 and A3) ───────────────────────────────────────
    pre_check_verdict: Dict[str, Any]

    # ── A3 — Rule Engine Agent ──────────────────────────────────────────────
    rule_verdict: Dict[str, Any]

    # ── A4 — Conflict Detector ──────────────────────────────────────────────
    audit_verdict: Dict[str, Any]
    conflict_report: Dict[str, Any]
    escalated: bool

    # ── A5 — Disclosure Agent ───────────────────────────────────────────────
    suitability_report: Dict[str, Any]

    # ── Pipeline control ────────────────────────────────────────────────────
    halt: bool
    halt_reason: str

    # ── Audit metadata (not forwarded to frontend) ──────────────────────────
    _retries: Dict[str, int]
    _outputs: Dict[str, str]
    _validations: Dict[str, Any]
