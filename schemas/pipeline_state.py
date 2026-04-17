from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from schemas.client_profile import ClientProfile
from schemas.product_profile import ProductProfile
from schemas.rule_verdict import RuleVerdict


@dataclass
class AuditEntry:
    timestamp: datetime
    agent: str      # e.g. "A1_client_profiler", "orchestrator"
    event: str      # short description of what happened


@dataclass
class PipelineState:
    # Raw input as received — preserved unchanged for audit purposes
    raw_input: Optional[Dict[str, Any]] = None

    # Structured outputs populated by each agent in turn
    client_profile: Optional[ClientProfile] = None
    product_profile: Optional[ProductProfile] = None
    rule_verdict: Optional[RuleVerdict] = None

    # Conflict flags raised by the conflict-detection agent
    conflict_flags: List[str] = field(default_factory=list)

    # Final human-readable report produced at the end of the pipeline
    final_report: Optional[str] = None

    # Append-only log of timestamped events across all agents
    audit_log: List[AuditEntry] = field(default_factory=list)

    def log(self, agent: str, event: str) -> None:
        self.audit_log.append(AuditEntry(
            timestamp=datetime.now(timezone.utc),
            agent=agent,
            event=event,
        ))

# schemas/pipeline_state.py
"""
pipeline_state is the shared dict passed through the entire pipeline.
Three rule engine verdict slots enforce the bypass-proof architecture.
"""

PIPELINE_STATE_KEYS = {
    # Profiles (set by A1 and A2)
    "client_profile",
    "product_profile",
    # Three independent rule engine contacts
    "pre_check_verdict",    # A0 — set immediately after A1+A2, before A3 runs
    "rule_verdict",         # A3 — formal tool call result
    "audit_verdict",        # A4 — independent re-run for cross-check
    # Conflict report and final report
    "conflict_report",
    "suitability_report",
    # Pipeline control
    "escalated",
    "halted",
    "halt_reason",
}


def make_empty_pipeline_state() -> dict:
    """Return a pipeline_state with all keys present, all values None."""
    return {k: None for k in PIPELINE_STATE_KEYS}
