# orchestrator/audit.py

import datetime
import json
import uuid


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
    from rule_engine.rule_engine import RULE_ENGINE_VERSION

    # Map agent IDs to their reasoning trace keys in pipeline_state
    _reasoning_keys = {
        "A1": "a1_reasoning",
        "A2": "a2_reasoning",
        "A3": "a3_reasoning",
        "A4": "a4_reasoning",
        "A5": "a5_reasoning",
    }

    return {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "rule_engine_version": RULE_ENGINE_VERSION,
        "final_decision": (
            (pipeline_state.get("suitability_report") or {})
            .get("decision", "UNKNOWN")
        ),
        "escalated": pipeline_state.get("escalated", False),
        "halted": pipeline_state.get("halt", False),
        "halt_reason": pipeline_state.get("halt_reason"),
        "warnings": pipeline_state.get("_warnings", []),
        "error_chain": pipeline_state.get("_error_chain", []),
        "stages": {
            stage: {
                "raw_output":        agent_outputs.get(stage),
                "validation_passed": validation_results.get(stage, (None, ""))[0],
                "validation_error":  validation_results.get(stage, (None, ""))[1],
                "retry_count":       retry_counts.get(stage, 0),
                # Explainability: per-agent chain of thought
                "reasoning_trace":   pipeline_state.get(_reasoning_keys[stage]),
            }
            for stage in ["A1", "A2", "A3", "A4", "A5"]
        },
    }


def build_full_audit_record(
    pipeline_state: dict,
    retry_counts: dict,
    agent_outputs: dict,
    validation_results: dict,
    *,
    client_text: str,
    product_text: str,
    model_version: dict,
    schema_version: int = 1,
    human_reviewer_id: str | None = None,
    human_reviewed_at: str | None = None,
) -> dict:
    """
    Build the complete EU AI Act-compliant audit record.

    Covers every field required for high-risk AI suitability assessments:
      - verbatim inputs preserved exactly as received
      - every intermediate output (A1 profile, A2 profile, rule verdicts,
        conflict report, suitability report)
      - every validation that ran and whether it passed
      - every retry: original output, verifier feedback, corrected output
      - both agent and verifier model versions
      - final decision and its full regulatory basis
      - optional human reviewer ID and review timestamp

    The returned dict is a strict superset of build_audit_log() — it includes
    the same ``timestamp``, ``stages``, and ``final_decision`` keys so that
    all existing callers continue to work without modification.
    """
    base = build_audit_log(pipeline_state, retry_counts, agent_outputs, validation_results)
    report = (pipeline_state.get("suitability_report") or {})

    return {
        # ── Identity ────────────────────────────────────────────────────────
        "assessment_id":      str(uuid.uuid4()),
        "schema_version":     schema_version,
        "rule_engine_version": base["rule_engine_version"],

        # ── Backward-compatible keys from build_audit_log ───────────────────
        "timestamp":        base["timestamp"],   # ISO 8601 string ending in "Z"
        "final_decision":   base["final_decision"],
        "escalated":        base["escalated"],
        "halted":           base["halted"],
        "halt_reason":      base["halt_reason"],
        "stages":           base["stages"],       # per-stage raw output + validation

        # ── Model traceability ───────────────────────────────────────────────
        # Both specialist and verifier models are recorded so the exact inference
        # stack that produced each output is reproducible by an auditor.
        "model_version":    model_version,

        # ── Verbatim inputs (preserved exactly as received) ──────────────────
        "client_text":      client_text,
        "product_text":     product_text,

        # ── Intermediate outputs: profiling ──────────────────────────────────
        # initial_profile = raw LLM output before the feedback loop
        # client_profile  = final profile after verifier corrections + deterministic overrides
        "a1_initial_profile":    pipeline_state.get("a1_initial_profile"),
        "client_profile":        pipeline_state.get("client_profile"),
        "a2_initial_profile":    pipeline_state.get("a2_initial_profile"),
        "product_profile":       pipeline_state.get("product_profile"),

        # ── Intermediate outputs: rule verdicts ──────────────────────────────
        "pre_check_verdict":     pipeline_state.get("pre_check_verdict"),
        "rule_verdict":          pipeline_state.get("rule_verdict"),
        "audit_verdict":         pipeline_state.get("audit_verdict"),

        # ── Intermediate outputs: conflict + final report ────────────────────
        "conflict_report":       pipeline_state.get("conflict_report"),
        "suitability_report":    pipeline_state.get("suitability_report"),

        # ── Validation results per stage ─────────────────────────────────────
        "validations": {
            stage: {
                "passed": validation_results.get(stage, (None, ""))[0],
                "error":  validation_results.get(stage, (None, ""))[1],
            }
            for stage in ["A1", "A2", "A3", "A4", "A5"]
        },

        # ── Consistency checks (deterministic layer) ─────────────────────────
        "a1_consistency_issues":    pipeline_state.get("a1_consistency_issues"),
        "a2_consistency_issues":    pipeline_state.get("a2_consistency_issues"),
        "cross_consistency_issues": pipeline_state.get("cross_consistency_issues"),

        # ── Verifier feedback loops (full retry trail) ───────────────────────
        # verification        = first verifier pass on the LLM output
        # correction          = {original, corrected, fields_fixed, attempts}
        # final_verification  = verifier pass after correction (present only when
        #                       at least one retry ran)
        "a1_verification":       pipeline_state.get("a1_verification"),
        "a1_correction":         pipeline_state.get("a1_correction"),
        "a1_final_verification": pipeline_state.get("a1_final_verification"),
        "a2_verification":       pipeline_state.get("a2_verification"),
        "a2_correction":         pipeline_state.get("a2_correction"),
        "a2_final_verification": pipeline_state.get("a2_final_verification"),

        # ── Retry counts per stage ───────────────────────────────────────────
        "retry_counts":          retry_counts,

        # ── Regulatory basis ─────────────────────────────────────────────────
        "regulatory_basis":      report.get("regulatory_basis"),

        # ── Provenance: how each profile was produced ────────────────────────
        "client_profile_provenance":  pipeline_state.get("client_profile_provenance"),
        "product_profile_provenance": pipeline_state.get("product_profile_provenance"),

        # ── Warnings and error chain ─────────────────────────────────────────
        # warnings: non-halting anomalies (e.g. score drift between pre-check and A3)
        # error_chain: ordered log of every correction, override, and warning applied
        "warnings":   pipeline_state.get("_warnings", []),
        "error_chain": pipeline_state.get("_error_chain", []),

        # ── Per-agent reasoning traces (explainability / chain of thought) ───
        # Each trace documents: inputs considered, decision factors, governance
        # layers applied, corrections made, and the final output per agent.
        # EU AI Act Art. 13 (transparency) and Art. 14 (human oversight).
        "a1_reasoning": pipeline_state.get("a1_reasoning"),
        "a2_reasoning": pipeline_state.get("a2_reasoning"),
        "a3_reasoning": pipeline_state.get("a3_reasoning"),
        "a4_reasoning": pipeline_state.get("a4_reasoning"),
        "a5_reasoning": pipeline_state.get("a5_reasoning"),

        # ── Human review (populated post-pipeline when a reviewer acts) ──────
        "human_reviewer_id":     human_reviewer_id,
        "human_reviewed_at":     human_reviewed_at,
    }


def _safe_jsonb(obj) -> str:
    """Serialize obj to a JSON string, converting non-serializable values to str."""
    def _default(o):
        if isinstance(o, datetime.datetime):
            return o.isoformat()
        if isinstance(o, set):
            return sorted(o)
        return str(o)
    return json.dumps(obj, default=_default)


async def persist_audit_log(
    full_record: dict,
    *,
    pool=None,
) -> str:
    """
    Insert full_record into the PostgreSQL ``audit_log`` table.

    Returns the ``assessment_id`` UUID string of the inserted row.

    Prerequisites
    -------------
    - asyncpg must be installed (``pip install asyncpg``)
    - DATABASE_URL environment variable must point to a reachable PostgreSQL instance
    - apply_migrations() must have been called at least once to create the table

    The function is deliberately narrow: it only inserts; it never updates or
    deletes.  The immutability trigger in the database provides a second line of
    defence, but compliance starts here.
    """
    from orchestrator.db import get_pool as _get_pool

    _pool = pool if pool is not None else await _get_pool()

    # Columns stored as top-level scalars for fast querying.
    assessment_id     = full_record["assessment_id"]
    schema_version    = full_record.get("schema_version", 1)
    model_version     = full_record.get("model_version", {})
    client_text       = full_record["client_text"]
    product_text      = full_record["product_text"]
    final_decision    = full_record.get("final_decision")
    regulatory_basis  = full_record.get("regulatory_basis")
    human_reviewer_id = full_record.get("human_reviewer_id")
    human_reviewed_at = full_record.get("human_reviewed_at")

    # Parse the ISO-8601 timestamp stored in the record.
    ts_str   = full_record.get("timestamp", datetime.datetime.utcnow().isoformat() + "Z")
    created_at = datetime.datetime.fromisoformat(ts_str.rstrip("Z")).replace(
        tzinfo=datetime.timezone.utc
    )

    # state_snapshot captures everything not already in a dedicated column so
    # an auditor can reconstruct the complete run from a single JSONB blob.
    _snapshot_exclude = {
        "client_text", "product_text", "model_version",
        "schema_version", "assessment_id", "timestamp",
        "final_decision", "regulatory_basis",
        "human_reviewer_id", "human_reviewed_at",
    }
    state_snapshot_obj = {k: v for k, v in full_record.items()
                          if k not in _snapshot_exclude}
    # Safe-serialize then re-parse so asyncpg receives a plain Python object
    # (handles datetimes, sets, and other non-JSON-native types).
    state_snapshot = json.loads(_safe_jsonb(state_snapshot_obj))

    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO audit_log (
                schema_version, assessment_id, created_at,
                model_version, client_text, product_text,
                state_snapshot, final_decision, regulatory_basis,
                human_reviewer_id, human_reviewed_at
            ) VALUES (
                $1, $2, $3,
                $4, $5, $6,
                $7, $8, $9,
                $10, $11
            )
            """,
            schema_version,
            uuid.UUID(assessment_id),
            created_at,
            model_version,          # asyncpg serialises dict → jsonb
            client_text,
            product_text,
            state_snapshot,         # asyncpg serialises dict → jsonb
            final_decision,
            regulatory_basis,
            human_reviewer_id,
            human_reviewed_at,
        )

    return assessment_id
