# tests/test_audit_pg.py
"""
Unit tests for build_full_audit_record and persist_audit_log.

No live database required — asyncpg is mocked at the connection pool level.
Tests verify:
  - build_full_audit_record returns all EU AI Act required fields
  - build_full_audit_record is a strict superset of build_audit_log
    (keeps timestamp, stages, final_decision so existing callers work)
  - persist_audit_log calls conn.execute with the right positional arguments
  - persist_audit_log returns the assessment_id from the record
  - persist_audit_log raises RuntimeError when DATABASE_URL is missing
  - apply_migrations reads SQL files from db/migrations/ and executes them
"""

import datetime
import json
import uuid
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.audit import build_audit_log, build_full_audit_record, persist_audit_log


# ── shared fixtures ──────────────────────────────────────────────────────────

STATE = {
    "client_profile": {"financial_knowledge": "basic", "risk_tolerance_score": 3},
    "product_profile": {"product_name": "X Fund", "risk_class": 3},
    "a1_initial_profile": {"financial_knowledge": "basic", "risk_tolerance_score": 2},
    "a2_initial_profile": {"product_name": "X Fund", "risk_class": 2},
    "pre_check_verdict": {"decision": "SUITABLE"},
    "rule_verdict": {"decision": "SUITABLE", "score": 80, "rules": [], "failed_rules": []},
    "audit_verdict": {"agreed": True},
    "conflict_report": {"flags": [], "escalate": False, "summary": "ok"},
    "suitability_report": {
        "decision": "SUITABLE",
        "regulatory_basis": "Article 25(2) MiFID II",
        "rule_findings": [{"rule_id": f"R{i}", "passed": True, "detail": ""} for i in range(1, 8)],
        "flags_addressed": [],
        "client_summary": "Suitable.",
    },
    "a1_consistency_issues": [],
    "a2_consistency_issues": [],
    "cross_consistency_issues": [],
    "a1_verification": {"passed": True, "confidence": 0.95, "issues": [], "field_checks": {}},
    "a1_correction": None,
    "a1_final_verification": None,
    "a2_verification": {"passed": True, "confidence": 0.92, "issues": [], "field_checks": {}},
    "a2_correction": None,
    "a2_final_verification": None,
    "escalated": False,
    "halt": False,
    "halt_reason": None,
}

RETRIES = {"A1": 0, "A2": 0, "A3": 0, "A4": 0, "A5": 0}
OUTPUTS = {
    "A1": '{"financial_knowledge":"basic"}',
    "A2": '{"product_name":"X Fund"}',
    "A3": '{"decision":"SUITABLE"}',
    "A4": '{"escalate":false}',
    "A5": '{"decision":"SUITABLE"}',
}
VALIDATIONS = {
    "A1": (True, ""),
    "A2": (True, ""),
    "A3": (True, ""),
    "A4": (True, ""),
    "A5": (True, ""),
}
MODEL_VERSION = {
    "agent_model": "llama-3.1-8b-instant",
    "verifier_model": "llama-3.3-70b-versatile",
    "provider": "groq",
}
CLIENT_TEXT = '{"financial_knowledge": "basic"}'
PRODUCT_TEXT = '{"product_name": "X Fund"}'


def _build() -> dict:
    return build_full_audit_record(
        STATE, RETRIES, OUTPUTS, VALIDATIONS,
        client_text=CLIENT_TEXT,
        product_text=PRODUCT_TEXT,
        model_version=MODEL_VERSION,
    )


# ── build_full_audit_record ───────────────────────────────────────────────────

def test_full_record_has_all_eu_ai_act_fields():
    rec = _build()
    required = [
        # identity
        "assessment_id", "schema_version",
        # timestamps
        "timestamp",
        # model traceability
        "model_version",
        # verbatim inputs
        "client_text", "product_text",
        # intermediate outputs
        "a1_initial_profile", "client_profile",
        "a2_initial_profile", "product_profile",
        "pre_check_verdict", "rule_verdict", "audit_verdict",
        "conflict_report", "suitability_report",
        # validations
        "validations",
        # consistency checks
        "a1_consistency_issues", "a2_consistency_issues", "cross_consistency_issues",
        # retry trail (verifier feedback loops)
        "a1_verification", "a1_correction", "a1_final_verification",
        "a2_verification", "a2_correction", "a2_final_verification",
        # retry counts
        "retry_counts",
        # decision + regulatory basis
        "final_decision", "regulatory_basis",
        # human review
        "human_reviewer_id", "human_reviewed_at",
    ]
    missing = [f for f in required if f not in rec]
    assert not missing, f"Missing EU AI Act required fields: {missing}"


def test_full_record_is_superset_of_build_audit_log():
    """Existing callers that check timestamp/stages/final_decision must still work."""
    base = build_audit_log(STATE, RETRIES, OUTPUTS, VALIDATIONS)
    rec = _build()
    for key in ("timestamp", "stages", "final_decision", "escalated", "halted", "halt_reason"):
        assert key in rec, f"Backward-compat key '{key}' missing from full record"
    # stages structure must be identical
    assert rec["stages"] == base["stages"]


def test_full_record_timestamp_is_iso8601_with_z():
    rec = _build()
    ts = rec["timestamp"]
    assert ts.endswith("Z"), f"timestamp must end with Z, got: '{ts}'"
    assert "T" in ts, f"timestamp must be ISO 8601, got: '{ts}'"


def test_full_record_verbatim_inputs_preserved():
    rec = _build()
    assert rec["client_text"] == CLIENT_TEXT
    assert rec["product_text"] == PRODUCT_TEXT


def test_full_record_model_version_captured():
    rec = _build()
    assert rec["model_version"]["agent_model"] == "llama-3.1-8b-instant"
    assert rec["model_version"]["verifier_model"] == "llama-3.3-70b-versatile"
    assert rec["model_version"]["provider"] == "groq"


def test_full_record_validations_per_stage():
    rec = _build()
    for stage in ["A1", "A2", "A3", "A4", "A5"]:
        v = rec["validations"][stage]
        assert "passed" in v and "error" in v


def test_full_record_final_decision_and_regulatory_basis():
    rec = _build()
    assert rec["final_decision"] == "SUITABLE"
    assert "Article 25" in rec["regulatory_basis"]


def test_full_record_assessment_id_is_valid_uuid():
    rec = _build()
    uuid.UUID(rec["assessment_id"])  # raises ValueError if invalid


def test_full_record_schema_version_defaults_to_1():
    rec = _build()
    assert rec["schema_version"] == 1


def test_full_record_schema_version_custom():
    rec = build_full_audit_record(
        STATE, RETRIES, OUTPUTS, VALIDATIONS,
        client_text=CLIENT_TEXT, product_text=PRODUCT_TEXT,
        model_version=MODEL_VERSION, schema_version=2,
    )
    assert rec["schema_version"] == 2


def test_full_record_human_reviewer_defaults_to_none():
    rec = _build()
    assert rec["human_reviewer_id"] is None
    assert rec["human_reviewed_at"] is None


def test_full_record_human_reviewer_can_be_set():
    rec = build_full_audit_record(
        STATE, RETRIES, OUTPUTS, VALIDATIONS,
        client_text=CLIENT_TEXT, product_text=PRODUCT_TEXT,
        model_version=MODEL_VERSION,
        human_reviewer_id="reviewer-007",
        human_reviewed_at="2026-05-22T12:00:00Z",
    )
    assert rec["human_reviewer_id"] == "reviewer-007"
    assert rec["human_reviewed_at"] == "2026-05-22T12:00:00Z"


def test_full_record_halted_state():
    # When the pipeline halts early (e.g. after A1), suitability_report is absent.
    halted_state = {
        **STATE,
        "halt": True,
        "halt_reason": "A1 failed twice.",
        "suitability_report": None,  # not yet produced
    }
    rec = build_full_audit_record(
        halted_state, {"A1": 1}, {}, {"A1": (False, "missing keys")},
        client_text=CLIENT_TEXT, product_text=PRODUCT_TEXT,
        model_version=MODEL_VERSION,
    )
    assert rec["halted"] is True
    assert rec["halt_reason"] == "A1 failed twice."
    assert rec["final_decision"] == "UNKNOWN"


def test_full_record_is_json_serializable():
    rec = _build()
    # Must not raise
    serialized = json.dumps(rec)
    parsed = json.loads(serialized)
    assert parsed["final_decision"] == "SUITABLE"


# ── persist_audit_log ────────────────────────────────────────────────────────

def _make_mock_pool():
    """Return a mock asyncpg pool whose acquire() context manager works."""
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=None)

    mock_acquire = MagicMock()
    mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_acquire.__aexit__ = AsyncMock(return_value=False)

    mock_pool = MagicMock()
    mock_pool.acquire = MagicMock(return_value=mock_acquire)
    return mock_pool, mock_conn


@pytest.mark.asyncio
async def test_persist_audit_log_returns_assessment_id():
    mock_pool, _ = _make_mock_pool()
    rec = _build()
    result = await persist_audit_log(rec, pool=mock_pool)
    assert result == rec["assessment_id"]


@pytest.mark.asyncio
async def test_persist_audit_log_calls_execute_once():
    mock_pool, mock_conn = _make_mock_pool()
    rec = _build()
    await persist_audit_log(rec, pool=mock_pool)
    mock_conn.execute.assert_called_once()


@pytest.mark.asyncio
async def test_persist_audit_log_passes_correct_scalars():
    mock_pool, mock_conn = _make_mock_pool()
    rec = _build()
    await persist_audit_log(rec, pool=mock_pool)

    call_args = mock_conn.execute.call_args
    # Positional args: SQL string + 11 bind params ($1–$11)
    _, *params = call_args.args
    schema_version, assessment_id_val, created_at_val, \
        model_version_val, client_text_val, product_text_val, \
        state_snapshot_val, final_decision_val, regulatory_basis_val, \
        human_reviewer_id_val, human_reviewed_at_val = params

    assert schema_version == 1
    assert str(assessment_id_val) == rec["assessment_id"]
    assert isinstance(created_at_val, datetime.datetime)
    assert created_at_val.tzinfo is not None  # must be timezone-aware
    assert model_version_val == json.dumps(MODEL_VERSION)  # serialized to JSON string for JSONB
    assert client_text_val == CLIENT_TEXT
    assert product_text_val == PRODUCT_TEXT
    assert isinstance(state_snapshot_val, str)  # serialized to JSON string for JSONB
    assert json.loads(state_snapshot_val)  # must be valid JSON
    assert final_decision_val == "SUITABLE"
    assert "Article 25" in regulatory_basis_val
    assert human_reviewer_id_val is None
    assert human_reviewed_at_val is None


@pytest.mark.asyncio
async def test_persist_audit_log_state_snapshot_excludes_scalar_columns():
    """Columns stored as top-level scalars must not be duplicated in state_snapshot."""
    mock_pool, mock_conn = _make_mock_pool()
    rec = _build()
    await persist_audit_log(rec, pool=mock_pool)

    call_args = mock_conn.execute.call_args
    _, *params = call_args.args
    state_snapshot = json.loads(params[6])  # $7 is now a JSON string; parse before key-checking

    for excluded_key in ("client_text", "product_text", "model_version",
                          "assessment_id", "final_decision", "regulatory_basis"):
        assert excluded_key not in state_snapshot, (
            f"'{excluded_key}' should not appear in state_snapshot "
            f"(it has its own column)"
        )


@pytest.mark.asyncio
async def test_persist_audit_log_state_snapshot_contains_intermediate_outputs():
    mock_pool, mock_conn = _make_mock_pool()
    rec = _build()
    await persist_audit_log(rec, pool=mock_pool)

    call_args = mock_conn.execute.call_args
    _, *params = call_args.args
    state_snapshot = json.loads(params[6])  # $7 is a JSON string; parse before key-checking

    for key in ("client_profile", "product_profile", "rule_verdict",
                 "conflict_report", "validations", "retry_counts"):
        assert key in state_snapshot, f"Expected '{key}' in state_snapshot"


@pytest.mark.asyncio
async def test_persist_audit_log_raises_without_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    # Reset the cached pool so get_pool() re-reads the env var
    import orchestrator.db as db_mod
    db_mod._pool = None

    rec = _build()
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        await persist_audit_log(rec)  # no pool= kwarg → falls through to get_pool()


# ── apply_migrations ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_apply_migrations_executes_sql_files():
    from orchestrator.db import apply_migrations

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=None)

    mock_acquire = MagicMock()
    mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_acquire.__aexit__ = AsyncMock(return_value=False)

    mock_pool = MagicMock()
    mock_pool.acquire = MagicMock(return_value=mock_acquire)

    await apply_migrations(pool=mock_pool)

    # At least one SQL file must have been executed
    assert mock_conn.execute.call_count >= 1
    # The executed SQL should mention audit_log
    called_sql = mock_conn.execute.call_args[0][0]
    assert "audit_log" in called_sql


@pytest.mark.asyncio
async def test_apply_migrations_sql_file_exists():
    migrations_dir = Path(__file__).parent.parent / "db" / "migrations"
    sql_files = sorted(migrations_dir.glob("*.sql"))
    assert len(sql_files) >= 1, "Expected at least one migration file"
    content = sql_files[0].read_text()
    assert "CREATE TABLE" in content
    assert "audit_log_no_update" in content
    assert "audit_log_no_delete" in content
