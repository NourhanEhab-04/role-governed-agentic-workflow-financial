-- db/migrations/001_create_audit_log.sql
--
-- EU AI Act Article 13 / MiFID II Article 25 compliant audit log.
-- Requirements: PostgreSQL 13+ (gen_random_uuid() is built-in from PG 13).
-- Retention requirement: 7 years — enforced at the infrastructure/backup level.
--
-- IMMUTABILITY: a trigger function raises an exception on any UPDATE or DELETE.
-- The application DB user should additionally be granted only INSERT + SELECT
-- so that the trigger is a second, independent line of defence.

CREATE TABLE IF NOT EXISTS audit_log (
    -- Surrogate primary key for internal ordering; never exposed externally.
    id                  BIGSERIAL       PRIMARY KEY,

    -- Schema version allows non-breaking column additions in future migrations.
    schema_version      INTEGER         NOT NULL DEFAULT 1,

    -- Stable UUID that uniquely identifies this suitability assessment end-to-end.
    assessment_id       UUID            NOT NULL DEFAULT gen_random_uuid(),

    -- Wall-clock time at which the audit record was committed.
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    -- Both the specialist-agent model and the verifier-model must be recorded
    -- so the exact inference stack that produced each output is reproducible.
    model_version       JSONB           NOT NULL,

    -- Verbatim, unmodified input texts as received from the caller.
    client_text         TEXT            NOT NULL,
    product_text        TEXT            NOT NULL,

    -- Full serialised pipeline state: all intermediate outputs, every validation
    -- result, every retry with original + corrected output + verifier feedback.
    state_snapshot      JSONB           NOT NULL,

    -- Top-level fields duplicated for fast querying without JSON path extraction.
    final_decision      TEXT,
    regulatory_basis    TEXT,

    -- Human review: populated post-pipeline when a reviewer approves/rejects.
    -- NULL means no human review was performed (automated assessment only).
    human_reviewer_id   TEXT,
    human_reviewed_at   TIMESTAMPTZ
);

-- ── Append-only enforcement ─────────────────────────────────────────────────
-- The trigger fires BEFORE UPDATE or DELETE and raises an exception,
-- preventing any modification regardless of the calling user's privileges.

CREATE OR REPLACE FUNCTION _audit_log_immutable()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION
        'audit_log is append-only: % is prohibited (EU AI Act compliance)',
        TG_OP;
END;
$$;

CREATE OR REPLACE TRIGGER audit_log_no_update
    BEFORE UPDATE ON audit_log
    FOR EACH ROW EXECUTE FUNCTION _audit_log_immutable();

CREATE OR REPLACE TRIGGER audit_log_no_delete
    BEFORE DELETE ON audit_log
    FOR EACH ROW EXECUTE FUNCTION _audit_log_immutable();

-- ── Indexes for compliance officer queries ──────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_audit_log_created_at
    ON audit_log (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_audit_log_assessment_id
    ON audit_log (assessment_id);

CREATE INDEX IF NOT EXISTS idx_audit_log_final_decision
    ON audit_log (final_decision);
