-- db/schema.sql
-- E-CIP v3.0 — Full PostgreSQL Schema
-- Auto-executed by Docker on first postgres container start.

-- ─── EXTENSIONS ────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ─── API KEYS ──────────────────────────────────────────────────
-- v3 Fix #14: Keys bcrypt-hashed, never plain text.
CREATE TABLE IF NOT EXISTS api_keys (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT        NOT NULL,
    key_hash    TEXT        NOT NULL UNIQUE,
    active      BOOL        NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used   TIMESTAMPTZ
);

-- ─── PREDICTION LOGS ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS prediction_logs (
    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id    TEXT        NOT NULL UNIQUE,
    module        TEXT        NOT NULL,
    model_version TEXT        NOT NULL,
    input_hash    TEXT        NOT NULL,
    prediction    JSONB       NOT NULL,
    latency_ms    INTEGER,
    api_key_id    UUID        REFERENCES api_keys(id),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes: module+time for dashboard queries, GIN for JSONB paths.
CREATE INDEX IF NOT EXISTS idx_pred_module_created
    ON prediction_logs (module, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_pred_gin
    ON prediction_logs USING GIN (prediction);

CREATE INDEX IF NOT EXISTS idx_pred_risk_band
    ON prediction_logs ((prediction->>'risk_band'))
    WHERE module = 'retention';

CREATE INDEX IF NOT EXISTS idx_pred_confidence
    ON prediction_logs ((prediction->>'confidence'))
    WHERE module = 'product';

-- ─── REVIEW QUEUE ──────────────────────────────────────────────
-- v3 Fix #38: Low-confidence and OOD predictions routed here.
CREATE TABLE IF NOT EXISTS review_queue (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id  TEXT        NOT NULL,
    module      TEXT        NOT NULL,
    trigger     TEXT        NOT NULL,   -- 'low_confidence' | 'ood_flagged' | 'drift_alert'
    payload     JSONB       NOT NULL,
    status      TEXT        NOT NULL DEFAULT 'pending',  -- 'pending' | 'resolved' | 'dismissed'
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reviewed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_review_status
    ON review_queue (status, created_at DESC);

-- ─── DRIFT EVENTS ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS drift_events (
    id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    module            TEXT        NOT NULL,
    feature_name      TEXT,                  -- v3: specific feature that drifted
    metric            TEXT        NOT NULL,  -- 'psi' | 'ks'
    metric_value      FLOAT       NOT NULL,
    threshold         FLOAT       NOT NULL,
    alert_triggered   BOOL        NOT NULL,
    reference_version TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_drift_module_created
    ON drift_events (module, created_at DESC);

-- ─── IMAGE STORAGE ─────────────────────────────────────────────
-- v3 Fix #9: Tracks uploaded product images and Grad-CAM outputs.
CREATE TABLE IF NOT EXISTS image_storage (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id  TEXT        NOT NULL UNIQUE,
    file_type   TEXT        NOT NULL,  -- 'upload' | 'gradcam'
    file_path   TEXT        NOT NULL,
    expires_at  TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_image_expires
    ON image_storage (expires_at)
    WHERE expires_at IS NOT NULL;