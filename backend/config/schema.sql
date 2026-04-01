-- ─────────────────────────────────────────────────────────────────────────────
-- schema.sql — ML News Classifier Database
-- Run: psql -d mlnews -f backend/config/schema.sql
-- ─────────────────────────────────────────────────────────────────────────────

-- Articles
CREATE TABLE IF NOT EXISTS articles (
    id           SERIAL PRIMARY KEY,
    text         TEXT NOT NULL,
    source       VARCHAR(100),
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- Classification results
CREATE TABLE IF NOT EXISTS classifications (
    id             SERIAL PRIMARY KEY,
    article_id     INT REFERENCES articles(id) ON DELETE CASCADE,
    category       VARCHAR(20) NOT NULL,
    label          SMALLINT NOT NULL,
    confidence     FLOAT NOT NULL,
    model_version  VARCHAR(50) DEFAULT 'ensemble_v1',
    proba_world    FLOAT,
    proba_sports   FLOAT,
    proba_business FLOAT,
    proba_tech     FLOAT,
    latency_ms     INT,
    classified_at  TIMESTAMPTZ DEFAULT NOW()
);

-- User feedback (label corrections)
CREATE TABLE IF NOT EXISTS feedback (
    id                SERIAL PRIMARY KEY,
    classification_id INT REFERENCES classifications(id),
    correct_label     SMALLINT NOT NULL,
    note              TEXT,
    submitted_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Training runs (experiment tracking)
CREATE TABLE IF NOT EXISTS training_runs (
    id               SERIAL PRIMARY KEY,
    model_name       VARCHAR(50) NOT NULL,
    model_version    VARCHAR(50),
    val_accuracy     FLOAT,
    test_accuracy    FLOAT,
    test_f1          FLOAT,
    hyperparams      JSONB,
    artifact_path    TEXT,
    started_at       TIMESTAMPTZ,
    finished_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Daily accuracy metrics (for monitoring dashboard)
CREATE TABLE IF NOT EXISTS daily_metrics (
    id          SERIAL PRIMARY KEY,
    date        DATE NOT NULL,
    accuracy    FLOAT,
    total_preds INT,
    world_pct   FLOAT,
    sports_pct  FLOAT,
    biz_pct     FLOAT,
    tech_pct    FLOAT
);

-- Drift alerts
CREATE TABLE IF NOT EXISTS drift_alerts (
    id            SERIAL PRIMARY KEY,
    alert_type    VARCHAR(50),
    description   TEXT,
    severity      VARCHAR(20) DEFAULT 'warning',
    resolved      BOOLEAN DEFAULT FALSE,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Chat sessions
CREATE TABLE IF NOT EXISTS chat_sessions (
    id         SERIAL PRIMARY KEY,
    session_id UUID DEFAULT gen_random_uuid(),
    started_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id          SERIAL PRIMARY KEY,
    session_id  UUID NOT NULL,
    role        VARCHAR(10) NOT NULL,  -- 'user' | 'assistant'
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_class_created  ON classifications(classified_at);
CREATE INDEX IF NOT EXISTS idx_class_category ON classifications(category);
CREATE INDEX IF NOT EXISTS idx_feedback_class ON feedback(classification_id);
CREATE INDEX IF NOT EXISTS idx_chat_session   ON chat_messages(session_id);
