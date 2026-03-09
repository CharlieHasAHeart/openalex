BEGIN;

CREATE SCHEMA IF NOT EXISTS openalex;

-- Production table: keep in public, only ensure required final-result columns exist.
CREATE TABLE IF NOT EXISTS public.authors_avatars (
    author_id BIGINT PRIMARY KEY,
    oss_object_key TEXT NOT NULL,
    oss_url TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    commons_file TEXT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE public.authors_avatars ADD COLUMN IF NOT EXISTS oss_object_key TEXT;
ALTER TABLE public.authors_avatars ADD COLUMN IF NOT EXISTS oss_url TEXT;
ALTER TABLE public.authors_avatars ADD COLUMN IF NOT EXISTS content_sha256 TEXT;
ALTER TABLE public.authors_avatars ADD COLUMN IF NOT EXISTS commons_file TEXT;
ALTER TABLE public.authors_avatars ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'avatar_pipeline_runs_status_check'
          AND conrelid = 'openalex.avatar_pipeline_runs'::regclass
    ) THEN
        ALTER TABLE openalex.avatar_pipeline_runs
            ADD CONSTRAINT avatar_pipeline_runs_status_check
            CHECK (status IN ('running', 'success', 'partial_failed', 'failed'));
    END IF;
EXCEPTION WHEN undefined_table THEN
    NULL;
END $$;

CREATE TABLE IF NOT EXISTS openalex.avatar_pipeline_runs (
    run_id UUID PRIMARY KEY,
    trigger_type TEXT NOT NULL,
    status TEXT NOT NULL,
    config_snapshot JSONB NULL,
    operator TEXT NULL,
    notes TEXT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'avatar_pipeline_runs_status_check'
          AND conrelid = 'openalex.avatar_pipeline_runs'::regclass
    ) THEN
        ALTER TABLE openalex.avatar_pipeline_runs
            ADD CONSTRAINT avatar_pipeline_runs_status_check
            CHECK (status IN ('running', 'success', 'partial_failed', 'failed'));
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS openalex.avatar_pipeline_author_runs (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES openalex.avatar_pipeline_runs(run_id) ON DELETE CASCADE,
    author_id BIGINT NOT NULL,
    status TEXT NOT NULL,
    error_code TEXT NULL,
    error_message TEXT NULL,
    selected_candidate_id BIGINT NULL,
    rule_score DOUBLE PRECISION NULL,
    llm_score DOUBLE PRECISION NULL,
    final_score DOUBLE PRECISION NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'avatar_pipeline_author_runs_status_check'
          AND conrelid = 'openalex.avatar_pipeline_author_runs'::regclass
    ) THEN
        ALTER TABLE openalex.avatar_pipeline_author_runs
            ADD CONSTRAINT avatar_pipeline_author_runs_status_check
            CHECK (status IN ('ok', 'no_image', 'invalid_image', 'ambiguous', 'no_match', 'error'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_avatar_pipeline_author_runs_author_id_created_at
    ON openalex.avatar_pipeline_author_runs (author_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_avatar_pipeline_author_runs_run_id
    ON openalex.avatar_pipeline_author_runs (run_id);
CREATE INDEX IF NOT EXISTS idx_avatar_pipeline_author_runs_status
    ON openalex.avatar_pipeline_author_runs (status);

CREATE TABLE IF NOT EXISTS openalex.avatar_candidate_images (
    candidate_id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES openalex.avatar_pipeline_runs(run_id) ON DELETE CASCADE,
    author_id BIGINT NOT NULL,
    query_used TEXT NULL,
    source_page_url TEXT NULL,
    source_image_url TEXT NULL,
    source_domain TEXT NULL,
    page_title TEXT NULL,
    snippet TEXT NULL,
    nearby_text TEXT NULL,
    image_alt TEXT NULL,
    mime_type TEXT NULL,
    width INT NULL,
    height INT NULL,
    size_bytes BIGINT NULL,
    face_count INT NULL,
    is_portrait BOOLEAN NULL,
    is_valid_image BOOLEAN NULL,
    invalid_reason TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_avatar_candidate_images_author_id_created_at
    ON openalex.avatar_candidate_images (author_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_avatar_candidate_images_run_id
    ON openalex.avatar_candidate_images (run_id);
CREATE INDEX IF NOT EXISTS idx_avatar_candidate_images_source_domain
    ON openalex.avatar_candidate_images (source_domain);

CREATE TABLE IF NOT EXISTS openalex.avatar_candidate_decisions (
    id BIGSERIAL PRIMARY KEY,
    candidate_id BIGINT NOT NULL REFERENCES openalex.avatar_candidate_images(candidate_id) ON DELETE CASCADE,
    run_id UUID NOT NULL REFERENCES openalex.avatar_pipeline_runs(run_id) ON DELETE CASCADE,
    author_id BIGINT NOT NULL,
    decision TEXT NOT NULL,
    rule_score DOUBLE PRECISION NULL,
    llm_score DOUBLE PRECISION NULL,
    final_score DOUBLE PRECISION NULL,
    decision_reason TEXT NULL,
    evidence JSONB NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'avatar_candidate_decisions_decision_check'
          AND conrelid = 'openalex.avatar_candidate_decisions'::regclass
    ) THEN
        ALTER TABLE openalex.avatar_candidate_decisions
            ADD CONSTRAINT avatar_candidate_decisions_decision_check
            CHECK (decision IN ('match', 'non_match', 'uncertain'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_avatar_candidate_decisions_author_id_created_at
    ON openalex.avatar_candidate_decisions (author_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_avatar_candidate_decisions_candidate_id
    ON openalex.avatar_candidate_decisions (candidate_id);
CREATE INDEX IF NOT EXISTS idx_avatar_candidate_decisions_run_id
    ON openalex.avatar_candidate_decisions (run_id);
CREATE INDEX IF NOT EXISTS idx_avatar_candidate_decisions_decision
    ON openalex.avatar_candidate_decisions (decision);

COMMIT;
