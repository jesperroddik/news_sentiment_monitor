-- Danish News Sentiment Monitor — database schema
-- Run with: psql $DATABASE_URL -f sql/schema.sql
--
-- One row per news article. Columns are populated in pipeline order:
--   fetch     -> source, title, summary, url, published_at, fetched_at
--   sentiment -> sentiment_score, sentiment_label
--   iptc      -> iptc_category, iptc_score (fixed IPTC Media Topics taxonomy;
--                drives every dashboard view, including the word clouds)

CREATE TABLE IF NOT EXISTS articles (
    id                  SERIAL PRIMARY KEY,

    -- Raw fields from the RSS feed (original Danish text).
    source              TEXT        NOT NULL,            -- e.g. 'DR', 'TV2'
    title               TEXT        NOT NULL,            -- original Danish headline
    summary             TEXT,                            -- original Danish summary
    url                 TEXT        NOT NULL UNIQUE,     -- dedup key
    published_at        TIMESTAMPTZ,                     -- article publication time

    -- Sentiment from a Danish transformer, scored on the original Danish text.
    sentiment_score     REAL,                            -- P(pos)-P(neg), -1.0 .. +1.0
    sentiment_label     TEXT,                            -- 'positive' | 'neutral' | 'negative'

    -- IPTC Media Topics top-level category (embedding similarity), with confidence.
    iptc_category       TEXT,                            -- e.g. 'Politik', 'Sport'
    iptc_score          REAL,                            -- best cosine similarity, 0..1

    -- One-sentence Danish digest written by a local LLM (Ollama). Filled lazily,
    -- only for articles that become a "top story" representative in the overview,
    -- and reused across runs (a recurring top story isn't re-summarized).
    digest              TEXT,

    fetched_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Idempotent migrations for databases created under an older schema.
ALTER TABLE articles ADD COLUMN IF NOT EXISTS iptc_category TEXT;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS iptc_score    REAL;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS digest        TEXT;
-- The NMF topic model was removed in favour of IPTC categories; drop its column.
ALTER TABLE articles DROP COLUMN IF EXISTS topic_id;

-- Precomputed "Nyhedsoverblik" snapshot rendered at the top of the dashboard.
-- Each pipeline run inserts one row: the top-6 IPTC topics, each with its 3 most
-- important (multi-source) stories and their LLM digests, as a single JSON blob.
-- The dashboard reads the most recent row, so it needs no model at render time.
CREATE TABLE IF NOT EXISTS news_overview (
    id           SERIAL PRIMARY KEY,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    payload      JSONB       NOT NULL
);

-- Dashboard and pipeline access patterns.
CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles (published_at);
CREATE INDEX IF NOT EXISTS idx_articles_source       ON articles (source);
CREATE INDEX IF NOT EXISTS idx_articles_iptc_category ON articles (iptc_category);

-- Partial indexes to speed up the "find pending work" queries in the NLP stages.
CREATE INDEX IF NOT EXISTS idx_articles_unscored
    ON articles (id) WHERE sentiment_score IS NULL;
CREATE INDEX IF NOT EXISTS idx_articles_unclassified
    ON articles (id) WHERE iptc_category IS NULL;

-- The dashboard reads the latest overview snapshot (ORDER BY generated_at DESC).
CREATE INDEX IF NOT EXISTS idx_news_overview_generated_at
    ON news_overview (generated_at DESC);
