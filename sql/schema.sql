-- Danish News Sentiment Monitor — database schema
-- Run with: psql $DATABASE_URL -f sql/schema.sql
--
-- One row per news article. Columns are populated in pipeline order:
--   fetch     -> source, title, summary, url, published_at, fetched_at
--   sentiment -> sentiment_score, sentiment_label
--   topics    -> topic_id

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

    -- LDA topic assignment (dominant topic for this article).
    topic_id            INTEGER,

    fetched_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Dashboard and pipeline access patterns.
CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles (published_at);
CREATE INDEX IF NOT EXISTS idx_articles_source       ON articles (source);
CREATE INDEX IF NOT EXISTS idx_articles_topic_id     ON articles (topic_id);

-- Partial index to speed up the "find unscored work" query in the sentiment stage.
CREATE INDEX IF NOT EXISTS idx_articles_unscored
    ON articles (id) WHERE sentiment_score IS NULL;
