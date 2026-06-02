-- Danish News Sentiment Monitor — analysis queries
-- These answer the "Key Questions" from the README and back the dashboard.

-- 1. Which topics are dominating over the past week?
SELECT topic_id,
       COUNT(*)                  AS article_count,
       ROUND(AVG(sentiment_score)::numeric, 3) AS avg_sentiment
FROM articles
WHERE published_at >= now() - INTERVAL '7 days'
  AND topic_id IS NOT NULL
GROUP BY topic_id
ORDER BY article_count DESC;

-- 2. Which sources publish the most negative / most positive content?
SELECT source,
       COUNT(*)                                 AS article_count,
       ROUND(AVG(sentiment_score)::numeric, 3)  AS avg_sentiment,
       SUM((sentiment_label = 'negative')::int) AS negative,
       SUM((sentiment_label = 'positive')::int) AS positive
FROM articles
WHERE sentiment_score IS NOT NULL
GROUP BY source
ORDER BY avg_sentiment ASC;   -- most negative first

-- 3. Per-topic sentiment by source — does coverage diverge or align?
SELECT topic_id,
       source,
       COUNT(*)                                 AS article_count,
       ROUND(AVG(sentiment_score)::numeric, 3)  AS avg_sentiment
FROM articles
WHERE topic_id IS NOT NULL AND sentiment_score IS NOT NULL
GROUP BY topic_id, source
ORDER BY topic_id, avg_sentiment;

-- 4. Topic spikes today, and whether they are driven by one outlet.
SELECT topic_id,
       COUNT(*)                       AS articles_today,
       COUNT(DISTINCT source)         AS distinct_sources
FROM articles
WHERE published_at >= date_trunc('day', now())
  AND topic_id IS NOT NULL
GROUP BY topic_id
ORDER BY articles_today DESC;

-- 5. Sentiment distribution per topic (share of pos/neutral/neg).
SELECT topic_id,
       sentiment_label,
       COUNT(*)                                                AS n,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY topic_id), 1) AS pct
FROM articles
WHERE topic_id IS NOT NULL AND sentiment_label IS NOT NULL
GROUP BY topic_id, sentiment_label
ORDER BY topic_id, sentiment_label;

-- 6. Daily sentiment trend per source (drives the dashboard line chart).
SELECT date_trunc('day', published_at) AS day,
       source,
       ROUND(AVG(sentiment_score)::numeric, 3) AS avg_sentiment,
       COUNT(*)                                AS article_count
FROM articles
WHERE sentiment_score IS NOT NULL
GROUP BY day, source
ORDER BY day, source;

-- ---------------------------------------------------------------------------
-- Storage maintenance — keep the database inside the Neon free-tier 0.5 GB cap.
-- ---------------------------------------------------------------------------

-- 7. Storage footprint of the articles table.
--    total_size = table + indexes + TOAST; compare against the 0.5 GB budget.
--    Run occasionally to track real growth vs. the ~1.2 KB/row estimate.
SELECT pg_size_pretty(pg_total_relation_size('articles')) AS total_size,
       pg_size_pretty(pg_relation_size('articles'))       AS table_size,
       pg_size_pretty(pg_indexes_size('articles'))         AS indexes_size,
       COUNT(*)                                            AS row_count,
       pg_size_pretty(
           pg_total_relation_size('articles') / NULLIF(COUNT(*), 0)
       )                                                   AS avg_row_size
FROM articles;

-- 8. Retention sweep — delete articles older than 18 months.
--    Bounds storage permanently; schedule this from scheduler.py (e.g. daily).
--    Adjust the interval to trade history depth against storage headroom.
DELETE FROM articles
WHERE published_at < now() - INTERVAL '18 months';

-- 9. Reclaim space after a large retention DELETE.
--    Plain DELETE leaves dead tuples that autovacuum reclaims lazily; on Neon
--    those still count toward the storage quota until vacuumed. Run this to
--    force reclamation after a big purge. (VACUUM cannot run inside a
--    transaction block, so execute it on its own.)
VACUUM (ANALYZE) articles;
