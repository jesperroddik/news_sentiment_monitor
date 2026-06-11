"""Build the "Nyhedsoverblik" snapshot the dashboard renders at the top.

For each of the top-6 most prevalent IPTC topics, find the 3 most *important*
stories and attach a one-sentence Danish digest to each. Importance is inferred
from **multi-source coverage**: we have no click/engagement data, so a story that
several outlets ran is treated as a top story. Within a topic we cluster
near-duplicate articles across sources by embedding cosine similarity (reusing
the sentence-transformer ``iptc`` already loads — no second model), then rank
clusters by how many distinct outlets covered them, breaking ties by recency.
Singleton stories therefore fall back to a recency ordering.

The result is written as a single JSON blob to ``news_overview`` so the dashboard
needs neither the transformer nor Ollama at render time. This runs as the last
pipeline stage (fetch -> store -> sentiment -> IPTC -> overview); like the
scrapers it logs and degrades rather than crashing the run.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

import db
import digest as digest_mod

FALLBACK_LABEL = "Øvrige"          # below-confidence IPTC bucket, excluded here
_WINDOW_HOURS = 72
_TOP_K = 6                          # topics shown
_PER_TOPIC = 3                      # stories per topic
_CLUSTER_DISTANCE = 0.35           # cosine distance; <=0.35 ~ "same story"
_KEEP_SNAPSHOTS = 10               # prune old news_overview rows beyond this


def _effective_time(row: dict) -> datetime:
    """published_at if present, else fetched_at — for recency ordering."""
    return row.get("published_at") or row.get("fetched_at") or datetime.min.replace(
        tzinfo=timezone.utc
    )


def _cluster_labels(texts: list[str]) -> list[int]:
    """Group similar texts; returns a cluster label per text.

    One label per text when there is nothing to cluster (0 or 1 text). Reuses
    the IPTC sentence-transformer so no extra model is loaded.
    """
    if len(texts) <= 1:
        return [0] * len(texts)

    import iptc
    from sklearn.cluster import AgglomerativeClustering

    emb = iptc._model().encode(texts, convert_to_numpy=True)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=_CLUSTER_DISTANCE,
        metric="cosine",
        linkage="average",
    )
    return clustering.fit_predict(emb).tolist()


def _representative(cluster: list[dict]) -> dict:
    """Pick the cluster member with the most informative summary (longest
    non-empty), breaking ties toward the outlet that published earliest."""
    return max(
        cluster,
        key=lambda r: (len((r.get("summary") or "").strip()),
                       -_effective_time(r).timestamp()),
    )


def _story_payload(cluster: list[dict]) -> dict:
    """Assemble one story dict (representative + the outlets covering it).

    Generates and caches the LLM digest on the representative's article row when
    it doesn't have one yet.
    """
    rep = _representative(cluster)
    sources = sorted({r["source"] for r in cluster})

    digest = rep.get("digest")
    if not digest:
        digest, from_llm = digest_mod.digest_text(rep["title"], rep.get("summary"))
        # Only cache genuine LLM digests; a teaser fallback (Ollama down) stays
        # un-cached so the next run retries the LLM for this story.
        if from_llm:
            with db.get_engine().begin() as conn:
                conn.execute(
                    text("UPDATE articles SET digest = :d WHERE id = :id"),
                    {"d": digest, "id": rep["id"]},
                )

    published = _effective_time(rep)
    return {
        "article_id": rep["id"],
        "title": rep["title"],
        "url": rep["url"],
        "source": rep["source"],
        "sources": sources,
        "n_sources": len(sources),
        "published_at": published.isoformat(),
        "sentiment_score": rep.get("sentiment_score"),
        "sentiment_label": rep.get("sentiment_label"),
        "digest": digest,
    }


def _topic_stories(rows: list[dict]) -> list[dict]:
    """Cluster a topic's articles, rank by coverage then recency, return top-N."""
    cluster_texts = [
        f"{r['title']}. {(r.get('summary') or '')[:160]}".strip() for r in rows
    ]
    labels = _cluster_labels(cluster_texts)

    clusters: dict[int, list[dict]] = {}
    for row, label in zip(rows, labels):
        clusters.setdefault(label, []).append(row)

    ranked = sorted(
        clusters.values(),
        key=lambda c: (
            len({r["source"] for r in c}),                       # coverage
            max(_effective_time(r) for r in c).timestamp(),      # recency
        ),
        reverse=True,
    )
    return [_story_payload(c) for c in ranked[:_PER_TOPIC]]


def build_overview(window_hours: int = _WINDOW_HOURS, top_k: int = _TOP_K) -> int:
    """Build the overview snapshot and store it. Returns the story count."""
    engine = db.get_engine()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)

    with engine.connect() as conn:
        rows = [
            dict(r)
            for r in conn.execute(
                text(
                    """
                    SELECT id, source, title, summary, url, published_at,
                           fetched_at, sentiment_score, sentiment_label,
                           iptc_category, digest
                    FROM articles
                    WHERE iptc_category IS NOT NULL
                      AND iptc_category <> :fallback
                      AND COALESCE(published_at, fetched_at) >= :cutoff
                    """
                ),
                {"fallback": FALLBACK_LABEL, "cutoff": cutoff},
            ).mappings()
        ]

    if not rows:
        print(f"[overview] no classified articles in the last {window_hours}h; "
              "snapshot not updated")
        return 0

    # Most prevalent categories first.
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["iptc_category"]] = counts.get(r["iptc_category"], 0) + 1
    top_cats = sorted(counts, key=counts.get, reverse=True)[:top_k]

    topics = []
    total_stories = 0
    for cat in top_cats:
        cat_rows = [r for r in rows if r["iptc_category"] == cat]
        stories = _topic_stories(cat_rows)
        topics.append({"category": cat, "stories": stories})
        total_stories += len(stories)
        print(f"[overview] {cat}: {len(stories)} stories from {len(cat_rows)} articles")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_hours": window_hours,
        "topics": topics,
    }

    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO news_overview (payload) VALUES (CAST(:p AS JSONB))"),
            {"p": json.dumps(payload, ensure_ascii=False)},
        )
        # Keep only the most recent snapshots.
        conn.execute(
            text(
                """
                DELETE FROM news_overview
                WHERE id NOT IN (
                    SELECT id FROM news_overview
                    ORDER BY generated_at DESC LIMIT :keep
                )
                """
            ),
            {"keep": _KEEP_SNAPSHOTS},
        )

    print(f"[overview] built snapshot: {len(topics)} topics, {total_stories} stories")
    return total_stories


if __name__ == "__main__":
    build_overview()
