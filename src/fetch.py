"""RSS collection + one-off pipeline entry point.

Running ``python src/fetch.py`` performs the full one-off pipeline:
    fetch -> sentiment -> topic assign -> store.

``run_pipeline()`` is reused by ``scheduler.py`` for the periodic job.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from html import unescape
from time import mktime

import feedparser
from sqlalchemy import text

import db

# RSS feeds (free, no key required). source label -> feed URL.
# TV2 and Berlingske dropped their public RSS feeds (feeds.tv2.dk no longer
# resolves; berlingske.dk/rss resets the connection) and are not indexed by
# NewsAPI's free tier either, so they are currently unavailable. Kristeligt
# Dagblad publishes no discoverable feed. DR, Politiken, and Information all
# serve working RSS.
RSS_FEEDS: dict[str, str] = {
    "DR": "https://www.dr.dk/nyheder/service/feeds/allenyheder",
    "Politiken": "https://politiken.dk/rss/senestenyt.rss",
    "Information": "https://www.information.dk/feed",
}

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_text(raw: str | None) -> str:
    """Strip HTML tags, unescape entities, and normalise whitespace."""
    if not raw:
        return ""
    no_tags = _TAG_RE.sub(" ", raw)
    return _WS_RE.sub(" ", unescape(no_tags)).strip()


def _parse_published(entry) -> datetime | None:
    """Best-effort parse of an entry's publication time to an aware datetime."""
    struct = getattr(entry, "published_parsed", None) or getattr(
        entry, "updated_parsed", None
    )
    if struct is None:
        return None
    return datetime.fromtimestamp(mktime(struct), tz=timezone.utc)


def fetch_feeds(feeds: dict[str, str] = RSS_FEEDS) -> list[dict]:
    """Parse all feeds and return a list of cleaned article dicts."""
    articles: list[dict] = []
    for source, url in feeds.items():
        parsed = feedparser.parse(url)
        if not parsed.entries:
            # Zero entries means a dead host/path (DNS failure, connection
            # reset, moved feed), not a quiet news day — surface it loudly so
            # it isn't mistaken for normal operation.
            reason = getattr(parsed, "bozo_exception", None) or "no entries returned"
            print(f"[error] {source}: feed returned 0 entries ({reason}); "
                  f"contributed nothing — check URL: {url}")
            continue
        if parsed.bozo:
            print(f"[warn] {source}: feed parsed with issues: {parsed.bozo_exception}")
        for entry in parsed.entries:
            link = getattr(entry, "link", None)
            title = clean_text(getattr(entry, "title", None))
            if not link or not title:
                continue
            articles.append(
                {
                    "source": source,
                    "title": title,
                    "summary": clean_text(getattr(entry, "summary", None)),
                    "url": link,
                    "published_at": _parse_published(entry),
                }
            )
        print(f"[fetch] {source}: {len(parsed.entries)} entries")
    return articles


def check_sources() -> dict[str, bool]:
    """Ping every configured feed and report which are reachable.

    Run at start-up so a dead feed surfaces immediately rather than silently
    contributing zero articles. Returns ``{source: healthy?}``. Raises
    ``RuntimeError`` if *every* feed is down — there is nothing to do.
    """
    status: dict[str, bool] = {}
    print("[startup] checking sources...")

    for source, url in RSS_FEEDS.items():
        parsed = feedparser.parse(url)
        ok = bool(parsed.entries)
        status[source] = ok
        if ok:
            print(f"  [ok]   {source}: {len(parsed.entries)} entries")
        else:
            reason = getattr(parsed, "bozo_exception", None) or "no entries"
            print(f"  [DEAD] {source}: {reason} ({url})")

    healthy = sum(status.values())
    print(f"[startup] {healthy}/{len(status)} sources healthy")
    if healthy == 0:
        raise RuntimeError("No healthy news sources — aborting pipeline.")
    return status


def store_articles(articles: list[dict]) -> int:
    """Insert new articles, deduplicating on URL. Returns rows inserted."""
    if not articles:
        return 0

    engine = db.get_engine()
    incoming = {a["url"]: a for a in articles}  # dedup within this batch too

    with engine.begin() as conn:
        existing = conn.execute(
            text("SELECT url FROM articles WHERE url = ANY(:urls)"),
            {"urls": list(incoming.keys())},
        ).scalars()
        for url in existing:
            incoming.pop(url, None)

        new_rows = list(incoming.values())
        if new_rows:
            conn.execute(
                text(
                    """
                    INSERT INTO articles (source, title, summary, url, published_at)
                    VALUES (:source, :title, :summary, :url, :published_at)
                    ON CONFLICT (url) DO NOTHING
                    """
                ),
                new_rows,
            )
    print(f"[store] inserted {len(new_rows)} new articles "
          f"({len(articles) - len(new_rows)} duplicates skipped)")
    return len(new_rows)


def run_pipeline() -> None:
    """Full one-off pipeline: fetch -> sentiment -> topic assign."""
    # Imported lazily so each stage can be developed/run in isolation.
    import sentiment
    import topics

    check_sources()
    inserted = store_articles(fetch_feeds())
    if inserted == 0:
        print("[pipeline] no new articles; running NLP on any pending rows anyway")

    sentiment.score_pending()
    topics.assign_pending()
    print("[pipeline] done")


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
