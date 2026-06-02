"""RSS collection + one-off pipeline entry point.

Running ``python src/fetch.py`` performs the full one-off pipeline:
    fetch -> translate -> sentiment -> topic assign -> store.

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
RSS_FEEDS: dict[str, str] = {
    "DR": "https://www.dr.dk/nyheder/service/feeds/allenyheder",
    "TV2": "https://feeds.tv2.dk/news/rss",
    "Berlingske": "https://www.berlingske.dk/rss",
    "Politiken": "https://politiken.dk/rss/senestenyt.rss",
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
        if parsed.bozo:
            print(f"[warn] {source}: feed parse issue: {parsed.bozo_exception}")
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


def fetch_newsapi() -> list[dict]:
    """Optional NewsAPI.org source. Not wired up yet (RSS-only for now)."""
    # TODO: collect via NewsAPI.org (language='da'), respecting 100 req/day.
    return []


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
    """Full one-off pipeline: fetch -> translate -> sentiment -> topic assign."""
    # Imported lazily so each stage can be developed/run in isolation.
    import sentiment
    import topics
    import translate

    inserted = store_articles(fetch_feeds())
    if inserted == 0:
        print("[pipeline] no new articles; running NLP on any pending rows anyway")

    translate.translate_pending()
    sentiment.score_pending()
    topics.assign_pending()
    print("[pipeline] done")


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
