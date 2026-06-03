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

# RSS feeds (free, no key required). source label -> list of feed URLs.
# A single source may expose more than one feed (e.g. Jyllands-Posten's
# top-stories and latest-news); store_articles() dedups on URL, so any overlap
# between a source's feeds is harmless. TV2's public feed (feeds.tv2.dk) no
# longer resolves, so it remains unavailable. Berlingske's legacy
# berlingske.dk/rss was dead, but its next-api "alle" feed works (it declares
# us-ascii yet serves utf-8, so feedparser flags bozo — benign, the entries
# parse fine). Kristeligt Dagblad's /rss/nyheder feed delivers HTML-escaped
# summaries, which clean_text() unescapes-then-strips.
RSS_FEEDS: dict[str, list[str]] = {
    "DR": ["https://www.dr.dk/nyheder/service/feeds/allenyheder"],
    "Politiken": ["https://politiken.dk/rss/senestenyt.rss"],
    "Information": ["https://www.information.dk/feed"],
    "Jyllands-Posten": [
        "https://newsletter-proxy.aws.jyllands-posten.dk/v1/top-stories/jyllands-posten.dk",
        "https://newsletter-proxy.aws.jyllands-posten.dk/v1/latestNewsRss/jyllands-posten.dk?count=10",
    ],
    "Berlingske": ["https://www.berlingske.dk/next-api/feeds/alle"],
    "Kristeligt Dagblad": ["https://www.kristeligt-dagblad.dk/rss/nyheder"],
}

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_text(raw: str | None) -> str:
    """Unescape entities, strip HTML tags, and normalise whitespace.

    Unescape happens *before* tag stripping so that feeds delivering
    HTML-escaped markup (e.g. Kristeligt Dagblad sends ``&lt;p&gt;`` in its
    summaries) get their tags removed too, instead of leaking ``<p>``/``dir``
    /``ltr`` noise into the NLP once unescaped.
    """
    if not raw:
        return ""
    no_tags = _TAG_RE.sub(" ", unescape(raw))
    return _WS_RE.sub(" ", no_tags).strip()


def _parse_published(entry) -> datetime | None:
    """Best-effort parse of an entry's publication time to an aware datetime."""
    struct = getattr(entry, "published_parsed", None) or getattr(
        entry, "updated_parsed", None
    )
    if struct is None:
        return None
    return datetime.fromtimestamp(mktime(struct), tz=timezone.utc)


def fetch_feeds(feeds: dict[str, list[str]] = RSS_FEEDS) -> list[dict]:
    """Parse all feeds and return a list of cleaned article dicts."""
    articles: list[dict] = []
    for source, urls in feeds.items():
        for url in urls:
            parsed = feedparser.parse(url)
            if not parsed.entries:
                # Zero entries means a dead host/path (DNS failure, connection
                # reset, moved feed), not a quiet news day — surface it loudly
                # so it isn't mistaken for normal operation.
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
            print(f"[fetch] {source}: {len(parsed.entries)} entries <- {url}")
    return articles


def check_sources() -> dict[str, bool]:
    """Ping every configured feed and report which are reachable.

    Run at start-up so a dead feed surfaces immediately rather than silently
    contributing zero articles. Returns ``{feed_url: healthy?}``. Raises
    ``RuntimeError`` if *every* feed is down — there is nothing to do.
    """
    status: dict[str, bool] = {}
    print("[startup] checking sources...")

    for source, urls in RSS_FEEDS.items():
        for url in urls:
            parsed = feedparser.parse(url)
            ok = bool(parsed.entries)
            status[url] = ok
            if ok:
                print(f"  [ok]   {source}: {len(parsed.entries)} entries")
            else:
                reason = getattr(parsed, "bozo_exception", None) or "no entries"
                print(f"  [DEAD] {source}: {reason} ({url})")

    healthy = sum(status.values())
    print(f"[startup] {healthy}/{len(status)} feeds healthy")
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
