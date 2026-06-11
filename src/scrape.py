"""HTML scrapers for Danish sources without a usable RSS feed.

DR's `/nyheder` and TV2's `/live/kort-nyt` are both server-rendered with their
content embedded as structured JSON, so we parse that JSON rather than the
rendered DOM — robust and with no headless browser. Each scraper returns the
same article dict shape as ``fetch.fetch_feeds()`` (``source``, ``title``,
``summary``, ``url``, ``published_at``) so the rest of the pipeline is unchanged.

  - DR:  ``__NEXT_DATA__`` (Next.js) carries article items with a real ``summary``
         (the teaser the DR *RSS* feed omits) and a ``urlPathId`` that rebuilds
         the same URL the RSS feed used, so it dedups against existing rows.
  - TV2: JSON-LD ``LiveBlogPosting`` carries each "kort nyt" update as a
         ``BlogPosting`` with ``headline``/``articleBody``/``datePublished`` and a
         unique ``#entry=<uuid>`` URL for dedup.

Both shapes are framework-specific, so a structural change upstream will break a
scraper; each one logs a loud ``[error]`` and returns ``[]`` rather than raising,
and ``check_sources`` validates the JSON marker is present before a run.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import requests

from fetch import clean_text

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
_TIMEOUT = 20

# source label -> (page URL, substring that must be present for the scraper to work)
SCRAPE_SOURCES: dict[str, tuple[str, str]] = {
    "DR": ("https://www.dr.dk/nyheder", "__NEXT_DATA__"),
    "TV2": ("https://nyheder.tv2.dk/live/kort-nyt", "application/ld+json"),
}
_DR_BASE = "https://www.dr.dk"


def _get(url: str) -> str:
    resp = requests.get(url, headers={"User-Agent": _UA}, timeout=_TIMEOUT)
    resp.raise_for_status()
    # Both pages are UTF-8 but omit the charset in their Content-Type header, so
    # requests falls back to its ISO-8859-1 default and mojibakes Danish letters
    # (æøå) that appear as literal UTF-8 bytes — notably in the titles. Force UTF-8.
    resp.encoding = "utf-8"
    return resp.text


def _parse_iso(value: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp (with 'Z' or offset) to an aware UTC datetime."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def scrape_dr() -> list[dict]:
    """Scrape DR /nyheder via its ``__NEXT_DATA__`` blob. Returns article dicts."""
    html = _get(SCRAPE_SOURCES["DR"][0])
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    if not m:
        print("[error] DR: __NEXT_DATA__ not found — page structure changed; scraped nothing")
        return []
    data = json.loads(m.group(1))

    out: list[dict] = []
    seen: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            path = node.get("urlPathId")
            if (
                isinstance(path, str) and path.startswith("/")
                and node.get("title") and isinstance(node.get("summary"), str)
            ):
                url = _DR_BASE + path
                if url not in seen:
                    seen.add(url)
                    out.append(
                        {
                            "source": "DR",
                            "title": clean_text(node.get("title")),
                            "summary": clean_text(node.get("summary")),
                            "url": url,
                            "published_at": _parse_iso(node.get("startDate")),
                        }
                    )
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(data)
    print(f"[scrape] DR: {len(out)} articles <- {SCRAPE_SOURCES['DR'][0]}")
    return out


def scrape_tv2() -> list[dict]:
    """Scrape TV2 /live/kort-nyt via its JSON-LD LiveBlogPosting. Returns article dicts."""
    html = _get(SCRAPE_SOURCES["TV2"][0])
    blocks = re.findall(
        r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', html, re.S
    )
    if not blocks:
        print("[error] TV2: no JSON-LD found — page structure changed; scraped nothing")
        return []

    out: list[dict] = []
    seen: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            if (
                node.get("@type") in ("BlogPosting", "NewsArticle")
                and node.get("headline") and node.get("articleBody")
            ):
                url = node.get("url")
                if isinstance(url, str) and url not in seen:
                    seen.add(url)
                    out.append(
                        {
                            "source": "TV2",
                            "title": clean_text(node.get("headline")),
                            "summary": clean_text(node.get("articleBody")),
                            "url": url,
                            "published_at": _parse_iso(node.get("datePublished")),
                        }
                    )
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    for block in blocks:
        try:
            walk(json.loads(block))
        except json.JSONDecodeError:
            continue
    print(f"[scrape] TV2: {len(out)} articles <- {SCRAPE_SOURCES['TV2'][0]}")
    return out


_SCRAPERS = {"DR": scrape_dr, "TV2": scrape_tv2}


def scrape_all() -> list[dict]:
    """Run every scraper, isolating failures so one bad page can't sink the rest."""
    articles: list[dict] = []
    for name, fn in _SCRAPERS.items():
        try:
            articles += fn()
        except Exception as exc:  # network error, JSON error, structure change
            print(f"[error] {name} scrape failed: {exc!r}")
    return articles


def check_sources() -> dict[str, bool]:
    """Probe each scrape page for its expected JSON marker (cheap structural check).

    Returns ``{source: healthy?}``. A page that 200s but no longer contains its
    marker is reported unhealthy — that is exactly the upstream-redesign case the
    scrapers can't survive.
    """
    status: dict[str, bool] = {}
    for name, (url, marker) in SCRAPE_SOURCES.items():
        try:
            ok = marker in _get(url)
        except Exception as exc:
            ok = False
            print(f"  [DEAD] {name} (scrape): {exc!r} ({url})")
        else:
            print(f"  [{'ok' if ok else 'DEAD'}]   {name} (scrape): "
                  f"{'marker found' if ok else 'marker missing — structure changed'}")
        status[name] = ok
    return status
