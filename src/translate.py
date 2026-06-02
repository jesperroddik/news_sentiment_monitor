"""Translation layer (DeepL free tier).

Translates Danish article text to English *before* any NLP runs. Only rows
that have not yet been translated are processed, and a usage guard keeps the
pipeline inside the 500,000 chars/month free quota.
"""

from __future__ import annotations

import os

import deepl
from sqlalchemy import text

import db

# Stop translating once monthly usage crosses this fraction of the free limit,
# leaving headroom so a single batch can't push us over.
_USAGE_CEILING = 0.95
# How many articles to pull/translate per round-trip.
_BATCH_SIZE = 50

_translator: deepl.Translator | None = None


def get_translator() -> deepl.Translator:
    global _translator
    if _translator is None:
        key = os.environ.get("DEEPL_API_KEY")
        if not key:
            raise RuntimeError("DEEPL_API_KEY is not set (see .env.example).")
        _translator = deepl.Translator(key)
    return _translator


def _quota_remaining(translator: deepl.Translator) -> bool:
    """True if character usage is below the configured ceiling."""
    usage = translator.get_usage()
    if usage.character.valid and usage.character.limit:
        used, limit = usage.character.count, usage.character.limit
        print(f"[translate] DeepL usage: {used:,}/{limit:,} chars")
        return used < limit * _USAGE_CEILING
    return True


def translate_pending(batch_size: int = _BATCH_SIZE) -> int:
    """Translate untranslated articles DA->EN. Returns rows translated."""
    translator = get_translator()
    engine = db.get_engine()
    total = 0

    while True:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT id, title, summary
                    FROM articles
                    WHERE translated_title IS NULL
                    ORDER BY id
                    LIMIT :limit
                    """
                ),
                {"limit": batch_size},
            ).mappings().all()

        if not rows:
            break
        if not _quota_remaining(translator):
            print("[translate] quota ceiling reached — stopping.")
            break

        # One DeepL call for all titles, one for all summaries in the batch.
        titles = [r["title"] or "" for r in rows]
        summaries = [r["summary"] or "" for r in rows]

        translated_titles = translator.translate_text(
            titles, source_lang="DA", target_lang="EN-US"
        )
        translated_summaries = translator.translate_text(
            summaries, source_lang="DA", target_lang="EN-US"
        )

        updates = [
            {
                "id": r["id"],
                "tt": translated_titles[i].text,
                "ts": translated_summaries[i].text,
            }
            for i, r in enumerate(rows)
        ]
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE articles
                    SET translated_title = :tt, translated_summary = :ts
                    WHERE id = :id
                    """
                ),
                updates,
            )
        total += len(rows)
        print(f"[translate] translated {len(rows)} (running total {total})")

    print(f"[translate] done — {total} articles translated")
    return total


if __name__ == "__main__":
    translate_pending()
