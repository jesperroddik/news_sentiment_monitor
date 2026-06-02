"""Sentiment scoring (VADER).

VADER is English-only, so this stage runs on the translated columns. ``score_text``
is a pure function (no DB) so it can be unit-tested directly.
"""

from __future__ import annotations

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sqlalchemy import text

import db

# Standard VADER thresholds for labelling the compound score.
_POS_THRESHOLD = 0.05
_NEG_THRESHOLD = -0.05

_analyzer: SentimentIntensityAnalyzer | None = None


def _get_analyzer() -> SentimentIntensityAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def label_for(compound: float) -> str:
    if compound >= _POS_THRESHOLD:
        return "positive"
    if compound <= _NEG_THRESHOLD:
        return "negative"
    return "neutral"


def score_text(text_in: str) -> tuple[float, str]:
    """Return (compound score in -1..+1, label) for an English string."""
    compound = _get_analyzer().polarity_scores(text_in or "")["compound"]
    return compound, label_for(compound)


def score_pending() -> int:
    """Score translated-but-unscored articles. Returns rows scored."""
    engine = db.get_engine()

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT id, translated_title, translated_summary
                FROM articles
                WHERE sentiment_score IS NULL
                  AND translated_title IS NOT NULL
                ORDER BY id
                """
            )
        ).mappings().all()

    if not rows:
        print("[sentiment] nothing to score")
        return 0

    updates = []
    for r in rows:
        combined = f"{r['translated_title']} {r['translated_summary'] or ''}".strip()
        score, label = score_text(combined)
        updates.append({"id": r["id"], "score": score, "label": label})

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE articles
                SET sentiment_score = :score, sentiment_label = :label
                WHERE id = :id
                """
            ),
            updates,
        )
    print(f"[sentiment] scored {len(updates)} articles")
    return len(updates)


if __name__ == "__main__":
    score_pending()
