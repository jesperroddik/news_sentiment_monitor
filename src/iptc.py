"""IPTC category classification (embedding similarity, on the original Danish text).

Assigns each article one **IPTC Media Topics top-level category** (Politik,
Økonomi, Sport, ...) — a fixed, human-curated taxonomy — which is the dashboard's
topic axis. No training data of our own: each article and each candidate category
phrase is embedded with a multilingual sentence-transformer, and the article gets
the category with the highest cosine similarity.

This replaced an earlier zero-shot-NLI approach, which needed one transformer
forward pass *per candidate label* (17×) and was too slow/unstable on CPU.
Embedding similarity is one forward pass per article — ~17× cheaper — and stable.

``classify_pending`` mirrors ``sentiment.score_pending``: it loads the model once
and writes ``iptc_category`` (+ confidence ``iptc_score``) back to articles that
don't have one yet. Heavy imports (torch/sentence-transformers)
are deferred to the first call so importing this module stays cheap.
"""

from __future__ import annotations

import functools

from sqlalchemy import text

import db

# Small multilingual sentence-transformer (~120 MB) with solid Danish support;
# fast on CPU. Downloads from the Hugging Face hub on first use and is cached.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# IPTC Media Topics top-level subjects. Maps a short Danish *display name*
# (stored in the DB and shown in the dashboard) to the natural-language Danish
# *phrase* that is embedded and matched against — richer phrases embed better
# than single keywords.
IPTC_LABELS: dict[str, str] = {
    "Politik": "politik, regering og valg",
    "Økonomi": "økonomi, erhverv og finans",
    "Kriminalitet": "kriminalitet, politi og retsvæsen",
    "Krig & konflikt": "krig, militær og væbnet konflikt",
    "Ulykker & katastrofer": "ulykker, naturkatastrofer og redningsindsats",
    "Sundhed": "sundhed, sygdom og sundhedsvæsen",
    "Videnskab & teknologi": "videnskab, forskning og teknologi",
    "Miljø & klima": "miljø, klima og natur",
    "Uddannelse": "uddannelse, skole og forskning",
    "Arbejdsmarked": "arbejdsmarked, job og fagforeninger",
    "Sport": "sport, fodbold og atletik",
    "Kultur & medier": "kultur, kunst, musik og medier",
    "Religion": "religion, tro og kirke",
    "Samfund": "samfund, sociale forhold og menneskerettigheder",
    "Livsstil & fritid": "livsstil, mad, rejser og fritid",
    "Menneskelig interesse": "menneskelige historier og berømtheder",
    "Vejr": "vejr og vejrudsigt",
}

# Below this best-match cosine similarity the assignment is too weak to trust.
# Kept low because a news headline and a keyword-phrase aren't paraphrases, so
# even a correct match scores modestly (~0.2-0.35) with this model; the floor is
# only here to bucket genuinely off-topic text, not to second-guess the argmax.
CONFIDENCE_FLOOR = 0.15
FALLBACK_LABEL = "Øvrige"

_CHUNK = 64     # texts per encode + per DB commit (commit cadence = resumability)
_MAX_CHARS = 500  # premise cap; the model truncates ~128 tokens anyway


@functools.lru_cache(maxsize=1)
def _model():
    """Load the sentence-transformer once (lazy heavy imports)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


@functools.lru_cache(maxsize=1)
def _label_space():
    """Return ``(display_names, label_embeddings)`` with normalized embeddings,
    computed once so every batch reuses them."""
    model = _model()
    names = list(IPTC_LABELS)
    phrases = [IPTC_LABELS[n] for n in names]
    emb = model.encode(phrases, normalize_embeddings=True, convert_to_numpy=True)
    return names, emb


def classify_texts(texts: list[str]) -> list[tuple[str, float]]:
    """Classify a batch of Danish strings into IPTC categories.

    Returns ``[(category, score), ...]``. ``category`` is a display name from
    ``IPTC_LABELS`` (or ``FALLBACK_LABEL`` when the best cosine similarity is
    below ``CONFIDENCE_FLOOR``); ``score`` is that best cosine similarity.
    """
    if not texts:
        return []
    model = _model()
    names, label_emb = _label_space()
    premises = [(t or "")[:_MAX_CHARS] for t in texts]

    emb = model.encode(premises, normalize_embeddings=True, convert_to_numpy=True)
    sims = emb @ label_emb.T            # cosine sim (both sides normalized)
    best = sims.argmax(axis=1)

    results: list[tuple[str, float]] = []
    for row, j in zip(sims, best):
        score = float(row[j])
        if score < CONFIDENCE_FLOOR:
            results.append((FALLBACK_LABEL, score))
        else:
            results.append((names[j], score))
    return results


def classify_text(text_in: str) -> tuple[str, float]:
    """Return (category, score) for a single Danish string."""
    return classify_texts([text_in])[0]


def classify_pending() -> int:
    """Classify articles missing an IPTC category. Returns rows classified."""
    engine = db.get_engine()

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT id, title, summary
                FROM articles
                WHERE iptc_category IS NULL
                ORDER BY id
                """
            )
        ).mappings().all()

    if not rows:
        print("[iptc] nothing to classify")
        return 0

    update_sql = text(
        """
        UPDATE articles
        SET iptc_category = :category, iptc_score = :score
        WHERE id = :id
        """
    )

    # Classify and commit in chunks so a crash mid-run loses at most one chunk;
    # re-running resumes (only NULL iptc_category rows are selected).
    done = 0
    for start in range(0, len(rows), _CHUNK):
        batch = rows[start : start + _CHUNK]
        docs = [f"{r['title']} {r['summary'] or ''}".strip() for r in batch]
        labelled = classify_texts(docs)
        updates = [
            {"id": r["id"], "category": cat, "score": score}
            for r, (cat, score) in zip(batch, labelled)
        ]
        with engine.begin() as conn:
            conn.execute(update_sql, updates)
        done += len(updates)
        print(f"[iptc] classified {done}/{len(rows)}")

    print(f"[iptc] classified {done} articles ({MODEL_NAME})")
    return done


if __name__ == "__main__":
    classify_pending()
