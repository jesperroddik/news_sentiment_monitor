"""Topic modelling (LDA over the original Danish corpus).

``train_model`` fits a CountVectorizer (with Danish stop words) +
LatentDirichletAllocation on the raw Danish ``title``/``summary`` and persists
both to models/lda_model.pkl. ``assign_pending`` loads that model and writes the
dominant topic_id back to articles that don't have one yet. Retrain periodically
as the corpus grows.
"""

from __future__ import annotations

import functools
from pathlib import Path

import joblib
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sqlalchemy import text

import db

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = _PROJECT_ROOT / "models" / "lda_model.pkl"
# Optional override: one lowercase stop word per line ('#' for comments).
STOPWORDS_PATH = _PROJECT_ROOT / "data" / "danish_stopwords.txt"

# Number of topics. README suggests experimenting with k = 5..10.
DEFAULT_K = 7
_MIN_DOCS = 10  # don't bother fitting LDA on a near-empty corpus


@functools.lru_cache(maxsize=1)
def danish_stopwords() -> list[str]:
    """Danish stop words for the vectorizer.

    Prefers ``data/danish_stopwords.txt`` if present; otherwise falls back to
    NLTK's Danish list (downloading the corpus on first use).
    """
    if STOPWORDS_PATH.exists():
        words = [
            line.strip().lower()
            for line in STOPWORDS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if words:
            return sorted(set(words))

    import nltk

    try:
        from nltk.corpus import stopwords

        words = stopwords.words("danish")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords

        words = stopwords.words("danish")
    return sorted({w.lower() for w in words})


def _build_corpus_text(title: str | None, summary: str | None) -> str:
    return f"{title or ''} {summary or ''}".strip()


def _load_all_docs() -> list[dict]:
    engine = db.get_engine()
    with engine.connect() as conn:
        return conn.execute(
            text(
                """
                SELECT id, title, summary
                FROM articles
                ORDER BY id
                """
            )
        ).mappings().all()


def train_model(k: int = DEFAULT_K, save: bool = True):
    """Fit vectorizer + LDA on the full Danish corpus. Returns (vec, lda)."""
    rows = _load_all_docs()
    docs = [_build_corpus_text(r["title"], r["summary"]) for r in rows]
    if len(docs) < _MIN_DOCS:
        raise RuntimeError(
            f"Only {len(docs)} docs — need at least {_MIN_DOCS} to train LDA."
        )

    vec, lda = fit_lda(docs, k=k)
    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": vec, "lda": lda, "k": k}, MODEL_PATH)
        print(f"[topics] saved model (k={k}) to {MODEL_PATH}")
    return vec, lda


def fit_lda(docs: list[str], k: int = DEFAULT_K):
    """Pure fit helper (no DB) — usable in tests on an in-memory corpus."""
    vec = CountVectorizer(stop_words=danish_stopwords(), max_df=0.95, min_df=2)
    dtm = vec.fit_transform(docs)
    lda = LatentDirichletAllocation(
        n_components=k, learning_method="batch", random_state=42
    )
    lda.fit(dtm)
    return vec, lda


def top_terms(vec, lda, n: int = 10) -> dict[int, list[str]]:
    """Top-N terms per topic, for inspection."""
    features = vec.get_feature_names_out()
    return {
        topic_idx: [features[i] for i in comp.argsort()[: -n - 1 : -1]]
        for topic_idx, comp in enumerate(lda.components_)
    }


def _load_model():
    if not MODEL_PATH.exists():
        return None
    bundle = joblib.load(MODEL_PATH)
    return bundle["vectorizer"], bundle["lda"]


def topic_term_weights(n: int = 40) -> dict[int, dict[str, float]]:
    """Top-``n`` ``term -> weight`` per topic from the saved model.

    Used to render per-topic word clouds. Returns ``{}`` if no model exists.
    """
    model = _load_model()
    if model is None:
        return {}
    vec, lda = model
    features = vec.get_feature_names_out()
    return {
        int(topic_idx): {
            features[i]: float(comp[i]) for i in comp.argsort()[: -n - 1 : -1]
        }
        for topic_idx, comp in enumerate(lda.components_)
    }


def topic_labels(n: int = 3, sep: str = " · ") -> dict[int, str]:
    """Human-readable label per topic: its top-``n`` terms joined with ``sep``.

    Derived from the saved model's term weights, so labels regenerate
    automatically on every retrain — when a story takes a new turn, the next
    ``train_model`` shifts the top terms and the names follow. Returns ``{}``
    if no model exists yet.
    """
    return {
        tid: sep.join(list(freqs)[:n])
        for tid, freqs in topic_term_weights(n).items()
    }


def assign_pending() -> int:
    """Assign topic_id to articles missing one. Trains a model if none exists."""
    model = _load_model()
    if model is None:
        print("[topics] no saved model — training one first")
        train_model()
        model = _load_model()
    vec, lda = model

    engine = db.get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT id, title, summary
                FROM articles
                WHERE topic_id IS NULL
                ORDER BY id
                """
            )
        ).mappings().all()

    if not rows:
        print("[topics] nothing to assign")
        return 0

    docs = [_build_corpus_text(r["title"], r["summary"]) for r in rows]
    topic_dist = lda.transform(vec.transform(docs))
    dominant = topic_dist.argmax(axis=1)

    updates = [
        {"id": r["id"], "topic_id": int(dominant[i])} for i, r in enumerate(rows)
    ]
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE articles SET topic_id = :topic_id WHERE id = :id"),
            updates,
        )
    print(f"[topics] assigned topics to {len(updates)} articles")
    return len(updates)


if __name__ == "__main__":
    vec, lda = train_model()
    for tid, terms in top_terms(vec, lda).items():
        print(f"Topic {tid}: {', '.join(terms)}")
    assign_pending()
