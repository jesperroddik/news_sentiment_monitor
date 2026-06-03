"""Topic modelling (NMF over a TF-IDF view of the original Danish corpus).

``train_model`` fits a TF-IDF vectorizer (lemmatized Danish tokens) + NMF on
the raw Danish ``title``/``summary`` and persists both to models/lda_model.pkl.
``assign_pending`` loads that model and writes the dominant topic_id back to
articles that don't have one yet. Retrain periodically as the corpus grows.

NMF over TF-IDF is used instead of LDA because it yields sharper, more coherent
topics on this small corpus of short Danish headlines.
"""

from __future__ import annotations

import functools
from pathlib import Path

import joblib
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import text

import db

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = _PROJECT_ROOT / "models" / "lda_model.pkl"
# Optional override: one lowercase stop word per line ('#' for comments).
STOPWORDS_PATH = _PROJECT_ROOT / "data" / "danish_stopwords.txt"

# Number of topics. Fewer topics stay more coherent on a small corpus.
DEFAULT_K = 5
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


@functools.lru_cache(maxsize=1)
def _lemmatized_stopwords() -> frozenset[str]:
    """Stop words plus their Danish lemmas, so filtering still matches after
    the tokenizer lemmatizes the corpus (e.g. ``regeringen`` -> ``regering``)."""
    import simplemma

    base = danish_stopwords()
    lemmas = {simplemma.lemmatize(w.lower(), lang="da").lower() for w in base}
    return frozenset(set(base) | lemmas)


def lemmatize_tokens(text_in: str) -> list[str]:
    """CountVectorizer tokenizer: split, lemmatize to Danish base forms,
    lowercase, and drop non-alphabetic tokens and (lemmatized) stop words.

    Collapses inflected forms (definite ``-en/-et``, genitive ``-s``, plurals)
    onto one term so LDA topics aren't fragmented across surface forms.
    """
    import simplemma

    stops = _lemmatized_stopwords()
    tokens: list[str] = []
    for tok in simplemma.simple_tokenizer(text_in):
        if not tok.isalpha():
            continue
        lemma = simplemma.lemmatize(tok.lower(), lang="da").lower()
        if len(lemma) > 1 and lemma not in stops:
            tokens.append(lemma)
    return tokens


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
    """Fit vectorizer + NMF on the full Danish corpus. Returns (vec, model)."""
    rows = _load_all_docs()
    docs = [_build_corpus_text(r["title"], r["summary"]) for r in rows]
    if len(docs) < _MIN_DOCS:
        raise RuntimeError(
            f"Only {len(docs)} docs — need at least {_MIN_DOCS} to train the model."
        )

    vec, model = fit_topics(docs, k=k)
    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": vec, "model": model, "k": k}, MODEL_PATH)
        print(f"[topics] saved model (k={k}) to {MODEL_PATH}")
    return vec, model


def fit_topics(docs: list[str], k: int = DEFAULT_K):
    """Pure fit helper (no DB) — usable in tests on an in-memory corpus.

    TF-IDF down-weights ubiquitous terms, and NMF factorises it into ``k``
    additive parts, giving tighter topics than LDA on short headlines. Stop
    words are filtered inside ``lemmatize_tokens`` (against lemmatized forms),
    so no separate ``stop_words`` list is passed here. ``min_df=3`` drops the
    long tail of words seen in only one or two articles.
    """
    vec = TfidfVectorizer(
        tokenizer=lemmatize_tokens,
        token_pattern=None,
        lowercase=False,
        max_df=0.9,
        min_df=3,
    )
    dtm = vec.fit_transform(docs)
    model = NMF(
        n_components=k, init="nndsvd", random_state=42, max_iter=400
    )
    model.fit(dtm)
    return vec, model


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
    return bundle["vectorizer"], bundle["model"]


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
