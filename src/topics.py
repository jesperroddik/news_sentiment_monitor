"""Danish text preprocessing — a lemmatizing tokenizer + stop-word list.

Originally this module fitted an NMF-over-TF-IDF topic model, but the dashboard
now categorizes articles with the IPTC taxonomy (see ``iptc.py``), so the topic
model was removed. What survives is the shared Danish tokenizer: ``app.py`` reuses
``lemmatize_tokens`` to build the per-IPTC-category word clouds, lemmatizing
inflected forms onto one base term (e.g. ``regeringen``/``regeringens`` ->
``regering``) and dropping Danish stop words.
"""

from __future__ import annotations

import functools
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Optional override: one lowercase stop word per line ('#' for comments).
STOPWORDS_PATH = _PROJECT_ROOT / "data" / "danish_stopwords.txt"


@functools.lru_cache(maxsize=1)
def danish_stopwords() -> list[str]:
    """Danish stop words for the tokenizer.

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
    """Tokenizer for ``TfidfVectorizer``: split, lemmatize to Danish base forms,
    lowercase, and drop non-alphabetic tokens and (lemmatized) stop words.

    Collapses inflected forms (definite ``-en/-et``, genitive ``-s``, plurals)
    onto one term so the word-cloud terms aren't fragmented across surface forms.
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
