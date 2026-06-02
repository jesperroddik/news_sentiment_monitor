"""Sentiment scoring with a Danish transformer (replaces VADER).

Scores the *original Danish* ``title``/``summary`` directly — no translation
step is required. A fine-tuned Danish sentiment model produces 3-class polarity;
we map ``argmax`` to the ``positive``/``neutral``/``negative`` label and use
``P(positive) - P(negative)`` as the ``sentiment_score`` (still in -1..+1, so the
existing schema and dashboard are unchanged).

The model (~0.4 GB) is downloaded from the Hugging Face hub on first use and
cached under ``~/.cache/huggingface``. Heavy imports (torch/transformers) are
deferred to first scoring call so importing this module stays cheap.
"""

from __future__ import annotations

import functools

from sqlalchemy import text

import db

# Fine-tuned Danish 3-class sentiment model. Reads original Danish text.
MODEL_NAME = "alexandrainst/da-sentiment-base"

_MAX_TOKENS = 512
_BATCH = 16


@functools.lru_cache(maxsize=1)
def _load():
    """Load tokenizer + model once and resolve the pos/neg label indices.

    Returns ``(tokenizer, model, torch, pos_idx, neg_idx)``. The positive and
    negative class indices are read from ``model.config.id2label`` so the code
    works across Danish models whose labels are spelled 'positiv'/'positive'
    etc. Raises if the labels can't be identified (e.g. generic 'LABEL_0').
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    pos_idx = neg_idx = None
    for idx, label in model.config.id2label.items():
        s = str(label).lower()
        if "pos" in s:
            pos_idx = int(idx)
        elif "neg" in s:
            neg_idx = int(idx)
    if pos_idx is None or neg_idx is None:
        raise RuntimeError(
            f"Could not infer positive/negative classes from {MODEL_NAME} "
            f"(id2label={model.config.id2label}). Set the mapping manually."
        )
    return tokenizer, model, torch, pos_idx, neg_idx


def score_texts(texts: list[str]) -> list[tuple[float, str]]:
    """Score a batch of Danish strings. Returns [(score in -1..+1, label), ...]."""
    if not texts:
        return []
    tokenizer, model, torch, pos_idx, neg_idx = _load()

    results: list[tuple[float, str]] = []
    with torch.no_grad():
        for start in range(0, len(texts), _BATCH):
            chunk = [t or "" for t in texts[start : start + _BATCH]]
            enc = tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=_MAX_TOKENS,
                padding=True,
            )
            probs = torch.softmax(model(**enc).logits, dim=-1)
            for row in probs:
                vals = row.tolist()
                score = float(vals[pos_idx] - vals[neg_idx])
                winner = int(row.argmax())
                if winner == pos_idx:
                    label = "positive"
                elif winner == neg_idx:
                    label = "negative"
                else:
                    label = "neutral"
                results.append((score, label))
    return results


def score_text(text_in: str) -> tuple[float, str]:
    """Return (score in -1..+1, label) for a single Danish string."""
    return score_texts([text_in])[0]


def score_pending() -> int:
    """Score unscored articles from their raw Danish text. Returns rows scored."""
    engine = db.get_engine()

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT id, title, summary
                FROM articles
                WHERE sentiment_score IS NULL
                ORDER BY id
                """
            )
        ).mappings().all()

    if not rows:
        print("[sentiment] nothing to score")
        return 0

    docs = [f"{r['title']} {r['summary'] or ''}".strip() for r in rows]
    scored = score_texts(docs)
    updates = [
        {"id": r["id"], "score": score, "label": label}
        for r, (score, label) in zip(rows, scored)
    ]

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
    print(f"[sentiment] scored {len(updates)} articles ({MODEL_NAME})")
    return len(updates)


if __name__ == "__main__":
    score_pending()
