"""One-sentence Danish news digest written by a local LLM (Ollama).

The dashboard's "Nyhedsoverblik" banner shows, per topic, a single meaningful
Danish sentence describing each top story. We only collect each article's
``title`` + publisher ``summary`` (a teaser), not the full body, so a small
local model rewrites that into one crisp, uniform Danish sentence.

Everything stays local: Ollama runs as a service on ``localhost:11434`` — no
external API, consistent with the rest of the pipeline (sentiment + IPTC are
local Hugging Face models). The model is swappable via ``OLLAMA_MODEL``
(default ``gemma2:2b`` — small and CPU-friendly with decent Danish).

Resilience mirrors ``scrape.py``: if Ollama isn't installed/running, we log a
loud ``[digest]`` warning and fall back to the teaser's first sentence rather
than crashing the pipeline. The ``ollama`` import is deferred so importing this
module stays cheap and works even when the package isn't installed.
"""

from __future__ import annotations

import os
import re

MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")

# Keep the prompt input bounded — the teaser is short anyway, and a tight cap
# keeps CPU generation fast.
_MAX_INPUT_CHARS = 600

_PROMPT = (
    "Du er nyhedsredaktør. Skriv ÉN kort, neutral dansk sætning (højst 30 ord) "
    "der opsummerer nyheden nedenfor. Brug kun oplysningerne i overskrift og "
    "resumé — find ikke på noget. Svar KUN med selve sætningen, uden "
    "anførselstegn eller indledning.\n\n"
    "Overskrift: {title}\nResumé: {summary}"
)

# True once we've already logged that Ollama is unreachable, so a full run with
# many representatives doesn't spam the same warning ~18 times.
_warned = False


def _first_sentence(text: str) -> str:
    """First sentence of *text* (split on . ! ? followed by space/end)."""
    text = text.strip()
    if not text:
        return ""
    m = re.search(r"[.!?](?:\s|$)", text)
    return text[: m.end()].strip() if m else text


def _teaser_fallback(title: str, summary: str | None) -> str:
    """Best non-LLM digest: the teaser's first sentence, else the title."""
    sentence = _first_sentence(summary or "")
    return sentence or title.strip()


def _clean_one_sentence(raw: str, title: str, summary: str | None) -> str:
    """Normalise the model output to a single clean sentence.

    Strips wrapping quotes / stray leading labels and keeps the first sentence.
    Empty/degenerate output falls back to the teaser.
    """
    out = raw.strip().strip('"').strip("'").strip()
    # Some small models prefix "Sætning:" / "Digest:" etc — drop a short label.
    out = re.sub(r"^\s*[\wæøåÆØÅ ]{0,20}:\s*", "", out, count=1)
    out = _first_sentence(out) or out
    return out if len(out) >= 10 else _teaser_fallback(title, summary)


def digest_text(title: str, summary: str | None) -> tuple[str, bool]:
    """Return ``(sentence, from_llm)`` — a one-sentence Danish digest for a story.

    Uses Ollama if reachable; otherwise logs once and falls back to the teaser.
    ``from_llm`` lets the caller cache only genuine LLM digests, so a run while
    Ollama is down doesn't poison the cache with teaser text that would never be
    upgraded once Ollama comes back.
    """
    global _warned
    title = (title or "").strip()
    summary = (summary or "").strip()
    prompt = _PROMPT.format(
        title=title[:_MAX_INPUT_CHARS], summary=summary[:_MAX_INPUT_CHARS]
    )

    try:
        import ollama

        resp = ollama.generate(
            model=MODEL,
            prompt=prompt,
            options={"temperature": 0.2, "num_predict": 80},
        )
        return _clean_one_sentence(resp.get("response", ""), title, summary), True
    except Exception as exc:  # not installed, server down, model missing, ...
        if not _warned:
            print(f"[digest] Ollama unavailable ({exc!r}); falling back to teaser "
                  f"sentences. Install Ollama and `ollama pull {MODEL}` for "
                  f"LLM-written digests.")
            _warned = True
        return _teaser_fallback(title, summary), False
