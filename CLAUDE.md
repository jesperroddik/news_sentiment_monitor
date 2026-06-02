# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

This is a **Danish News Sentiment Monitor** — an end-to-end NLP pipeline collecting headlines from Danish news RSS feeds (DR, Politiken, Information), scoring sentiment and modelling topics (LDA) **directly on the original Danish text**, and presenting results in a Streamlit dashboard. There is no translation step — all NLP runs on Danish. The README describes the full spec.

## Intended Repository Layout

```
src/
  db.py          # SQLAlchemy/psycopg2 connection to Neon PostgreSQL
  fetch.py       # feedparser RSS collector + start-up health check; entry point for one-off fetch
  sentiment.py   # Danish transformer sentiment (-1 to +1) on raw Danish + label
  topics.py      # sklearn LDA on raw Danish (Danish stop words); per-article topic assignment; saves .pkl
  scheduler.py   # APScheduler job running full pipeline every 2 hours
sql/
  schema.sql     # articles table definition (sentiment_*, topic_id)
  analysis_queries.sql
notebooks/
  01_exploration.ipynb
app.py           # Streamlit dashboard
.env             # DATABASE_URL (see .env.example)
requirements.txt
```

## Environment Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in DATABASE_URL
psql $DATABASE_URL -f sql/schema.sql
```

## Common Commands

| Task | Command |
|---|---|
| One-off fetch + process | `python src/fetch.py` |
| Launch dashboard | `streamlit run app.py` |
| Run scheduler (every 2 h) | `python src/scheduler.py` |

## Architecture

The pipeline runs in this order: **fetch → sentiment → topic assign → store**.

- `db.py` owns the SQLAlchemy engine/session; all other modules import from it.
- `fetch.py` deduplicates on URL before inserting to avoid re-processing. `check_sources()` runs first and aborts if every feed is unreachable; a feed returning 0 entries is logged loudly as an error rather than treated as a quiet news day.
- `sentiment.py` scores the **original Danish** `title`/`summary` with a fine-tuned Danish transformer (`MODEL_NAME` in the module). Maps `argmax` to the label and `P(pos)-P(neg)` to the -1..+1 score. Model downloads from Hugging Face on first run (~0.4 GB, cached under `~/.cache/huggingface`).
- `topics.py` trains LDA on the **raw Danish** `title`/`summary`, filtering Danish stop words (`danish_stopwords()` — NLTK's Danish list, overridable by `data/danish_stopwords.txt`); k=5–10 topics. Persists the fitted model as a `.pkl` via `joblib` and writes `topic_id` back to each article row. Topic terms/word clouds are therefore in Danish.
- `app.py` reads directly from Neon; no local state. Expects the pipeline columns (`sentiment_score`, `sentiment_label`, `topic_id`) to be populated before launch; the UI is in Danish and shows per-topic word clouds (no English translation displayed).

## Key Constraints

- **All NLP runs on Danish**: both sentiment (Danish transformer) and LDA (Danish stop words) operate on the raw Danish `title`/`summary`. There is no translation step or DeepL dependency — `translate.py` and the `translated_*` columns were removed.
- **Neon connection**: Use `sslmode=require` in `DATABASE_URL`. Neon auto-suspends on inactivity; handle reconnect in `db.py`.
- **Sources**: RSS-only (DR, Politiken, Information), no API key required. TV2 and Berlingske dropped their public feeds and Kristeligt Dagblad publishes none; NewsAPI was evaluated as a fallback but its free tier has near-zero Danish coverage, so it is not used. `NEWS_API_KEY` may exist in `.env` but is currently unused.
