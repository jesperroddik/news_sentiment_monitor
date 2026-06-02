# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

This is a **Danish News Sentiment Monitor** — an end-to-end NLP pipeline collecting headlines from Danish news RSS feeds (DR, Politiken, Information), translating them via DeepL, scoring sentiment with VADER, modelling topics with LDA, and presenting results in a Streamlit dashboard. The README describes the full spec.

## Intended Repository Layout

```
src/
  db.py          # SQLAlchemy/psycopg2 connection to Neon PostgreSQL
  fetch.py       # feedparser RSS collector + start-up health check; entry point for one-off fetch
  translate.py   # DeepL free-tier translation (500k chars/month budget)
  sentiment.py   # VADER compound scoring (-1 to +1) + label assignment
  topics.py      # sklearn LDA training and per-article topic assignment; saves .pkl
  scheduler.py   # APScheduler job running full pipeline every 2 hours
sql/
  schema.sql     # articles table definition (includes translated_title, sentiment_*, topic_id)
  analysis_queries.sql
notebooks/
  01_exploration.ipynb
app.py           # Streamlit dashboard
.env             # DATABASE_URL, DEEPL_API_KEY (see .env.example)
requirements.txt
```

## Environment Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in DATABASE_URL, DEEPL_API_KEY
psql $DATABASE_URL -f sql/schema.sql
```

## Common Commands

| Task | Command |
|---|---|
| One-off fetch + process | `python src/fetch.py` |
| Launch dashboard | `streamlit run app.py` |
| Run scheduler (every 2 h) | `python src/scheduler.py` |

## Architecture

The pipeline runs in this order: **fetch → translate → sentiment → topic assign → store**.

- `db.py` owns the SQLAlchemy engine/session; all other modules import from it.
- `fetch.py` deduplicates on URL before inserting to avoid re-processing. `check_sources()` runs first and aborts if every feed is unreachable; a feed returning 0 entries is logged loudly as an error rather than treated as a quiet news day.
- `translate.py` batches requests and respects the 500k chars/month DeepL free quota; translate *before* any NLP.
- `sentiment.py` scores the English-translated text (VADER is English-only).
- `topics.py` trains LDA on the full translated corpus; k=5–10 topics. Persists the fitted model as a `.pkl` via `joblib` and writes `topic_id` back to each article row.
- `app.py` reads directly from Neon; no local state. Expects all pipeline columns (`translated_title`, `sentiment_score`, `sentiment_label`, `topic_id`) to be populated before launch.

## Key Constraints

- **Translation-first NLP**: VADER and LDA operate on English text only. Never run them on raw Danish text.
- **DeepL quota**: 500k characters/month on the free tier. Batch calls and skip already-translated rows.
- **Neon connection**: Use `sslmode=require` in `DATABASE_URL`. Neon auto-suspends on inactivity; handle reconnect in `db.py`.
- **Sources**: RSS-only (DR, Politiken, Information), no API key required. TV2 and Berlingske dropped their public feeds and Kristeligt Dagblad publishes none; NewsAPI was evaluated as a fallback but its free tier has near-zero Danish coverage, so it is not used. `NEWS_API_KEY` may exist in `.env` but is currently unused.
