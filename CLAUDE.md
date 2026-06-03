# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

This is a **Danish News Sentiment Monitor** — an end-to-end NLP pipeline collecting headlines from Danish news RSS feeds (DR, Politiken, Information, Jyllands-Posten, Berlingske, Kristeligt Dagblad), scoring sentiment and modelling topics (NMF over TF-IDF, on lemmatized text) **directly on the original Danish text**, and presenting results in a Streamlit dashboard. There is no translation step — all NLP runs on Danish. The README describes the full spec.

## Intended Repository Layout

```
src/
  db.py          # SQLAlchemy/psycopg2 connection to Neon PostgreSQL
  fetch.py       # feedparser RSS collector + start-up health check; entry point for one-off fetch
  sentiment.py   # Danish transformer sentiment (-1 to +1) on raw Danish + label
  topics.py      # sklearn NMF over TF-IDF on lemmatized Danish (simplemma + stop words); per-article topic assignment; saves .pkl
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
- `topics.py` fits **NMF over a TF-IDF** view of the **Danish** `title`/`summary`. A custom tokenizer (`lemmatize_tokens`) lemmatizes to Danish base forms via `simplemma` (so `regeringen`/`regeringens` collapse onto `regering`) and filters stop words (`danish_stopwords()` — the union of NLTK's Danish list and an extended list in `data/danish_stopwords.txt`, which **overrides** the NLTK fallback when present); `DEFAULT_K=5`, `min_df=3`. NMF over TF-IDF gives sharper topics than LDA on this small corpus of short headlines. Persists the fitted vectorizer+model as a `.pkl` via `joblib` (bundle keys `vectorizer`/`model`/`k`) and writes `topic_id` back to each article row. `topic_labels()` derives a name per topic from its top terms, regenerated on every retrain. The saved model references `topics.lemmatize_tokens`, so `simplemma` is required wherever the model is loaded **and applied** (`assign_pending`, the scheduler). Topic terms/word clouds/names are therefore in Danish.
- `app.py` reads directly from Neon; no local state. Expects the pipeline columns (`sentiment_score`, `sentiment_label`, `topic_id`) to be populated before launch; the UI is in Danish and shows per-topic word clouds, top-term topic names, a topic-distribution bar chart, a source×topic mean-sentiment heatmap, and a filterable article table (no English translation displayed). Editing `topics.py`/`db.py` requires a Streamlit restart (already-imported modules are cached in `sys.modules`); editing `app.py` itself hot-reloads.

## Key Constraints

- **All NLP runs on Danish**: both sentiment (Danish transformer) and topic modelling (NMF over TF-IDF on lemmatized Danish text, Danish stop words) operate on the Danish `title`/`summary`. There is no translation step or DeepL dependency — `translate.py` and the `translated_*` columns were removed.
- **Neon connection**: Use `sslmode=require` in `DATABASE_URL`. Neon auto-suspends on inactivity; handle reconnect in `db.py`.
- **Sources**: RSS-only (DR, Politiken, Information, Jyllands-Posten, Berlingske, Kristeligt Dagblad), no API key required. `RSS_FEEDS` maps each source to a *list* of feed URLs, since Jyllands-Posten exposes two (top-stories + latest-news); `store_articles()` dedups on URL. Berlingske's `next-api` "alle" feed declares us-ascii but serves utf-8, so feedparser flags `bozo` — benign, the entries parse fine. Kristeligt Dagblad's `/rss/nyheder` feed sends HTML-escaped summaries, so `clean_text()` unescapes **before** stripping tags (otherwise `<p>`/`dir`/`ltr` leak into the NLP). TV2's public feed no longer resolves; NewsAPI was evaluated as a fallback but its free tier has near-zero Danish coverage, so it is not used. `NEWS_API_KEY` may exist in `.env` but is currently unused.
