# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

This is a **Danish News Sentiment Monitor** — an end-to-end NLP pipeline collecting headlines from Danish news sources (Politiken, Information, Jyllands-Posten, Berlingske and Kristeligt Dagblad via RSS; DR and TV2 via HTML scrapers), scoring sentiment and classifying each article into a fixed **IPTC Media Topics** category (embedding similarity) — all **directly on the original Danish text** — and presenting results in a Streamlit dashboard. There is no translation step — all NLP runs on Danish. The README describes the full spec.

The dashboard is **entirely IPTC-based**: the **IPTC category** (`iptc_category`) drives every view, including the word clouds (TF-IDF over each category's Danish text). An earlier NMF-over-TF-IDF topic model was removed; `topics.py` now contains only the shared Danish lemmatizing tokenizer (`lemmatize_tokens`), which the IPTC word clouds reuse. There is no `topic_id` column or `.pkl` model anymore.

## Intended Repository Layout

```
src/
  db.py          # SQLAlchemy/psycopg2 connection to Neon PostgreSQL
  fetch.py       # feedparser RSS collector + scrape orchestration + start-up health check; entry point for one-off fetch
  scrape.py      # HTML scrapers for DR (/nyheder __NEXT_DATA__) and TV2 (/live/kort-nyt JSON-LD); same article dict shape as RSS
  sentiment.py   # Danish transformer sentiment (-1 to +1) on raw Danish + label
  topics.py      # shared Danish text preprocessing: lemmatizing tokenizer (simplemma) + stop words; reused by the IPTC word clouds
  iptc.py        # IPTC Media Topics category on raw Danish via embedding similarity (sentence-transformer + cosine); per-article iptc_category + confidence
  digest.py      # one-sentence Danish digest of a story via a local LLM (Ollama); falls back to the publisher teaser when Ollama is unavailable
  overview.py    # builds the dashboard's "Nyhedsoverblik" snapshot: top-6 topics x 3 most multi-source-covered stories + their digests, stored in news_overview
  scheduler.py   # APScheduler job running full pipeline every 2 hours
sql/
  schema.sql     # articles table (sentiment_*, iptc_*, digest) + news_overview snapshot table
  analysis_queries.sql
scripts/
  scheduled_run.py # Windows Task Scheduler launcher: runs the pipeline once under pythonw, logs to logs/pipeline.log, toasts on failure
app.py           # Streamlit dashboard
.streamlit/
  config.toml    # pins dark theme (word clouds need a dark bg) + disables usage stats
.env             # DATABASE_URL (see .env.example)
requirements.txt
```

## Environment Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in DATABASE_URL
psql $DATABASE_URL -f sql/schema.sql
```

The digest stage (`digest.py`/`overview.py`) needs a local **Ollama** install — install the Ollama app (auto-starts a `localhost:11434` service) and pull a model: `ollama pull gemma2:2b` (override with `OLLAMA_MODEL`). It's optional: without it the pipeline still runs and the overview falls back to publisher-teaser sentences.

## Common Commands

| Task | Command |
|---|---|
| One-off fetch + process | `python src/fetch.py` |
| Launch dashboard | `streamlit run app.py` |
| Run scheduler (every 2 h) | `python src/scheduler.py` |
| Scheduled one-shot (Windows task) | `pythonw scripts/scheduled_run.py` |

On this Windows machine, prefer `python -m streamlit run app.py` (the bare `streamlit` launcher is flaky here), and use PowerShell equivalents for the setup steps (`Copy-Item .env.example .env`; pipe `schema.sql` to `psql`).

## Architecture

The pipeline runs in this order: **fetch → store → sentiment → IPTC classify → overview**.

- `db.py` owns the SQLAlchemy engine/session; all other modules import from it.
- `fetch.py` collects RSS feeds (`fetch_feeds`) and merges in `scrape.scrape_all()` (DR + TV2), then `store_articles()` deduplicates on URL before inserting. `check_sources()` runs first — it pings every RSS feed *and* probes each scrape page for its JSON marker via `scrape.check_sources()` — and aborts only if *every* source is unreachable; a feed returning 0 entries is logged loudly as an error rather than treated as a quiet news day.
- `scrape.py` collects DR and TV2 — sources without a usable RSS feed — by parsing the **embedded JSON** their server-rendered pages ship (DR's Next.js `__NEXT_DATA__`, TV2's JSON-LD `LiveBlogPosting`), not the rendered DOM, so no headless browser is needed. Each scraper returns the same `source`/`title`/`summary`/`url`/`published_at` dict shape as the RSS path and reuses `fetch.clean_text`. DR's `urlPathId` rebuilds the same URL the old RSS feed used (so it dedups against existing rows); TV2 uses each post's unique `#entry=<uuid>` URL. The JSON shapes are framework-specific, so a scraper logs a loud `[error]` and returns `[]` on a structural change rather than raising. **Import note:** `scrape.py` does `from fetch import clean_text` at module top, so `fetch.py` imports `scrape` *lazily* (inside `check_sources`/`run_pipeline`) to avoid a circular import.
- `sentiment.py` scores the **original Danish** `title`/`summary` with a fine-tuned Danish transformer (`MODEL_NAME` in the module). Maps `argmax` to the label and `P(pos)-P(neg)` to the -1..+1 score. Model downloads from Hugging Face on first run (~0.4 GB, cached under `~/.cache/huggingface`).
- `topics.py` is now just **shared Danish text preprocessing** (the NMF topic model it once held was removed). `lemmatize_tokens` lemmatizes tokens to Danish base forms via `simplemma` (so `regeringen`/`regeringens` collapse onto `regering`) and filters stop words (`danish_stopwords()` — the union of NLTK's Danish list and an extended list in `data/danish_stopwords.txt`, which **overrides** the NLTK fallback when present). `app.py` passes `lemmatize_tokens` as the tokenizer when building the per-IPTC-category word clouds, so their terms are lemmatized Danish. Requires `simplemma` wherever it is used.
- `iptc.py` assigns each article one **IPTC Media Topics top-level category** (`iptc_category`, e.g. `Politik`, `Sport`) via **embedding similarity** on the Danish `title`/`summary`. A multilingual sentence-transformer (`MODEL_NAME`, MiniLM) embeds the article and each of the 17 Danish category phrases in `IPTC_LABELS`; the article gets the category with the highest cosine similarity (both sides L2-normalized, so a dot product is the cosine). The best similarity is written back as confidence (`iptc_score`); below `CONFIDENCE_FLOOR` it is bucketed as `Øvrige`. The label embeddings are cached (`_label_space`). No training data and no `.pkl` — the model *is* the artifact (cached by HF, ~0.12 GB on first run). This replaced an earlier zero-shot-NLI approach that needed one forward pass *per label* (17×) and was too slow/unstable on CPU; embedding similarity is one pass per article. `classify_pending` encodes and commits per `_CHUNK` so a crash is resumable (only `iptc_category IS NULL` rows are selected). The IPTC category is the **only** topic axis surfaced in the dashboard; `topic_id` is no longer displayed.
- `digest.py` writes a **one-sentence Danish digest** of a single story via a **local LLM (Ollama)** — kept local/no-external-API like the rest of the NLP. `digest_text(title, summary)` returns `(sentence, from_llm)`; it prompts `OLLAMA_MODEL` (default `gemma2:2b`) over `localhost:11434` via the `ollama` package. If Ollama isn't installed/running it logs once and falls back to the publisher teaser's first sentence with `from_llm=False`, so the pipeline degrades instead of crashing (same stance as `scrape.py`). The `ollama` import is deferred so the module imports cheaply even without the package. **Ollama is an external prerequisite** (install the app + `ollama pull gemma2:2b`), unlike the pip-only Hugging Face models.
- `overview.py` builds the dashboard's **"Nyhedsoverblik"** snapshot. `build_overview()` loads recent classified articles (last 72 h, `Øvrige` excluded), takes the **top-6 most prevalent IPTC categories**, and within each clusters near-duplicate stories across outlets by **embedding cosine similarity** (reusing `iptc._model()` — no second model load; `sklearn.cluster.AgglomerativeClustering`, `metric="cosine"`, `distance_threshold≈0.35`). Clusters rank by **distinct-source count then recency** — so "importance" = multi-source coverage, with singletons falling back to recency — and the top 3 per topic are kept. The representative (longest summary, tie-broken by earliest publish) gets a digest via `digest.py`, **cached on `articles.digest` only when LLM-generated** (a teaser fallback stays un-cached so the next run retries Ollama). The whole thing is written as one JSON blob to `news_overview`; the pipeline wraps this stage in try/except so a digest hiccup can't fail the run.
- `app.py` reads directly from Neon; no local state. Expects the pipeline columns (`sentiment_score`, `sentiment_label`, `iptc_category`) to be populated before launch; the UI is in Danish and shows, **at the very top, the "Nyhedsoverblik" digest banner** (`load_overview` → the latest `news_overview` snapshot, rendered as top-6 topics × 3 stories, each a linked digest sentence + covering sources + sentiment chip + `N kilder` coverage count; this banner is a global "what's happening now" view and is **not** affected by the sidebar filters), then **per-IPTC-category word clouds** (`iptc_clouds`: the 6 most prevalent categories, each a TF-IDF over that category's Danish title+summary, lemmatized via `topics.lemmatize_tokens`, `Øvrige` excluded), an IPTC **category**-distribution bar chart, a source×category mean-sentiment heatmap, a category filter, and a filterable article table (no English translation displayed). Editing `topics.py`/`iptc.py`/`db.py`/`overview.py`/`digest.py` requires a Streamlit restart (already-imported modules are cached in `sys.modules`); editing `app.py` itself hot-reloads.
- `scheduler.py` (APScheduler `BlockingScheduler`, UTC) runs the full pipeline immediately on startup, then every 2 h, **and** a daily retention sweep (`db.purge_old_articles()`, `RETENTION_MONTHS`) at 03:00 UTC. Both jobs swallow exceptions so one failure can't kill the scheduler. For unattended runs on Windows, `scripts/scheduled_run.py` is the production entry point instead — it runs one pipeline pass under `pythonw` (redirecting `sys.stdout`/`stderr` to `logs/pipeline.log`, since `pythonw` leaves them `None`), pops a winotify toast and exits non-zero on failure.

## Key Constraints

- **All NLP runs on Danish**: sentiment (Danish transformer) and IPTC classification (embedding similarity against Danish category phrases) operate on the Danish `title`/`summary`; the word clouds tokenize Danish text with `lemmatize_tokens`. There is no translation step or DeepL dependency — `translate.py` and the `translated_*` columns were removed. IPTC classification is likewise local (a cached Hugging Face sentence-transformer), consistent with the no-external-API stance.
- **IPTC is the topic axis**: `iptc_category` (fixed IPTC taxonomy, embedding similarity) drives every dashboard view, word clouds included. The earlier NMF topic model and its `topic_id` column were removed; `topics.py` now exists only to provide the shared `lemmatize_tokens` tokenizer the word clouds reuse.
- **Neon connection**: Use `sslmode=require` in `DATABASE_URL`. Neon auto-suspends on inactivity; handle reconnect in `db.py`.
- **Sources**: no API key required. Five via RSS (`RSS_FEEDS`: Politiken, Information, Jyllands-Posten, Berlingske, Kristeligt Dagblad) and two via scraper (`scrape.SCRAPE_SOURCES`: DR, TV2). `RSS_FEEDS` maps each source to a *list* of feed URLs, since Jyllands-Posten exposes two (top-stories + latest-news); `store_articles()` dedups on URL across both RSS and scraped articles. Berlingske's `next-api` "alle" feed declares us-ascii but serves utf-8, so feedparser flags `bozo` — benign, the entries parse fine. Kristeligt Dagblad's `/rss/nyheder` feed sends HTML-escaped summaries, so `clean_text()` unescapes **before** stripping tags (otherwise `<p>`/`dir`/`ltr` leak into the NLP); it unescapes *repeatedly* until stable, so double-escaped entities like `&amp;nbsp;` fully decode instead of leaving a literal `nbsp` token. **DR** moved off RSS to the scraper because the DR RSS feed omits article summaries (its `/nyheder` page carries them in `__NEXT_DATA__`); a few DR top/breaking items still have an empty teaser, and the pre-existing RSS-era DR rows keep their empty summaries (not back-filled). **TV2** is scraped because its public RSS feed no longer resolves. NewsAPI was evaluated as a fallback but its free tier has near-zero Danish coverage, so it is not used. `NEWS_API_KEY` may exist in `.env` but is currently unused.
