# Danish News Sentiment Monitor
### Topic Tracking & Sentiment Analysis for Danish News Sources

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Neon-4169E1?logo=postgresql&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-FFD21E?logoColor=black)
![sentence-transformers](https://img.shields.io/badge/sentence--transformers-IPTC-2ca5e0)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF-F7931E?logo=scikit-learn&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=white)

---

## Project Overview

An end-to-end NLP and data pipeline project that collects headlines and article summaries from major Danish news sources in near real-time, performs sentiment analysis and classifies each article into a fixed **IPTC Media Topics** category (via multilingual embedding similarity) **directly on the Danish text**, and presents the results in an interactive Streamlit dashboard.

Built with a media monitoring background in mind, the project answers:

> *"Which topics are dominating Danish news right now — and what is the sentiment tone across sources?"*

---

## Tech Stack

| Category | Tool / Technology | Purpose |
|---|---|---|
| Data Collection | RSS feeds (`feedparser`) + HTML scrapers (`requests`) | Pull headlines and summaries from Politiken, Information, Jyllands-Posten, Berlingske, Kristeligt Dagblad (RSS) plus DR and TV2 (scraped from their embedded JSON) |
| Data Storage | PostgreSQL (Neon serverless) | Persist all articles, scores, and topic assignments |
| Sentiment Analysis | Danish transformer (`transformers`) | Sentiment scoring on the original Danish text (positive / neutral / negative) |
| Topic Classification | Multilingual sentence-transformer + cosine similarity (`sentence-transformers`) | Assign each article a fixed IPTC Media Topics category |
| Word Clouds | TF-IDF (`scikit-learn`) + `simplemma` lemmatizer | Characteristic Danish terms per category |
| Dashboard | Streamlit + Plotly | Interactive web app for exploration, filtering, and trend tracking |
| Scheduling | APScheduler | Automatically fetch new articles every 1–2 hours |
| Version Control | Git / GitHub | All code, SQL, and queries versioned publicly |
| Environment | python-dotenv | Credential management for API keys and DB connection strings |

---

## Repository Structure

```
danish-news-sentiment/
├── data/                        # danish_stopwords.txt (custom Danish stop-word list)
├── sql/
│   ├── schema.sql               # Table definitions
│   └── analysis_queries.sql     # Business queries for dashboard
├── src/
│   ├── db.py                    # Neon/PostgreSQL connection helper
│   ├── fetch.py                 # RSS collector + scrape orchestration + source health check
│   ├── scrape.py                # HTML scrapers for DR and TV2 (parse embedded JSON)
│   ├── sentiment.py             # Danish transformer sentiment scoring
│   ├── iptc.py                  # IPTC category classification (embedding similarity)
│   ├── topics.py                # Danish lemmatizing tokenizer (shared text preprocessing)
│   └── scheduler.py             # APScheduler job definitions
├── app.py                       # Streamlit dashboard
├── requirements.txt
├── .env.example                 # Template for API keys
└── README.md
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/danish-news-sentiment.git
cd danish-news-sentiment
pip install -r requirements.txt
```

### 2. Configure credentials

Copy `.env.example` to `.env` and fill in your keys:

```
DATABASE_URL=postgresql://user:password@ep-xxx.neon.tech/neondb?sslmode=require
```

> `NEWS_API_KEY` is no longer required — all sources are collected via RSS.

### 3. Set up the database

```bash
psql $DATABASE_URL -f sql/schema.sql
```

### 4. Run the first fetch and launch the dashboard

```bash
python src/fetch.py          # Fetch and process articles once
streamlit run app.py         # Launch the dashboard
```

### 5. Enable automatic scheduling

```bash
python src/scheduler.py      # Runs the full pipeline every 2 hours
```

---

## Build Steps

Follow these phases in order. Each builds on the last.

| Phase | Task | Description |
|---|---|---|
| 01 | RSS setup | Identify working RSS feed URLs for DR, Politiken, and Information. Write `src/fetch.py` using `feedparser` to pull headlines, publication date, source, and summary, with a start-up health check that flags dead feeds. |
| 02 | Database schema | Create a free Neon project. Write `schema.sql` with an `articles` table. Add columns for `sentiment_score`, `sentiment_label`, `iptc_category`, and `iptc_score`. |
| 03 | ETL pipeline | Extend fetch to deduplicate on URL, clean text (strip HTML, normalise whitespace), and load into Neon. Verify row counts. |
| 04 | Sentiment scoring | Write `src/sentiment.py` using a fine-tuned Danish transformer (Hugging Face). Score each Danish headline + summary directly. Store score (−1 to +1) and a label (positive / neutral / negative) back to Neon. |
| 05 | Topic classification | In `src/iptc.py`, embed each Danish article and the 17 IPTC Media Topics category phrases with a multilingual sentence-transformer, then assign each article the category with the highest cosine similarity (no training data; below a confidence floor → `Øvrige`). `src/topics.py` provides the shared Danish lemmatizing tokenizer reused for the word clouds. |
| 06 | Streamlit dashboard | Build `app.py` with KPI cards, per-category word clouds (the 6 most prevalent IPTC categories), a category distribution bar chart, source comparison view, a source×category mean-sentiment heatmap, and a filterable article table with sentiment labels. |
| 07 | Scheduling | Add APScheduler to `src/scheduler.py` to trigger the full pipeline (fetch → sentiment → IPTC classify) every 2 hours. |
| 08 | Documentation | Write a clean README, comment your SQL, and add a short project write-up. Push everything to GitHub. |

---

## Key Questions Answered

- Which topics are dominating Danish news today, and how has that shifted over the past week?
- Which news sources tend to publish the most negative or most positive content?
- Are sentiment trends for a given topic consistent across sources, or does coverage diverge?
- Which topics spike suddenly — and are they driven by one outlet or across all sources?
- What is the typical sentiment distribution for political vs. business vs. sports coverage?

---

## Data Sources

### RSS Feeds (free, no key required)

| Source | RSS URL | Notes |
|---|---|---|
| Politiken | `https://politiken.dk/rss/senestenyt.rss` | Centre-left broadsheet |
| Information | `https://www.information.dk/feed` | Independent daily |
| Jyllands-Posten | `https://newsletter-proxy.aws.jyllands-posten.dk/v1/top-stories/jyllands-posten.dk` | Major daily — top-stories feed |
| Jyllands-Posten | `https://newsletter-proxy.aws.jyllands-posten.dk/v1/latestNewsRss/jyllands-posten.dk?count=10` | Major daily — latest-news feed |
| Berlingske | `https://www.berlingske.dk/next-api/feeds/alle` | Conservative daily — declares us-ascii but serves utf-8 (benign `bozo` warning) |
| Kristeligt Dagblad | `https://www.kristeligt-dagblad.dk/rss/nyheder` | Christian daily — summaries arrive HTML-escaped (`clean_text` unescapes then strips) |

A single source may expose multiple feeds (Jyllands-Posten has both top-stories and latest-news), so `RSS_FEEDS` maps each source label to a *list* of feed URLs; `store_articles()` dedups on URL, so any overlap between a source's feeds is harmless.

### Scraped sources (`src/scrape.py`)

Two national sources don't offer a usable RSS feed, so `scrape.py` collects them by parsing the **embedded JSON** their server-rendered pages already ship (no headless browser):

| Source | Page | How |
|---|---|---|
| DR | `https://www.dr.dk/nyheder` | Parse the Next.js `__NEXT_DATA__` blob. DR's RSS feed omits article summaries — the page carries them. URLs rebuild to the RSS format, so rows dedup cleanly. |
| TV2 | `https://nyheder.tv2.dk/live/kort-nyt` | Parse the JSON-LD `LiveBlogPosting`; each "kort nyt" update has a headline, body, timestamp, and a unique `#entry=<uuid>` URL. TV2's public RSS feed no longer resolves. |

Scraped articles use the same `source`/`title`/`summary`/`url`/`published_at` shape as RSS, so the rest of the pipeline is identical. Because these JSON shapes are framework-specific, a scraper logs a loud error and yields nothing (rather than crashing) if the page is restructured.

`fetch.py` runs a start-up health check (`check_sources()`) that pings each RSS feed **and** probes each scrape page for its JSON marker, aborting only if *every* source is unreachable — so a dead feed or a restructured page surfaces immediately instead of silently contributing zero articles.

### A note on feeds that don't work

TV2 discontinued its public RSS feed (`feeds.tv2.dk` no longer resolves) and DR's RSS feed omits article summaries — both are now collected via the scrapers above instead. (Berlingske's legacy `berlingske.dk/rss` was dead, and Kristeligt Dagblad was previously thought to publish no feed, but their `next-api` "alle" and `/rss/nyheder` feeds respectively work and are included in the RSS table.) NewsAPI.org was evaluated as a fallback but its free tier returns almost no Danish coverage, so it is not used.

---

## requirements.txt

```
feedparser
requests
pandas
sqlalchemy
psycopg2-binary
python-dotenv
torch
transformers
sentencepiece
sentence-transformers
scikit-learn
nltk
simplemma
streamlit
plotly
wordcloud
apscheduler
```

---

## A Note on Danish-Language NLP

This project runs NLP **directly on Danish** — there is no translation step:

- **Sentiment** uses a fine-tuned Danish transformer (Hugging Face `transformers`), scoring the original `title`/`summary`. The 3-class model maps to a label and a −1..+1 score (`P(positive) − P(negative)`). The model (~0.4 GB) downloads on first run and is cached locally.
- **Topic classification** uses the fixed **IPTC Media Topics** taxonomy. A multilingual sentence-transformer embeds each Danish article and each of the 17 Danish category phrases, and the article is assigned the category with the highest cosine similarity — zero training data, one forward pass per article. Articles whose best match falls below a confidence floor are bucketed as `Øvrige`.
- **Word clouds** are built per category from a **TF-IDF** over each category's Danish text, tokenized with a `simplemma` lemmatizer (so inflected forms like `regeringen`/`regeringens` collapse onto `regering`) and filtered against Danish stop words — a union of NLTK's Danish list and an extended list in `data/danish_stopwords.txt`. The terms are therefore lemmatized Danish.

The design has evolved twice. An early version translated everything to English first (DeepL) so English-only tools like VADER could be used; that translate-first stage was removed in favour of native Danish models, avoiding translation drift and the external API dependency. Topics were then modelled with LDA, then NMF over TF-IDF, and finally replaced by IPTC embedding-similarity classification — which yields stable, human-readable categories and is far cheaper on CPU than per-label zero-shot NLI.

---

*Built as a portfolio project demonstrating end-to-end data engineering, NLP, and BI skills in a Danish-language context.*
