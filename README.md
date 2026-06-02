# Danish News Sentiment Monitor
### Topic Tracking & Sentiment Analysis for Danish News Sources

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Neon-4169E1?logo=postgresql&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-FFD21E?logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-LDA-F7931E?logo=scikit-learn&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=white)

---

## Project Overview

An end-to-end NLP and data pipeline project that collects headlines and article summaries from major Danish news sources in near real-time, performs sentiment analysis and LDA topic modelling **directly on the Danish text**, and presents the results in an interactive Streamlit dashboard.

Built with a media monitoring background in mind, the project answers:

> *"Which topics are dominating Danish news right now — and what is the sentiment tone across sources?"*

---

## Tech Stack

| Category | Tool / Technology | Purpose |
|---|---|---|
| Data Collection | RSS feeds (`feedparser`) | Pull headlines and summaries from DR, Politiken, Information |
| Data Storage | PostgreSQL (Neon serverless) | Persist all articles, scores, and topic assignments |
| Sentiment Analysis | Danish transformer (`transformers`) | Sentiment scoring on the original Danish text (positive / neutral / negative) |
| Topic Modelling | LDA (`scikit-learn`) | Identify recurring topic clusters across articles |
| Dashboard | Streamlit + Plotly | Interactive web app for exploration, filtering, and trend tracking |
| Scheduling | APScheduler | Automatically fetch new articles every 1–2 hours |
| Version Control | Git / GitHub | All code, SQL, and notebooks versioned publicly |
| Environment | python-dotenv | Credential management for API keys and DB connection strings |

---

## Repository Structure

```
danish-news-sentiment/
├── data/                        # Local CSV snapshots (optional)
├── notebooks/
│   └── 01_exploration.ipynb     # EDA: topic distributions, sentiment patterns
├── sql/
│   ├── schema.sql               # Table definitions
│   └── analysis_queries.sql     # Business queries for dashboard
├── src/
│   ├── db.py                    # Neon/PostgreSQL connection helper
│   ├── fetch.py                 # RSS collector + start-up source health check
│   ├── sentiment.py             # Danish transformer sentiment scoring
│   ├── topics.py                # LDA model training and inference (Danish)
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
| 02 | Database schema | Create a free Neon project. Write `schema.sql` with an `articles` table. Add columns for `sentiment_score`, `sentiment_label`, and `topic_id`. |
| 03 | ETL pipeline | Extend fetch to deduplicate on URL, clean text (strip HTML, normalise whitespace), and load into Neon. Verify row counts. |
| 04 | Sentiment scoring | Write `src/sentiment.py` using a fine-tuned Danish transformer (Hugging Face). Score each Danish headline + summary directly. Store score (−1 to +1) and a label (positive / neutral / negative) back to Neon. |
| 05 | LDA topic modelling | In `src/topics.py`, load all Danish text from Neon, preprocess (tokenise, remove Danish stopwords), and train an LDA model. Experiment with k=5–10 topics. Assign topic IDs back to each article and save the fitted model as a `.pkl` file. |
| 06 | EDA notebook | In `notebooks/01_exploration.ipynb`, explore topic distributions over time, sentiment by source, most common topic terms, and source-tone correlation. |
| 07 | Streamlit dashboard | Build `app.py` with KPI cards, per-topic word clouds, topic distribution bar chart, source comparison view, and a filterable article table with sentiment labels. |
| 08 | Scheduling | Add APScheduler to `src/scheduler.py` to trigger the full pipeline (fetch → sentiment → topic assign) every 2 hours. |
| 09 | Documentation | Write a clean README, comment your SQL, and add a short project write-up. Push everything to GitHub. |

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
| DR | `https://www.dr.dk/nyheder/service/feeds/allenyheder` | Danish public broadcaster |
| Politiken | `https://politiken.dk/rss/senestenyt.rss` | Centre-left broadsheet |
| Information | `https://www.information.dk/feed` | Independent daily |

`fetch.py` runs a start-up health check (`check_sources()`) that pings each feed and aborts if all are unreachable, so a dead feed surfaces immediately instead of silently contributing zero articles.

### Sources without a usable feed

TV2 and Berlingske discontinued their public RSS feeds (`feeds.tv2.dk` no longer resolves; `berlingske.dk/rss` resets the connection), and Kristeligt Dagblad publishes no discoverable feed. NewsAPI.org was evaluated as a fallback for these outlets but its free tier returns almost no Danish coverage, so it is not used. These sources are currently omitted.

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
scikit-learn
nltk
streamlit
plotly
wordcloud
apscheduler
joblib
```

---

## A Note on Danish-Language NLP

This project runs NLP **directly on Danish** — there is no translation step:

- **Sentiment** uses a fine-tuned Danish transformer (Hugging Face `transformers`), scoring the original `title`/`summary`. The 3-class model maps to a label and a −1..+1 score (`P(positive) − P(negative)`). The model (~0.4 GB) downloads on first run and is cached locally.
- **Topic modelling** trains LDA on the raw Danish text, filtering Danish stop words (NLTK's Danish list, overridable via `data/danish_stopwords.txt`). Topic terms and word clouds are therefore Danish.

An earlier design translated everything to English first (DeepL) so English-only tools like VADER could be used; that translate-first stage has since been removed in favour of native Danish models, which avoids translation drift and the external API dependency.

---

*Built as a portfolio project demonstrating end-to-end data engineering, NLP, and BI skills in a Danish-language context.*
