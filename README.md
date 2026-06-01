# Danish News Sentiment Monitor
### Topic Tracking & Sentiment Analysis for Danish News Sources

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Neon-4169E1?logo=postgresql&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![DeepL](https://img.shields.io/badge/DeepL-0F2B46?logo=deepl&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-LDA-F7931E?logo=scikit-learn&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=white)

---

## Project Overview

An end-to-end NLP and data pipeline project that collects headlines and article summaries from major Danish news sources in near real-time, translates them to English, performs sentiment analysis and LDA topic modelling, and presents the results in an interactive Streamlit dashboard.

Built with a media monitoring background in mind, the project answers:

> *"Which topics are dominating Danish news right now — and what is the sentiment tone across sources?"*

---

## Tech Stack

| Category | Tool / Technology | Purpose |
|---|---|---|
| Data Collection | RSS feeds (`feedparser`) + NewsAPI.org | Pull headlines and summaries from DR, TV2, Berlingske, Politiken |
| Data Storage | PostgreSQL (Neon serverless) | Persist all articles, translations, scores, and topic assignments |
| Translation | DeepL API (free tier) | Translate Danish text to English before NLP processing |
| Sentiment Analysis | VADER (`vaderSentiment`) | Sentence-level sentiment scoring (positive / neutral / negative) |
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
│   ├── fetch.py                 # RSS + NewsAPI collector
│   ├── translate.py             # DeepL translation pipeline
│   ├── sentiment.py             # VADER scoring logic
│   ├── topics.py                # LDA model training and inference
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
DEEPL_API_KEY=your-deepl-free-api-key
NEWS_API_KEY=your-newsapi-org-key   # optional
```

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
| 01 | RSS + NewsAPI setup | Identify RSS feed URLs for DR, TV2, Berlingske, and Politiken. Write `src/fetch.py` using `feedparser` to pull headlines, publication date, source, and summary. |
| 02 | Database schema | Create a free Neon project. Write `schema.sql` with an `articles` table. Add columns for `translated_title`, `sentiment_score`, `sentiment_label`, and `topic_id`. |
| 03 | ETL pipeline | Extend fetch to deduplicate on URL, clean text (strip HTML, normalise whitespace), and load into Neon. Verify row counts. |
| 04 | Translation layer | Write `src/translate.py` to call the DeepL free API. Store translations back to Neon. Add rate-limit handling and batch requests to stay within the free quota (500k chars/month). |
| 05 | Sentiment scoring | Write `src/sentiment.py` using `vaderSentiment`. Score each translated headline + summary. Store compound score (−1 to +1) and a label (positive / neutral / negative) back to Neon. |
| 06 | LDA topic modelling | In `src/topics.py`, load all translated text from Neon, preprocess (tokenise, remove stopwords), and train an LDA model. Experiment with k=5–10 topics. Assign topic IDs back to each article and save the fitted model as a `.pkl` file. |
| 07 | EDA notebook | In `notebooks/01_exploration.ipynb`, explore topic distributions over time, sentiment by source, most common topic terms, and source-tone correlation. |
| 08 | Streamlit dashboard | Build `app.py` with KPI cards, a sentiment trend line chart, topic distribution bar chart, source comparison view, and a filterable article table with sentiment labels. |
| 09 | Scheduling | Add APScheduler to `src/scheduler.py` to trigger the full pipeline (fetch → translate → sentiment → topic assign) every 2 hours. |
| 10 | Documentation | Write a clean README, comment your SQL, and add a short project write-up. Push everything to GitHub. |

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
| TV2 | `https://feeds.tv2.dk/news/rss` | Second major public outlet |
| Berlingske | `https://www.berlingske.dk/rss` | Centre-right broadsheet |
| Politiken | `https://politiken.dk/rss/senestenyt.rss` | Centre-left broadsheet |

### NewsAPI.org (optional, free tier)
- 100 requests/day, headlines + snippets
- Supports filtering by language (`da`) and source

---

## requirements.txt

```
feedparser
requests
pandas
sqlalchemy
psycopg2-binary
python-dotenv
deepl
vaderSentiment
scikit-learn
nltk
streamlit
plotly
apscheduler
joblib
```

---

## A Note on Danish-Language NLP

Most NLP models (including VADER) are trained on English. This project handles Danish via a **translate-first** approach:

- All Danish headlines and summaries are translated to English via the DeepL free API before NLP processing.
- DeepL's free tier allows 500,000 characters/month — sufficient for continuous collection from 3–4 sources.
- This keeps the pipeline simple and accurate without requiring specialised Danish models.

An alternative is to use a Danish BERT model (e.g. `danish-bert-botxo` on HuggingFace) which skips the translation step but requires more compute. This can be explored as an extension once the baseline pipeline is running.

---

*Built as a portfolio project demonstrating end-to-end data engineering, NLP, and BI skills in a Danish-language context.*
