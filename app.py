"""Streamlit dashboard for the Danish News Sentiment Monitor.

    streamlit run app.py

Reads directly from Neon (no local state) and expects the pipeline columns
(translated_title, sentiment_score, sentiment_label, topic_id) to be populated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Make src/ importable so we can reuse the shared engine.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import db  # noqa: E402

st.set_page_config(page_title="Danish News Sentiment Monitor", layout="wide")


@st.cache_data(ttl=600)
def load_articles() -> pd.DataFrame:
    """Pull all articles into a DataFrame (cached for 10 min)."""
    query = """
        SELECT id, source, title, translated_title, summary,
               url, published_at, sentiment_score, sentiment_label,
               topic_id, fetched_at
        FROM articles
        ORDER BY published_at DESC NULLS LAST
    """
    df = pd.read_sql(query, db.get_engine())
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df


st.title("🇩🇰 Danish News Sentiment Monitor")
st.caption("Topic tracking & sentiment analysis across DR, TV2, Berlingske and Politiken.")

df = load_articles()
if df.empty:
    st.warning("No articles yet. Run `python src/fetch.py` to populate the database.")
    st.stop()

# --- Sidebar filters -------------------------------------------------------
st.sidebar.header("Filters")

min_date = df["published_at"].min()
max_date = df["published_at"].max()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date.date(), max_date.date()) if pd.notna(min_date) else None,
)
sources = st.sidebar.multiselect(
    "Source", sorted(df["source"].dropna().unique()), default=None
)
labels = st.sidebar.multiselect(
    "Sentiment", ["positive", "neutral", "negative"], default=None
)
topics = st.sidebar.multiselect(
    "Topic", sorted(df["topic_id"].dropna().unique().astype(int)), default=None
)

mask = pd.Series(True, index=df.index)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0], tz="UTC"), pd.Timestamp(date_range[1], tz="UTC")
    mask &= df["published_at"].between(start, end + pd.Timedelta(days=1))
if sources:
    mask &= df["source"].isin(sources)
if labels:
    mask &= df["sentiment_label"].isin(labels)
if topics:
    mask &= df["topic_id"].isin(topics)

fdf = df[mask]

# --- KPI cards -------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Articles", f"{len(fdf):,}")
avg = fdf["sentiment_score"].mean()
c2.metric("Avg sentiment", f"{avg:+.3f}" if pd.notna(avg) else "—")
c3.metric("Topics", int(fdf["topic_id"].nunique()))
c4.metric("Sources", int(fdf["source"].nunique()))

st.divider()

# --- Sentiment trend over time --------------------------------------------
st.subheader("Sentiment trend")
trend = (
    fdf.dropna(subset=["published_at", "sentiment_score"])
    .set_index("published_at")
    .groupby([pd.Grouper(freq="D"), "source"])["sentiment_score"]
    .mean()
    .reset_index()
)
if not trend.empty:
    fig = px.line(
        trend, x="published_at", y="sentiment_score", color="source",
        markers=True, labels={"sentiment_score": "Avg sentiment", "published_at": "Date"},
    )
    fig.add_hline(y=0, line_dash="dot", opacity=0.4)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough scored data to plot a trend.")

# --- Topic distribution + source comparison -------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Topic distribution")
    topic_counts = fdf["topic_id"].dropna().astype(int).value_counts().sort_index()
    if not topic_counts.empty:
        fig = px.bar(
            x=topic_counts.index.astype(str), y=topic_counts.values,
            labels={"x": "Topic", "y": "Articles"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No topic assignments yet.")

with right:
    st.subheader("Sentiment by source")
    by_source = (
        fdf.dropna(subset=["sentiment_label"])
        .groupby(["source", "sentiment_label"]).size().reset_index(name="count")
    )
    if not by_source.empty:
        fig = px.bar(
            by_source, x="source", y="count", color="sentiment_label",
            barmode="group",
            color_discrete_map={"positive": "#2ca02c", "neutral": "#7f7f7f", "negative": "#d62728"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No labelled data yet.")

# --- Article table ---------------------------------------------------------
st.subheader("Articles")
st.dataframe(
    fdf[
        ["published_at", "source", "title", "translated_title",
         "sentiment_score", "sentiment_label", "topic_id", "url"]
    ],
    use_container_width=True,
    hide_index=True,
    column_config={"url": st.column_config.LinkColumn("Link")},
)
