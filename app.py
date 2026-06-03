"""Streamlit-dashboard til Dansk Nyhedssentiment-monitor.

    streamlit run app.py

Læser direkte fra Neon (ingen lokal tilstand) og forventer, at pipeline-kolonnerne
(sentiment_score, sentiment_label, topic_id) er udfyldt. Sentiment scores på den
oprindelige danske tekst, så dashboardet viser ikke den engelske oversættelse.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud

# Gør src/ importerbar, så vi kan genbruge den delte engine.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import db  # noqa: E402
import topics  # noqa: E402

st.set_page_config(page_title="Dansk Nyhedssentiment-monitor", layout="wide")

# DB-værdier (engelske) -> danske visningsetiketter.
SENTIMENT_DA = {"positive": "Positiv", "neutral": "Neutral", "negative": "Negativ"}
SENTIMENT_FARVER = {"Positiv": "#2ca02c", "Neutral": "#7f7f7f", "Negativ": "#d62728"}


@st.cache_data(ttl=600)
def load_articles() -> pd.DataFrame:
    """Hent alle artikler til en DataFrame (cachet i 10 min)."""
    query = """
        SELECT id, source, title, summary,
               url, published_at, sentiment_score, sentiment_label,
               topic_id, fetched_at
        FROM articles
        ORDER BY published_at DESC NULLS LAST
    """
    df = pd.read_sql(query, db.get_engine())
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df


@st.cache_data(ttl=600)
def topic_clouds(n: int = 40) -> dict[int, object]:
    """Render a word-cloud image per LDA topic from its term weights."""
    images: dict[int, object] = {}
    for tid, freqs in topics.topic_term_weights(n).items():
        if not freqs:
            continue
        wc = WordCloud(
            width=480, height=320, background_color="white",
            colormap="viridis", prefer_horizontal=0.9,
        ).generate_from_frequencies(freqs)
        images[tid] = wc.to_array()
    return images


@st.cache_data(ttl=600)
def topic_label_map(n: int = 3) -> dict[int, str]:
    """Top-term label per topic_id, regenerated whenever the model is retrained."""
    return topics.topic_labels(n)


def label_for(tid) -> str:
    """Display name for a topic_id, falling back to 'Emne N' if unlabelled."""
    name = topic_label_map().get(int(tid))
    return f"{int(tid)}: {name}" if name else f"Emne {int(tid)}"


st.title("🇩🇰 Dansk Nyhedssentiment-monitor")
st.caption(
    "Emnesporing og sentimentanalyse på tværs af DR, Politiken, Information, "
    "Jyllands-Posten, Berlingske og Kristeligt Dagblad."
)

df = load_articles()
if df.empty:
    st.warning("Ingen artikler endnu. Kør `python src/fetch.py` for at fylde databasen.")
    st.stop()

# --- Sidebjælke-filtre -----------------------------------------------------
st.sidebar.header("Filtre")

min_date = df["published_at"].min()
max_date = df["published_at"].max()
date_range = st.sidebar.date_input(
    "Datointerval",
    value=(min_date.date(), max_date.date()) if pd.notna(min_date) else None,
)
sources = st.sidebar.multiselect(
    "Kilde", sorted(df["source"].dropna().unique()), default=None
)
labels = st.sidebar.multiselect(
    "Stemning", ["positive", "neutral", "negative"], default=None,
    format_func=lambda x: SENTIMENT_DA[x],
)
selected_topics = st.sidebar.multiselect(
    "Emne", sorted(df["topic_id"].dropna().unique().astype(int)), default=None,
    format_func=label_for,
)

mask = pd.Series(True, index=df.index)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0], tz="UTC"), pd.Timestamp(date_range[1], tz="UTC")
    mask &= df["published_at"].between(start, end + pd.Timedelta(days=1))
if sources:
    mask &= df["source"].isin(sources)
if labels:
    mask &= df["sentiment_label"].isin(labels)
if selected_topics:
    mask &= df["topic_id"].isin(selected_topics)

fdf = df[mask]

# --- Nøgletal --------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Artikler", f"{len(fdf):,}")
avg = fdf["sentiment_score"].mean()
c2.metric("Gns. stemning", f"{avg:+.3f}" if pd.notna(avg) else "—")
c3.metric("Emner", int(fdf["topic_id"].nunique()))
c4.metric("Kilder", int(fdf["source"].nunique()))

st.divider()

# --- Emneskyer (ordsky pr. emne) -------------------------------------------
st.subheader("Emneskyer")
st.caption("Mest karakteristiske ord pr. emne fra emnemodellen (NMF, dansk korpus).")
clouds = topic_clouds()
if clouds:
    per_row = 3
    tids = sorted(clouds)
    for i in range(0, len(tids), per_row):
        row = st.columns(per_row)
        for col, tid in zip(row, tids[i : i + per_row]):
            col.image(clouds[tid], caption=label_for(tid), use_container_width=True)
else:
    st.info("Ingen emnemodel endnu. Kør topic-trinet for at generere emner.")

# --- Emnefordeling + kildesammenligning ------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Emnefordeling")
    topic_counts = fdf["topic_id"].dropna().astype(int).value_counts().sort_index()
    if not topic_counts.empty:
        fig = px.bar(
            x=[label_for(t) for t in topic_counts.index], y=topic_counts.values,
            labels={"x": "Emne", "y": "Artikler"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Ingen emnetildelinger endnu.")

with right:
    st.subheader("Stemning pr. kilde")
    by_source = (
        fdf.dropna(subset=["sentiment_label"])
        .groupby(["source", "sentiment_label"]).size().reset_index(name="count")
    )
    if not by_source.empty:
        by_source["Stemning"] = by_source["sentiment_label"].map(SENTIMENT_DA)
        fig = px.bar(
            by_source, x="source", y="count", color="Stemning",
            barmode="group",
            labels={"source": "Kilde", "count": "Antal"},
            color_discrete_map=SENTIMENT_FARVER,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Ingen mærkede data endnu.")

# --- Stemning pr. kilde og emne (matrix) -----------------------------------
st.subheader("Stemning pr. kilde og emne")
st.caption("Gennemsnitlig stemningsscore (−1 til +1) for hver kilde pr. emne.")
heat = fdf.dropna(subset=["topic_id", "sentiment_score"]).copy()
if not heat.empty:
    heat["topic_id"] = heat["topic_id"].astype(int)
    pivot = heat.pivot_table(
        index="source", columns="topic_id", values="sentiment_score", aggfunc="mean"
    ).sort_index(axis=1)
    pivot.columns = [label_for(c) for c in pivot.columns]
    fig = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn",
        zmin=-1, zmax=1,
        aspect="auto",
        text_auto=".2f",
        labels={"x": "Emne", "y": "Kilde", "color": "Gns. stemning"},
    )
    fig.update_xaxes(type="category")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Ingen stemningsdata pr. emne endnu.")

# --- Artikeltabel ----------------------------------------------------------
st.subheader("Artikler")
table = fdf.copy()
table["stemning"] = table["sentiment_label"].map(SENTIMENT_DA)
table["emne"] = table["topic_id"].map(
    lambda t: label_for(t) if pd.notna(t) else ""
)
st.dataframe(
    table[
        ["published_at", "source", "title",
         "sentiment_score", "stemning", "emne", "url"]
    ],
    use_container_width=True,
    hide_index=True,
    column_config={
        "published_at": st.column_config.DatetimeColumn("Udgivet", format="YYYY-MM-DD HH:mm"),
        "source": "Kilde",
        "title": "Overskrift",
        "sentiment_score": st.column_config.NumberColumn("Stemningsscore", format="%.3f"),
        "stemning": "Stemning",
        "emne": "Emne",
        "url": st.column_config.LinkColumn("Link"),
    },
)
