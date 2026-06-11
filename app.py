"""Streamlit-dashboard til Dansk Nyhedssentiment-monitor.

    streamlit run app.py

Læser direkte fra Neon (ingen lokal tilstand) og forventer, at pipeline-kolonnerne
(sentiment_score, sentiment_label, iptc_category) er udfyldt. Sentiment scores på den
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
               iptc_category, iptc_score, fetched_at
        FROM articles
        ORDER BY published_at DESC NULLS LAST
    """
    df = pd.read_sql(query, db.get_engine())
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df


@st.cache_data(ttl=600)
def iptc_clouds(top_k: int = 6, n: int = 40) -> dict[str, object]:
    """Word-cloud image for each of the ``top_k`` most prevalent IPTC categories.

    Each category is treated as one aggregated document (its articles' Danish
    title+summary, lemmatized via ``topics.lemmatize_tokens``); a TF-IDF across
    the categories then surfaces the terms most *distinctive* to each one. The
    ``Øvrige`` below-floor bucket is excluded. Insertion order follows
    prevalence, so the most common category renders first.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    rows = pd.read_sql(
        """
        SELECT iptc_category, title, summary
        FROM articles
        WHERE iptc_category IS NOT NULL AND iptc_category <> 'Øvrige'
        """,
        db.get_engine(),
    )
    if rows.empty:
        return {}

    rows["text"] = (
        rows["title"].fillna("") + " " + rows["summary"].fillna("")
    ).str.strip()
    top_cats = rows["iptc_category"].value_counts().head(top_k).index.tolist()
    docs = [" ".join(rows.loc[rows["iptc_category"] == cat, "text"]) for cat in top_cats]

    vec = TfidfVectorizer(
        tokenizer=topics.lemmatize_tokens, token_pattern=None, lowercase=False,
    )
    dtm = vec.fit_transform(docs)
    features = vec.get_feature_names_out()

    images: dict[str, object] = {}
    for i, cat in enumerate(top_cats):
        weights = dtm[i].toarray().ravel()
        top_idx = weights.argsort()[::-1][:n]
        freqs = {features[j]: float(weights[j]) for j in top_idx if weights[j] > 0}
        if not freqs:
            continue
        wc = WordCloud(
            width=480, height=320, background_color="white",
            colormap="viridis", prefer_horizontal=0.9,
        ).generate_from_frequencies(freqs)
        images[cat] = wc.to_array()
    return images


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
selected_categories = st.sidebar.multiselect(
    "Kategori (IPTC)", sorted(df["iptc_category"].dropna().unique()), default=None,
)

mask = pd.Series(True, index=df.index)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0], tz="UTC"), pd.Timestamp(date_range[1], tz="UTC")
    mask &= df["published_at"].between(start, end + pd.Timedelta(days=1))
if sources:
    mask &= df["source"].isin(sources)
if labels:
    mask &= df["sentiment_label"].isin(labels)
if selected_categories:
    mask &= df["iptc_category"].isin(selected_categories)

fdf = df[mask]

# --- Nøgletal --------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Artikler", f"{len(fdf):,}")
avg = fdf["sentiment_score"].mean()
c2.metric("Gns. stemning", f"{avg:+.3f}" if pd.notna(avg) else "—")
c3.metric("Kategorier", int(fdf["iptc_category"].nunique()))
c4.metric("Kilder", int(fdf["source"].nunique()))

st.divider()

# --- Emneskyer (ordsky pr. kategori) ---------------------------------------
st.subheader("Emneskyer")
st.caption("Mest karakteristiske ord for de 6 mest udbredte IPTC-kategorier (dansk korpus).")
clouds = iptc_clouds()
if clouds:
    per_row = 3
    cats = list(clouds)
    for i in range(0, len(cats), per_row):
        row = st.columns(per_row)
        for col, cat in zip(row, cats[i : i + per_row]):
            col.image(clouds[cat], caption=cat, use_container_width=True)
else:
    st.info("Ingen kategorier endnu. Kør pipelinen for at klassificere artikler.")

# --- Emnefordeling + kildesammenligning ------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Kategorifordeling")
    cat_counts = fdf["iptc_category"].dropna().value_counts()
    if not cat_counts.empty:
        fig = px.bar(
            x=cat_counts.index, y=cat_counts.values,
            labels={"x": "Kategori", "y": "Artikler"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Ingen kategoritildelinger endnu.")

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

# --- Stemning pr. kilde og kategori (matrix) -------------------------------
st.subheader("Stemning pr. kilde og kategori")
st.caption("Gennemsnitlig stemningsscore (−1 til +1) for hver kilde pr. IPTC-kategori.")
heat = fdf.dropna(subset=["iptc_category", "sentiment_score"]).copy()
if not heat.empty:
    pivot = heat.pivot_table(
        index="source", columns="iptc_category", values="sentiment_score", aggfunc="mean"
    ).sort_index(axis=1)
    fig = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn",
        zmin=-1, zmax=1,
        aspect="auto",
        text_auto=".2f",
        labels={"x": "Kategori", "y": "Kilde", "color": "Gns. stemning"},
    )
    fig.update_xaxes(type="category")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Ingen stemningsdata pr. kategori endnu.")

# --- Artikeltabel ----------------------------------------------------------
st.subheader("Artikler")
table = fdf.copy()
table["stemning"] = table["sentiment_label"].map(SENTIMENT_DA)
table["kategori"] = table["iptc_category"].fillna("")
st.dataframe(
    table[
        ["published_at", "source", "title",
         "sentiment_score", "stemning", "kategori", "url"]
    ],
    use_container_width=True,
    hide_index=True,
    column_config={
        "published_at": st.column_config.DatetimeColumn("Udgivet", format="YYYY-MM-DD HH:mm"),
        "source": "Kilde",
        "title": "Overskrift",
        "sentiment_score": st.column_config.NumberColumn("Stemningsscore", format="%.3f"),
        "stemning": "Stemning",
        "kategori": "Kategori",
        "url": st.column_config.LinkColumn("Link"),
    },
)
