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
import streamlit.components.v1 as components
from wordcloud import WordCloud

# Gør src/ importerbar, så vi kan genbruge den delte engine.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import db  # noqa: E402
import topics  # noqa: E402

st.set_page_config(page_title="Dansk Nyhedssentiment-monitor", layout="wide")

# The whole app is Danish, but Streamlit ships the page as <html lang="en">,
# so screen readers mispronounce it. A 0-height component runs same-origin and
# can reach the parent document to correct the language tag.
components.html(
    "<script>window.parent.document.documentElement.lang = 'da';</script>",
    height=0,
)

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
def load_overview() -> dict | None:
    """Latest precomputed 'Nyhedsoverblik' snapshot, or None if none built yet."""
    from sqlalchemy import text

    with db.get_engine().connect() as conn:
        row = conn.execute(
            text(
                "SELECT payload FROM news_overview "
                "ORDER BY generated_at DESC LIMIT 1"
            )
        ).scalar()
    return row  # JSONB comes back as a dict (or None)


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
            width=480, height=320, mode="RGBA", background_color=None,
            colormap="Set2", prefer_horizontal=0.9,
        ).generate_from_frequencies(freqs)
        images[cat] = wc.to_array()
    return images


def render_story(s: dict) -> None:
    """Render one overview story: the digest sentence (linked) + a meta caption."""
    da = SENTIMENT_DA.get(s.get("sentiment_label"), "")
    color = SENTIMENT_FARVER.get(da, "#7f7f7f")
    digest = s.get("digest") or s.get("title", "")
    coverage = f" · {s['n_sources']} kilder" if s.get("n_sources", 1) > 1 else ""
    meta = (
        f"<div style='font-size:0.82em;color:#666;margin:-4px 0 12px 0'>"
        f"{', '.join(s.get('sources', []))}"
        f"<span style='color:{color}'> · ● {da}</span>{coverage}</div>"
    )
    st.markdown(f"**[{digest}]({s['url']})**")
    st.markdown(meta, unsafe_allow_html=True)


def overview_built_at(overview: dict | None) -> str | None:
    """Local-time string for when the snapshot was built, or None.

    ``generated_at`` is stored as an ISO-8601 UTC string; ``astimezone()`` with
    no argument converts it to the machine's local timezone (no tzdata needed).
    """
    from datetime import datetime

    raw = (overview or {}).get("generated_at")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).astimezone().strftime("%d-%m-%Y %H:%M")
    except ValueError:
        return None


def render_overview(overview: dict | None, per_row: int = 3) -> None:
    """Render the 'Nyhedsoverblik' banner from a stored snapshot."""
    topics = (overview or {}).get("topics") if overview else None
    if not topics:
        st.info("Intet overblik endnu. Kør pipelinen (`python src/fetch.py`) "
                "for at bygge det.")
        return
    for i in range(0, len(topics), per_row):
        cols = st.columns(per_row)
        for col, topic in zip(cols, topics[i : i + per_row]):
            with col:
                st.markdown(f"##### {topic['category']}")
                for story in topic["stories"]:
                    render_story(story)


st.title("Dansk Nyhedssentiment-monitor")
st.caption(
    "Emnesporing og sentimentanalyse på tværs af DR, TV2, Politiken, Information, "
    "Jyllands-Posten, Berlingske og Kristeligt Dagblad."
)

df = load_articles()
if df.empty:
    st.warning("Ingen artikler endnu. Kør `python src/fetch.py` for at fylde databasen.")
    st.stop()

# --- Nyhedsoverblik (digest af de vigtigste historier pr. emne) ------------
overview = load_overview()
st.subheader("📰 Nyhedsoverblik")
caption = (
    "De vigtigste historier pr. emne lige nu — flest kilder først, med et "
    "kort dansk resumé. Bygges med pipelinen og påvirkes ikke af filtrene."
)
built = overview_built_at(overview)
if built:
    caption += f" Sidst opdateret: {built}."
st.caption(caption)
render_overview(overview)
st.divider()

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
    "Sentiment", ["positive", "neutral", "negative"], default=None,
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
    st.subheader("Sentiment pr. kilde")
    by_source = (
        fdf.dropna(subset=["sentiment_label"])
        .groupby(["source", "sentiment_label"]).size().reset_index(name="count")
    )
    if not by_source.empty:
        by_source["Sentiment"] = by_source["sentiment_label"].map(SENTIMENT_DA)
        fig = px.bar(
            by_source, x="source", y="count", color="Sentiment",
            barmode="group",
            labels={"source": "Kilde", "count": "Antal"},
            color_discrete_map=SENTIMENT_FARVER,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Ingen mærkede data endnu.")

# --- Sentiment pr. kilde og kategori (matrix) -------------------------------
st.subheader("Sentiment pr. kilde og kategori")
st.caption("Gennemsnitlig sentimentscore (−1 til +1) for hver kilde pr. IPTC-kategori.")
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
        labels={"x": "Kategori", "y": "Kilde", "color": "Gns. sentiment"},
    )
    fig.update_xaxes(type="category")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Ingen sentimentdata pr. kategori endnu.")

# --- Artikeltabel ----------------------------------------------------------
st.subheader("Artikler")
table = fdf.copy()
table["sentiment"] = table["sentiment_label"].map(SENTIMENT_DA)
table["kategori"] = table["iptc_category"].fillna("")
st.dataframe(
    table[
        ["published_at", "source", "title",
         "sentiment_score", "sentiment", "kategori", "url"]
    ],
    use_container_width=True,
    hide_index=True,
    column_config={
        "published_at": st.column_config.DatetimeColumn("Udgivet", format="YYYY-MM-DD HH:mm"),
        "source": "Kilde",
        "title": "Overskrift",
        "sentiment_score": st.column_config.NumberColumn("Sentimentscore", format="%.3f"),
        "sentiment": "Sentiment",
        "kategori": "Kategori",
        "url": st.column_config.LinkColumn("Link"),
    },
)
