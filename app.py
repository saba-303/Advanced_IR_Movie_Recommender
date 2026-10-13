import random
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from recommender import (
    load_and_prepare_data, create_ir_model, get_movie_results,
    get_query_results, personalized_recommendations, search_movies,
    get_analytics
)

st.set_page_config(page_title="Advanced Movie Recommender System",
                   page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.stApp { background: #0b0e14; color: #f5f7fb; }
.block-container { padding-top: 1.4rem; max-width: 1450px; }
h1, h2, h3 { color: #f5f7fb !important; }
div[data-testid="stMetric"] { background: #151b27; padding: 14px; border-radius: 12px; border: 1px solid #283244; }
.movie-card { background: #151b27; border: 1px solid #30394b; border-radius: 14px; padding: 16px; margin: 8px 0; }
.small-muted { color: #a9b4c8; font-size: 0.9rem; }
.stButton > button[kind="primary"] { background: #ff4b5c; border: 0; color: white; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_engine():
    df = load_and_prepare_data()
    model = create_ir_model(df)
    return df, model

st.title("🎬 Advanced Movie Recommender System")
st.caption("Powered by Information Retrieval Algorithms")

try:
    with st.spinner("Loading movie dataset and building TF-IDF / LSI / BM25 indexes..."):
        movies, model = load_engine()
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.info("Place movies_metadata.csv, credits.csv, and keywords.csv in the same folder as app.py.")
    st.stop()

if "favorites" not in st.session_state:
    st.session_state.favorites = []
if "last_search" not in st.session_state:
    st.session_state.last_search = ""
if "last_results" not in st.session_state:
    st.session_state.last_results = pd.DataFrame()

tabs = st.tabs(["🔎 Search & Recommend", "🎯 Personalized Recommendations", "📊 Analytics", "🎲 Random Discovery", "⭐ Favorites"])

def render_movie_card(row, key_prefix):
    title = str(row.get("title", "Unknown title"))
    year = row.get("release_year", None)
    year_text = str(int(year)) if pd.notna(year) else "Year unknown"
    rating = float(row.get("vote_average", 0) or 0)
    runtime = row.get("runtime", None)
    runtime_text = f"{int(runtime)} min" if pd.notna(runtime) and runtime > 0 else "Runtime unavailable"
    genres = ", ".join(row.get("genres_list", [])[:4]) or "Genre unavailable"
    cast = ", ".join(row.get("cast_list", [])[:3]) or "Cast unavailable"
    director = str(row.get("director", "")) or "Director unavailable"
    overview = str(row.get("overview", "")).strip() or "No overview available."
    with st.container(border=True):
        left, right = st.columns([5, 1])
        with left:
            st.subheader(title)
            st.caption(f"🗓️ {year_text}   ⭐ {rating:.1f}/10   ⏱️ {runtime_text}")
            st.markdown(f"**Genres:** {genres}")
            st.markdown(f"**Cast:** {cast}  ·  **Director:** {director}")
            with st.expander("Overview"):
                st.write(overview)
        with right:
            if st.button("⭐ Favorite", key=f"fav_{key_prefix}_{title}"):
                if title not in st.session_state.favorites:
                    st.session_state.favorites.append(title)
                    st.toast(f"Added {title} to favorites")
                else:
                    st.toast("Already in favorites")
            if st.button("🔎 Similar", key=f"similar_{key_prefix}_{title}"):
                st.session_state.similar_title = title
                st.toast(f"Selected {title}. Open Personalized Recommendations.")
    return title

with tabs[0]:
    st.subheader("Search movies by title, actor, director, or genre")
    query = st.text_input("Search query", placeholder="e.g., The Conjuring, Vera Farmiga, James Wan, horror")
    search_mode = st.radio("Retrieval mode", ["Hybrid IR ranking", "TF-IDF only", "BM25 keyword ranking"], horizontal=True)
    limit = st.slider("Number of results", 5, 20, 10)
    if st.button("Search", type="primary", key="search_btn"):
        if not query.strip():
            st.warning("Enter a movie title, actor, director, genre, or query.")
        elif search_mode == "Hybrid IR ranking":
            results = get_query_results(query, movies, model, k=limit)
            st.session_state.last_results = results
            st.session_state.last_search = query
        elif search_mode == "TF-IDF only":
            qvec = model["vectorizer"].transform([query])
            from sklearn.metrics.pairwise import cosine_similarity
            scores = cosine_similarity(qvec, model["tfidf_matrix"]).ravel()
            idx = scores.argsort()[-limit:][::-1]
            results = movies.iloc[idx].copy()
            results["retrieval_score"] = scores[idx]
            st.session_state.last_results = results
            st.session_state.last_search = query
        else:
            scores = model["bm25"].score_query(query)
            idx = scores.argsort()[-limit:][::-1]
            results = movies.iloc[idx].copy()
            results["retrieval_score"] = scores[idx]
            st.session_state.last_results = results
            st.session_state.last_search = query
    if isinstance(st.session_state.last_results, pd.DataFrame) and not st.session_state.last_results.empty:
        st.info(f"Showing results for: **{st.session_state.last_search}**")
        st.caption(f"Found {len(st.session_state.last_results)} movie(s)")
        for i, (_, row) in enumerate(st.session_state.last_results.iterrows()):
            render_movie_card(row, f"search_{i}")

with tabs[1]:
    st.subheader("Get Personalized Recommendations")
    options = movies["title"].dropna().astype(str).drop_duplicates().sort_values().tolist()
    default_title = st.session_state.get("similar_title", "Toy Story")
    default_index = options.index(default_title) if default_title in options else 0
    chosen_title = st.selectbox("Choose a movie you like", options, index=default_index)
    rec_count = st.slider("Number of recommendations", 5, 20, 10, key="rec_count")
    if st.button("Recommend similar movies", type="primary", key="personal_btn"):
        matched, recs = personalized_recommendations(chosen_title, movies, model, k=rec_count)
        if matched and not recs.empty:
            st.success(f"Recommendations based on: {matched}")
            st.caption("Hybrid score combines TF-IDF, latent semantic (LSI), BM25 relevance, quality re-ranking, and MMR diversity.")
            for i, (_, row) in enumerate(recs.iterrows()):
                render_movie_card(row, f"personal_{i}")
        else:
            st.warning("No recommendations found for that title.")

with tabs[2]:
    st.subheader("Movie Analytics Dashboard")
    analytics = get_analytics(movies)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Movies", f"{analytics['total_movies']:,}")
    c2.metric("Average Rating", f"{analytics['average_rating']:.2f}/10")
    c3.metric("Total Genres", analytics["total_genres"])
    c4.metric("Total Directors", f"{analytics['total_directors']:,}")
    left, right = st.columns(2)
    with left:
        st.markdown("### 🎭 Top Genres")
        if not analytics["genre_df"].empty:
            st.bar_chart(analytics["genre_df"].set_index("Genre"))
        st.markdown("### ⭐ Rating Distribution")
        rating_hist = pd.cut(
            analytics["ratings"],
            bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            include_lowest=True
        ).value_counts().sort_index()

        rating_hist.index = rating_hist.index.astype(str)
        rating_hist = rating_hist.rename_axis("Rating Range").reset_index(name="Movies")
        st.bar_chart(rating_hist, x="Rating Range", y="Movies")

    with right:
        st.markdown("### 📅 Movies by Year")
        years = analytics["year_df"]
        years = years[(years["Year"] >= 1900) & (years["Year"] <= 2035)]
        if not years.empty:
            st.line_chart(years.set_index("Year"))
        st.markdown("### 🏆 Highest Rated (with votes)")
        top = movies[movies["vote_count"] >= 100].copy()
        top["weighted_rating"] = (top["vote_count"] / (top["vote_count"] + top["vote_count"].quantile(.7))) * top["vote_average"] + (top["vote_count"].quantile(.7) / (top["vote_count"] + top["vote_count"].quantile(.7))) * top["vote_average"].mean()
        st.dataframe(top.sort_values("weighted_rating", ascending=False)[["title","vote_average","vote_count","release_year"]].head(10), use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Random Discovery")
    st.write("Discover something unexpected from the movie catalogue.")
    sample_size = st.slider("How many random movies?", 3, 12, 6, key="random_count")
    if st.button("🎲 Discover movies", type="primary", key="random_btn"):
        st.session_state.random_movies = random.sample(range(len(movies)), min(sample_size, len(movies)))
    if "random_movies" in st.session_state:
        for i, idx in enumerate(st.session_state.random_movies):
            render_movie_card(movies.iloc[idx], f"random_{i}")

with tabs[4]:
    st.subheader("Your Favorite Movies")
    if not st.session_state.favorites:
        st.info("No favorites yet. Use the Favorite button on any movie card to save a movie during this session.")
    else:
        for i, title in enumerate(st.session_state.favorites):
            row = movies[movies["title"] == title]
            if not row.empty:
                render_movie_card(row.iloc[0], f"favorite_{i}")
            if st.button(f"Remove {title}", key=f"remove_{i}"):
                st.session_state.favorites.remove(title)
                st.rerun()

st.divider()
st.caption("IR methods in this application: TF-IDF, cosine similarity, BM25, LSI/SVD, Rocchio-style query expansion, MMR diversity re-ranking, fuzzy title matching, and IMDb weighted rating.")
