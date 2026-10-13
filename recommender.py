import ast
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process


def _parse_json_list(value):
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def _names(value, limit=None):
    parsed = _parse_json_list(value)
    names = [str(item.get("name", "")).strip() for item in parsed
             if isinstance(item, dict) and item.get("name")]
    return names[:limit] if limit else names


def _director(value):
    for person in _parse_json_list(value):
        if isinstance(person, dict) and person.get("job") == "Director":
            return str(person.get("name", ""))
    return ""


def _clean(value):
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9\s]", " ", str(value or "").lower())).strip()


def _safe_merge_id(frame, id_col="id"):
    frame = frame.copy()
    if id_col in frame.columns:
        frame[id_col] = pd.to_numeric(frame[id_col], errors="coerce")
        frame = frame.dropna(subset=[id_col])
        frame[id_col] = frame[id_col].astype("int64")
    return frame


def load_and_prepare_data(metadata_path="movies_metadata.csv",
                          credits_path="credits.csv",
                          keywords_path="keywords.csv"):
    for path in (metadata_path, credits_path, keywords_path):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Required dataset file '{path}' was not found. Download it from "
                "The Movies Dataset on Kaggle and place it beside app.py."
            )

    meta = pd.read_csv(metadata_path, low_memory=False)
    credits = pd.read_csv(credits_path, low_memory=False)
    keywords = pd.read_csv(keywords_path, low_memory=False)

    # The three files use movie IDs as their join key.
    meta = _safe_merge_id(meta)
    credits = _safe_merge_id(credits)
    keywords = _safe_merge_id(keywords)
    required = {"title", "id"}
    if not required.issubset(meta.columns):
        raise ValueError("movies_metadata.csv must contain 'id' and 'title' columns.")

    # Avoid duplicate merge rows where source data contains repeated IDs.
    credits = credits.drop_duplicates("id", keep="first")
    keywords = keywords.drop_duplicates("id", keep="first")
    df = meta.merge(credits, on="id", how="left", suffixes=("", "_credits"))
    df = df.merge(keywords, on="id", how="left", suffixes=("", "_keywords"))

    for col in ["title", "overview", "tagline", "vote_average", "vote_count",
                "runtime", "release_date", "genres", "cast", "crew", "keywords"]:
        if col not in df.columns:
            df[col] = "" if col in ["title", "overview", "tagline", "release_date"] else np.nan

    df["title"] = df["title"].fillna("").astype(str)
    df = df[df["title"].str.strip().ne("")].copy()
    df["overview"] = df["overview"].fillna("").astype(str)
    df["tagline"] = df["tagline"].fillna("").astype(str)
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0.0)
    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")

    df["genres_list"] = df["genres"].apply(_names)
    df["keywords_list"] = df["keywords"].apply(_names)
    df["cast_list"] = df["cast"].apply(lambda x: _names(x, limit=3))
    df["director"] = df["crew"].apply(_director)
    df["genres_text"] = df["genres_list"].apply(lambda x: " ".join(x))
    df["keywords_text"] = df["keywords_list"].apply(lambda x: " ".join(x))
    df["cast_text"] = df["cast_list"].apply(lambda x: " ".join(x))
    df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year

    # Repeated title/director/cast terms are explicit feature weighting.
    df["content"] = df.apply(
        lambda r: " ".join([
            (_clean(r["title"]) + " ") * 4,
            (_clean(r["director"]) + " ") * 3,
            (_clean(r["cast_text"]) + " ") * 2,
            _clean(r["genres_text"]),
            _clean(r["keywords_text"]),
            _clean(r["overview"]),
            _clean(r["tagline"]),
        ]),
        axis=1,
    )
    df = df.drop_duplicates(subset="title", keep="first").reset_index(drop=True)
    if df.empty:
        raise ValueError("No usable movie rows were found in the supplied dataset.")
    return df


def create_ir_model(df, max_features=30000, lsi_components=100):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features,
                                 ngram_range=(1, 2), min_df=1, max_df=0.98,
                                 sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(df["content"].fillna(""))
    # Dense all-pairs matrices are costly on 45k movies; compute similarities on demand.
    n_components = min(lsi_components, max(2, tfidf_matrix.shape[0] - 1),
                       max(2, tfidf_matrix.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    lsi_matrix = svd.fit_transform(tfidf_matrix)
    lsi_norms = np.linalg.norm(lsi_matrix, axis=1, keepdims=True)
    lsi_matrix = lsi_matrix / np.maximum(lsi_norms, 1e-12)
    title_to_index = {}
    for i, title in enumerate(df["title"].astype(str)):
        title_to_index.setdefault(title, i)
    return {
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "lsi_matrix": lsi_matrix,
        "svd": svd,
        "title_to_index": title_to_index,
        "bm25": BM25Index(df["content"].fillna("").tolist()),
    }


class BM25Index:
    """Small, transparent BM25 implementation for tokenized movie documents."""
    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tokens = [re.findall(r"\b\w+\b", str(doc).lower()) for doc in documents]
        self.lengths = np.array([len(doc) for doc in self.tokens], dtype=float)
        self.avgdl = float(self.lengths.mean()) if len(self.lengths) else 0.0
        self.doc_freqs = []
        df = Counter()
        for doc in self.tokens:
            counts = Counter(doc)
            self.doc_freqs.append(counts)
            df.update(counts.keys())
        n = len(self.tokens)
        self.idf = {term: np.log(1 + (n - freq + 0.5) / (freq + 0.5))
                    for term, freq in df.items()}

    def score_query(self, query):
        terms = re.findall(r"\b\w+\b", str(query).lower())
        scores = np.zeros(len(self.tokens), dtype=float)
        if not terms or not self.tokens:
            return scores
        for i, counts in enumerate(self.doc_freqs):
            dl = self.lengths[i]
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if tf == 0 or term not in self.idf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-12))
                score += self.idf[term] * tf * (self.k1 + 1) / denom
            scores[i] = score
        return scores


def _weighted_rating(df):
    votes = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0).astype(float)
    ratings = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0).astype(float)
    m = float(votes.quantile(0.70)) if len(votes) else 0.0
    c = float(ratings.mean()) if len(ratings) else 0.0
    denominator = votes + m
    return np.where(denominator > 0, (votes / np.maximum(denominator, 1e-12)) * ratings
                    + (m / np.maximum(denominator, 1e-12)) * c, c)


def _top_indices(scores, k=10, exclude=None):
    scores = np.asarray(scores, dtype=float).copy()
    if exclude is not None and 0 <= int(exclude) < len(scores):
        scores[int(exclude)] = -np.inf
    k = min(k, int(np.isfinite(scores).sum()))
    if k <= 0:
        return []
    candidates = np.argpartition(scores, -k)[-k:]
    return candidates[np.argsort(scores[candidates])[::-1]].tolist()


def find_title_match(query, df, threshold=60):
    titles = df["title"].astype(str).tolist()
    if not query.strip():
        return None
    # Prefer exact case-insensitive match before fuzzy matching.
    lookup = {title.casefold(): title for title in titles}
    if query.casefold() in lookup:
        return lookup[query.casefold()]
    match = process.extractOne(query, titles)
    if match and match[1] >= threshold:
        return match[0]
    return None


def _result_frame(df, indices, scores=None):
    results = df.iloc[indices].copy()
    if scores is not None:
        results["similarity"] = [float(scores[i]) for i in indices]
    results["weighted_rating"] = _weighted_rating(results)
    return results


def _hybrid_query_scores(query, df, model, use_rocchio=True):
    qvec = model["vectorizer"].transform([query])
    tfidf_scores = cosine_similarity(qvec, model["tfidf_matrix"]).ravel()
    bm25_scores = model["bm25"].score_query(query)
    if bm25_scores.max() > 0:
        bm25_scores = bm25_scores / bm25_scores.max()
    lsi_q = qvec @ model["vectorizer"].idf_.reshape(-1, 1) if False else None
    # Project the TF-IDF query into the same latent space learned from documents.
    svd = model.get("svd")
    if svd is not None:
        q_lsi = svd.transform(qvec)
        q_lsi = q_lsi / max(float(np.linalg.norm(q_lsi)), 1e-12)
        lsi_scores = np.clip(model["lsi_matrix"] @ q_lsi.ravel(), 0, 1)
    else:
        lsi_scores = tfidf_scores
    # Report's stated 70% TF-IDF + 30% LSI hybrid; BM25 adds a modest query relevance signal.
    scores = 0.65 * tfidf_scores + 0.25 * lsi_scores + 0.10 * bm25_scores
    if use_rocchio:
        top = np.argsort(scores)[-5:]
        if len(top):
            mean_vec = model["tfidf_matrix"][top].mean(axis=0)
            expanded = qvec + 0.35 * mean_vec
            expanded = np.asarray(expanded)
            expanded_scores = cosine_similarity(expanded, model["tfidf_matrix"]).ravel()
            scores = 0.75 * scores + 0.25 * expanded_scores
    return scores, tfidf_scores


def get_query_results(query, df, model, k=10):
    scores, tfidf_scores = _hybrid_query_scores(query, df, model)
    # Rank by hybrid retrieval score then apply weighted rating as a quality tie-break/reranker.
    candidates = _top_indices(scores, k=min(max(k * 4, k), len(df)))
    if not candidates:
        return df.iloc[[]].copy()
    results = df.iloc[candidates].copy()
    results["retrieval_score"] = scores[candidates]
    results["tfidf_score"] = tfidf_scores[candidates]
    results["weighted_rating"] = _weighted_rating(results)
    results["final_score"] = 0.8 * results["retrieval_score"] + 0.2 * (
        results["weighted_rating"] / 10.0
    )
    results = results.sort_values("final_score", ascending=False).head(k)
    return results


def get_movie_results(query, df, model, k=10):
    matched = find_title_match(query, df)
    if not matched:
        return None, df.iloc[[]].copy()
    idx = model["title_to_index"][matched]
    qvec = model["tfidf_matrix"][idx]
    tfidf_scores = cosine_similarity(qvec, model["tfidf_matrix"]).ravel()
    lsi_scores = np.clip(model["lsi_matrix"] @ model["lsi_matrix"][idx], 0, 1)
    bm25_scores = model["bm25"].score_query(df.iloc[idx]["content"])
    if bm25_scores.max() > 0:
        bm25_scores = bm25_scores / bm25_scores.max()
    scores = 0.65 * tfidf_scores + 0.25 * lsi_scores + 0.10 * bm25_scores
    scores[idx] = -1
    candidate_idx = _top_indices(scores, k=min(max(k * 4, k), len(df)), exclude=idx)
    if not candidate_idx:
        return matched, df.iloc[[]].copy()
    candidates = df.iloc[candidate_idx].copy()
    candidates["retrieval_score"] = scores[candidate_idx]
    candidates["tfidf_score"] = tfidf_scores[candidate_idx]
    candidates["weighted_rating"] = _weighted_rating(candidates)
    candidates["final_score"] = 0.8 * candidates["retrieval_score"] + 0.2 * (
        candidates["weighted_rating"] / 10.0
    )
    # MMR re-ranks candidates to trade off relevance and diversity.
    chosen = mmr_rerank(candidates["final_score"].to_numpy(),
                        model["tfidf_matrix"][candidate_idx], k=k, lambda_param=0.75)
    return matched, candidates.iloc[chosen].copy().reset_index(drop=True)


def mmr_rerank(relevance_scores, candidate_matrix, k=10, lambda_param=0.75):
    """Maximal Marginal Relevance: balance query relevance with result diversity."""
    relevance = np.asarray(relevance_scores, dtype=float)
    if len(relevance) == 0:
        return []
    relevance = (relevance - relevance.min()) / max(relevance.max() - relevance.min(), 1e-12)
    sim = cosine_similarity(candidate_matrix, candidate_matrix)
    selected, remaining = [], set(range(len(relevance)))
    while remaining and len(selected) < min(k, len(relevance)):
        best, best_score = None, -np.inf
        for idx in remaining:
            redundancy = max((sim[idx, chosen] for chosen in selected), default=0.0)
            score = lambda_param * relevance[idx] - (1 - lambda_param) * redundancy
            if score > best_score:
                best, best_score = idx, score
        selected.append(best)
        remaining.remove(best)
    return selected


def personalized_recommendations(title, df, model, k=10):
    matched, results = get_movie_results(title, df, model, k=k)
    return matched, results


def search_movies(query, df, limit=20):
    q = str(query).strip().casefold()
    if not q:
        return df.iloc[[]].copy()
    title_mask = df["title"].astype(str).str.casefold().str.contains(re.escape(q), na=False)
    cast_mask = df["cast_text"].astype(str).str.casefold().str.contains(re.escape(q), na=False)
    director_mask = df["director"].astype(str).str.casefold().str.contains(re.escape(q), na=False)
    genre_mask = df["genres_text"].astype(str).str.casefold().str.contains(re.escape(q), na=False)
    matches = df[title_mask | cast_mask | director_mask | genre_mask].copy()
    if matches.empty:
        fuzzy = find_title_match(query, df)
        if fuzzy:
            matches = df[df["title"] == fuzzy].copy()
    matches["weighted_rating"] = _weighted_rating(matches)
    return matches.sort_values(["vote_count", "weighted_rating"], ascending=False).head(limit)


def get_analytics(df):
    total = len(df)
    avg_rating = float(df["vote_average"].mean()) if total else 0.0
    genres = Counter(g for items in df["genres_list"] for g in items)
    genre_df = pd.DataFrame(genres.most_common(15), columns=["Genre", "Movies"])
    ratings = df["vote_average"].clip(0, 10)
    years = df["release_year"].dropna().astype(int)
    year_df = years.value_counts().sort_index().rename_axis("Year").reset_index(name="Movies")
    directors = df["director"].replace("", np.nan).nunique()
    return {
        "total_movies": total,
        "average_rating": avg_rating,
        "total_genres": len(genres),
        "total_directors": int(directors),
        "genre_df": genre_df,
        "ratings": ratings,
        "year_df": year_df,
    }
