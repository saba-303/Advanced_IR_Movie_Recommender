# Advanced IR-Based Movie Recommender System

A Streamlit application demonstrating content-based movie retrieval and recommendation using classical Information Retrieval techniques.

## Features
- Search by title, actor, director, genre, or descriptive query
- TF-IDF and cosine-similarity retrieval
- BM25 keyword ranking
- Latent Semantic Indexing (LSI) using TruncatedSVD
- Rocchio-style query expansion from top retrieved documents
- Hybrid ranking combining TF-IDF, LSI, and BM25 scores
- Maximal Marginal Relevance (MMR) for diversity in movie recommendations
- Fuzzy title matching for spelling errors
- IMDb-style weighted rating for quality-aware re-ranking
- Analytics dashboard with movie counts, genres, ratings, directors, and release years
- Random Discovery and session-based favorites
- Movie cards with overview, genres, cast, director, runtime, and rating

## Dataset
Download **The Movies Dataset** from Kaggle:
https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

Place these files in the same directory as `app.py`:
- `movies_metadata.csv`
- `credits.csv`
- `keywords.csv`

The large `credits.csv` dataset is intentionally excluded from Git tracking. Download it separately. Dataset files are not included in this source archive.

## Setup (Linux/macOS)
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, usually http://localhost:8501.

## Windows
```powershell
py -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Implementation notes
- Title, director, and cast terms are repeated in the feature text to weight them more heavily.
- TF-IDF and LSI scores are combined with normalized BM25 scores for hybrid retrieval.
- Rocchio-style query expansion adds a weighted centroid of top retrieved documents to the query vector.
- MMR trades off relevance and redundancy when selecting similar-movie recommendations.
- IMDb-style weighted ratings are used as a quality signal; they do not replace relevance scoring.
- Favorites are stored in Streamlit session state and reset when the session ends.

## Project structure
```text
app.py
recommender.py
requirements.txt
README.md
.gitignore
movies_metadata.csv   # downloaded separately
credits.csv           # downloaded separately; excluded from Git
keywords.csv           # downloaded separately
```

## Attribution
This project is based on an existing public IR movie-recommender repository. Preserve the original attribution and document your own modifications appropriately.
