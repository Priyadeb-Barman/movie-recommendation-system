"""
app.py — Movie Recommendation System (Streamlit Web Application)

A content-based movie recommendation app that suggests similar movies
based on the user's selection. Uses cosine similarity on text features
(genres, keywords, cast, crew, overview) to find the most similar movies.

Usage:
    streamlit run app.py

Author: Your Name
"""

import os
import pickle
import subprocess
import sys

import pandas as pd
import streamlit as st

from poster_api import fetch_poster


# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Movie Recommender | ML Project",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# Custom CSS for Premium UI
# ============================================================
st.markdown("""
<style>
    /* ---- Import Google Font ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ---- Global Styles ---- */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---- Hide Streamlit Defaults ---- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ---- Hero Section ---- */
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0;
        background: linear-gradient(135deg, #E50914, #FF6B6B, #FFA07A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        text-align: center;
        color: #8899A6;
        margin-top: 8px;
        margin-bottom: 40px;
        font-weight: 300;
    }

    /* ---- Movie Card Styles ---- */
    .movie-card {
        background: linear-gradient(145deg, #1A1C2C, #232640);
        border-radius: 16px;
        padding: 12px;
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.05);
        height: 100%;
    }

    .movie-card:hover {
        transform: translateY(-4px);
        border-color: rgba(229, 9, 20, 0.3);
        box-shadow: 0 12px 40px rgba(229, 9, 20, 0.15);
    }

    .movie-card img {
        border-radius: 12px;
        width: 100%;
        aspect-ratio: 2/3;
        object-fit: cover;
    }

    .movie-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #FAFAFA;
        margin-top: 12px;
        line-height: 1.3;
    }

    /* ---- Recommend Button ---- */
    .stButton > button {
        background: linear-gradient(135deg, #E50914, #B20710) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 48px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #FF1A25, #E50914) !important;
        box-shadow: 0 8px 25px rgba(229, 9, 20, 0.4) !important;
        transform: translateY(-2px) !important;
    }

    /* ---- Selectbox Styling ---- */
    .stSelectbox > div > div {
        border-radius: 12px !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
    }

    /* ---- Divider ---- */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(229, 9, 20, 0.3), transparent);
        margin: 40px 0;
        border: none;
    }

    /* ---- Footer ---- */
    .custom-footer {
        text-align: center;
        color: #555;
        font-size: 0.8rem;
        margin-top: 60px;
        padding: 20px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }

    .custom-footer a {
        color: #E50914;
        text-decoration: none;
    }

    /* ---- Tech Badge ---- */
    .tech-badge {
        display: inline-block;
        background: rgba(229, 9, 20, 0.1);
        color: #E50914;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
        margin: 2px;
        border: 1px solid rgba(229, 9, 20, 0.2);
    }

    /* ---- Info Box ---- */
    .info-box {
        background: linear-gradient(145deg, #1A1C2C, #232640);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Model Loading
# ============================================================
MODEL_DIR = "models"
MOVIE_DICT_PATH = os.path.join(MODEL_DIR, "movie_dict.pkl")
SIMILARITY_PATH = os.path.join(MODEL_DIR, "similarity.pkl")


@st.cache_data
def build_model():
    """
    Build the recommendation model by running setup.py.
    This is called only when pickle files are not found.
    """
    st.info(" Building recommendation model for the first time. This may take a minute...")
    result = subprocess.run(
        [sys.executable, "setup.py"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        st.error(f" Model build failed:\n{result.stderr}")
        st.stop()


@st.cache_data
def load_model():
    """
    Load the preprocessed movie data and similarity matrix.

    Returns:
        tuple: (movies DataFrame, similarity matrix numpy array)
    """
    # Auto-build model if pickle files don't exist
    if not os.path.exists(MOVIE_DICT_PATH) or not os.path.exists(SIMILARITY_PATH):
        # Check if data files exist
        if not os.path.exists("data/tmdb_5000_movies.csv"):
            return None, None
        build_model()

    movie_dict = pickle.load(open(MOVIE_DICT_PATH, "rb"))
    movies = pd.DataFrame(movie_dict)
    similarity = pickle.load(open(SIMILARITY_PATH, "rb"))

    return movies, similarity


# ============================================================
# Recommendation Engine
# ============================================================
def recommend(movie_title, movies_df, similarity_matrix):
    """
    Recommend the top 5 similar movies based on cosine similarity.

    Args:
        movie_title (str): Title of the selected movie.
        movies_df (pd.DataFrame): Processed movies dataframe.
        similarity_matrix (np.ndarray): Precomputed cosine similarity matrix.

    Returns:
        tuple: (list of movie titles, list of movie IDs)
    """
    # Find the index of the selected movie
    movie_index = movies_df[movies_df["title"] == movie_title].index[0]

    # Get similarity scores for all movies relative to the selected one
    distances = similarity_matrix[movie_index]

    # Sort by similarity (descending) and get top 6 (skip first = itself)
    similar_movies = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    # Extract titles and IDs
    recommended_titles = []
    recommended_ids = []

    for idx, score in similar_movies:
        recommended_titles.append(movies_df.iloc[idx]["title"])
        recommended_ids.append(movies_df.iloc[idx]["movie_id"])

    return recommended_titles, recommended_ids


# ============================================================
# Main Application UI
# ============================================================
def main():
    """Render the Streamlit application."""

    # ---- Hero Section ----
    st.markdown('<h1 class="hero-title">🎬 Movie Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">'
        'Discover your next favorite movie — powered by Machine Learning & Cosine Similarity'
        '</p>',
        unsafe_allow_html=True
    )

    # ---- Load Model ----
    movies, similarity = load_model()

    # Handle missing data files
    if movies is None:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.error(" Dataset not found!")
        st.markdown("""
        **To get started, follow these steps:**

        1. Download the **TMDB 5000 Movies Dataset** from Kaggle:
           [https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

        2. Place these files in the `data/` directory:
           - `tmdb_5000_movies.csv`
           - `tmdb_5000_credits.csv`

        3. Restart the app — the model will be built automatically!
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    # ---- Movie Selection ----
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        selected_movie = st.selectbox(
            " Type or select a movie you like",
            movies["title"].values,
            index=None,
            placeholder="Search for a movie...",
        )

        # ---- Recommend Button ----
        recommend_clicked = st.button(" Get Recommendations", use_container_width=True)

    # ---- Display Recommendations ----
    if recommend_clicked and selected_movie:
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        with st.spinner(" Finding similar movies..."):
            titles, ids = recommend(selected_movie, movies, similarity)
            posters = [fetch_poster(movie_id) for movie_id in ids]

        # Section header
        st.markdown(
            f'<p style="text-align:center; color:#8899A6; font-size:0.95rem; margin-bottom:24px;">'
            f'Because you liked <strong style="color:#E50914;">{selected_movie}</strong>, '
            f'you might also enjoy:</p>',
            unsafe_allow_html=True
        )

        # Display 5 movie cards in columns
        cols = st.columns(5, gap="medium")

        for i, col in enumerate(cols):
            with col:
                st.markdown(
                    f'''
                    <div class="movie-card">
                        <img src="{posters[i]}" alt="{titles[i]}" />
                        <p class="movie-title">{titles[i]}</p>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

    elif recommend_clicked and not selected_movie:
        st.warning(" Please select a movie first!")

    # ---- How It Works Section ----
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    with st.expander(" How does this work?", expanded=False):
        st.markdown("""
        This recommendation system uses **Content-Based Filtering** with the following pipeline:

        1. **Feature Extraction** — Extracts genres, keywords, cast, crew, and overview from each movie
        2. **Text Preprocessing** — Lowercasing, space removal, and Porter Stemming
        3. **Vectorization** — Converts text tags into numerical vectors using CountVectorizer (Bag of Words)
        4. **Cosine Similarity** — Measures the angle between movie vectors to determine similarity
        5. **Ranking** — Sorts movies by similarity score and returns the top 5

        **Dataset:** TMDB 5000 Movies Dataset from Kaggle
        """)

    
    


if __name__ == "__main__":
    main()
