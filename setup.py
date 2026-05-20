"""
setup.py — Data Preprocessing & Model Building Script

This script processes the TMDB 5000 Movies dataset and builds the
recommendation model (cosine similarity matrix).

Steps:
    1. Load and merge movie + credits datasets
    2. Extract relevant features (genres, keywords, cast, crew, overview)
    3. Preprocess text data (lowercase, remove spaces, stemming)
    4. Vectorize using CountVectorizer
    5. Compute cosine similarity matrix
    6. Save processed data and similarity matrix as pickle files

Usage:
    python setup.py

Prerequisites:
    Place these files in the data/ directory:
        - tmdb_5000_movies.csv
        - tmdb_5000_credits.csv

    Download from: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
"""

import os
import ast
import pickle

import numpy as np
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Configuration
# ============================================================
DATA_DIR = "data"
MODEL_DIR = "models"
MOVIES_FILE = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
CREDITS_FILE = os.path.join(DATA_DIR, "tmdb_5000_credits.csv")
MOVIE_DICT_PATH = os.path.join(MODEL_DIR, "movie_dict.pkl")
SIMILARITY_PATH = os.path.join(MODEL_DIR, "similarity.pkl")

# CountVectorizer settings
MAX_FEATURES = 5000
STOP_WORDS = "english"


def ensure_nltk_data():
    """Download required NLTK data if not already present."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("[INFO] Downloading NLTK punkt tokenizer...")
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        print("[INFO] Downloading NLTK punkt_tab tokenizer...")
        nltk.download("punkt_tab", quiet=True)


def load_data():
    """
    Load and merge the movies and credits datasets.

    Returns:
        pd.DataFrame: Merged dataframe with relevant columns.
    """
    print("[1/6] Loading datasets...")

    if not os.path.exists(MOVIES_FILE):
        raise FileNotFoundError(
            f"'{MOVIES_FILE}' not found!\n"
            f"Please download the TMDB 5000 Movies Dataset from Kaggle:\n"
            f"https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata\n"
            f"and place the CSV files in the '{DATA_DIR}/' directory."
        )

    if not os.path.exists(CREDITS_FILE):
        raise FileNotFoundError(
            f"'{CREDITS_FILE}' not found!\n"
            f"Please download the TMDB 5000 Movies Dataset from Kaggle:\n"
            f"https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata\n"
            f"and place the CSV files in the '{DATA_DIR}/' directory."
        )

    movies = pd.read_csv(MOVIES_FILE)
    credits = pd.read_csv(CREDITS_FILE)

    # Merge on title column
    movies = movies.merge(credits, on="title")

    # Keep only the columns we need for recommendation
    movies = movies[[
        "movie_id", "title", "overview",
        "genres", "keywords", "cast", "crew"
    ]]

    # Drop rows with missing values
    movies.dropna(inplace=True)

    print(f" Loaded {len(movies)} movies")
    return movies


def parse_json_column(text):
    """
    Parse a JSON-like string column and extract the 'name' field.

    Args:
        text (str): JSON string (e.g., '[{"id": 28, "name": "Action"}]')

    Returns:
        list: List of name strings (e.g., ['Action'])
    """
    try:
        items = ast.literal_eval(text)
        return [item["name"] for item in items]
    except (ValueError, KeyError):
        return []


def extract_top_cast(text, n=3):
    """
    Extract the top N cast members from the cast JSON string.

    Args:
        text (str): JSON string containing cast information.
        n (int): Number of top cast members to extract.

    Returns:
        list: List of top cast member names.
    """
    try:
        cast = ast.literal_eval(text)
        return [member["name"] for member in cast[:n]]
    except (ValueError, KeyError):
        return []


def extract_director(text):
    """
    Extract the director's name from the crew JSON string.

    Args:
        text (str): JSON string containing crew information.

    Returns:
        list: Single-element list with director's name, or empty list.
    """
    try:
        crew = ast.literal_eval(text)
        for member in crew:
            if member.get("job") == "Director":
                return [member["name"]]
        return []
    except (ValueError, KeyError):
        return []


def preprocess_features(movies):
    """
    Extract and preprocess features from the merged dataset.

    Transforms JSON columns into clean lists and creates a
    combined 'tags' column for vectorization.

    Args:
        movies (pd.DataFrame): Merged movies dataframe.

    Returns:
        pd.DataFrame: Processed dataframe with 'tags' column.
    """
    print("[2/6] Extracting features...")

    # Parse JSON columns
    movies["genres"] = movies["genres"].apply(parse_json_column)
    movies["keywords"] = movies["keywords"].apply(parse_json_column)
    movies["cast"] = movies["cast"].apply(extract_top_cast)
    movies["crew"] = movies["crew"].apply(extract_director)

    # Convert overview from string to list of words
    movies["overview"] = movies["overview"].apply(lambda x: x.split())

    print("[3/6] Preprocessing text...")

    # Remove spaces from multi-word names to avoid collisions
    # e.g., "Sam Mendes" → "sammendes" so it's treated as one token
    # This prevents "Sam" from matching "Sam Worthington"
    for col in ["genres", "keywords", "cast", "crew"]:
        movies[col] = movies[col].apply(
            lambda items: [name.replace(" ", "").lower() for name in items]
        )

    # Create combined tags column
    movies["tags"] = (
        movies["overview"]
        + movies["genres"]
        + movies["keywords"]
        + movies["cast"]
        + movies["crew"]
    )

    # Keep only necessary columns
    new_df = movies[["movie_id", "title", "tags"]].copy()

    # Join tags list into a single lowercase string
    new_df.loc[:, "tags"] = new_df["tags"].apply(
        lambda x: " ".join(x).lower()
    )

    return new_df


def apply_stemming(df):
    """
    Apply Porter Stemming to the tags column.

    Stemming reduces words to their root form:
        'loving' → 'love', 'dancing' → 'danc'

    This helps match similar words during vectorization.

    Args:
        df (pd.DataFrame): Dataframe with 'tags' column.

    Returns:
        pd.DataFrame: Dataframe with stemmed tags.
    """
    print("[4/6] Applying stemming...")

    ps = PorterStemmer()

    def stem_text(text):
        return " ".join([ps.stem(word) for word in text.split()])

    df.loc[:, "tags"] = df["tags"].apply(stem_text)
    return df


def build_similarity_matrix(df):
    """
    Build the cosine similarity matrix using CountVectorizer.

    Steps:
        1. Convert tags text to feature vectors using CountVectorizer
        2. Compute pairwise cosine similarity between all movies

    Args:
        df (pd.DataFrame): Dataframe with processed 'tags' column.

    Returns:
        np.ndarray: Cosine similarity matrix of shape (n_movies, n_movies).
    """
    print("[5/6] Building similarity matrix...")

    # Vectorize the tags using Bag of Words (CountVectorizer)
    cv = CountVectorizer(max_features=MAX_FEATURES, stop_words=STOP_WORDS)
    vectors = cv.fit_transform(df["tags"]).toarray()

    print(f"     Feature matrix shape: {vectors.shape}")

    # Compute cosine similarity between all movie vectors
    similarity = cosine_similarity(vectors)

    print(f"     Similarity matrix shape: {similarity.shape}")
    return similarity


def save_model(df, similarity):
    """
    Save the processed movie data and similarity matrix as pickle files.

    Args:
        df (pd.DataFrame): Processed movies dataframe.
        similarity (np.ndarray): Cosine similarity matrix.
    """
    print("[6/6] Saving model files...")

    # Create models directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save movie data as dictionary for fast lookup
    pickle.dump(df.to_dict(), open(MOVIE_DICT_PATH, "wb"))
    print(f"     Saved: {MOVIE_DICT_PATH}")

    # Save similarity matrix
    pickle.dump(similarity, open(SIMILARITY_PATH, "wb"))
    print(f"     Saved: {SIMILARITY_PATH}")


def main():
    """Main pipeline: Load → Process → Vectorize → Save."""
    print("=" * 55)
    print("  Movie Recommendation System — Model Builder")
    print("=" * 55)
    print()

    # Ensure NLTK data is available
    ensure_nltk_data()

    # Step 1: Load and merge datasets
    movies = load_data()

    # Step 2-3: Extract and preprocess features
    processed_df = preprocess_features(movies)

    # Step 4: Apply stemming
    processed_df = apply_stemming(processed_df)

    # Step 5: Build cosine similarity matrix
    similarity = build_similarity_matrix(processed_df)

    # Step 6: Save model files
    save_model(processed_df, similarity)

    print()
    print(" Model built successfully!")
    print(f"   Movies processed: {len(processed_df)}")
    print(f"   Model files saved to: {MODEL_DIR}/")
    print()
    print("Next step: Run the Streamlit app with:")
    print("   streamlit run app.py")


if __name__ == "__main__":
    main()
