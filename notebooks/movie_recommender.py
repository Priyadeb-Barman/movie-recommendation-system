"""
Movie Recommendation System — Data Exploration & Preprocessing Notebook
========================================================================

This script walks through the complete data exploration and ML pipeline
step by step. It's designed to be converted into a Jupyter notebook.

To convert to a Jupyter notebook in VS Code:
    1. Open this file
    2. Right-click → "Open in Jupyter Notebook"
    Or use: jupytext --to notebook movie_recommender.py

Sections:
    1. Import Libraries
    2. Load Datasets
    3. Exploratory Data Analysis (EDA)
    4. Data Preprocessing
    5. Feature Engineering
    6. Text Vectorization
    7. Cosine Similarity
    8. Recommendation Function
    9. Testing Recommendations
"""

# %% [markdown]
# # 🎬 Movie Recommendation System using Machine Learning
# ---
# A content-based recommendation system that suggests similar movies
# based on movie metadata (genres, keywords, cast, crew, overview).

# %% [markdown]
# ## 1. Import Libraries

# %%
import ast
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# %% [markdown]
# ## 2. Load Datasets
# We use the TMDB 5000 Movies Dataset which consists of two CSV files:
# - `tmdb_5000_movies.csv` — Movie metadata (genres, keywords, overview, etc.)
# - `tmdb_5000_credits.csv` — Cast and crew information

# %%
# Load the datasets
movies = pd.read_csv("../data/tmdb_5000_movies.csv")
credits = pd.read_csv("../data/tmdb_5000_credits.csv")

# %% [markdown]
# ### Quick look at the data

# %%
print("Movies Dataset Shape:", movies.shape)
print("Credits Dataset Shape:", credits.shape)

# %%
movies.head(2)

# %%
credits.head(2)

# %%
# Check column names
print("Movies columns:", movies.columns.tolist())
print("\nCredits columns:", credits.columns.tolist())

# %% [markdown]
# ## 3. Exploratory Data Analysis (EDA)

# %%
# Check for missing values in movies dataset
print("Missing values in Movies dataset:")
print(movies.isnull().sum())

# %%
# Check for duplicates
print(f"\nDuplicate movies: {movies.duplicated().sum()}")
print(f"Duplicate credits: {credits.duplicated().sum()}")

# %%
# Basic statistics
print(f"\nTotal movies: {len(movies)}")
print(f"Average vote: {movies['vote_average'].mean():.2f}")
print(f"Most popular movie: {movies.loc[movies['popularity'].idxmax(), 'title']}")

# %% [markdown]
# ## 4. Data Preprocessing
# ### 4.1 Merge datasets on 'title'

# %%
# Merge movies and credits on title
movies = movies.merge(credits, on="title")

print(f"Shape after merge: {movies.shape}")

# %%
# Keep only relevant columns for our recommendation system
movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
movies.head()

# %%
# Check for missing values after filtering
print("Missing values:")
print(movies.isnull().sum())

# %%
# Drop rows with missing overview
movies.dropna(inplace=True)
print(f"Shape after dropping NaN: {movies.shape}")

# %% [markdown]
# ### 4.2 Parse JSON columns
# The genres, keywords, cast, and crew columns contain JSON strings.
# We need to extract the relevant names from them.

# %%
# Let's look at what a genres entry looks like
print("Sample genres entry:")
print(movies.iloc[0]["genres"])

# %%
# Function to extract names from JSON-like strings
def parse_json_column(text):
    """Extract 'name' values from a JSON-like string."""
    try:
        items = ast.literal_eval(text)
        return [item["name"] for item in items]
    except (ValueError, KeyError):
        return []

# Apply to genres and keywords
movies["genres"] = movies["genres"].apply(parse_json_column)
movies["keywords"] = movies["keywords"].apply(parse_json_column)

# %%
# Let's see the parsed genres
print("Parsed genres for Avatar:")
print(movies.iloc[0]["genres"])

# %% [markdown]
# ### 4.3 Extract top 3 cast members

# %%
def extract_top_cast(text, n=3):
    """Extract top N cast members."""
    try:
        cast = ast.literal_eval(text)
        return [member["name"] for member in cast[:n]]
    except (ValueError, KeyError):
        return []

movies["cast"] = movies["cast"].apply(extract_top_cast)

# %%
# Let's see the cast for Avatar
print("Top 3 cast for Avatar:")
print(movies.iloc[0]["cast"])

# %% [markdown]
# ### 4.4 Extract director from crew

# %%
def extract_director(text):
    """Extract only the director from the crew."""
    try:
        crew = ast.literal_eval(text)
        for member in crew:
            if member.get("job") == "Director":
                return [member["name"]]
        return []
    except (ValueError, KeyError):
        return []

movies["crew"] = movies["crew"].apply(extract_director)

# %%
# Let's see the director for Avatar
print("Director of Avatar:")
print(movies.iloc[0]["crew"])

# %% [markdown]
# ## 5. Feature Engineering
# ### 5.1 Create the 'tags' column
# We combine all text features into a single 'tags' column.

# %%
# Convert overview to list of words
movies["overview"] = movies["overview"].apply(lambda x: x.split())

# %%
# Remove spaces from multi-word names to prevent collisions
# "Sam Mendes" → "sammendes" (so it doesn't match "Sam Worthington")
for col in ["genres", "keywords", "cast", "crew"]:
    movies[col] = movies[col].apply(
        lambda items: [name.replace(" ", "").lower() for name in items]
    )

# %%
# Combine all features into tags
movies["tags"] = (
    movies["overview"]
    + movies["genres"]
    + movies["keywords"]
    + movies["cast"]
    + movies["crew"]
)

# %%
# Create a new clean dataframe
new_df = movies[["movie_id", "title", "tags"]].copy()

# Join tags list into a string
new_df.loc[:, "tags"] = new_df["tags"].apply(lambda x: " ".join(x).lower())

# %%
# Let's see what the tags look like for Avatar
print("Tags for Avatar (first 300 chars):")
print(new_df.iloc[0]["tags"][:300])

# %% [markdown]
# ### 5.2 Apply Stemming
# Stemming reduces words to their root form:
# - "loving" → "love"
# - "dancing" → "danc"
# - "action" → "action"

# %%
ps = PorterStemmer()

def stem_text(text):
    """Apply Porter Stemming to text."""
    return " ".join([ps.stem(word) for word in text.split()])

new_df.loc[:, "tags"] = new_df["tags"].apply(stem_text)

# %%
# Let's see the stemmed tags for Avatar
print("Stemmed tags for Avatar (first 300 chars):")
print(new_df.iloc[0]["tags"][:300])

# %% [markdown]
# ## 6. Text Vectorization (Bag of Words)
# We use CountVectorizer to convert text into numerical vectors.
# Each movie becomes a vector of 5000 dimensions (word frequencies).

# %%
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(new_df["tags"]).toarray()

print(f"Vectors shape: {vectors.shape}")
print(f"Number of features (unique words): {len(cv.get_feature_names_out())}")

# %%
# Let's see some of the feature names (words)
print("Sample feature names:")
print(cv.get_feature_names_out()[:20])

# %% [markdown]
# ## 7. Cosine Similarity
# Cosine similarity measures the angle between two vectors.
# - 1.0 = identical direction (very similar)
# - 0.0 = perpendicular (no similarity)

# %%
similarity = cosine_similarity(vectors)
print(f"Similarity matrix shape: {similarity.shape}")

# %%
# Let's check the similarity between Avatar (index 0) and itself
print(f"Avatar vs Avatar: {similarity[0][0]:.4f}")  # Should be 1.0

# %% [markdown]
# ## 8. Recommendation Function

# %%
def recommend(movie_title):
    """
    Recommend top 5 similar movies.

    Args:
        movie_title: Title of the movie to find recommendations for.

    Returns:
        List of 5 recommended movie titles.
    """
    # Find the index of the movie
    try:
        movie_index = new_df[new_df["title"] == movie_title].index[0]
    except IndexError:
        print(f"❌ Movie '{movie_title}' not found in the dataset.")
        return []

    # Get similarity scores
    distances = similarity[movie_index]

    # Sort by similarity (descending), skip first (itself)
    similar_movies = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    # Print recommendations
    print(f"\n🎬 Movies similar to '{movie_title}':\n")
    recommendations = []
    for rank, (idx, score) in enumerate(similar_movies, 1):
        title = new_df.iloc[idx]["title"]
        print(f"  {rank}. {title} (similarity: {score:.4f})")
        recommendations.append(title)

    return recommendations

# %% [markdown]
# ## 9. Testing Recommendations

# %%
# Test with popular movies
recommend("Avatar")

# %%
recommend("The Dark Knight")

# %%
recommend("Inception")

# %%
recommend("Titanic")

# %%
recommend("The Avengers")

# %% [markdown]
# ## 10. Save Model (Optional)
# This is handled by setup.py, but shown here for reference.

# %%
import pickle
import os

# os.makedirs("../models", exist_ok=True)
# pickle.dump(new_df.to_dict(), open("../models/movie_dict.pkl", "wb"))
# pickle.dump(similarity, open("../models/similarity.pkl", "wb"))
# print("✅ Model saved!")

# %% [markdown]
# ## Summary
#
# **Pipeline:**
# 1. Load & merge TMDB datasets (movies + credits)
# 2. Extract features: genres, keywords, top 3 cast, director, overview
# 3. Create combined "tags" column
# 4. Apply Porter Stemming for word normalization
# 5. Vectorize using CountVectorizer (5000 features, English stop words removed)
# 6. Compute cosine similarity matrix
# 7. Recommend top 5 movies by highest similarity score
#
# **Key Concepts:**
# - Content-Based Filtering
# - Bag of Words (CountVectorizer)
# - Cosine Similarity
# - Text Preprocessing (Stemming)
# - Feature Engineering
