"""
poster_api.py — TMDB Poster Fetching Module

Fetches movie poster images from The Movie Database (TMDB) API.
Used by the Streamlit app to display poster images alongside recommendations.

Usage:
    from poster_api import fetch_poster
    poster_url = fetch_poster(movie_id=550)
"""

import requests
import streamlit as st


# TMDB API base URL and image CDN
TMDB_API_BASE = "https://api.themoviedb.org/3/movie/"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# Fallback placeholder when poster is unavailable
PLACEHOLDER_POSTER = (
    "https://via.placeholder.com/500x750.png"
    "?text=No+Poster+Available"
)


def _get_api_key():
    """
    Retrieve the TMDB API key from Streamlit secrets.

    For local development, create .streamlit/secrets.toml with:
        TMDB_API_KEY = "your_api_key_here"

    For Streamlit Cloud deployment, add the secret via the dashboard.

    Returns:
        str or None: The API key, or None if not configured.
    """
    try:
        return st.secrets["TMDB_API_KEY"]
    except (KeyError, FileNotFoundError):
        return None


def fetch_poster(movie_id):
    """
    Fetch the poster URL for a given TMDB movie ID.

    Args:
        movie_id (int): The TMDB movie ID.

    Returns:
        str: URL of the movie poster image, or a placeholder if unavailable.

    Example:
        >>> fetch_poster(550)  # Fight Club
        'https://image.tmdb.org/t/p/w500/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg'
    """
    api_key = _get_api_key()

    # If no API key is configured, return placeholder
    if not api_key:
        return PLACEHOLDER_POSTER

    try:
        # Make API request to TMDB
        url = f"{TMDB_API_BASE}{movie_id}?api_key={api_key}&language=en-US"
        response = requests.get(url, timeout=5)
        response.raise_for_status()

        data = response.json()
        poster_path = data.get("poster_path")

        # Return full poster URL if available, otherwise placeholder
        if poster_path:
            return f"{TMDB_IMAGE_BASE}{poster_path}"
        else:
            return PLACEHOLDER_POSTER

    except (requests.RequestException, ValueError, KeyError):
        # Gracefully handle any API errors
        return PLACEHOLDER_POSTER
