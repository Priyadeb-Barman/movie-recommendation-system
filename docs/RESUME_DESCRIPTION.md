# 🎬 Movie Recommendation System — Resume-Ready Project Description

> A ready-to-use reference for resumes, LinkedIn, GitHub, cover letters, and interviews.
> Copy-paste the version that fits your need.

---

## 1. 📝 Resume Bullet Points

### Short (1 line) — For space-constrained resumes

> **Built** a content-based movie recommendation system using Python and Scikit-learn that suggests similar movies from a dataset of 4,800+ titles using cosine similarity on text features.

### Medium (2 lines) — For standard resume format

> **Developed** a content-based movie recommendation system using Python, Pandas, and Scikit-learn, processing 4,800+ movies from the TMDB dataset. Engineered text features from genres, keywords, cast, crew, and overview, applied Porter Stemming and CountVectorizer (5,000 features), and computed cosine similarity to power a Streamlit web app deployed on the cloud.

### Detailed (3–4 lines) — For project-focused resume sections

> **Engineered** an end-to-end content-based movie recommendation system processing the TMDB 5000 Movies dataset. Extracted and combined features from genres, keywords, top 3 cast members, director, and plot overview into a unified text representation. Applied Porter Stemming for text normalization and CountVectorizer (5,000 features) to build a Bag-of-Words model, then computed pairwise cosine similarity across all movies. Built an interactive Streamlit web application with TMDB API integration for real-time poster display, and deployed it on Streamlit Community Cloud.

---

## 2. 💼 LinkedIn Project Description

**Paste this directly into LinkedIn → Projects section.**

---

**Movie Recommendation System**

Built a content-based movie recommendation system that suggests similar movies based on metadata analysis of 4,800+ titles from the TMDB 5000 Movies dataset.

**What it does:** Select any movie, and the system recommends the 5 most similar movies — complete with poster images fetched in real time from the TMDB API.

**How it works:**
- Extracted features from genres, keywords, top 3 cast members, director, and plot overview
- Applied Porter Stemming for text normalization and CountVectorizer (Bag of Words, 5,000 features) for vectorization
- Computed pairwise cosine similarity to rank and surface the most relevant recommendations

**Tech stack:** Python · Pandas · NumPy · Scikit-learn · NLTK · Streamlit · TMDB API

**Deployed on:** Streamlit Community Cloud

---

## 3. 🔗 GitHub Repository Description

**One-liner for the GitHub repo description field (max ~350 chars):**

> A content-based movie recommendation system built with Python and Scikit-learn. Uses CountVectorizer and cosine similarity on TMDB 5000 movie metadata (genres, keywords, cast, crew, overview) to suggest similar movies via a Streamlit web app with real-time poster display.

**Shorter version (under 150 chars):**

> Content-based movie recommendation system using cosine similarity on TMDB movie metadata. Built with Scikit-learn & Streamlit.

---

## 4. 🛠️ Skills to Highlight

### Technical Skills Demonstrated

| Skill Area | Specific Skills | Maps to Job Requirement |
|---|---|---|
| **Programming** | Python, modular code structure, OOP principles | Python development, scripting |
| **Data Processing** | Pandas, NumPy, JSON parsing (`ast.literal_eval`), data cleaning | Data wrangling, ETL pipelines |
| **Machine Learning** | Content-based filtering, CountVectorizer (BoW), cosine similarity | ML fundamentals, NLP basics |
| **NLP / Text Processing** | Tokenization, Porter Stemming, stop word removal, feature engineering | Text preprocessing, NLP |
| **Web Development** | Streamlit (frontend + backend), REST API integration (TMDB API) | Full-stack, API integration |
| **Deployment** | Streamlit Community Cloud, pickle serialization, caching (`@st.cache_data`) | Cloud deployment, MLOps basics |
| **Software Engineering** | Separation of concerns (setup/app/api modules), error handling, docstrings | Clean code, production readiness |
| **Version Control** | Git, `.gitignore`, project structuring | Git, collaboration |

### Keywords for ATS (Applicant Tracking Systems)

```
Python, Pandas, NumPy, Scikit-learn, NLTK, Streamlit, Machine Learning,
Natural Language Processing, Content-Based Filtering, Cosine Similarity,
CountVectorizer, Bag of Words, Feature Engineering, Text Preprocessing,
REST API, Data Preprocessing, Model Deployment, Cloud Deployment
```

---

## 5. 🎤 How to Talk About It in Interviews

### 30-Second Elevator Pitch

> "I built a content-based movie recommendation system using the TMDB 5000 Movies dataset. The core idea is straightforward — I combined metadata like genres, keywords, cast, crew, and plot overview into a single text representation for each movie, then used CountVectorizer and cosine similarity to find the most similar movies. I wrapped the whole thing in a Streamlit web app where users can select a movie and instantly see the top 5 recommendations with posters. It's deployed on Streamlit Community Cloud."

### 2-Minute Detailed Explanation

> "The goal was to build a recommendation system that doesn't rely on user ratings or collaborative filtering — instead, it uses the movie's own content to find similar titles.
>
> **Data:** I used the TMDB 5000 Movies dataset, which has two CSV files — one with movie metadata and one with cast/crew information. After merging them, I had about 4,800 movies to work with.
>
> **Feature Engineering:** This was the most important part. I extracted genres and keywords from JSON columns, pulled out the top 3 cast members and the director from nested data structures, and split the overview into tokens. I removed spaces from multi-word names like 'Sam Mendes' to prevent false matches with 'Sam Worthington' — small details like these matter for accuracy.
>
> **Text Processing:** I combined all five feature types into a single 'tags' string per movie, applied Porter Stemming so words like 'loving' and 'loved' are treated as the same root word, then used CountVectorizer with 5,000 features and English stop word removal to create a Bag-of-Words representation.
>
> **Similarity:** I computed pairwise cosine similarity, which gives a score between 0 and 1 for every pair of movies. When a user selects a movie, I sort all other movies by their similarity score and return the top 5.
>
> **Deployment:** The frontend is a Streamlit app. It loads the precomputed similarity matrix from pickle files, and fetches movie posters in real time via the TMDB API. I deployed it on Streamlit Community Cloud so anyone can try it."

### What to Emphasize by Role

#### For ML Engineer / Data Scientist roles:
- The **content-based filtering approach** and why you chose it over collaborative filtering (no cold-start problem, works without user interaction history)
- **Feature engineering decisions** — why top 3 cast, why director only, why remove spaces from names
- **CountVectorizer vs. TF-IDF** — you chose CountVectorizer because all features are equally important; TF-IDF would down-weight common genres
- **Cosine similarity vs. Euclidean distance** — cosine is better for high-dimensional sparse text vectors because it measures direction, not magnitude

#### For Software Development (SDE) roles:
- **Modular architecture** — separate files for setup pipeline (`setup.py`), web app (`app.py`), and API integration (`poster_api.py`)
- **Error handling** — graceful fallbacks when API keys are missing or posters are unavailable
- **Caching** — `@st.cache_data` to avoid reloading the model on every interaction
- **Auto-build pipeline** — the app detects missing model files and triggers `setup.py` automatically

#### For Data Analyst / Business Analyst roles:
- **Data cleaning and merging** — handling missing values, merging two datasets on a common key
- **JSON parsing** — extracting structured data from nested JSON columns
- **End-user focus** — the web app makes the ML model accessible to non-technical users

---

## 6. 📨 Project Summary for Cover Letter

### Version A — Technical focus

> As part of my ML portfolio, I built a content-based movie recommendation system that processes the TMDB 5000 Movies dataset. I engineered text features from movie metadata, applied NLP techniques like stemming and Bag-of-Words vectorization, and used cosine similarity to power an interactive Streamlit web application — demonstrating end-to-end ML project execution from data processing to cloud deployment.

### Version B — Impact focus

> I developed and deployed a movie recommendation web application that suggests similar movies from a catalog of 4,800+ titles. The project involved building a complete ML pipeline — from feature engineering and text preprocessing to model deployment on the cloud — which strengthened my practical understanding of content-based filtering and NLP fundamentals.

---

## 7. 📊 Quick-Reference Project Card

| Field | Details |
|---|---|
| **Project** | Movie Recommendation System |
| **Type** | Content-Based Filtering (ML) |
| **Dataset** | TMDB 5000 Movies (4,800+ movies after cleaning) |
| **Features Used** | Genres, keywords, top 3 cast, director, overview |
| **ML Technique** | CountVectorizer (5,000 features) → Cosine Similarity |
| **NLP** | Porter Stemming, stop word removal, tokenization |
| **Frontend** | Streamlit web app with TMDB API poster integration |
| **Deployment** | Streamlit Community Cloud |
| **Tech Stack** | Python, Pandas, NumPy, Scikit-learn, NLTK, Streamlit |
| **Architecture** | Modular — `setup.py` (pipeline), `app.py` (UI), `poster_api.py` (API) |

---

> [!TIP]
> **Usage advice:** Pick the version that matches the space you have. For a one-page resume, use the short bullet point. For a two-page resume or a projects section, use the medium or detailed version. Always tailor the emphasis to the role you're applying for.
