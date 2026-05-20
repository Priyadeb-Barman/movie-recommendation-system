#  Movie Recommendation System using Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/Scikit--Learn-1.6-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.2-150458?style=for-the-badge&logo=pandas&logoColor=white)
![TMDB](https://img.shields.io/badge/TMDB-API-01D277?style=for-the-badge&logo=themoviedatabase&logoColor=white)

**A content-based movie recommendation web app that suggests similar movies using cosine similarity on text features.**

[Live Demo](#deployment) · [How It Works](#how-it-works) · [Installation](#installation) · [Interview Q&A](docs/INTERVIEW_QA.md)

</div>

---

##  Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [How It Works](#-how-it-works)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Screenshots](#-screenshots)
- [Deployment](#-deployment)
- [Future Scope](#-future-scope)
- [Documentation](#-documentation)
- [Author](#-author)

---

##  Overview

This project is a **Content-Based Movie Recommendation System** that recommends movies similar to a user-selected movie title. It analyzes movie metadata (genres, keywords, cast, crew, and overview) to find patterns and suggest the top 5 most similar movies.

Built using the **TMDB 5000 Movies Dataset** from Kaggle, this application demonstrates practical machine learning concepts including text preprocessing, feature engineering, vectorization, and similarity computation — all wrapped in a clean Streamlit web interface.

---

##  Features

| Feature | Description |
|---------|-------------|
|  **Smart Search** | Search/select from 4800+ movies with autocomplete |
|  **Top 5 Recommendations** | Get 5 similar movies ranked by cosine similarity |
|  **Movie Posters** | Beautiful poster display via TMDB API integration |
|  **ML-Powered** | Content-based filtering using CountVectorizer + Cosine Similarity |
|  **Premium UI** | Dark-themed Streamlit interface with custom styling |
|  **Fast Response** | Pre-computed similarity matrix for instant recommendations |
|  **Auto Model Build** | Automatically builds the model on first run |
|  **Responsive** | Works on desktop and mobile browsers |

---

##  Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.10+** | Core programming language |
| **Pandas** | Data manipulation and preprocessing |
| **NumPy** | Numerical computations |
| **Scikit-learn** | CountVectorizer and cosine similarity |
| **NLTK** | Text preprocessing (Porter Stemmer) |
| **Streamlit** | Web application framework |
| **Requests** | TMDB API calls for movie posters |
| **Pickle** | Model serialization |

---

##  How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. LOAD DATA                                               │
│     tmdb_5000_movies.csv + tmdb_5000_credits.csv            │
│                          │                                  │
│  2. MERGE               ▼                                   │
│     Join on 'title' column                                  │
│                          │                                  │
│  3. EXTRACT FEATURES     ▼                                  │
│     genres + keywords + top 3 cast + director + overview    │
│                          │                                  │
│  4. PREPROCESS           ▼                                  │
│     Lowercase → Remove spaces → Porter Stemming             │
│                          │                                  │
│  5. VECTORIZE            ▼                                  │
│     CountVectorizer (5000 features, English stop words)     │
│                          │                                  │
│  6. SIMILARITY           ▼                                  │
│     Cosine Similarity Matrix (4800 × 4800)                  │
│                          │                                  │
│  7. RECOMMEND            ▼                                  │
│     Sort by similarity → Return top 5 movies                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Concepts

- **Content-Based Filtering**: Recommends movies with similar content (metadata), not based on user behavior
- **Bag of Words (CountVectorizer)**: Converts text into numerical feature vectors based on word frequency
- **Cosine Similarity**: Measures the angle between two vectors — closer angle means more similar movies
- **Porter Stemming**: Reduces words to their root form ("loving" → "love") for better matching

---

##  Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/movie-recommendation-system.git
cd movie-recommendation-system

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset from Kaggle
#    URL: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
#    Place these files in the data/ directory:
#      - tmdb_5000_movies.csv
#      - tmdb_5000_credits.csv

# 5. Build the recommendation model
python setup.py

# 6. Run the Streamlit app
streamlit run app.py
```

### Optional: Set Up Movie Posters

To display movie posters, you need a free TMDB API key:

1. Create an account at [themoviedb.org](https://www.themoviedb.org/)
2. Go to Settings → API → Request API Key
3. Create `.streamlit/secrets.toml`:

```toml
TMDB_API_KEY = "your_api_key_here"
```

---

##  Usage

1. **Open the app** in your browser (usually `http://localhost:8501`)
2. **Search for a movie** using the dropdown/search box
3. **Click "Get Recommendations"** to see 5 similar movies
4. **Explore results** — each recommendation shows the movie poster and title

---

##  Project Structure

```
movie-recommendation-system/
│
├── app.py                          #  Streamlit web application
├── poster_api.py                   #  TMDB poster fetching module
├── setup.py                        #  Data preprocessing & model builder
├── requirements.txt                #  Python dependencies
├── README.md                       #  Project documentation
├── .gitignore                      #  Git ignore rules
│
├── .streamlit/
│   └── config.toml                 #  Streamlit theme configuration
│
├── data/
│   ├── tmdb_5000_movies.csv        #  Movies dataset (from Kaggle)
│   └── tmdb_5000_credits.csv       #  Credits dataset (from Kaggle)
│
├── models/
│   ├── movie_dict.pkl              #  Processed movie data
│   └── similarity.pkl              #  Cosine similarity matrix
│
├── notebooks/
│   └── movie_recommender.py        #  Data exploration script
│
├── docs/
│   ├── INTERVIEW_QA.md             #  Interview questions & answers
│   ├── ARCHITECTURE.md             #  System architecture
│   ├── DEPLOYMENT.md               #  Deployment guide
│   └── RESUME_DESCRIPTION.md       #  Resume-ready descriptions
│
├── screenshots/                    #  UI screenshots
└── utils/
    └── __init__.py                 #  Utility functions
```

---

##  Screenshots

> **Note:** Add screenshots of your running application here.

| Home Page | Recommendations |
|-----------|-----------------|
| *Screenshot of the homepage with movie selector* | *Screenshot showing 5 recommended movies with posters* |

To take screenshots:
1. Run the app: `streamlit run app.py`
2. Take screenshots of the UI
3. Save them in the `screenshots/` directory
4. Update the image paths above

---

##  Deployment

### Streamlit Community Cloud (Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Click "New app" → Select your repository
4. Set main file path to `app.py`
5. Add your TMDB API key in Settings → Secrets
6. Click "Deploy"

For detailed instructions, see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

---

##  Future Scope

- [ ] **Collaborative Filtering** — Add user-based recommendations using Matrix Factorization
- [ ] **Hybrid Model** — Combine content-based and collaborative filtering
- [ ] **More Features** — Include runtime, release year, and ratings in similarity
- [ ] **User Authentication** — Let users save their favorites and history
- [ ] **Movie Details** — Show rating, genre, year, and overview in results
- [ ] **Search History** — Track and display recent searches
- [ ] **Genre Filter** — Filter recommendations by specific genres
- [ ] **TF-IDF Vectorizer** — Compare results with TF-IDF instead of CountVectorizer
- [ ] **Deep Learning** — Experiment with neural collaborative filtering

---

##  Documentation

| Document | Description |
|----------|-------------|
| [Interview Q&A](docs/INTERVIEW_QA.md) | 25+ interview questions with detailed answers |
| [Architecture](docs/ARCHITECTURE.md) | System design and data flow diagrams |
| [Deployment Guide](docs/DEPLOYMENT.md) | Step-by-step deployment instructions |
| [Resume Description](docs/RESUME_DESCRIPTION.md) | Resume bullet points and interview pitches |

---

##  Author

**Priyadeb Barman**

- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Name](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

##  License

This project is open source and available under the [MIT License](LICENSE).

---

##  Acknowledgments

- **Dataset**: [TMDB 5000 Movies Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) by Kaggle
- **Poster API**: [The Movie Database (TMDB)](https://www.themoviedb.org/)
- **Framework**: [Streamlit](https://streamlit.io/)
- **Inspiration**: Content-based recommendation systems in production

---

<div align="center">

 **Star this repository if you found it helpful!** 

</div>
