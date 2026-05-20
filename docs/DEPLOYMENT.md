# 🚀 Deployment & Setup Guide — Movie Recommendation System

> **A complete, beginner-friendly guide to set up, run, and deploy the Movie Recommendation System.**
> Follow each section step by step — every command is copy-paste ready.

---

## 📑 Table of Contents

| # | Section | Description |
|---|---------|-------------|
| 1 | [Local Setup](#-1-local-setup-step-by-step) | Get the app running on your machine |
| 2 | [GitHub Upload](#-2-github-upload-steps) | Push your project to GitHub |
| 3 | [Streamlit Cloud Deployment](#-3-streamlit-community-cloud-deployment) | Deploy the app live for free |
| 4 | [TMDB API Key Setup](#-4-tmdb-api-key-setup) | Enable movie poster images |
| 5 | [Troubleshooting](#-5-troubleshooting) | Common errors and fixes |
| 6 | [Terminal Commands Reference](#-6-terminal-commands-reference) | All commands in one place |

---

## 🛠️ 1. Local Setup (Step by Step)

### 1.1 Prerequisites

Make sure you have the following installed on your machine:

| Tool | Minimum Version | Check Command |
|------|----------------|---------------|
| Python | 3.10+ | `python --version` |
| pip | Latest | `pip --version` |
| Git | Any | `git --version` |

> **💡 Tip:** Download Python from [python.org](https://www.python.org/downloads/). Make sure to check **"Add Python to PATH"** during installation on Windows.

---

### 1.2 Clone the Repository

```bash
git clone https://github.com/<your-username>/movie-recommendation-system.git
cd movie-recommendation-system
```

> If you haven't uploaded to GitHub yet, skip this step and work in your local project folder.

---

### 1.3 Create a Virtual Environment

Creating an isolated environment keeps your project dependencies clean.

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal prompt.

---

### 1.4 Install Dependencies

```bash
pip install -r requirements.txt
```

This installs the following packages:

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | 1.45.1 | Web application framework |
| `pandas` | 2.2.3 | Data manipulation |
| `numpy` | 2.2.6 | Numerical computations |
| `scikit-learn` | 1.6.1 | ML algorithms (CountVectorizer, Cosine Similarity) |
| `nltk` | 3.9.1 | Natural Language Processing (Porter Stemmer) |
| `requests` | 2.32.3 | HTTP requests for TMDB API |

---

### 1.5 Download the Dataset

1. Go to the Kaggle dataset page:
   **[https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)**

2. Click the **"Download"** button (you'll need a free Kaggle account).

3. Extract the ZIP file — you'll get two CSV files:
   - `tmdb_5000_movies.csv`
   - `tmdb_5000_credits.csv`

4. Create the `data/` directory (if it doesn't exist) and place both files inside:

```
movie-recommendation-system/
└── data/
    ├── tmdb_5000_movies.csv    ← Place here
    └── tmdb_5000_credits.csv   ← Place here
```

**Using the terminal:**
```bash
mkdir data
# Then manually move/copy the downloaded CSV files into the data/ folder
```

---

### 1.6 Build the Recommendation Model

Run the setup script to preprocess the data and generate the model files:

```bash
python setup.py
```

**Expected output:**
```
=======================================================
  Movie Recommendation System — Model Builder
=======================================================

[1/6] Loading datasets...
    → Loaded 4806 movies
[2/6] Extracting features...
[3/6] Preprocessing text...
[4/6] Applying stemming...
[5/6] Building similarity matrix...
    → Feature matrix shape: (4806, 5000)
    → Similarity matrix shape: (4806, 4806)
[6/6] Saving model files...
    → Saved: models/movie_dict.pkl
    → Saved: models/similarity.pkl

✅ Model built successfully!
   Movies processed: 4806
   Model files saved to: models/
```

This creates two files in the `models/` directory:

| File | Description | Approximate Size |
|------|-------------|-----------------|
| `movie_dict.pkl` | Processed movie data (ID, title, tags) | ~1 MB |
| `similarity.pkl` | Cosine similarity matrix (4806 × 4806) | ~88 MB |

---

### 1.7 Run the Streamlit App

```bash
streamlit run app.py
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open **http://localhost:8501** in your browser. The app is now running! 🎉

> **To stop the app:** Press `Ctrl + C` in the terminal.

---

### 1.8 (Optional) Set Up TMDB API Key

The app works without an API key — it shows placeholder images instead of real posters. To enable real movie poster images, see [Section 4: TMDB API Key Setup](#-4-tmdb-api-key-setup).

---

## 📤 2. GitHub Upload Steps

### 2.1 Initialize Git and Make Your First Commit

```bash
git init
git add .
git commit -m "Initial commit: Movie Recommendation System"
```

### 2.2 Create a Repository on GitHub

1. Go to [github.com/new](https://github.com/new)
2. **Repository name:** `movie-recommendation-system`
3. **Description:** `Content-based Movie Recommendation System using Machine Learning & Streamlit`
4. **Visibility:** Public ✅
5. **Do NOT** initialize with README, .gitignore, or license (you already have these locally)
6. Click **"Create repository"**

### 2.3 Push to GitHub

```bash
git remote add origin https://github.com/<your-username>/movie-recommendation-system.git
git branch -M main
git push -u origin main
```

> **📝 Replace** `<your-username>` with your actual GitHub username.

---

### 2.4 Recommended Commit History

For a professional-looking repository, you can restructure your commit history into **6 clean commits**. This is great for interview discussions and portfolio showcasing.

| # | Commit Message | What to Include |
|---|---------------|-----------------|
| 1 | `chore: project setup with requirements and gitignore` | `requirements.txt`, `.gitignore`, `.streamlit/config.toml` |
| 2 | `feat: add data preprocessing and model builder (setup.py)` | `setup.py` |
| 3 | `feat: add TMDB poster API module` | `poster_api.py` |
| 4 | `feat: build Streamlit app with recommendation engine` | `app.py` |
| 5 | `docs: add project documentation and notebook` | `README.md`, `docs/`, `notebooks/` |
| 6 | `feat: add dataset files for cloud deployment` | `data/*.csv` (if deploying to Streamlit Cloud) |

**How to create clean commits (if starting fresh):**

```bash
# Commit 1 — Project Setup
git add requirements.txt .gitignore .streamlit/config.toml
git commit -m "chore: project setup with requirements and gitignore"

# Commit 2 — Model Builder
git add setup.py
git commit -m "feat: add data preprocessing and model builder (setup.py)"

# Commit 3 — Poster API
git add poster_api.py
git commit -m "feat: add TMDB poster API module"

# Commit 4 — Streamlit App
git add app.py
git commit -m "feat: build Streamlit app with recommendation engine"

# Commit 5 — Documentation
git add README.md docs/ notebooks/
git commit -m "docs: add project documentation and notebook"

# Commit 6 — Dataset (for cloud deployment)
git add -f data/*.csv
git commit -m "feat: add dataset files for cloud deployment"

# Push everything
git push -u origin main
```

> **⚠️ Note:** The `-f` flag in Commit 6 force-adds the CSV files even though they're in `.gitignore`. This is necessary for Streamlit Cloud deployment (see Section 3).

---

## ☁️ 3. Streamlit Community Cloud Deployment

Deploy your app for **free** on [Streamlit Community Cloud](https://streamlit.io/cloud) — no server management, no cost.

### 3.1 Prerequisites for Cloud Deployment

> **⚠️ IMPORTANT: Data Files Must Be in the Repository**
>
> By default, `data/*.csv` is listed in `.gitignore`, which means the CSV files are **not pushed to GitHub**. However, Streamlit Cloud needs these files to build the model on first run.
>
> **You have two options:**

**Option A — Force-add the CSV files (Recommended):**
```bash
git add -f data/tmdb_5000_movies.csv data/tmdb_5000_credits.csv
git commit -m "feat: add dataset files for cloud deployment"
git push
```

**Option B — Remove the CSV line from `.gitignore`:**
```bash
# Edit .gitignore and remove or comment out the line: data/*.csv
# Then:
git add data/*.csv
git commit -m "feat: add dataset files for cloud deployment"
git push
```

> **💡 Why this matters:** The app's `app.py` automatically runs `setup.py` to build the model if the `.pkl` files are missing. But it needs the CSV files in the repo to do this. Without them, the app will show a "Dataset not found" error on the cloud.

---

### 3.2 Deploy Step by Step

**Step 1 — Sign In**
- Go to **[share.streamlit.io](https://share.streamlit.io)**
- Sign in with your **GitHub account**

**Step 2 — Create New App**
- Click **"New app"** (top-right corner)

**Step 3 — Configure the App**

| Field | Value |
|-------|-------|
| **Repository** | `<your-username>/movie-recommendation-system` |
| **Branch** | `main` |
| **Main file path** | `app.py` |

**Step 4 — Add Secrets (TMDB API Key)**
- Click **"Advanced settings"** before deploying
- In the **"Secrets"** text area, add:

```toml
TMDB_API_KEY = "your_tmdb_api_key_here"
```

> Don't have an API key yet? See [Section 4](#-4-tmdb-api-key-setup) for how to get one.

**Step 5 — Deploy!**
- Click **"Deploy!"**
- Streamlit will install dependencies, build the model, and launch the app
- First deployment takes **3-5 minutes** (model building + dependency installation)
- Subsequent deployments are much faster

**Step 6 — Your App is Live! 🎉**

Your app will be available at:
```
https://<your-username>-movie-recommendation-system-app-<hash>.streamlit.app
```

---

### 3.3 How Auto-Build Works on the Cloud

The deployment flow is automatic:

```
Streamlit Cloud starts app.py
        │
        ▼
  Are .pkl model files present?
        │
   ┌────┴────┐
   │ NO      │ YES
   ▼         ▼
Runs setup.py    Loads model
(builds model    directly
 from CSVs)         │
   │                │
   ▼                ▼
 App is ready & serving
```

> **📝 Note:** The `.pkl` files are in `.gitignore` and will NOT be in the repo. The app automatically builds them on the cloud from the CSV files during the first run.

---

### 3.4 Updating Your Deployed App

Every time you push to the `main` branch on GitHub, Streamlit Cloud **automatically redeploys** your app.

```bash
# Make changes locally, then:
git add .
git commit -m "update: improve UI styling"
git push
```

The live app will update within ~1-2 minutes.

---

## 🔑 4. TMDB API Key Setup

The TMDB API key enables **real movie poster images** in the app. Without it, the app still works perfectly — it just shows placeholder images.

### 4.1 Get Your Free API Key

1. Go to **[https://www.themoviedb.org/signup](https://www.themoviedb.org/signup)** and create a free account

2. After signing in, go to **[https://www.themoviedb.org/settings/api](https://www.themoviedb.org/settings/api)**

3. Click **"Create"** → select **"Developer"**

4. Fill in the application form:

   | Field | What to Enter |
   |-------|--------------|
   | Type of Use | Personal |
   | Application Name | Movie Recommendation System |
   | Application URL | http://localhost (or your Streamlit Cloud URL) |
   | Application Summary | A content-based movie recommendation system for learning ML |

5. Accept the terms and submit

6. Copy your **API Key (v3 auth)** — it looks like: `8265bd1679663a7ea12ac168da84d2e8`

---

### 4.2 For Local Development

Create a file at `.streamlit/secrets.toml` in your project root:

```bash
# Create the file (the .streamlit/ directory should already exist)
```

Add this content to `.streamlit/secrets.toml`:

```toml
TMDB_API_KEY = "your_api_key_here"
```

> **🔒 Security Note:** This file is already listed in `.gitignore` — it will **never** be pushed to GitHub. Your API key stays private.

**Example:**
```toml
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"
```

Then restart the Streamlit app to see real poster images.

---

### 4.3 For Streamlit Cloud Deployment

1. Open your app on [share.streamlit.io](https://share.streamlit.io)
2. Click the **⋮ (three dots)** menu in the bottom-right corner of your deployed app
3. Click **"Settings"**
4. Go to the **"Secrets"** tab
5. Paste the following:

```toml
TMDB_API_KEY = "your_api_key_here"
```

6. Click **"Save"** — the app will restart with poster images enabled

---

## 🔧 5. Troubleshooting

### ❌ `ModuleNotFoundError: No module named 'streamlit'`

**Cause:** Dependencies not installed, or virtual environment not activated.

**Fix:**
```bash
# Activate virtual environment first
.\venv\Scripts\Activate          # Windows
source venv/bin/activate          # macOS/Linux

# Then install dependencies
pip install -r requirements.txt
```

---

### ❌ `FileNotFoundError: 'data/tmdb_5000_movies.csv' not found!`

**Cause:** The dataset CSV files are missing from the `data/` directory.

**Fix:**
1. Download from [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
2. Extract and place both CSV files in the `data/` folder:
   ```
   data/
   ├── tmdb_5000_movies.csv
   └── tmdb_5000_credits.csv
   ```

---

### ❌ `FileNotFoundError: models/movie_dict.pkl not found`

**Cause:** The model hasn't been built yet.

**Fix:**
```bash
python setup.py
```

> This generates `movie_dict.pkl` and `similarity.pkl` in the `models/` directory.

---

### ❌ API Errors / Poster Images Not Loading

**Cause:** TMDB API key is missing, invalid, or rate-limited.

**Fix:**
- Verify your API key is correct in `.streamlit/secrets.toml`
- Check you're using the **v3 API key** (not v4 token)
- TMDB allows ~40 requests per 10 seconds — if you hit the rate limit, wait a few seconds
- The app works fine without an API key — it shows placeholder images

---

### ❌ `pickle.UnpicklingError` or `ModuleNotFoundError` When Loading `.pkl` Files

**Cause:** The pickle files were created with a different Python version or different package versions.

**Fix:**
```bash
# Delete old model files and rebuild
rm models/movie_dict.pkl models/similarity.pkl     # macOS/Linux
del models\movie_dict.pkl models\similarity.pkl     # Windows

# Rebuild the model
python setup.py
```

---

### ❌ `LookupError: Resource punkt not found`

**Cause:** NLTK tokenizer data is missing.

**Fix:** This is handled automatically by `setup.py`, but if it persists:
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

---

### ❌ Streamlit Cloud: App Crashes on Deploy

**Common causes and fixes:**

| Cause | Fix |
|-------|-----|
| CSV files not in repo | `git add -f data/*.csv && git commit -m "add data" && git push` |
| Wrong main file path | Set main file to `app.py` in Streamlit Cloud settings |
| Python version mismatch | Add `python = "3.10"` in a `runtime.txt` file at the project root |
| Missing dependency | Ensure all packages are in `requirements.txt` |

**To create a `runtime.txt` (optional):**
```bash
echo python-3.10 > runtime.txt
```

---

### ❌ `MemoryError` on Streamlit Cloud

**Cause:** The similarity matrix (~88 MB) exceeds the free tier memory limit.

**Fix:** Streamlit Community Cloud provides ~1 GB RAM for free apps. If you still hit limits:
- The app uses `@st.cache_data` to avoid reloading — this should keep memory in check
- Consider reducing `MAX_FEATURES` in `setup.py` (e.g., from 5000 to 3000)

---

## 📋 6. Terminal Commands Reference

All commands in one place for quick copy-paste.

### 🏗️ Initial Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/movie-recommendation-system.git
cd movie-recommendation-system

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate                  # Windows (PowerShell)
source venv/bin/activate                  # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### 📊 Dataset & Model

```bash
# Create data directory
mkdir data

# Build the recommendation model (after placing CSVs in data/)
python setup.py
```

### 🚀 Run the App

```bash
# Start the Streamlit app
streamlit run app.py

# Stop the app: Ctrl + C
```

### 📤 Git & GitHub

```bash
# Initialize and push to GitHub
git init
git add .
git commit -m "Initial commit: Movie Recommendation System"
git remote add origin https://github.com/<your-username>/movie-recommendation-system.git
git branch -M main
git push -u origin main

# Force-add CSV files for cloud deployment
git add -f data/tmdb_5000_movies.csv data/tmdb_5000_credits.csv
git commit -m "feat: add dataset files for cloud deployment"
git push
```

### 🔧 Troubleshooting Commands

```bash
# Check Python version
python --version

# Check installed packages
pip list

# Rebuild model (delete old files first)
# Windows:
del models\movie_dict.pkl models\similarity.pkl
# macOS/Linux:
rm models/movie_dict.pkl models/similarity.pkl

# Then rebuild
python setup.py

# Fix NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## 📁 Project Structure Reference

```
movie-recommendation-system/
│
├── app.py                      # 🎬 Streamlit web application
├── poster_api.py               # 🖼️ TMDB poster fetching module
├── setup.py                    # ⚙️ ML pipeline (builds model from CSVs)
├── requirements.txt            # 📦 Python dependencies
├── .gitignore                  # 🚫 Git ignore rules
│
├── .streamlit/
│   ├── config.toml             # 🎨 Streamlit theme configuration
│   └── secrets.toml            # 🔑 API keys (git-ignored, create manually)
│
├── data/
│   ├── tmdb_5000_movies.csv    # 📊 Movie metadata (from Kaggle)
│   └── tmdb_5000_credits.csv   # 📊 Cast & crew data (from Kaggle)
│
├── models/
│   ├── movie_dict.pkl          # 🧠 Processed movie data (generated)
│   └── similarity.pkl          # 🧠 Cosine similarity matrix (generated)
│
├── notebooks/
│   └── movie_recommender.py    # 📓 Jupyter/exploration notebook
│
└── docs/
    └── DEPLOYMENT.md           # 📖 This guide
```

---

<div align="center">

**Built with ❤️ using Python, Scikit-learn & Streamlit**

*Movie data provided by [TMDB](https://www.themoviedb.org/)*

</div>
