# Streamlit DNA Wellness Dashboard

This repo contains a single-file Streamlit app (`app.py`) and sample data to run an interactive dashboard for:
- EDA & interactive charts
- Association rule mining (Apriori)
- Clustering & segmentation (KMeans)
- Regression (Linear, Ridge, Lasso, RandomForest)
- Classification (Decision Tree, Gradient Boosting)

## How to deploy
1. Ensure `data/dna_survey_synthetic_600.csv` and/or `data/DNA_Wellness_Survey_Synthetic_Data.csv` exist in `data/`.
2. Deploy to Streamlit Cloud: https://share.streamlit.io (Sign in with GitHub → New app → select this repo → `app.py`).
3. If you prefer local run:
   - `python -m venv .venv` (activate)
   - `pip install -r requirements.txt`
   - `streamlit run app.py`

## Notes
- Column names in the CSV must match your survey file. Edit `app.py` if your column names differ.
- The app automatically detects binary item columns for association mining. You can also manually enter columns for Apriori.
