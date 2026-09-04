# Tourism Experience Analytics: Classification, Prediction, and Recommendation System

Full pipeline built on the **real dataset**: data cleaning → EDA → regression
(rating prediction) → classification (visit mode prediction) → recommendation
system (collaborative + content-based) → Streamlit app.

## ⚠️ Known data limitation

The Drive folder provided did not include a `User.xlsx` demographics table —
only `Transaction`, `Item`, `Type`, `Mode`, `City`, `Country`, `Region`, and
`Continent`. That means there's no record of *where each tourist lives*, only
*where each attraction is located*. All models and the app use attraction
location + a user's own visit history as substitute signal. If you later find
a User file, send it over and the pipeline can be extended to include real
user geography features.

## Real data summary

- 33,530 unique users
- 30 unique attractions
- 49,208 cleaned transactions (after removing ~3,722 duplicates)
- Ratings 1–5, Visit years 2013–2022
- 17 attraction types, 5 visit modes (Business/Couples/Family/Friends/Solo)

## Model results (on real data)

**Regression (predict Rating):** best model GradientBoosting, R²=0.11, RMSE=0.91.
Rating is inherently noisy/subjective, so this level of fit is typical for
this kind of task without richer user features.

**Classification (predict VisitMode):** best model LightGBM, accuracy ~47%,
weighted F1=0.41 (vs. ~20% random baseline for 5 classes). Business and
non-common cold-start classes ("-"/other) are hardest to predict — expected,
since attraction attributes alone don't fully capture travel-party context.

**Recommendation:** SVD-based collaborative filtering (user-item matrix) plus
content-based similarity (attraction type + location) as a cold-start fallback.

## Setup

```bash
pip install -r requirements.txt
```

## Run the full pipeline

```bash
python src/data_cleaning.py             # -> data/processed/master_dataset.csv
python src/feature_engineering.py       # -> data/processed/model_features.csv
python src/eda.py                       # -> outputs/eda/*.png
python src/train_regression.py          # -> models/regression_model.joblib
python src/train_classification.py      # -> models/classification_model.joblib
python src/train_recommender.py         # -> models/recommender_bundle.joblib
```

## Run the app

```bash
streamlit run app.py
```

## Project structure

```
tourism_project/
├── data/
│   ├── raw/                  # real source CSVs (converted from your .xlsx uploads)
│   ├── synthetic_reference/  # earlier placeholder data, kept for reference only
│   └── processed/            # cleaned + feature-engineered datasets
├── models/                   # trained model artifacts (.joblib)
├── outputs/
│   └── eda/                  # chart PNGs
├── src/
│   ├── data_cleaning.py           # real-data cleaning (primary)
│   ├── feature_engineering.py     # real-data features (primary)
│   ├── data_cleaning_synthetic.py # old synthetic-data version, reference only
│   ├── feature_engineering_synthetic.py
│   ├── generate_synthetic_data.py
│   ├── eda.py
│   ├── train_regression.py
│   ├── train_classification.py
│   └── train_recommender.py
├── app.py                    # Streamlit app
├── requirements.txt
└── README.md
```

## Deploying it

1. Push this folder to a GitHub repo.
2. Go to https://share.streamlit.io, connect your GitHub, point it at `app.py`.
3. It builds automatically from `requirements.txt` and gives you a public URL.

## Still to do

- A short written report summarizing approach/findings (ask for this — it can
  be generated from the real metrics above)
- If a User demographics file turns up, re-run the pipeline with it added for
  richer, more accurate features
