"""
Feature Engineering
Builds user-level and attraction-level aggregate features on top of
master_dataset.csv, then encodes categoricals for modeling.
Output: data/processed/model_features.csv
"""
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import joblib

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)


def build_features():
    df = pd.read_csv(PROCESSED / "master_dataset.csv")

    # ---- User-level aggregates (their typical behavior, avg rating given) ----
    user_avg_rating = df.groupby("UserId")["Rating"].transform("mean")
    df["UserAvgRating"] = user_avg_rating

    user_visit_count = df.groupby("UserId")["TransactionId"].transform("count")
    df["UserVisitCount"] = user_visit_count

    # ---- Attraction-level aggregates (historical popularity/quality) ----
    df["AttractionAvgRating"] = df.groupby("AttractionId")["Rating"].transform("mean")
    df["AttractionVisitCount"] = df.groupby("AttractionId")["TransactionId"].transform("count")

    # ---- Encode categoricals, saving encoders for the Streamlit app ----
    cat_cols = ["Continent", "Region", "Country", "CityName", "VisitMode", "AttractionType"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    joblib.dump(encoders, MODELS / "label_encoders.joblib")

    df.to_csv(PROCESSED / "model_features.csv", index=False)
    print("Saved model_features.csv, shape:", df.shape)
    return df


if __name__ == "__main__":
    build_features()
