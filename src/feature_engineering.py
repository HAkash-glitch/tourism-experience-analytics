"""
Feature Engineering — REAL DATASET VERSION
No user demographics available, so features are built from:
 - user behavior history (avg rating given, visit count)
 - attraction identity/location/popularity
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

    df["UserAvgRating"] = df.groupby("UserId")["Rating"].transform("mean")
    df["UserVisitCount"] = df.groupby("UserId")["TransactionId"].transform("count")

    df["AttractionAvgRating"] = df.groupby("AttractionId")["Rating"].transform("mean")
    df["AttractionVisitCount"] = df.groupby("AttractionId")["TransactionId"].transform("count")

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
