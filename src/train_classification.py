"""
Classification Task: Predict Visit Mode
"""
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

FEATURES = [
    "Continent_enc", "Region_enc", "Country_enc", "CityName_enc",
    "AttractionType_enc", "VisitYear", "VisitMonth",
    "UserAvgRating", "UserVisitCount", "AttractionAvgRating", "AttractionVisitCount",
]
TARGET = "VisitMode_enc"


def main():
    df = pd.read_csv(PROCESSED / "model_features.csv")
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=14, random_state=42, n_jobs=-1),
    }
    if HAS_LGBM:
        candidates["LightGBM"] = LGBMClassifier(random_state=42, verbosity=-1)

    best_name, best_model, best_f1 = None, None, -1
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        print(f"{name:18s}  Acc={acc:.4f}  Prec={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
        if f1 > best_f1:
            best_name, best_model, best_f1 = name, model, f1

    print(f"\nBest model: {best_name} (F1={best_f1:.4f})")
    print(classification_report(y_test, best_model.predict(X_test), zero_division=0))

    joblib.dump(best_model, MODELS / "classification_model.joblib")
    joblib.dump(FEATURES, MODELS / "classification_features.joblib")
    print("Saved to models/classification_model.joblib")


if __name__ == "__main__":
    main()
