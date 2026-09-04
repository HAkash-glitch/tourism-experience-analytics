"""
Regression Task: Predict Attraction Rating
"""
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

FEATURES = [
    "Continent_enc", "Region_enc", "Country_enc", "CityName_enc",
    "VisitMode_enc", "AttractionType_enc", "VisitYear", "VisitMonth",
    "UserVisitCount", "AttractionAvgRating", "AttractionVisitCount",
]
TARGET = "Rating"


def main():
    df = pd.read_csv(PROCESSED / "model_features.csv")
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    results = {}
    best_name, best_model, best_r2 = None, None, -1e9
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        mae = mean_absolute_error(y_test, preds)
        results[name] = {"R2": round(r2, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4)}
        print(f"{name:18s}  R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")
        if r2 > best_r2:
            best_name, best_model, best_r2 = name, model, r2

    joblib.dump(best_model, MODELS / "regression_model.joblib")
    joblib.dump(FEATURES, MODELS / "regression_features.joblib")
    print(f"\nBest model: {best_name} (R2={best_r2:.4f}) saved to models/regression_model.joblib")
    return results


if __name__ == "__main__":
    main()
