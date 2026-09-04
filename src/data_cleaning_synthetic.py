"""
Data Cleaning & Consolidation
- Loads the 9 raw tables (transaction, user, city, country, region, continent,
  type, visit_mode, item)
- Handles missing values, duplicates, invalid categorical entries
- Joins everything into one consolidated, model-ready dataset
Output: data/processed/master_dataset.csv
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


def load_raw():
    return {
        "transaction": pd.read_csv(RAW / "transaction.csv"),
        "user": pd.read_csv(RAW / "user.csv"),
        "city": pd.read_csv(RAW / "city.csv"),
        "country": pd.read_csv(RAW / "country.csv"),
        "region": pd.read_csv(RAW / "region.csv"),
        "continent": pd.read_csv(RAW / "continent.csv"),
        "type": pd.read_csv(RAW / "type.csv"),
        "visit_mode": pd.read_csv(RAW / "visit_mode.csv"),
        "item": pd.read_csv(RAW / "item.csv"),
    }


def clean(tables: dict) -> pd.DataFrame:
    tx = tables["transaction"].copy()
    log = []

    # 1. Drop exact duplicate transactions
    before = len(tx)
    tx = tx.drop_duplicates(subset=[c for c in tx.columns if c != "TransactionId"])
    log.append(f"Removed {before - len(tx)} duplicate transaction rows")

    # 2. Fix invalid VisitMonth (0 or >12) -> treat as missing, then impute with mode
    invalid_month = ~tx["VisitMonth"].between(1, 12)
    log.append(f"Fixed {invalid_month.sum()} invalid VisitMonth values")
    tx.loc[invalid_month, "VisitMonth"] = tx["VisitMonth"][~invalid_month].mode()[0]

    # 3. Handle missing Rating -> drop rows with no target for supervised tasks,
    #    but keep a copy for recommendation use where possible.
    missing_rating = tx["Rating"].isna()
    log.append(f"Dropped {missing_rating.sum()} rows with missing Rating")
    tx = tx[~missing_rating].copy()
    tx["Rating"] = tx["Rating"].astype(int)

    # 4. Outlier check on Rating (should be 1-5)
    out_of_range = ~tx["Rating"].between(1, 5)
    log.append(f"Dropped {out_of_range.sum()} rows with out-of-range Rating")
    tx = tx[~out_of_range]

    # 5. Join dimension tables.
    # User already carries the authoritative Continent/Region/Country/City FK chain,
    # so drop the redundant FK columns from each dimension table before merging
    # to avoid _x/_y collisions.
    city_dim = tables["city"][["CityId", "CityName"]]
    country_dim = tables["country"][["CountryId", "Country"]]
    region_dim = tables["region"][["RegionId", "Region"]]
    continent_dim = tables["continent"][["ContinentId", "Continent"]]
    item_dim = tables["item"]  # AttractionId, AttractionCityId, AttractionTypeId, Attraction, AttractionAddress
    type_dim = tables["type"][["AttractionTypeId", "AttractionType"]]

    df = tx.merge(tables["user"], on="UserId", how="left")
    df = df.merge(city_dim, on="CityId", how="left")
    df = df.merge(country_dim, on="CountryId", how="left")
    df = df.merge(region_dim, on="RegionId", how="left")
    df = df.merge(continent_dim, on="ContinentId", how="left")
    df = df.merge(tables["visit_mode"], on="VisitModeId", how="left")
    df = df.merge(item_dim, on="AttractionId", how="left")
    df = df.merge(type_dim, on="AttractionTypeId", how="left")

    # 6. Drop rows that failed to join (broken FK) — report count
    before = len(df)
    df = df.dropna(subset=["VisitMode", "AttractionType", "CityName"])
    log.append(f"Dropped {before - len(df)} rows with unresolved foreign keys")

    # 7. Standardize text categorical columns
    for col in ["CityName", "Country", "Region", "Continent", "VisitMode", "AttractionType", "Attraction"]:
        df[col] = df[col].astype(str).str.strip().str.title()

    for line in log:
        print(" -", line)

    return df.reset_index(drop=True)


if __name__ == "__main__":
    tables = load_raw()
    master = clean(tables)
    out_path = PROCESSED / "master_dataset.csv"
    master.to_csv(out_path, index=False)
    print(f"\nSaved cleaned & joined dataset: {out_path}  shape={master.shape}")
    print(master.head())
