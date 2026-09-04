"""
Data Cleaning & Consolidation — REAL DATASET VERSION
Schema differs slightly from the brief's description:
 - No standalone User demographics table (only UserId in Transaction)
 - Transaction.VisitMode is the VisitModeId (FK into Mode.csv)
 - Region/Country/City/Continent/Mode each carry a placeholder row
   (id=0, name="-") representing "unknown"
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
        "item": pd.read_csv(RAW / "item.csv"),
        "type": pd.read_csv(RAW / "type.csv"),
        "mode": pd.read_csv(RAW / "mode.csv"),
        "city": pd.read_csv(RAW / "city.csv"),
        "country": pd.read_csv(RAW / "country.csv"),
        "region": pd.read_csv(RAW / "region.csv"),
        "continent": pd.read_csv(RAW / "continent.csv"),
    }


def clean(tables: dict) -> pd.DataFrame:
    tx = tables["transaction"].copy()
    log = []

    # 1. Duplicate rows
    before = len(tx)
    tx = tx.drop_duplicates(subset=[c for c in tx.columns if c != "TransactionId"])
    log.append(f"Removed {before - len(tx)} duplicate transaction rows")

    # 2. Rating sanity check
    bad_rating = ~tx["Rating"].between(1, 5)
    log.append(f"Dropped {bad_rating.sum()} rows with out-of-range Rating")
    tx = tx[~bad_rating]

    # 3. VisitMonth / VisitYear sanity check
    bad_month = ~tx["VisitMonth"].between(1, 12)
    log.append(f"Dropped {bad_month.sum()} rows with invalid VisitMonth")
    tx = tx[~bad_month]

    # 4. Rename VisitMode -> VisitModeId for a clean join
    tx = tx.rename(columns={"VisitMode": "VisitModeId"})

    # 5. Build the attraction -> geography chain (City -> Country -> Region -> Continent)
    item = tables["item"].merge(
        tables["city"], left_on="AttractionCityId", right_on="CityId", how="left"
    )
    item = item.merge(tables["country"], on="CountryId", how="left")
    item = item.merge(tables["region"], on="RegionId", how="left")
    item = item.merge(tables["continent"], on="ContinentId", how="left")
    item = item.merge(tables["type"], on="AttractionTypeId", how="left")

    # 6. Join transaction -> item (+ geography) -> mode
    df = tx.merge(item, on="AttractionId", how="left")
    df = df.merge(tables["mode"], on="VisitModeId", how="left")

    # 7. Drop rows with unresolved FKs
    before = len(df)
    df = df.dropna(subset=["Attraction", "VisitMode", "AttractionType"])
    log.append(f"Dropped {before - len(df)} rows with unresolved foreign keys")

    # 8. Treat placeholder "-" categories as "Unknown"
    for col in ["CityName", "Country", "Region", "Continent"]:
        df[col] = df[col].replace("-", "Unknown").fillna("Unknown")

    # 9. Standardize text
    for col in ["CityName", "Country", "Region", "Continent", "VisitMode", "AttractionType", "Attraction"]:
        df[col] = df[col].astype(str).str.strip()

    for line in log:
        print(" -", line)

    return df.reset_index(drop=True)


if __name__ == "__main__":
    tables = load_raw()
    master = clean(tables)
    out_path = PROCESSED / "master_dataset.csv"
    master.to_csv(out_path, index=False)
    print(f"\nSaved cleaned & joined dataset: {out_path}  shape={master.shape}")
    print(master.columns.tolist())
    print(master.head())
