"""
Generate synthetic data that matches the schema described in the project brief:
Transaction, User, City, Type, VisitMode, Continent, Country, Region, Item (Attraction).

This is a STAND-IN for the real dataset (linked via Google Drive in the brief).
Once you have the real CSVs, put them in data/raw/ with the same filenames/columns
and every downstream script (cleaning, EDA, modeling, app) works unchanged.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- Continent
continents = pd.DataFrame({
    "ContinentId": range(1, 6),
    "Continent": ["Asia", "Europe", "North America", "Africa", "Oceania"],
})

# -------------------------------------------------------------------- Region
region_names = {
    1: ["South Asia", "Southeast Asia", "East Asia"],
    2: ["Western Europe", "Eastern Europe", "Southern Europe"],
    3: ["Northern America", "Central America"],
    4: ["East Africa", "West Africa", "North Africa"],
    5: ["Australia & NZ", "Pacific Islands"],
}
rows = []
rid = 1
for cid, names in region_names.items():
    for name in names:
        rows.append({"RegionId": rid, "Region": name, "ContinentId": cid})
        rid += 1
regions = pd.DataFrame(rows)

# ------------------------------------------------------------------- Country
country_pool = {
    "South Asia": ["India", "Sri Lanka", "Nepal"],
    "Southeast Asia": ["Indonesia", "Thailand", "Vietnam"],
    "East Asia": ["Japan", "South Korea"],
    "Western Europe": ["France", "Germany", "Netherlands"],
    "Eastern Europe": ["Poland", "Romania"],
    "Southern Europe": ["Italy", "Spain", "Greece"],
    "Northern America": ["USA", "Canada"],
    "Central America": ["Mexico", "Costa Rica"],
    "East Africa": ["Kenya", "Tanzania"],
    "West Africa": ["Nigeria", "Ghana"],
    "North Africa": ["Egypt", "Morocco"],
    "Australia & NZ": ["Australia", "New Zealand"],
    "Pacific Islands": ["Fiji"],
}
rows = []
cid_counter = 1
region_lookup = regions.set_index("Region")["RegionId"].to_dict()
for region_name, countries in country_pool.items():
    for c in countries:
        rows.append({"CountryId": cid_counter, "Country": c, "RegionId": region_lookup[region_name]})
        cid_counter += 1
countries_df = pd.DataFrame(rows)

# --------------------------------------------------------------------- City
city_pool = {
    "India": ["Mumbai", "Delhi", "Jaipur", "Goa"],
    "Sri Lanka": ["Colombo", "Kandy"],
    "Nepal": ["Kathmandu"],
    "Indonesia": ["Bali", "Yogyakarta", "Jakarta"],
    "Thailand": ["Bangkok", "Phuket"],
    "Vietnam": ["Hanoi", "Da Nang"],
    "Japan": ["Tokyo", "Kyoto"],
    "South Korea": ["Seoul"],
    "France": ["Paris", "Nice"],
    "Germany": ["Berlin", "Munich"],
    "Netherlands": ["Amsterdam"],
    "Poland": ["Warsaw"],
    "Romania": ["Bucharest"],
    "Italy": ["Rome", "Venice"],
    "Spain": ["Barcelona", "Madrid"],
    "Greece": ["Athens", "Santorini"],
    "USA": ["New York", "Los Angeles", "Orlando"],
    "Canada": ["Toronto", "Vancouver"],
    "Mexico": ["Cancun"],
    "Costa Rica": ["San Jose"],
    "Kenya": ["Nairobi"],
    "Tanzania": ["Zanzibar"],
    "Nigeria": ["Lagos"],
    "Ghana": ["Accra"],
    "Egypt": ["Cairo"],
    "Morocco": ["Marrakech"],
    "Australia": ["Sydney", "Melbourne"],
    "New Zealand": ["Auckland"],
    "Fiji": ["Suva"],
}
rows = []
city_counter = 1
country_lookup = countries_df.set_index("Country")["CountryId"].to_dict()
for country, cities in city_pool.items():
    for city in cities:
        rows.append({"CityId": city_counter, "CityName": city, "CountryId": country_lookup[country]})
        city_counter += 1
cities_df = pd.DataFrame(rows)

# ---------------------------------------------------------------- VisitMode
visit_modes = pd.DataFrame({
    "VisitModeId": range(1, 6),
    "VisitMode": ["Business", "Couples", "Family", "Friends", "Solo"],
})

# ---------------------------------------------------------------- Attraction Type
attraction_types = pd.DataFrame({
    "AttractionTypeId": range(1, 9),
    "AttractionType": [
        "Beach", "Museum", "Historical Site", "Park", "Religious Site",
        "Adventure/Theme Park", "Nature Reserve", "Shopping/Market",
    ],
})

# ---------------------------------------------------------------- Item (Attraction)
n_attractions = 150
attractions = pd.DataFrame({
    "AttractionId": range(1, n_attractions + 1),
    "AttractionCityId": RNG.choice(cities_df["CityId"], size=n_attractions),
    "AttractionTypeId": RNG.choice(attraction_types["AttractionTypeId"], size=n_attractions),
})
attractions["Attraction"] = [f"Attraction_{i}" for i in attractions["AttractionId"]]
attractions["AttractionAddress"] = [f"{RNG.integers(1,999)} Main Rd" for _ in range(n_attractions)]

# ---------------------------------------------------------------------- User
n_users = 2000
user_city = RNG.choice(cities_df["CityId"], size=n_users)
city_to_country = cities_df.set_index("CityId")["CountryId"].to_dict()
country_to_region = countries_df.set_index("CountryId")["RegionId"].to_dict()
region_to_continent = regions.set_index("RegionId")["ContinentId"].to_dict()

user_country = [city_to_country[c] for c in user_city]
user_region = [country_to_region[c] for c in user_country]
user_continent = [region_to_continent[r] for r in user_region]

users = pd.DataFrame({
    "UserId": range(1, n_users + 1),
    "ContinentId": user_continent,
    "RegionId": user_region,
    "CountryId": user_country,
    "CityId": user_city,
})

# ---------------------------------------------------------------- Transaction
n_tx = 20000
tx_user = RNG.choice(users["UserId"], size=n_tx)
tx_attraction = RNG.choice(attractions["AttractionId"], size=n_tx)
tx_year = RNG.choice(range(2019, 2025), size=n_tx)
tx_month = RNG.integers(1, 13, size=n_tx)
tx_mode = RNG.choice(visit_modes["VisitModeId"], size=n_tx, p=[0.15, 0.25, 0.30, 0.20, 0.10])

# Ratings correlated loosely with attraction type popularity + noise
attr_type_map = attractions.set_index("AttractionId")["AttractionTypeId"].to_dict()
type_bias = {1: 0.6, 2: 0.1, 3: 0.2, 4: 0.3, 5: 0.0, 6: 0.5, 7: 0.4, 8: -0.2}
base_rating = 3.3 + np.array([type_bias[attr_type_map[a]] for a in tx_attraction])
ratings = np.clip(np.round(base_rating + RNG.normal(0, 0.9, size=n_tx)), 1, 5).astype(int)

transactions = pd.DataFrame({
    "TransactionId": range(1, n_tx + 1),
    "UserId": tx_user,
    "VisitYear": tx_year,
    "VisitMonth": tx_month,
    "VisitMode": tx_mode,   # note: FK to VisitModeId, will rename after merge in cleaning
    "AttractionId": tx_attraction,
    "Rating": ratings,
})
transactions = transactions.rename(columns={"VisitMode": "VisitModeId"})

# Inject a few realistic messiness artifacts for the cleaning step to fix
missing_idx = RNG.choice(transactions.index, size=200, replace=False)
transactions.loc[missing_idx, "Rating"] = np.nan
dup_rows = transactions.sample(50, random_state=1)
transactions = pd.concat([transactions, dup_rows], ignore_index=True)
bad_month_idx = RNG.choice(transactions.index, size=30, replace=False)
transactions.loc[bad_month_idx, "VisitMonth"] = 0  # invalid month

# ---------------------------------------------------------------------- Save
continents.to_csv(RAW / "continent.csv", index=False)
regions.to_csv(RAW / "region.csv", index=False)
countries_df.to_csv(RAW / "country.csv", index=False)
cities_df.to_csv(RAW / "city.csv", index=False)
visit_modes.to_csv(RAW / "visit_mode.csv", index=False)
attraction_types.to_csv(RAW / "type.csv", index=False)
attractions.to_csv(RAW / "item.csv", index=False)
users.to_csv(RAW / "user.csv", index=False)
transactions.to_csv(RAW / "transaction.csv", index=False)

print("Synthetic data written to", RAW)
for f in sorted(RAW.glob("*.csv")):
    print(" -", f.name, pd.read_csv(f).shape)
