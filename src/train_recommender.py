"""
Recommendation Task: Personalized Attraction Suggestions
Implements:
  1. Collaborative filtering (user-item matrix + cosine similarity, SVD-based)
  2. Content-based filtering (attraction-type / location similarity)
Saves a lightweight artifact bundle the Streamlit app can load instantly.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def build_collaborative(df: pd.DataFrame):
    user_ids = df["UserId"].astype("category")
    attr_ids = df["AttractionId"].astype("category")

    user_item = csr_matrix(
        (df["Rating"], (user_ids.cat.codes, attr_ids.cat.codes)),
        shape=(len(user_ids.cat.categories), len(attr_ids.cat.categories)),
    )

    n_components = min(30, user_item.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(user_item)
    attr_factors = svd.components_.T  # (n_attractions, n_components)

    return {
        "user_categories": list(user_ids.cat.categories),
        "attr_categories": list(attr_ids.cat.categories),
        "user_factors": user_factors,
        "attr_factors": attr_factors,
        "svd": svd,
    }


def build_content_based(item_df: pd.DataFrame):
    # One-hot encode AttractionType + AttractionCityId as content features
    features = pd.get_dummies(item_df[["AttractionTypeId", "AttractionCityId"]].astype(str))
    sim_matrix = cosine_similarity(features.values)
    return {
        "attraction_ids": item_df["AttractionId"].tolist(),
        "similarity_matrix": sim_matrix,
    }


def main():
    df = pd.read_csv(PROCESSED / "master_dataset.csv")
    item_df = df.drop_duplicates("AttractionId")[
        ["AttractionId", "AttractionTypeId", "AttractionCityId", "Attraction", "AttractionType", "CityName"]
    ].reset_index(drop=True)

    collab = build_collaborative(df)
    content = build_content_based(item_df)

    bundle = {
        "collaborative": collab,
        "content": content,
        "item_lookup": item_df,
    }
    joblib.dump(bundle, MODELS / "recommender_bundle.joblib")
    print("Saved recommender bundle -> models/recommender_bundle.joblib")
    print(f"  Users: {len(collab['user_categories'])}, Attractions: {len(collab['attr_categories'])}")


def recommend_for_user(user_id: int, bundle: dict, top_n: int = 5):
    """Collaborative-filtering recommendation: score = user_factors @ attr_factors.T"""
    collab = bundle["collaborative"]
    if user_id not in collab["user_categories"]:
        return []
    u_idx = collab["user_categories"].index(user_id)
    scores = collab["user_factors"][u_idx] @ collab["attr_factors"].T
    top_idx = np.argsort(-scores)[:top_n]
    attr_ids = [collab["attr_categories"][i] for i in top_idx]
    lookup = bundle["item_lookup"].set_index("AttractionId")
    return lookup.loc[attr_ids].reset_index()


def similar_attractions(attraction_id: int, bundle: dict, top_n: int = 5):
    """Content-based recommendation: attractions similar in type/location."""
    content = bundle["content"]
    if attraction_id not in content["attraction_ids"]:
        return []
    idx = content["attraction_ids"].index(attraction_id)
    sims = content["similarity_matrix"][idx]
    top_idx = np.argsort(-sims)[1: top_n + 1]  # skip itself
    attr_ids = [content["attraction_ids"][i] for i in top_idx]
    lookup = bundle["item_lookup"].set_index("AttractionId")
    return lookup.loc[attr_ids].reset_index()


if __name__ == "__main__":
    main()
    bundle = joblib.load(MODELS / "recommender_bundle.joblib")
    sample_user = bundle["collaborative"]["user_categories"][0]
    print(f"\nSample collaborative recs for user {sample_user}:")
    print(recommend_for_user(sample_user, bundle))
    sample_attr = bundle["item_lookup"]["AttractionId"].iloc[0]
    print(f"\nSample content-based recs similar to attraction {sample_attr}:")
    print(similar_attractions(sample_attr, bundle))
