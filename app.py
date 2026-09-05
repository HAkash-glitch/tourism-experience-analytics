"""
Tourism Experience Analytics — Streamlit App
Run with: streamlit run app.py

NOTE: This app trains its models fresh on every deployment (cached after first
load) rather than loading pre-saved .joblib files. This avoids scikit-learn /
lightgbm pickle version-compatibility errors across different environments
(local machine vs. Streamlit Cloud), at the cost of a few seconds' startup time.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

try:
    from lightgbm import LGBMClassifier
    CLF_MODEL_NAME = "LightGBM"
except ImportError:
    from sklearn.ensemble import RandomForestClassifier as LGBMClassifier
    CLF_MODEL_NAME = "Random Forest"

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"

st.set_page_config(page_title="Waypoint — Tourism Analytics", layout="wide", page_icon="🧭")

# ============================================================== Visual design
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #FBF8F3; }
    .waypoint-hero {
        background: linear-gradient(120deg, #0E3B43 0%, #14545F 100%);
        border-radius: 4px; padding: 2.6rem 2.8rem; margin-bottom: 1.6rem; color: #FBF8F3;
    }
    .waypoint-hero h1 {
        font-family: 'Fraunces', serif; font-weight: 600; font-size: 2.6rem;
        margin: 0 0 0.4rem 0; color: #FBF8F3; letter-spacing: -0.01em;
    }
    .waypoint-hero p { font-size: 1.02rem; color: #CFE3E1; max-width: 640px; margin: 0; line-height: 1.5; }
    .waypoint-hero .accent { color: #E4572E; }
    h2, h3 { font-family: 'Fraunces', serif; font-weight: 600; color: #0E3B43; }
    div[data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #E8E1D3; border-left: 3px solid #E4572E;
        border-radius: 3px; padding: 0.9rem 1.1rem;
    }
    div[data-testid="stMetricLabel"] { color: #5A6B6E; font-size: 0.82rem; }
    div[data-testid="stMetricValue"] { color: #0E3B43; font-family: 'Fraunces', serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 1.8rem; border-bottom: 1px solid #E8E1D3; }
    .stTabs [data-baseweb="tab"] {
        height: 44px; background: transparent; font-family: 'Inter', sans-serif;
        font-weight: 500; color: #5A6B6E;
    }
    .stTabs [aria-selected="true"] { color: #E4572E !important; border-bottom: 2px solid #E4572E !important; }
    .stButton > button {
        background-color: #E4572E; color: #FFFFFF; border: none; border-radius: 3px;
        font-weight: 500; padding: 0.5rem 1.3rem;
    }
    .stButton > button:hover { background-color: #C94A24; color: #FFFFFF; }
    div[data-testid="stAlertContainer"] { background-color: #EFF6F5; border-left: 3px solid #14545F; }
    footer, #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- Data loading
@st.cache_data
def load_data():
    return pd.read_csv(PROCESSED / "model_features.csv")


# ---------------------------------------------------------------- Train everything fresh, once
@st.cache_resource
def train_all(df: pd.DataFrame):
    # Encoders (re-fit on the fly — cheap, and avoids pickle version issues entirely)
    cat_cols = ["Continent", "Region", "Country", "CityName", "VisitMode", "AttractionType"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le

    # ---------------- Regression: predict Rating
    reg_features = [
        "Continent_enc", "Region_enc", "Country_enc", "CityName_enc",
        "VisitMode_enc", "AttractionType_enc", "VisitYear", "VisitMonth",
        "UserVisitCount", "AttractionAvgRating", "AttractionVisitCount",
    ]
    Xr, yr = df[reg_features], df["Rating"]
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    reg_model = GradientBoostingRegressor(random_state=42)
    reg_model.fit(Xr_tr, yr_tr)
    reg_r2 = r2_score(yr_te, reg_model.predict(Xr_te))
    reg_rmse = mean_squared_error(yr_te, reg_model.predict(Xr_te)) ** 0.5

    # ---------------- Classification: predict VisitMode
    clf_features = [
        "Continent_enc", "Region_enc", "Country_enc", "CityName_enc",
        "AttractionType_enc", "VisitYear", "VisitMonth",
        "UserAvgRating", "UserVisitCount", "AttractionAvgRating", "AttractionVisitCount",
    ]
    Xc, yc = df[clf_features], df["VisitMode_enc"]
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)
    clf_model = LGBMClassifier(random_state=42, verbosity=-1) if CLF_MODEL_NAME == "LightGBM" else LGBMClassifier(random_state=42, n_estimators=300)
    clf_model.fit(Xc_tr, yc_tr)
    clf_acc = (clf_model.predict(Xc_te) == yc_te).mean()

    # ---------------- Recommender: collaborative (SVD) + content-based
    user_ids = df["UserId"].astype("category")
    attr_ids = df["AttractionId"].astype("category")
    user_item = csr_matrix(
        (df["Rating"], (user_ids.cat.codes, attr_ids.cat.codes)),
        shape=(len(user_ids.cat.categories), len(attr_ids.cat.categories)),
    )
    n_components = min(30, user_item.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(user_item)
    attr_factors = svd.components_.T

    item_df = df.drop_duplicates("AttractionId")[
        ["AttractionId", "AttractionTypeId", "AttractionCityId", "Attraction", "AttractionType", "CityName"]
    ].reset_index(drop=True)
    content_features = pd.get_dummies(item_df[["AttractionTypeId", "AttractionCityId"]].astype(str))
    sim_matrix = cosine_similarity(content_features.values)

    rec_bundle = {
        "user_categories": list(user_ids.cat.categories),
        "attr_categories": list(attr_ids.cat.categories),
        "user_factors": user_factors,
        "attr_factors": attr_factors,
        "attraction_ids": item_df["AttractionId"].tolist(),
        "similarity_matrix": sim_matrix,
        "item_lookup": item_df,
    }

    metrics = {"reg_r2": reg_r2, "reg_rmse": reg_rmse, "clf_acc": clf_acc, "clf_model_name": CLF_MODEL_NAME}

    return encoders, reg_model, reg_features, clf_model, clf_features, rec_bundle, metrics


df = load_data()
with st.spinner("Training models (first load only, cached after)..."):
    encoders, reg_model, reg_features, clf_model, clf_features, rec_bundle, metrics = train_all(df)

POPULAR_ATTRACTIONS = (
    df.groupby(["AttractionId", "Attraction", "AttractionType", "CityName"])["Rating"]
    .agg(["mean", "count"])
    .reset_index()
    .sort_values(["count", "mean"], ascending=False)
)

st.markdown(
    """
    <div class="waypoint-hero">
        <h1>Waypoint <span class="accent">·</span> Tourism Experience Analytics</h1>
        <p>Every trip in this dataset is a signal — where people went, how they traveled,
        and how they felt about it. This tool turns that history into three things:
        a rating forecast, a travel-mode read, and a next-stop recommendation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "This dataset has no user home-location table — only attraction location is known. "
    "Predictions use each attraction's own location/type plus a user's past behavior, "
    "not where the tourist lives.",
    icon="🧭",
)

tab_overview, tab_predict, tab_classify, tab_recommend = st.tabs(
    ["Overview", "Predict Rating", "Predict Visit Mode", "Recommendations"]
)

# ================================================================== Overview
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users", f"{df['UserId'].nunique():,}")
    c2.metric("Attractions", f"{df['AttractionId'].nunique():,}")
    c3.metric("Transactions", f"{len(df):,}")
    c4.metric("Avg Rating", f"{df['Rating'].mean():.2f} / 5")

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Transactions by continent")
        st.bar_chart(df["Continent"].value_counts(), color="#0E3B43")
    with col2:
        st.subheader("Visit mode split")
        st.bar_chart(df["VisitMode"].value_counts(), color="#E4572E")

    st.subheader("Average rating by attraction type")
    top_types = df.groupby("AttractionType")["Rating"].mean().sort_values(ascending=False)
    st.bar_chart(top_types, color="#14545F")

    st.subheader("Most visited attractions")
    st.dataframe(
        POPULAR_ATTRACTIONS.head(10).rename(columns={"mean": "Avg Rating", "count": "Visit Count"}),
        use_container_width=True, hide_index=True,
    )

    with st.expander("Model performance (this session)"):
        mcol1, mcol2 = st.columns(2)
        mcol1.metric("Regression R²", f"{metrics['reg_r2']:.3f}")
        mcol1.metric("Regression RMSE", f"{metrics['reg_rmse']:.3f}")
        mcol2.metric(f"{metrics['clf_model_name']} Accuracy", f"{metrics['clf_acc']:.1%}")

# ================================================================== Predict rating (Regression)
with tab_predict:
    st.subheader("Predict how a user might rate a visit")
    st.caption("Pick an attraction and trip details; the model estimates the likely rating (1–5).")

    colA, colB = st.columns(2)
    with colA:
        attraction_name = st.selectbox("Attraction", sorted(df["Attraction"].unique()), key="pred_attraction")
        attr_row = df[df["Attraction"] == attraction_name].iloc[0]
        visit_mode = st.selectbox("Visit Mode", sorted(df["VisitMode"].unique()), key="pred_mode")
    with colB:
        year = st.slider("Visit Year", int(df["VisitYear"].min()), int(df["VisitYear"].max()), 2022, key="pred_year")
        month = st.slider("Visit Month", 1, 12, 6, key="pred_month")

    if st.button("Predict rating", type="primary"):
        try:
            row = {
                "Continent_enc": encoders["Continent"].transform([attr_row["Continent"]])[0],
                "Region_enc": encoders["Region"].transform([attr_row["Region"]])[0],
                "Country_enc": encoders["Country"].transform([attr_row["Country"]])[0],
                "CityName_enc": encoders["CityName"].transform([attr_row["CityName"]])[0],
                "VisitMode_enc": encoders["VisitMode"].transform([visit_mode])[0],
                "AttractionType_enc": encoders["AttractionType"].transform([attr_row["AttractionType"]])[0],
                "VisitYear": year,
                "VisitMonth": month,
                "UserVisitCount": df["UserVisitCount"].median(),
                "AttractionAvgRating": attr_row["AttractionAvgRating"],
                "AttractionVisitCount": attr_row["AttractionVisitCount"],
            }
            X = pd.DataFrame([row])[reg_features]
            pred = float(np.clip(reg_model.predict(X)[0], 1, 5))
            st.success(f"Predicted rating: **{pred:.2f} / 5**")
            st.caption(
                f"This attraction's historical average is {attr_row['AttractionAvgRating']:.2f} "
                f"across {int(attr_row['AttractionVisitCount'])} visits."
            )
        except Exception as e:
            st.error(f"Couldn't generate a prediction for this combination: {e}")

# ================================================================== Predict visit mode (Classification)
with tab_classify:
    st.subheader("Predict a user's likely visit mode")
    st.caption("Given an attraction and trip timing, estimate whether the visit is Business, Family, etc.")

    colA, colB = st.columns(2)
    with colA:
        attraction_name2 = st.selectbox("Attraction ", sorted(df["Attraction"].unique()), key="clf_attraction")
        attr_row2 = df[df["Attraction"] == attraction_name2].iloc[0]
    with colB:
        year2 = st.slider("Visit Year ", int(df["VisitYear"].min()), int(df["VisitYear"].max()), 2022, key="clf_year")
        month2 = st.slider("Visit Month ", 1, 12, 6, key="clf_month")

    if st.button("Predict visit mode", type="primary"):
        try:
            row = {
                "Continent_enc": encoders["Continent"].transform([attr_row2["Continent"]])[0],
                "Region_enc": encoders["Region"].transform([attr_row2["Region"]])[0],
                "Country_enc": encoders["Country"].transform([attr_row2["Country"]])[0],
                "CityName_enc": encoders["CityName"].transform([attr_row2["CityName"]])[0],
                "AttractionType_enc": encoders["AttractionType"].transform([attr_row2["AttractionType"]])[0],
                "VisitYear": year2,
                "VisitMonth": month2,
                "UserAvgRating": df["UserAvgRating"].median(),
                "UserVisitCount": df["UserVisitCount"].median(),
                "AttractionAvgRating": attr_row2["AttractionAvgRating"],
                "AttractionVisitCount": attr_row2["AttractionVisitCount"],
            }
            X = pd.DataFrame([row])[clf_features]
            pred_enc = clf_model.predict(X)[0]
            pred_label = encoders["VisitMode"].inverse_transform([pred_enc])[0]
            st.success(f"Predicted visit mode: **{pred_label}**")

            proba = clf_model.predict_proba(X)[0]
            classes = encoders["VisitMode"].inverse_transform(clf_model.classes_)
            proba_df = pd.DataFrame({"Visit Mode": classes, "Probability": proba}).sort_values(
                "Probability", ascending=False
            )
            st.dataframe(proba_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Couldn't generate a prediction for this combination: {e}")

# ================================================================== Recommendations
with tab_recommend:
    st.subheader("Get personalized attraction recommendations")
    mode = st.radio(
        "Recommendation type",
        ["By existing User ID (collaborative)", "By an attraction you liked (content-based)"],
    )

    lookup = rec_bundle["item_lookup"].set_index("AttractionId")

    if mode.startswith("By existing"):
        known_users = set(rec_bundle["user_categories"])
        user_id = st.number_input(
            "Enter a User ID", min_value=int(df["UserId"].min()), max_value=int(df["UserId"].max()), step=1
        )
        if st.button("Recommend attractions", type="primary"):
            if user_id not in known_users:
                st.warning(
                    "No history found for this User ID — showing the most popular attractions instead "
                    "(cold-start fallback)."
                )
                st.dataframe(
                    POPULAR_ATTRACTIONS.head(5)[["Attraction", "AttractionType", "CityName"]],
                    use_container_width=True, hide_index=True,
                )
            else:
                u_idx = rec_bundle["user_categories"].index(user_id)
                scores = rec_bundle["user_factors"][u_idx] @ rec_bundle["attr_factors"].T
                top_idx = np.argsort(-scores)[:5]
                attr_ids = [rec_bundle["attr_categories"][i] for i in top_idx]
                st.dataframe(
                    lookup.loc[attr_ids][["Attraction", "AttractionType", "CityName"]],
                    use_container_width=True, hide_index=True,
                )
    else:
        attraction_names = rec_bundle["item_lookup"][["AttractionId", "Attraction"]]
        choice = st.selectbox("Select an attraction", attraction_names["Attraction"])
        if st.button("Find similar attractions", type="primary"):
            attraction_id = int(attraction_names[attraction_names["Attraction"] == choice]["AttractionId"].iloc[0])
            idx = rec_bundle["attraction_ids"].index(attraction_id)
            sims = rec_bundle["similarity_matrix"][idx]
            top_idx = np.argsort(-sims)[1:6]
            attr_ids = [rec_bundle["attraction_ids"][i] for i in top_idx]
            st.dataframe(
                lookup.loc[attr_ids][["Attraction", "AttractionType", "CityName"]],
                use_container_width=True, hide_index=True,
            )

st.divider()
st.caption(
    "Waypoint · Tourism Experience Analytics — data cleaning → EDA → regression / classification / "
    "recommendation → Streamlit"
)
