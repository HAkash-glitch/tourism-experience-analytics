"""
Waypoint — Tourism Experience Analytics
Robust Streamlit App

Run locally:
    streamlit run app.py
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# ==============================================================
# PATHS
# ==============================================================

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"


# ==============================================================
# STREAMLIT CONFIG
# ==============================================================

st.set_page_config(
    page_title="Waypoint — Tourism Analytics",
    page_icon="🧭",
    layout="wide",
)


# ==============================================================
# CUSTOM CSS
# ==============================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #FBF8F3;
    }

    .waypoint-hero {
        background: linear-gradient(120deg, #0E3B43 0%, #14545F 100%);
        border-radius: 8px;
        padding: 2.6rem 2.8rem;
        margin-bottom: 1.6rem;
        color: #FBF8F3;
    }

    .waypoint-hero h1 {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 2.6rem;
        margin: 0 0 0.5rem 0;
        color: #FBF8F3;
        letter-spacing: -0.01em;
    }

    .waypoint-hero p {
        font-size: 1.02rem;
        color: #CFE3E1;
        max-width: 760px;
        margin: 0;
        line-height: 1.6;
    }

    .waypoint-hero .accent {
        color: #E4572E;
    }

    h2, h3 {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        color: #0E3B43;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E8E1D3;
        border-left: 3px solid #E4572E;
        border-radius: 5px;
        padding: 0.9rem 1.1rem;
    }

    div[data-testid="stMetricLabel"] {
        color: #5A6B6E;
        font-size: 0.82rem;
    }

    div[data-testid="stMetricValue"] {
        color: #0E3B43;
        font-family: 'Fraunces', serif;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 1.8rem;
        border-bottom: 1px solid #E8E1D3;
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px;
        background: transparent;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: #5A6B6E;
    }

    .stTabs [aria-selected="true"] {
        color: #E4572E !important;
        border-bottom: 2px solid #E4572E !important;
    }

    .stButton > button {
        background-color: #E4572E;
        color: #FFFFFF;
        border: none;
        border-radius: 5px;
        font-weight: 500;
        padding: 0.5rem 1.3rem;
    }

    .stButton > button:hover {
        background-color: #C94A24;
        color: #FFFFFF;
    }

    footer, #MainMenu {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================
# LOAD DATA
# ==============================================================

@st.cache_data(show_spinner=False)
def load_data():

    data_path = PROCESSED / "model_features.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{data_path}"
        )

    df = pd.read_csv(data_path)

    required_columns = [
        "Attraction",
        "AttractionId",
        "AttractionType",
        "CityName",
        "Continent",
        "Region",
        "Country",
        "VisitMode",
        "VisitYear",
        "VisitMonth",
        "Rating",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in model_features.csv: {missing_columns}"
        )

    # Numeric columns
    numeric_columns = [
        "AttractionId",
        "VisitYear",
        "VisitMonth",
        "Rating",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # ----------------------------------------------------------
    # Attraction statistics
    # ----------------------------------------------------------

    if "AttractionAvgRating" not in df.columns:

        df["AttractionAvgRating"] = (
            df.groupby("AttractionId")["Rating"]
            .transform("mean")
        )

    if "AttractionVisitCount" not in df.columns:

        df["AttractionVisitCount"] = (
            df.groupby("AttractionId")["AttractionId"]
            .transform("count")
        )

    # ----------------------------------------------------------
    # User statistics
    # ----------------------------------------------------------

    if "UserVisitCount" not in df.columns:

        if "UserId" in df.columns:

            df["UserVisitCount"] = (
                df.groupby("UserId")["UserId"]
                .transform("count")
            )

        else:

            df["UserVisitCount"] = 1

    if "UserAvgRating" not in df.columns:

        if "UserId" in df.columns:

            df["UserAvgRating"] = (
                df.groupby("UserId")["Rating"]
                .transform("mean")
            )

        else:

            df["UserAvgRating"] = df["Rating"].median()

    # ----------------------------------------------------------
    # Fill missing numerical values
    # ----------------------------------------------------------

    stat_columns = [
        "AttractionAvgRating",
        "AttractionVisitCount",
        "UserVisitCount",
        "UserAvgRating",
    ]

    for col in stat_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        median_value = df[col].median()

        if pd.isna(median_value):
            median_value = 0

        df[col] = df[col].fillna(
            median_value
        )

    rating_median = df["Rating"].median()

    if pd.isna(rating_median):
        rating_median = 3.0

    df["Rating"] = df["Rating"].fillna(
        rating_median
    )

    # ----------------------------------------------------------
    # Categorical columns
    # ----------------------------------------------------------

    categorical_columns = [
        "Attraction",
        "AttractionType",
        "CityName",
        "Continent",
        "Region",
        "Country",
        "VisitMode",
    ]

    for col in categorical_columns:

        df[col] = (
            df[col]
            .fillna("Unknown")
            .astype(str)
        )

    df = df.dropna(
        subset=[
            "Attraction",
            "AttractionId",
            "VisitMode",
        ]
    )

    df = df.reset_index(drop=True)

    return df


# ==============================================================
# ENCODERS
# ==============================================================

ENCODER_COLUMNS = [
    "Continent",
    "Region",
    "Country",
    "CityName",
    "VisitMode",
    "AttractionType",
]


def build_encoders(df):

    encoders = {}

    for column in ENCODER_COLUMNS:

        encoder = LabelEncoder()

        encoder.fit(
            df[column]
            .fillna("Unknown")
            .astype(str)
        )

        encoders[column] = encoder

    return encoders


# ==============================================================
# FEATURE DEFINITIONS
# ==============================================================

REGRESSION_FEATURES = [
    "Continent_enc",
    "Region_enc",
    "Country_enc",
    "CityName_enc",
    "VisitMode_enc",
    "AttractionType_enc",
    "VisitYear",
    "VisitMonth",
    "UserVisitCount",
    "AttractionAvgRating",
    "AttractionVisitCount",
]


CLASSIFICATION_FEATURES = [
    "Continent_enc",
    "Region_enc",
    "Country_enc",
    "CityName_enc",
    "AttractionType_enc",
    "VisitYear",
    "VisitMonth",
    "UserAvgRating",
    "UserVisitCount",
    "AttractionAvgRating",
    "AttractionVisitCount",
]


# ==============================================================
# SAFE MODEL LOADING
# ==============================================================

def safe_load(path):

    if not path.exists():
        return None

    try:

        return joblib.load(path)

    except Exception:

        return None


# ==============================================================
# FALLBACK TRAINING
# ==============================================================

def train_fallback_models(df):

    encoders = build_encoders(df)

    work_df = df.copy()

    # Create encoded columns
    for column in ENCODER_COLUMNS:

        work_df[f"{column}_enc"] = (
            encoders[column]
            .transform(
                work_df[column]
                .fillna("Unknown")
                .astype(str)
            )
        )

    # ----------------------------------------------------------
    # Regression
    # ----------------------------------------------------------

    X_reg = (
        work_df[REGRESSION_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    y_reg = (
        pd.to_numeric(
            work_df["Rating"],
            errors="coerce"
        )
        .fillna(
            work_df["Rating"].median()
        )
    )

    reg_model = RandomForestRegressor(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    reg_model.fit(
        X_reg,
        y_reg
    )

    # ----------------------------------------------------------
    # Classification
    # ----------------------------------------------------------

    visit_mode_encoder = LabelEncoder()

    y_clf = visit_mode_encoder.fit_transform(
        work_df["VisitMode"].astype(str)
    )

    encoders["VisitMode"] = visit_mode_encoder

    # Re-create encoded VisitMode
    work_df["VisitMode_enc"] = (
        visit_mode_encoder.transform(
            work_df["VisitMode"].astype(str)
        )
    )

    X_clf = (
        work_df[CLASSIFICATION_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    clf_model = RandomForestClassifier(
        n_estimators=120,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    clf_model.fit(
        X_clf,
        y_clf
    )

    return (
        encoders,
        reg_model,
        REGRESSION_FEATURES,
        clf_model,
        CLASSIFICATION_FEATURES,
    )


# ==============================================================
# LOAD MODELS
# ==============================================================

@st.cache_resource(show_spinner=False)
def load_artifacts(df):

    fallback_used = False

    encoders = safe_load(
        MODELS / "label_encoders.joblib"
    )

    reg_model = safe_load(
        MODELS / "regression_model.joblib"
    )

    reg_features = safe_load(
        MODELS / "regression_features.joblib"
    )

    clf_model = safe_load(
        MODELS / "classification_model.joblib"
    )

    clf_features = safe_load(
        MODELS / "classification_features.joblib"
    )

    # ----------------------------------------------------------
    # If regression/classification model cannot be loaded,
    # train safe fallback models.
    # ----------------------------------------------------------

    if (
        reg_model is None
        or clf_model is None
    ):

        fallback_used = True

        (
            encoders,
            reg_model,
            reg_features,
            clf_model,
            clf_features,
        ) = train_fallback_models(df)

    else:

        if not isinstance(
            encoders,
            dict
        ):
            encoders = build_encoders(df)

        if not isinstance(
            reg_features,
            (list, tuple, np.ndarray)
        ):
            reg_features = REGRESSION_FEATURES
        else:
            reg_features = list(
                reg_features
            )

        if not isinstance(
            clf_features,
            (list, tuple, np.ndarray)
        ):
            clf_features = CLASSIFICATION_FEATURES
        else:
            clf_features = list(
                clf_features
            )

    return (
        encoders,
        reg_model,
        reg_features,
        clf_model,
        clf_features,
        fallback_used,
    )


# ==============================================================
# RECOMMENDER
# ==============================================================

@st.cache_resource(show_spinner=False)
def build_recommender(df):

    item_columns = [
        "AttractionId",
        "Attraction",
        "AttractionType",
        "CityName",
    ]

    items = (
        df[item_columns]
        .drop_duplicates(
            subset=["AttractionId"]
        )
        .reset_index(drop=True)
    )

    stats = (
        df.groupby("AttractionId")
        .agg(
            AvgRating=("Rating", "mean"),
            VisitCount=("AttractionId", "count"),
        )
        .reset_index()
    )

    items = items.merge(
        stats,
        on="AttractionId",
        how="left",
    )

    items["AvgRating"] = (
        items["AvgRating"]
        .fillna(
            df["Rating"].median()
        )
    )

    items["VisitCount"] = (
        items["VisitCount"]
        .fillna(0)
    )

    content = pd.get_dummies(
        items[
            [
                "AttractionType",
                "CityName",
            ]
        ]
        .fillna("Unknown")
        .astype(str)
    ).astype(float)

    if len(items) == 0:

        similarity = np.empty(
            (0, 0)
        )

    elif content.shape[1] == 0:

        similarity = np.eye(
            len(items)
        )

    else:

        similarity = cosine_similarity(
            content
        )

    return items, similarity


# ==============================================================
# LOAD EVERYTHING
# ==============================================================

try:

    df = load_data()

    (
        encoders,
        reg_model,
        reg_features,
        clf_model,
        clf_features,
        fallback_used,
    ) = load_artifacts(df)

    (
        recommendation_items,
        recommendation_similarity,
    ) = build_recommender(df)

except Exception as error:

    st.error(
        "The application could not start."
    )

    st.code(
        str(error)
    )

    st.stop()


# ==============================================================
# FALLBACK NOTICE
# ==============================================================

if fallback_used:

    st.warning(
        "A saved model could not be loaded with the current "
        "environment, so the app is using an automatically "
        "trained fallback model."
    )


# ==============================================================
# POPULAR ATTRACTIONS
# ==============================================================

POPULAR_ATTRACTIONS = (
    df.groupby(
        [
            "AttractionId",
            "Attraction",
            "AttractionType",
            "CityName",
        ],
        dropna=False,
    )["Rating"]
    .agg(
        ["mean", "count"]
    )
    .reset_index()
    .sort_values(
        [
            "count",
            "mean",
        ],
        ascending=False,
    )
)


# ==============================================================
# HERO
# ==============================================================

st.markdown(
    """
    <div class="waypoint-hero">
        <h1>
            Waypoint
            <span class="accent">·</span>
            Tourism Experience Analytics
        </h1>

        <p>
            Every trip in this dataset is a signal —
            where people went, how they traveled,
            and how they felt about it.
            This tool turns that history into
            rating forecasts, travel-mode predictions,
            and attraction recommendations.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================
# INFORMATION
# ==============================================================

st.info(
    """
    This dataset has no user home-location table —
    only attraction location is known.

    Predictions use attraction location/type and
    user behavior information available in the dataset.
    """,
    icon="🧭",
)


# ==============================================================
# TABS
# ==============================================================

(
    tab_overview,
    tab_predict,
    tab_classify,
    tab_recommend,
) = st.tabs(
    [
        "Overview",
        "Predict Rating",
        "Predict Visit Mode",
        "Recommendations",
    ]
)


# ==============================================================
# OVERVIEW
# ==============================================================

with tab_overview:

    c1, c2, c3, c4 = st.columns(4)

    # Users

    if "UserId" in df.columns:

        c1.metric(
            "Users",
            f"{df['UserId'].nunique():,}",
        )

    else:

        c1.metric(
            "Users",
            "N/A",
        )

    # Attractions

    c2.metric(
        "Attractions",
        f"{df['AttractionId'].nunique():,}",
    )

    # Transactions

    c3.metric(
        "Transactions",
        f"{len(df):,}",
    )

    # Average rating

    c4.metric(
        "Avg Rating",
        f"{df['Rating'].mean():.2f} / 5",
    )

    st.write("")

    col1, col2 = st.columns(2)

    # ----------------------------------------------------------
    # Continent
    # ----------------------------------------------------------

    with col1:

        st.subheader(
            "Transactions by continent"
        )

        st.bar_chart(
            df["Continent"]
            .value_counts()
        )

    # ----------------------------------------------------------
    # Visit mode
    # ----------------------------------------------------------

    with col2:

        st.subheader(
            "Visit mode split"
        )

        st.bar_chart(
            df["VisitMode"]
            .value_counts()
        )

    # ----------------------------------------------------------
    # Attraction type
    # ----------------------------------------------------------

    st.subheader(
        "Average rating by attraction type"
    )

    top_types = (
        df.groupby(
            "AttractionType"
        )["Rating"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    st.bar_chart(
        top_types
    )

    # ----------------------------------------------------------
    # Popular attractions
    # ----------------------------------------------------------

    st.subheader(
        "Most visited attractions"
    )

    popular_display = (
        POPULAR_ATTRACTIONS
        .head(10)
        .rename(
            columns={
                "mean": "Avg Rating",
                "count": "Visit Count",
            }
        )
    )

    st.dataframe(
        popular_display,
        use_container_width=True,
        hide_index=True,
    )


# ==============================================================
# SAFE ENCODING
# ==============================================================

def safe_encode(
    encoders,
    column,
    value,
):

    encoder = encoders[column]

    value = str(value)

    known_values = set(
        str(x)
        for x in encoder.classes_
    )

    if value not in known_values:
        return 0

    return int(
        encoder.transform(
            [value]
        )[0]
    )


# ==============================================================
# REGRESSION INPUT
# ==============================================================

def make_regression_input(
    attr_row,
    visit_mode,
    year,
    month,
):

    row = {

        "Continent_enc":
            safe_encode(
                encoders,
                "Continent",
                attr_row["Continent"],
            ),

        "Region_enc":
            safe_encode(
                encoders,
                "Region",
                attr_row["Region"],
            ),

        "Country_enc":
            safe_encode(
                encoders,
                "Country",
                attr_row["Country"],
            ),

        "CityName_enc":
            safe_encode(
                encoders,
                "CityName",
                attr_row["CityName"],
            ),

        "VisitMode_enc":
            safe_encode(
                encoders,
                "VisitMode",
                visit_mode,
            ),

        "AttractionType_enc":
            safe_encode(
                encoders,
                "AttractionType",
                attr_row["AttractionType"],
            ),

        "VisitYear":
            year,

        "VisitMonth":
            month,

        "UserVisitCount":
            float(
                df["UserVisitCount"]
                .median()
            ),

        "AttractionAvgRating":
            float(
                attr_row[
                    "AttractionAvgRating"
                ]
            ),

        "AttractionVisitCount":
            float(
                attr_row[
                    "AttractionVisitCount"
                ]
            ),
    }

    return pd.DataFrame(
        [row]
    )


# ==============================================================
# REGRESSION TAB
# ==============================================================

with tab_predict:

    st.subheader(
        "Predict how a user might rate a visit"
    )

    st.caption(
        "Choose an attraction and trip details to estimate the expected rating."
    )

    colA, colB = st.columns(2)

    with colA:

        attraction_name = st.selectbox(
            "Attraction",
            sorted(
                df["Attraction"]
                .astype(str)
                .unique()
            ),
            key="pred_attraction",
        )

        attr_row = (
            df[
                df["Attraction"].astype(str)
                == attraction_name
            ]
            .iloc[0]
        )

        visit_mode = st.selectbox(
            "Visit Mode",
            sorted(
                df["VisitMode"]
                .astype(str)
                .unique()
            ),
            key="pred_mode",
        )

    with colB:

        min_year = int(
            df["VisitYear"].min()
        )

        max_year = int(
            df["VisitYear"].max()
        )

        default_year = min(
            max(
                2022,
                min_year,
            ),
            max_year,
        )

        year = st.slider(
            "Visit Year",
            min_year,
            max_year,
            default_year,
            key="pred_year",
        )

        month = st.slider(
            "Visit Month",
            1,
            12,
            6,
            key="pred_month",
        )

    if st.button(
        "Predict rating",
        type="primary",
    ):

        try:

            X = make_regression_input(
                attr_row,
                visit_mode,
                year,
                month,
            )

            X = X.reindex(
                columns=reg_features,
                fill_value=0,
            )

            prediction = float(
                np.asarray(
                    reg_model.predict(X)
                )[0]
            )

            prediction = float(
                np.clip(
                    prediction,
                    1,
                    5,
                )
            )

            st.success(
                f"Predicted rating: **{prediction:.2f} / 5**"
            )

            st.caption(
                f"This attraction's historical average is "
                f"{float(attr_row['AttractionAvgRating']):.2f} "
                f"across "
                f"{int(attr_row['AttractionVisitCount'])} visits."
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# ==============================================================
# CLASSIFICATION INPUT
# ==============================================================

def make_classification_input(
    attr_row,
    year,
    month,
):

    row = {

        "Continent_enc":
            safe_encode(
                encoders,
                "Continent",
                attr_row["Continent"],
            ),

        "Region_enc":
            safe_encode(
                encoders,
                "Region",
                attr_row["Region"],
            ),

        "Country_enc":
            safe_encode(
                encoders,
                "Country",
                attr_row["Country"],
            ),

        "CityName_enc":
            safe_encode(
                encoders,
                "CityName",
                attr_row["CityName"],
            ),

        "AttractionType_enc":
            safe_encode(
                encoders,
                "AttractionType",
                attr_row["AttractionType"],
            ),

        "VisitYear":
            year,

        "VisitMonth":
            month,

        "UserAvgRating":
            float(
                df["UserAvgRating"]
                .median()
            ),

        "UserVisitCount":
            float(
                df["UserVisitCount"]
                .median()
            ),

        "AttractionAvgRating":
            float(
                attr_row[
                    "AttractionAvgRating"
                ]
            ),

        "AttractionVisitCount":
            float(
                attr_row[
                    "AttractionVisitCount"
                ]
            ),
    }

    return pd.DataFrame(
        [row]
    )


# ==============================================================
# CLASSIFICATION TAB
# ==============================================================

with tab_classify:

    st.subheader(
        "Predict a user's likely visit mode"
    )

    st.caption(
        "Given an attraction and trip timing, estimate the likely visit mode."
    )

    colA, colB = st.columns(2)

    with colA:

        attraction_name2 = st.selectbox(
            "Attraction",
            sorted(
                df["Attraction"]
                .astype(str)
                .unique()
            ),
            key="clf_attraction",
        )

        attr_row2 = (
            df[
                df["Attraction"].astype(str)
                == attraction_name2
            ]
            .iloc[0]
        )

    with colB:

        min_year2 = int(
            df["VisitYear"].min()
        )

        max_year2 = int(
            df["VisitYear"].max()
        )

        default_year2 = min(
            max(
                2022,
                min_year2,
            ),
            max_year2,
        )

        year2 = st.slider(
            "Visit Year",
            min_year2,
            max_year2,
            default_year2,
            key="clf_year",
        )

        month2 = st.slider(
            "Visit Month",
            1,
            12,
            6,
            key="clf_month",
        )

    if st.button(
        "Predict visit mode",
        type="primary",
    ):

        try:

            X = make_classification_input(
                attr_row2,
                year2,
                month2,
            )

            X = X.reindex(
                columns=clf_features,
                fill_value=0,
            )

            prediction_encoded = int(
                np.asarray(
                    clf_model.predict(X)
                )[0]
            )

            visit_encoder = (
                encoders["VisitMode"]
            )

            prediction_label = (
                visit_encoder
                .inverse_transform(
                    [prediction_encoded]
                )[0]
            )

            st.success(
                f"Predicted visit mode: **{prediction_label}**"
            )

            if hasattr(
                clf_model,
                "predict_proba",
            ):

                probabilities = (
                    np.asarray(
                        clf_model
                        .predict_proba(X)
                    )[0]
                )

                classes = np.asarray(
                    clf_model.classes_,
                    dtype=int,
                )

                labels = []

                valid_probabilities = []

                for index, class_id in enumerate(
                    classes
                ):

                    if (
                        0
                        <= class_id
                        < len(
                            visit_encoder.classes_
                        )
                    ):

                        labels.append(
                            visit_encoder
                            .classes_[
                                class_id
                            ]
                        )

                        valid_probabilities.append(
                            probabilities[index]
                        )

                probability_df = pd.DataFrame(
                    {
                        "Visit Mode":
                            labels,

                        "Probability":
                            valid_probabilities,
                    }
                )

                probability_df = (
                    probability_df
                    .sort_values(
                        "Probability",
                        ascending=False,
                    )
                )

                probability_df[
                    "Probability"
                ] = (
                    probability_df[
                        "Probability"
                    ]
                    .round(4)
                )

                st.dataframe(
                    probability_df,
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as error:

            st.error(
                f"Visit-mode prediction failed: {error}"
            )


# ==============================================================
# RECOMMENDATION TAB
# ==============================================================

with tab_recommend:

    st.subheader(
        "Get personalized attraction recommendations"
    )

    recommendation_type = st.radio(
        "Recommendation type",
        [
            "By existing User ID (collaborative)",
            "By an attraction you liked (content-based)",
        ],
    )

    # ==========================================================
    # EXISTING USER
    # ==========================================================

    if recommendation_type.startswith(
        "By existing"
    ):

        if "UserId" not in df.columns:

            st.warning(
                "UserId is not available in the dataset."
            )

            st.dataframe(
                POPULAR_ATTRACTIONS.head(5)[
                    [
                        "Attraction",
                        "AttractionType",
                        "CityName",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

        else:

            known_users = set(
                df["UserId"]
                .dropna()
                .astype(int)
                .unique()
            )

            user_min = int(
                df["UserId"].min()
            )

            user_max = int(
                df["UserId"].max()
            )

            user_id = st.number_input(
                "Enter a User ID",
                min_value=user_min,
                max_value=user_max,
                step=1,
            )

            if st.button(
                "Recommend attractions",
                type="primary",
            ):

                user_id = int(
                    user_id
                )

                # ------------------------------------------------
                # Unknown user
                # ------------------------------------------------

                if user_id not in known_users:

                    st.warning(
                        "No history found for this User ID. "
                        "Showing popular attractions instead."
                    )

                    st.dataframe(
                        POPULAR_ATTRACTIONS.head(5)[
                            [
                                "Attraction",
                                "AttractionType",
                                "CityName",
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

                # ------------------------------------------------
                # Known user
                # ------------------------------------------------

                else:

                    user_history = df[
                        df["UserId"]
                        .astype(int)
                        == user_id
                    ]

                    seen_ids = set(
                        user_history[
                            "AttractionId"
                        ]
                        .dropna()
                        .astype(int)
                        .unique()
                    )

                    candidates = (
                        recommendation_items
                        .copy()
                    )

                    candidates[
                        "AttractionId"
                    ] = (
                        candidates[
                            "AttractionId"
                        ]
                        .astype(int)
                    )

                    candidates = candidates[
                        ~candidates[
                            "AttractionId"
                        ]
                        .isin(
                            seen_ids
                        )
                    ]

                    if candidates.empty:

                        candidates = (
                            recommendation_items
                            .copy()
                        )

                    candidates["Score"] = (
                        candidates[
                            "AvgRating"
                        ].fillna(0)
                        * 0.7
                        +
                        np.log1p(
                            candidates[
                                "VisitCount"
                            ].fillna(0)
                        )
                        * 0.3
                    )

                    recommendations = (
                        candidates
                        .sort_values(
                            "Score",
                            ascending=False,
                        )
                        .head(5)
                    )

                    st.dataframe(
                        recommendations[
                            [
                                "Attraction",
                                "AttractionType",
                                "CityName",
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

    # ==========================================================
    # CONTENT BASED
    # ==========================================================

    else:

        attraction_options = (
            recommendation_items[
                "Attraction"
            ]
            .astype(str)
            .tolist()
        )

        selected_attraction = (
            st.selectbox(
                "Select an attraction",
                attraction_options,
            )
        )

        if st.button(
            "Find similar attractions",
            type="primary",
        ):

            try:

                matching_rows = (
                    recommendation_items[
                        recommendation_items[
                            "Attraction"
                        ].astype(str)
                        == str(
                            selected_attraction
                        )
                    ]
                )

                if matching_rows.empty:

                    st.warning(
                        "Attraction not found."
                    )

                else:

                    attraction_id = int(
                        matching_rows[
                            "AttractionId"
                        ].iloc[0]
                    )

                    all_ids = (
                        recommendation_items[
                            "AttractionId"
                        ]
                        .astype(int)
                        .tolist()
                    )

                    index = all_ids.index(
                        attraction_id
                    )

                    similarities = (
                        recommendation_similarity[
                            index
                        ]
                        .copy()
                    )

                    similarities[index] = -np.inf

                    top_indices = (
                        np.argsort(
                            -similarities
                        )[:5]
                    )

                    recommendations = (
                        recommendation_items
                        .iloc[
                            top_indices
                        ]
                    )

                    st.dataframe(
                        recommendations[
                            [
                                "Attraction",
                                "AttractionType",
                                "CityName",
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

            except Exception as error:

                st.error(
                    f"Recommendation failed: {error}"
                )


# ==============================================================
# FOOTER
# ==============================================================

st.divider()

st.caption(
    "Waypoint · Tourism Experience Analytics "
    "— data cleaning → EDA → regression / "
    "classification / recommendation → Streamlit"
)
