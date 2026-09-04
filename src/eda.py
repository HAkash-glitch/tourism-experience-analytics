"""
Exploratory Data Analysis
Generates key charts into outputs/eda/ :
 - user distribution by continent
 - attraction type popularity vs avg rating
 - visit mode distribution
 - rating distribution
 - correlation heatmap of numeric features
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "eda"
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def main():
    df = pd.read_csv(PROCESSED / "master_dataset.csv")

    # 1. Transaction volume by attraction continent
    # NOTE: no user-home-location data is available in this dataset (no User table),
    # so this reflects WHERE THE ATTRACTIONS ARE, not where tourists live.
    plt.figure(figsize=(7, 4))
    df["Continent"].value_counts().plot(kind="bar", color="#4C72B0")
    plt.title("Transactions by Attraction Continent")
    plt.ylabel("Number of Transactions")
    plt.tight_layout()
    plt.savefig(OUT / "transactions_by_continent.png", dpi=120)
    plt.close()

    # 2. Attraction type popularity vs avg rating
    grp = df.groupby("AttractionType").agg(
        Visits=("TransactionId", "count"), AvgRating=("Rating", "mean")
    ).sort_values("Visits", ascending=False)
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.bar(grp.index, grp["Visits"], color="#55A868")
    ax1.set_ylabel("Visit Count")
    ax1.tick_params(axis="x", rotation=40)
    ax2 = ax1.twinx()
    ax2.plot(grp.index, grp["AvgRating"], color="#C44E52", marker="o")
    ax2.set_ylabel("Avg Rating")
    plt.title("Attraction Type: Popularity vs Avg Rating")
    plt.tight_layout()
    plt.savefig(OUT / "attraction_type_popularity_rating.png", dpi=120)
    plt.close()

    # 3. Visit mode distribution
    plt.figure(figsize=(6, 4))
    df["VisitMode"].value_counts().plot(kind="pie", autopct="%1.0f%%", ylabel="")
    plt.title("Visit Mode Distribution")
    plt.tight_layout()
    plt.savefig(OUT / "visit_mode_distribution.png", dpi=120)
    plt.close()

    # 4. Rating distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Rating", data=df, palette="viridis")
    plt.title("Rating Distribution")
    plt.tight_layout()
    plt.savefig(OUT / "rating_distribution.png", dpi=120)
    plt.close()

    # 5. Correlation heatmap (numeric)
    num_cols = ["VisitYear", "VisitMonth", "Rating"]
    plt.figure(figsize=(5, 4))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUT / "correlation_heatmap.png", dpi=120)
    plt.close()

    print("EDA charts saved to", OUT)


if __name__ == "__main__":
    main()
