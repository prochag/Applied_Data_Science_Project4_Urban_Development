"""
╔══════════════════════════════════════════════════════════════════════╗
║         PROJECT STEP 3 — EXPLORATORY DATA ANALYSIS (EDA)           ║
║         NYC Urban Development Prediction                            ║
╠══════════════════════════════════════════════════════════════════════╣
║  Purpose : Understand the structure, distributions, spatial         ║
║            patterns, and relationships in the cleaned dataset.      ║
║            Incorporates unsupervised learning (KMeans clustering)   ║
║            to discover neighborhood development archetypes.         ║
║  Input   : data/final/nyc_urban_features.csv                        ║
║            data/processed/*.csv                                     ║
║  Output  : outputs/eda/  — all figures as .png files                ║
║  Run     : python step3_eda.py                                      ║
╚══════════════════════════════════════════════════════════════════════╝

EDA Sections:
  [3.1] Dataset Overview & Descriptive Statistics
  [3.2] Target Variable Distribution
  [3.3] Permit Trends Over Time
  [3.4] Borough-Level Comparisons
  [3.5] Land Use & Vacancy Analysis
  [3.6] Transit Proximity vs. Development
  [3.7] Property Price Signals
  [3.8] Correlation Heatmap
  [3.9] Unsupervised Learning — KMeans Neighborhood Clustering
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
FINAL_DIR   = ROOT / "data" / "final"
PROC_DIR    = ROOT / "data" / "processed"
OUTPUT_DIR  = ROOT / "outputs" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Consistent color palette
BOROUGH_COLORS = {
    "Manhattan":    "#E63946",
    "Brooklyn":     "#457B9D",
    "Queens":       "#2A9D8F",
    "Bronx":        "#E9C46A",
    "Staten Island":"#9C6644",
}
PALETTE = list(BOROUGH_COLORS.values())

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
def section(num: str, title: str):
    print(f"\n{'═' * 60}")
    print(f"  EDA SECTION {num} — {title}")
    print(f"{'═' * 60}")

def save_fig(filename: str):
    path = OUTPUT_DIR / filename
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved → outputs/eda/{filename}")


# ─────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────
def load_data():
    final   = pd.read_csv(FINAL_DIR / "nyc_urban_features.csv")
    permits = pd.read_csv(PROC_DIR  / "permits_clean.csv",      low_memory=False)
    pluto   = pd.read_csv(PROC_DIR  / "pluto_clean.csv",        low_memory=False)
    sales   = pd.read_csv(PROC_DIR  / "sales_clean.csv",        low_memory=False)

    # Parse dates
    permits["filing_date"] = pd.to_datetime(permits["filing_date"], errors="coerce")
    sales["sale_date"]     = pd.to_datetime(sales["sale_date"],     errors="coerce")

    print(f"  Final dataset  : {final.shape[0]:,} rows × {final.shape[1]} features")
    print(f"  Permits        : {len(permits):,} rows")
    print(f"  PLUTO          : {len(pluto):,} rows")
    print(f"  Sales          : {len(sales):,} rows")
    return final, permits, pluto, sales


# ═════════════════════════════════════════════════════════════════════
# EDA 3.1 — DATASET OVERVIEW & DESCRIPTIVE STATISTICS
# ═════════════════════════════════════════════════════════════════════
def eda_overview(df: pd.DataFrame):
    section("3.1", "Dataset Overview & Descriptive Statistics")

    print("\n  Shape:", df.shape)
    print("\n  Column types:\n", df.dtypes.value_counts().to_string())
    print("\n  Null counts (top 10):\n",
          df.isnull().sum().sort_values(ascending=False).head(10).to_string())

    desc = df.describe().T
    print("\n  Descriptive statistics (numeric features):\n", desc.to_string())

    # Save descriptive stats to CSV for the report
    desc.to_csv(OUTPUT_DIR / "descriptive_stats.csv")
    print(f"  ✓ Saved → outputs/eda/descriptive_stats.csv")


# ═════════════════════════════════════════════════════════════════════
# EDA 3.2 — TARGET VARIABLE DISTRIBUTION
# ═════════════════════════════════════════════════════════════════════
def eda_target(df: pd.DataFrame):
    section("3.2", "Target Variable Distribution")

    counts = df["high_development"].value_counts()
    labels = ["Low Development\n(0)", "High Development\n(1)"]
    colors = ["#ADB5BD", "#E63946"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Target Variable: 'high_development'", fontsize=14, fontweight="bold")

    # Bar chart
    axes[0].bar(labels, counts.values, color=colors, edgecolor="white", width=0.5)
    axes[0].set_title("Class Counts")
    axes[0].set_ylabel("Number of Districts × Years")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 1, str(v), ha="center", fontweight="bold")

    # Pie chart
    axes[1].pie(counts.values, labels=labels, colors=colors,
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Class Proportions")

    plt.tight_layout()
    save_fig("3_2_target_distribution.png")

    # By borough
    fig, ax = plt.subplots(figsize=(10, 5))
    borough_target = (
        df.groupby("borough")["high_development"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    bars = ax.bar(
        borough_target["borough"],
        borough_target["high_development"] * 100,
        color=[BOROUGH_COLORS.get(b, "#888") for b in borough_target["borough"]],
        edgecolor="white"
    )
    ax.set_title("% of Years Classified as High Development by Borough", fontsize=13)
    ax.set_ylabel("% High Development")
    ax.set_ylim(0, 100)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{bar.get_height():.1f}%", ha="center", fontsize=10)
    plt.tight_layout()
    save_fig("3_2_target_by_borough.png")


# ═════════════════════════════════════════════════════════════════════
# EDA 3.3 — PERMIT TRENDS OVER TIME
# ═════════════════════════════════════════════════════════════════════
def eda_permit_trends(permits_df: pd.DataFrame):
    section("3.3", "Permit Trends Over Time")

    # Monthly citywide permit volume
    monthly = (
        permits_df
        .dropna(subset=["filing_date"])
        .set_index("filing_date")
        .resample("M")["permit_type"]
        .count()
        .reset_index()
        .rename(columns={"permit_type": "count", "filing_date": "month"})
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle("NYC Building Permit Activity Over Time", fontsize=14, fontweight="bold")

    # Citywide monthly volume
    axes[0].plot(monthly["month"], monthly["count"], color="#E63946", linewidth=1.5)
    axes[0].fill_between(monthly["month"], monthly["count"], alpha=0.2, color="#E63946")
    axes[0].set_title("Monthly Permit Volume (Citywide)")
    axes[0].set_ylabel("Number of Permits")
    axes[0].xaxis.set_major_locator(mticker.MaxNLocator(10))

    # By borough
    borough_annual = (
        permits_df
        .groupby(["borough", "permit_year"])
        .size()
        .reset_index(name="count")
    )
    for borough, color in BOROUGH_COLORS.items():
        sub = borough_annual[borough_annual["borough"] == borough]
        if len(sub):
            axes[1].plot(sub["permit_year"], sub["count"],
                         label=borough, color=color, marker="o", linewidth=2)
    axes[1].set_title("Annual Permit Volume by Borough")
    axes[1].set_ylabel("Number of Permits")
    axes[1].set_xlabel("Year")
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    save_fig("3_3_permit_trends.png")

    # NB vs A1 split
    type_annual = (
        permits_df
        .groupby(["permit_year", "permit_type"])
        .size()
        .reset_index(name="count")
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    for ptype, color in [("NB", "#E63946"), ("A1", "#457B9D")]:
        sub = type_annual[type_annual["permit_type"] == ptype]
        ax.bar(sub["permit_year"] + (0.2 if ptype == "A1" else -0.2),
               sub["count"], width=0.4, label=ptype, color=color, alpha=0.85)
    ax.set_title("Annual Permits: New Buildings (NB) vs. Major Alterations (A1)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    save_fig("3_3_permit_type_split.png")


# ═════════════════════════════════════════════════════════════════════
# EDA 3.4 — BOROUGH-LEVEL COMPARISONS
# ═════════════════════════════════════════════════════════════════════
def eda_borough_comparison(df: pd.DataFrame):
    section("3.4", "Borough-Level Comparisons")

    metrics = [
        ("total_permits",          "Total Permits"),
        ("pct_vacant",             "% Vacant Lots"),
        ("avg_assessed_value",     "Avg Assessed Value ($)"),
        ("pct_within_800m_subway", "% Lots Within 800m of Subway"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Borough-Level Feature Comparisons", fontsize=14, fontweight="bold")

    for ax, (col, label) in zip(axes.flatten(), metrics):
        agg = df.groupby("borough")[col].mean().sort_values(ascending=False)
        colors = [BOROUGH_COLORS.get(b, "#888") for b in agg.index]
        bars = ax.barh(agg.index, agg.values, color=colors, edgecolor="white")
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.invert_yaxis()
        for bar in bars:
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():,.1f}", va="center", fontsize=9)

    plt.tight_layout()
    save_fig("3_4_borough_comparisons.png")


# ═════════════════════════════════════════════════════════════════════
# EDA 3.5 — LAND USE & VACANCY ANALYSIS
# ═════════════════════════════════════════════════════════════════════
def eda_land_use(pluto_df: pd.DataFrame):
    section("3.5", "Land Use & Vacancy Analysis")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("PLUTO Land Use Distribution", fontsize=14, fontweight="bold")

    # Land use breakdown citywide
    landuse_counts = (
        pluto_df["landuse_desc"]
        .value_counts()
        .head(10)
    )
    axes[0].barh(landuse_counts.index, landuse_counts.values,
                 color=sns.color_palette("Blues_r", len(landuse_counts)),
                 edgecolor="white")
    axes[0].set_title("Top 10 Land Use Categories (Lot Count)")
    axes[0].set_xlabel("Number of Lots")
    axes[0].invert_yaxis()

    # Vacant lots by borough
    vacant_by_borough = (
        pluto_df.groupby("borough")["is_vacant"]
        .agg(["sum", "mean"])
        .reset_index()
        .rename(columns={"sum": "vacant_count", "mean": "vacant_pct"})
    )
    colors = [BOROUGH_COLORS.get(b, "#888") for b in vacant_by_borough["borough"]]
    axes[1].bar(
        vacant_by_borough["borough"],
        vacant_by_borough["vacant_pct"] * 100,
        color=colors, edgecolor="white"
    )
    axes[1].set_title("% Vacant Lots by Borough")
    axes[1].set_ylabel("% Vacant")
    for i, row in vacant_by_borough.iterrows():
        axes[1].text(i, row["vacant_pct"] * 100 + 0.1,
                     f"{row['vacant_count']:,}", ha="center", fontsize=9)

    plt.tight_layout()
    save_fig("3_5_land_use_vacancy.png")


# ═════════════════════════════════════════════════════════════════════
# EDA 3.6 — TRANSIT PROXIMITY VS. DEVELOPMENT
# ═════════════════════════════════════════════════════════════════════
def eda_transit(df: pd.DataFrame):
    section("3.6", "Transit Proximity vs. Development")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Transit Proximity & Urban Development", fontsize=14, fontweight="bold")

    # Avg permits by transit access group
    df["transit_group"] = pd.cut(
        df["avg_dist_to_subway_m"],
        bins=[0, 400, 800, 1500, np.inf],
        labels=["<400m", "400–800m", "800–1500m", ">1500m"]
    )
    transit_permits = df.groupby("transit_group")["total_permits"].mean().reset_index()
    axes[0].bar(transit_permits["transit_group"].astype(str),
                transit_permits["total_permits"],
                color=["#E63946", "#F4A261", "#2A9D8F", "#457B9D"],
                edgecolor="white")
    axes[0].set_title("Avg Permits by Distance to Nearest Subway")
    axes[0].set_xlabel("Distance to Subway")
    axes[0].set_ylabel("Avg Total Permits")

    # High development rate by TOD flag
    tod_agg = df.groupby("within_800m_subway")["high_development"].mean().reset_index()
    tod_agg["label"] = tod_agg["within_800m_subway"].map({
        0.0: "Outside 800m\n(Non-TOD)",
        1.0: "Within 800m\n(TOD Zone)"
    })
    # Handle case where column may be averaged (float between 0-1)
    # Map to nearest integer for label lookup
    tod_agg["label"] = tod_agg["within_800m_subway"].round().map({
        0: "Outside 800m\n(Non-TOD)",
        1: "Within 800m\n(TOD Zone)"
    })
    axes[1].bar(tod_agg["label"], tod_agg["high_development"] * 100,
                color=["#ADB5BD", "#E63946"], edgecolor="white")
    axes[1].set_title("High Development Rate: TOD vs. Non-TOD Zones")
    axes[1].set_ylabel("% Districts Classified as High Development")
    axes[1].set_ylim(0, 100)

    plt.tight_layout()
    save_fig("3_6_transit_proximity.png")

    # Scatter: distance to subway vs permits
    fig, ax = plt.subplots(figsize=(10, 6))
    for borough, color in BOROUGH_COLORS.items():
        sub = df[df["borough"] == borough]
        ax.scatter(sub["avg_dist_to_subway_m"], sub["total_permits"],
                   label=borough, color=color, alpha=0.5, s=30)
    ax.set_xlabel("Avg Distance to Nearest Subway (m)")
    ax.set_ylabel("Total Permits")
    ax.set_title("Distance to Subway vs. Permit Activity by Borough")
    ax.legend()
    plt.tight_layout()
    save_fig("3_6_transit_scatter.png")


# ═════════════════════════════════════════════════════════════════════
# EDA 3.7 — PROPERTY PRICE SIGNALS
# ═════════════════════════════════════════════════════════════════════
def eda_prices(sales_df: pd.DataFrame, df: pd.DataFrame):
    section("3.7", "Property Price Signals")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Property Market Signals", fontsize=14, fontweight="bold")

    # Median sale price by borough over time
    price_trend = (
        sales_df
        .groupby(["borough", "sale_year"])["sale_price"]
        .median()
        .reset_index()
    )
    for borough, color in BOROUGH_COLORS.items():
        sub = price_trend[price_trend["borough"] == borough]
        if len(sub):
            axes[0].plot(sub["sale_year"], sub["sale_price"] / 1e6,
                         label=borough, color=color, marker="o", linewidth=2)
    axes[0].set_title("Median Sale Price Over Time (by Borough)")
    axes[0].set_ylabel("Median Price ($M)")
    axes[0].set_xlabel("Year")
    axes[0].legend(fontsize=9)

    # Price per sqft distribution
    clip_val = sales_df["price_per_sqft"].quantile(0.95)
    clean_psf = sales_df[sales_df["price_per_sqft"].between(1, clip_val)]
    for borough, color in BOROUGH_COLORS.items():
        sub = clean_psf[clean_psf["borough"] == borough]["price_per_sqft"]
        if len(sub):
            axes[1].hist(sub, bins=40, alpha=0.5, label=borough, color=color)
    axes[1].set_title("Price per Sqft Distribution by Borough")
    axes[1].set_xlabel("Price per Sqft ($)")
    axes[1].set_ylabel("Frequency")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    save_fig("3_7_price_signals.png")

    # Price appreciation vs high development
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, color in [(0, "#ADB5BD"), (1, "#E63946")]:
        sub = df[df["high_development"] == label]
        ax.scatter(
            sub["borough_price_appreciation"],
            sub["total_permits"],
            alpha=0.4, s=25, color=color,
            label=f"{'High' if label else 'Low'} Development"
        )
    ax.set_xlabel("Borough-Level Price Appreciation (YoY)")
    ax.set_ylabel("Total Permits")
    ax.set_title("Price Appreciation vs. Permit Activity")
    ax.legend()
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    save_fig("3_7_appreciation_vs_permits.png")


# ═════════════════════════════════════════════════════════════════════
# EDA 3.8 — CORRELATION HEATMAP
# ═════════════════════════════════════════════════════════════════════
def eda_correlation(df: pd.DataFrame):
    section("3.8", "Correlation Heatmap")

    feature_cols = [
        "total_permits", "new_buildings", "major_alterations",
        "permit_growth_yoy", "permits_3yr_avg",
        "avg_lot_area", "avg_bldg_area", "avg_floors",
        "avg_building_age", "pct_vacant",
        "avg_assessed_value", "avg_value_per_sqft",
        "avg_dist_to_subway_m", "pct_within_800m_subway",
        "borough_median_price", "borough_price_appreciation",
        "high_development"
    ]
    available = [c for c in feature_cols if c in df.columns]
    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, linewidths=0.5, ax=ax,
        annot_kws={"size": 8}, vmin=-1, vmax=1
    )
    ax.set_title("Feature Correlation Heatmap\n(lower triangle only)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig("3_8_correlation_heatmap.png")

    # Top correlations with target
    target_corr = corr["high_development"].drop("high_development").sort_values()
    print("\n  Top correlations with 'high_development':")
    print(pd.concat([target_corr.head(5), target_corr.tail(5)]).to_string())


# ═════════════════════════════════════════════════════════════════════
# EDA 3.9 — UNSUPERVISED LEARNING: KMEANS NEIGHBORHOOD CLUSTERING
# ═════════════════════════════════════════════════════════════════════
def eda_clustering(df: pd.DataFrame):
    """
    Apply KMeans clustering to identify distinct neighborhood development
    archetypes. This unsupervised step helps us:
      1. Discover natural groupings not captured by the binary target
      2. Validate that our features separate meaningfully
      3. Generate insights for the EDA narrative (e.g., 'transit-rich
         high-value zones' vs. 'suburban low-density zones')

    Steps:
      - Select and scale numeric features
      - Use Elbow method + Silhouette score to choose optimal K
      - Fit final KMeans model
      - Visualize in 2D using PCA
      - Profile each cluster
    """
    section("3.9", "Unsupervised Learning — KMeans Neighborhood Clustering")

    cluster_features = [
        "total_permits", "pct_vacant", "avg_assessed_value",
        "avg_dist_to_subway_m", "pct_within_800m_subway",
        "avg_building_age", "avg_floors", "permits_3yr_avg",
        "borough_median_price",
    ]
    available = [c for c in cluster_features if c in df.columns]
    X = df[available].dropna()
    idx = X.index

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Elbow + Silhouette ────────────────────────────────────────────
    print("  Finding optimal K (2–8) ...")
    inertias, silhouettes = [], []
    K_range = range(2, 9)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        print(f"    K={k}  inertia={km.inertia_:,.0f}  silhouette={silhouette_score(X_scaled, labels):.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("KMeans: Choosing Optimal Number of Clusters", fontsize=13, fontweight="bold")

    axes[0].plot(K_range, inertias, marker="o", color="#E63946", linewidth=2)
    axes[0].set_title("Elbow Method (Inertia)")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(K_range, silhouettes, marker="o", color="#457B9D", linewidth=2)
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Silhouette Score")

    plt.tight_layout()
    save_fig("3_9_kmeans_elbow.png")

    # ── Fit final model (K=4) ─────────────────────────────────────────
    # K=4 is a strong choice for NYC: typically maps to
    # Manhattan core / outer borough transit / suburban / industrial
    OPTIMAL_K = 4
    print(f"\n  Fitting final KMeans model with K={OPTIMAL_K} ...")
    km_final = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
    cluster_labels = km_final.fit_predict(X_scaled)
    df.loc[idx, "cluster"] = cluster_labels

    # ── PCA for 2D visualization ──────────────────────────────────────
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_ * 100
    print(f"  PCA variance explained: PC1={var_explained[0]:.1f}%, PC2={var_explained[1]:.1f}%")

    cluster_colors = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A"]
    fig, ax = plt.subplots(figsize=(10, 7))
    for k in range(OPTIMAL_K):
        mask = cluster_labels == k
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            label=f"Cluster {k}", color=cluster_colors[k], alpha=0.6, s=40
        )
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}% variance)")
    ax.set_title(f"KMeans Clusters (K={OPTIMAL_K}) — PCA 2D Projection",
                 fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    save_fig("3_9_kmeans_pca.png")

    # ── Cluster profiles ──────────────────────────────────────────────
    print("\n  Cluster profiles (mean values):")
    profile = (
        df.loc[idx]
        .assign(cluster=cluster_labels)
        .groupby("cluster")[available + ["high_development"]]
        .mean()
        .round(2)
    )
    print(profile.T.to_string())
    profile.T.to_csv(OUTPUT_DIR / "cluster_profiles.csv")
    print(f"  ✓ Saved → outputs/eda/cluster_profiles.csv")

    # ── Cluster composition by borough ────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    cluster_borough = (
        df.loc[idx]
        .assign(cluster=cluster_labels)
        .groupby(["cluster", "borough"])
        .size()
        .unstack(fill_value=0)
    )
    cluster_borough.plot(
        kind="bar", stacked=True, ax=ax,
        color=[BOROUGH_COLORS.get(c, "#888") for c in cluster_borough.columns],
        edgecolor="white"
    )
    ax.set_title("Cluster Composition by Borough", fontsize=13, fontweight="bold")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of District-Year Observations")
    ax.legend(title="Borough", bbox_to_anchor=(1.01, 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    save_fig("3_9_cluster_borough_composition.png")

    # ── High development rate per cluster ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    hd_by_cluster = (
        df.loc[idx]
        .assign(cluster=cluster_labels)
        .groupby("cluster")["high_development"]
        .mean() * 100
    )
    ax.bar(
        [f"Cluster {k}" for k in hd_by_cluster.index],
        hd_by_cluster.values,
        color=cluster_colors[:OPTIMAL_K], edgecolor="white"
    )
    ax.set_title("% High Development by Cluster", fontsize=13, fontweight="bold")
    ax.set_ylabel("% Districts Labeled High Development")
    ax.set_ylim(0, 100)
    for i, v in enumerate(hd_by_cluster.values):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    save_fig("3_9_cluster_development_rate.png")

    return df


# ═════════════════════════════════════════════════════════════════════
# EDA SUMMARY
# ═════════════════════════════════════════════════════════════════════
def print_eda_summary():
    figs = list(OUTPUT_DIR.glob("*.png"))
    csvs = list(OUTPUT_DIR.glob("*.csv"))
    print("\n" + "═" * 60)
    print("  STEP 3 COMPLETE — EDA SUMMARY")
    print("═" * 60)
    print(f"  Figures saved  : {len(figs)}")
    print(f"  Tables saved   : {len(csvs)}")
    print(f"  Output folder  : outputs/eda/")
    print("\n  Figures generated:")
    for f in sorted(figs):
        print(f"    • {f.name}")
    print("═" * 60)


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "═" * 60)
    print("  PROJECT STEP 3 — EXPLORATORY DATA ANALYSIS")
    print("  NYC Urban Development Prediction")
    print("═" * 60)

    # Load
    section("LOAD", "Reading Cleaned Datasets")
    df, permits, pluto, sales = load_data()

    # Run all EDA sections
    eda_overview(df)
    eda_target(df)
    eda_permit_trends(permits)
    eda_borough_comparison(df)
    eda_land_use(pluto)
    eda_transit(df)
    eda_prices(sales, df)
    eda_correlation(df)
    eda_clustering(df)

    print_eda_summary()


if __name__ == "__main__":
    main()
