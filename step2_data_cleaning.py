"""
╔══════════════════════════════════════════════════════════════════════╗
║         PROJECT STEP 2 — DATA CLEANING & PREPARATION               ║
║         NYC Urban Development Prediction                            ║
╠══════════════════════════════════════════════════════════════════════╣
║  Purpose : Clean each raw dataset, engineer base features, compute  ║
║            spatial proximity, and merge into one model-ready file.  ║
║  Input   : data/raw/*.csv   (produced by step1_data_collection.py)  ║
║  Output  : data/processed/*.csv  — cleaned individual datasets      ║
║            data/final/nyc_urban_features.csv  — merged final table  ║
║  Run     : python step2_data_cleaning.py                            ║
╚══════════════════════════════════════════════════════════════════════╝

Cleaning operations per dataset:
  [1] Permits  — date parsing, coordinate validation, borough normalization,
                 deduplication, temporal feature extraction
  [2] PLUTO    — numeric coercion, invalid record removal, land use labeling,
                 vacancy flagging, outlier capping, engineered ratios
  [3] Subway   — coordinate coercion, null removal
  [4] Sales    — price filtering (remove $0/$1 transfers), outlier removal,
                 borough normalization, price-per-sqft calculation
  [5] Spatial  — subway proximity distance (meters) + 800m TOD flag via
                 GeoPandas spatial join (EPSG:2263 projection)
  [6–8] Merge  — aggregate to community board level, join all datasets,
                 define binary target variable 'high_development'
"""

import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# DIRECTORY SETUP
# ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent
RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FINAL_DIR     = ROOT / "data" / "final"

for d in [PROCESSED_DIR, FINAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
def log(msg: str):
    print(f"  {msg}")

def section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")

def load_raw(filename: str) -> pd.DataFrame:
    path = RAW_DIR / filename
    df = pd.read_csv(path, low_memory=False)
    log(f"Loaded {filename}  →  {len(df):,} rows, {df.shape[1]} columns")
    return df

def save_processed(df: pd.DataFrame, filename: str):
    path = PROCESSED_DIR / filename
    df.to_csv(path, index=False)
    log(f"✓ Saved → data/processed/{filename}  ({len(df):,} rows)")

def report_cleaning(label: str, before: int, after: int):
    dropped = before - after
    pct = (dropped / before * 100) if before > 0 else 0
    log(f"  [{label}] Rows before: {before:,}  →  after: {after:,}  "
        f"(dropped {dropped:,} = {pct:.1f}%)")


# ═════════════════════════════════════════════════════════════════════
# CLEAN 1 — DOB BUILDING PERMITS
# ═════════════════════════════════════════════════════════════════════
def clean_permits(df: pd.DataFrame) -> pd.DataFrame:
    section("CLEANING 1 — DOB Building Permits")
    before = len(df)

    # ── Rename ambiguous column ───────────────────────────────────────
    df = df.rename(columns={"bin__": "bin"})

    # ── Parse dates ───────────────────────────────────────────────────
    # filing_date: when permit was applied for (used as primary time signal)
    # issuance_date: when permit was approved
    for col in ["filing_date", "issuance_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    log(f"Date nulls after parse — filing: {df['filing_date'].isna().sum()}, "
        f"issuance: {df['issuance_date'].isna().sum()}")

    # ── Drop rows with no location or borough ─────────────────────────
    df = df.dropna(subset=["latitude", "longitude", "borough"])

    # ── Numeric cast coordinates ──────────────────────────────────────
    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])

    # ── Remove coordinates outside NYC bounding box ───────────────────
    # Valid NYC range: lat 40.4–41.0, lon -74.3 to -73.6
    df = df[
        df["latitude"].between(40.4, 41.0) &
        df["longitude"].between(-74.3, -73.6)
    ]
    report_cleaning("Coordinate filter", before, len(df))

    # ── Normalize borough names to title case ─────────────────────────
    # Raw data contains abbreviations (MN, BK) and all-caps strings
    borough_map = {
        "MANHATTAN": "Manhattan", "MN": "Manhattan",
        "BROOKLYN":  "Brooklyn",  "BK": "Brooklyn",
        "QUEENS":    "Queens",    "QN": "Queens",
        "BRONX":     "Bronx",     "BX": "Bronx",
        "STATEN ISLAND": "Staten Island", "SI": "Staten Island",
    }
    df["borough"] = (
        df["borough"].str.upper().str.strip()
        .map(borough_map)
        .fillna(df["borough"])
    )

    # ── Temporal features ─────────────────────────────────────────────
    df["permit_year"]  = df["filing_date"].dt.year
    df["permit_month"] = df["filing_date"].dt.month

    # Days between filing and issuance (approval lag — can signal demand)
    df["approval_lag_days"] = (
        df["issuance_date"] - df["filing_date"]
    ).dt.days.clip(lower=0)

    # ── Encode permit type ────────────────────────────────────────────
    # NB = New Building (strongest development signal)
    # A1 = Major Alteration (significant renovation/expansion)
    df["is_new_building"] = (df["permit_type"] == "NB").astype(int)

    # ── Remove duplicates ─────────────────────────────────────────────
    # Same building (bin) can have multiple filings; deduplicate on
    # building + date + type to avoid double-counting
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["bin", "filing_date", "permit_type"])
    log(f"Duplicates removed: {before_dedup - len(df):,}")

    report_cleaning("Total permits", before, len(df))
    save_processed(df, "permits_clean.csv")
    return df


# ═════════════════════════════════════════════════════════════════════
# CLEAN 2 — PLUTO (Land Use)
# ═════════════════════════════════════════════════════════════════════
def clean_pluto(df: pd.DataFrame) -> pd.DataFrame:
    section("CLEANING 2 — PLUTO Land Use Data")
    before = len(df)

    # ── Numeric coercion ──────────────────────────────────────────────
    numeric_cols = ["lotarea", "bldgarea", "numfloors",
                    "yearbuilt", "assesstot", "latitude", "longitude"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Remove implausible records ────────────────────────────────────
    # yearbuilt < 1800 are placeholder/error values in PLUTO
    df = df[df["yearbuilt"] > 1800]
    # lotarea = 0 means lot has no usable area (data error)
    df = df[df["lotarea"] > 0]
    # bbl is the primary key — must not be null
    df = df.dropna(subset=["latitude", "longitude", "bbl"])
    report_cleaning("Invalid records", before, len(df))

    # ── Coordinate filter ─────────────────────────────────────────────
    df = df[
        df["latitude"].between(40.4, 41.0) &
        df["longitude"].between(-74.3, -73.6)
    ]

    # ── Land use labels ───────────────────────────────────────────────
    # NYC standard land use codes (01–11)
    landuse_map = {
        "01": "One/Two Family",         "02": "Multi-Family Walkup",
        "03": "Multi-Family Elevator",   "04": "Mixed Residential/Commercial",
        "05": "Commercial/Office",       "06": "Industrial",
        "07": "Transportation",          "08": "Public Facilities",
        "09": "Open Space",              "10": "Parking",
        "11": "Vacant Land",
    }
    df["landuse_desc"] = (
        df["landuse"].astype(str).str.zfill(2)
        .map(landuse_map).fillna("Other")
    )

    # ── Vacancy flag ──────────────────────────────────────────────────
    # Vacant land (code 11) is the most actionable development opportunity
    df["is_vacant"] = (df["landuse"].astype(str).str.zfill(2) == "11").astype(int)
    log(f"Vacant lots identified: {df['is_vacant'].sum():,} "
        f"({df['is_vacant'].mean()*100:.1f}% of total)")

    # ── Engineered features ───────────────────────────────────────────
    current_year = pd.Timestamp.now().year
    df["building_age"] = (current_year - df["yearbuilt"]).clip(lower=0)

    # Assessed value per square foot (economic density signal)
    df["value_per_sqft"] = np.where(
        df["lotarea"] > 0,
        df["assesstot"] / df["lotarea"],
        np.nan
    )

    # Floor Area Ratio proxy: building area / lot area
    df["far_proxy"] = np.where(
        df["lotarea"] > 0,
        df["bldgarea"] / df["lotarea"],
        np.nan
    )

    # ── Outlier capping (99th percentile) ─────────────────────────────
    # NYC has extreme outliers in assessed value (e.g. One World Trade)
    # Capping prevents these from distorting model training
    for col in ["assesstot", "value_per_sqft", "bldgarea", "far_proxy"]:
        q99 = df[col].quantile(0.99)
        capped = (df[col] > q99).sum()
        df[col] = df[col].clip(upper=q99)
        if capped > 0:
            log(f"  Capped {capped:,} outliers in '{col}' (threshold: {q99:,.1f})")

    report_cleaning("Total PLUTO", before, len(df))
    save_processed(df, "pluto_clean.csv")
    return df


# ═════════════════════════════════════════════════════════════════════
# CLEAN 3 — MTA SUBWAY STATIONS
# ═════════════════════════════════════════════════════════════════════
def clean_subway(df: pd.DataFrame) -> pd.DataFrame:
    section("CLEANING 3 — MTA Subway Stations")
    before = len(df)

    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])

    report_cleaning("Subway stations", before, len(df))
    save_processed(df, "subway_stations_clean.csv")
    return df


# ═════════════════════════════════════════════════════════════════════
# CLEAN 4 — NYC PROPERTY ROLLING SALES
# ═════════════════════════════════════════════════════════════════════
def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    section("CLEANING 4 — Property Rolling Sales")
    before = len(df)

    # ── Numeric coercion ──────────────────────────────────────────────
    df["sale_price"]       = pd.to_numeric(df["sale_price"],       errors="coerce")
    df["land_square_feet"] = pd.to_numeric(df["land_square_feet"], errors="coerce")
    df["gross_square_feet"]= pd.to_numeric(df["gross_square_feet"],errors="coerce")
    df["year_built"]       = pd.to_numeric(df["year_built"],       errors="coerce")
    df["sale_date"]        = pd.to_datetime(df["sale_date"],        errors="coerce")

    # ── Remove non-arm's-length transfers ────────────────────────────
    # Sales at $0 or $1 are inter-family transfers, foreclosures,
    # or legal transfers — not real market transactions
    before_price = len(df)
    df = df[df["sale_price"] > 1]
    log(f"Non-arm's-length transfers removed: {before_price - len(df):,}")

    # ── Remove top 1% price outliers ──────────────────────────────────
    upper = df["sale_price"].quantile(0.99)
    df = df[df["sale_price"] <= upper]

    # ── Drop rows with no neighborhood or date ────────────────────────
    df = df.dropna(subset=["neighborhood", "sale_date"])

    # ── Temporal features ─────────────────────────────────────────────
    df["sale_year"]    = df["sale_date"].dt.year
    df["sale_quarter"] = df["sale_date"].dt.quarter

    # ── Normalize borough codes ───────────────────────────────────────
    # Finance data uses numeric codes: 1=Manhattan, 2=Bronx, etc.
    borough_map = {
        "1": "Manhattan", "2": "Bronx",   "3": "Brooklyn",
        "4": "Queens",    "5": "Staten Island"
    }
    df["borough"] = df["borough"].astype(str).map(borough_map).fillna(df["borough"])

    # ── Price per square foot ─────────────────────────────────────────
    df["price_per_sqft"] = np.where(
        df["gross_square_feet"] > 0,
        df["sale_price"] / df["gross_square_feet"],
        np.nan
    )

    report_cleaning("Total sales", before, len(df))
    save_processed(df, "sales_clean.csv")
    return df


# ═════════════════════════════════════════════════════════════════════
# SPATIAL FEATURE — SUBWAY PROXIMITY
# ═════════════════════════════════════════════════════════════════════
def compute_subway_proximity(
    pluto_df: pd.DataFrame,
    subway_df: pd.DataFrame,
    radius_m: float = 800
) -> pd.DataFrame:
    """
    For every PLUTO tax lot, compute:
      - dist_to_subway_m     : distance (meters) to the nearest subway station
      - within_800m_subway   : 1 if within 800m (~half mile), else 0

    Method: Projects both datasets to NY State Plane (EPSG:2263, units = feet)
    for accurate Euclidean distance, then uses GeoPandas sjoin_nearest.

    800m is the standard Transit-Oriented Development (TOD) catchment radius
    used by urban planners to define walkable transit access zones.
    """
    section("SPATIAL FEATURE — Subway Proximity")
    log("Projecting to EPSG:2263 (NY State Plane — feet) for distance calculation ...")

    gdf_pluto = gpd.GeoDataFrame(
        pluto_df,
        geometry=gpd.points_from_xy(pluto_df["longitude"], pluto_df["latitude"]),
        crs="EPSG:4326"
    ).to_crs("EPSG:2263")

    gdf_subway = gpd.GeoDataFrame(
        subway_df,
        geometry=gpd.points_from_xy(subway_df["longitude"], subway_df["latitude"]),
        crs="EPSG:4326"
    ).to_crs("EPSG:2263")

    # Nearest station per lot
    joined = gpd.sjoin_nearest(
        gdf_pluto[["bbl", "geometry"]],
        gdf_subway[["name", "geometry"]],
        how="left",
        distance_col="dist_to_subway_ft"
    )

    # Convert feet → meters
    joined["dist_to_subway_m"] = joined["dist_to_subway_ft"] * 0.3048

    # TOD flag
    radius_ft = radius_m / 0.3048
    joined["within_800m_subway"] = (joined["dist_to_subway_ft"] <= radius_ft).astype(int)
    pct_tod = joined["within_800m_subway"].mean() * 100
    log(f"Lots within 800m of subway: {joined['within_800m_subway'].sum():,} ({pct_tod:.1f}%)")

    proximity = joined[["bbl", "dist_to_subway_m", "within_800m_subway"]].drop_duplicates("bbl")
    pluto_df  = pluto_df.merge(proximity, on="bbl", how="left")

    return pluto_df


# ═════════════════════════════════════════════════════════════════════
# AGGREGATION 1 — PERMITS → COMMUNITY BOARD LEVEL
# ═════════════════════════════════════════════════════════════════════
def aggregate_permits(permits_df: pd.DataFrame) -> pd.DataFrame:
    section("AGGREGATION 1 — Permits → Community Board × Year")

    agg = (
        permits_df
        .groupby(["borough", "community_board", "permit_year"])
        .agg(
            total_permits      = ("permit_type",    "count"),
            new_buildings      = ("is_new_building", "sum"),
            major_alterations  = ("is_new_building", lambda x: (x == 0).sum()),
            avg_approval_lag   = ("approval_lag_days", "mean"),
        )
        .reset_index()
    )

    # Year-over-year permit growth rate
    agg = agg.sort_values(["borough", "community_board", "permit_year"])
    agg["permit_growth_yoy"] = (
        agg.groupby(["borough", "community_board"])["total_permits"]
        .pct_change()
    )

    # Rolling 3-year average (smooths one-off spikes)
    agg["permits_3yr_avg"] = (
        agg.groupby(["borough", "community_board"])["total_permits"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    save_processed(agg, "permits_aggregated.csv")
    return agg


# ═════════════════════════════════════════════════════════════════════
# AGGREGATION 2 — SALES → NEIGHBORHOOD LEVEL
# ═════════════════════════════════════════════════════════════════════
def aggregate_sales(sales_df: pd.DataFrame) -> pd.DataFrame:
    section("AGGREGATION 2 — Sales → Neighborhood × Year")

    agg = (
        sales_df
        .groupby(["borough", "neighborhood", "sale_year"])
        .agg(
            median_sale_price    = ("sale_price",     "median"),
            median_price_per_sqft= ("price_per_sqft", "median"),
            num_sales            = ("sale_price",      "count"),
        )
        .reset_index()
    )

    agg = agg.sort_values(["borough", "neighborhood", "sale_year"])
    agg["price_appreciation_yoy"] = (
        agg.groupby(["borough", "neighborhood"])["median_sale_price"]
        .pct_change()
    )

    save_processed(agg, "sales_aggregated.csv")
    return agg


# ═════════════════════════════════════════════════════════════════════
# FINAL MERGE — BUILD MODEL-READY DATASET
# ═════════════════════════════════════════════════════════════════════
def build_final_dataset(
    permits_agg: pd.DataFrame,
    pluto_clean: pd.DataFrame,
    sales_agg:   pd.DataFrame,
) -> pd.DataFrame:
    section("FINAL MERGE — Building Model-Ready Dataset")

    # ── PLUTO: aggregate to community district level ──────────────────
    pluto_agg = (
        pluto_clean
        .groupby(["borough", "cd"])
        .agg(
            avg_lot_area           = ("lotarea",            "mean"),
            avg_bldg_area          = ("bldgarea",           "mean"),
            avg_floors             = ("numfloors",          "mean"),
            avg_building_age       = ("building_age",       "mean"),
            pct_vacant             = ("is_vacant",          "mean"),
            avg_assessed_value     = ("assesstot",          "mean"),
            avg_value_per_sqft     = ("value_per_sqft",     "mean"),
            avg_far_proxy          = ("far_proxy",          "mean"),
            avg_dist_to_subway_m   = ("dist_to_subway_m",   "mean"),
            pct_within_800m_subway = ("within_800m_subway", "mean"),
            num_lots               = ("bbl",                "count"),
        )
        .reset_index()
        .rename(columns={"cd": "community_board"})
    )

    # Normalize join key
    pluto_agg["community_board"]    = pluto_agg["community_board"].astype(str).str.strip()
    permits_agg["community_board"]  = permits_agg["community_board"].astype(str).str.strip()

    # ── Sales: roll up to borough × year (broadest reliable join key) ─
    sales_borough = (
        sales_agg
        .groupby(["borough", "sale_year"])
        .agg(
            borough_median_price       = ("median_sale_price",    "median"),
            borough_price_appreciation = ("price_appreciation_yoy","mean"),
            borough_num_sales          = ("num_sales",             "sum"),
        )
        .reset_index()
        .rename(columns={"sale_year": "permit_year"})
    )

    # ── Merge all datasets ────────────────────────────────────────────
    df = permits_agg.merge(pluto_agg,    on=["borough", "community_board"], how="left")
    df = df.merge(sales_borough,         on=["borough", "permit_year"],     how="left")

    # ── Define target variable ────────────────────────────────────────
    # 'high_development' = 1 if district's permit count is in the top
    # quartile for that year. This makes the label year-relative,
    # which is more robust than a fixed absolute threshold.
    yearly_q75 = df.groupby("permit_year")["total_permits"].transform(
        lambda x: x.quantile(0.75)
    )
    df["high_development"] = (df["total_permits"] >= yearly_q75).astype(int)
    log(f"Target distribution — high: {df['high_development'].sum():,}  "
        f"low: {(df['high_development'] == 0).sum():,}  "
        f"({df['high_development'].mean()*100:.1f}% positive)")

    # ── Final imputation ──────────────────────────────────────────────
    df = df.dropna(subset=["borough", "community_board"])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    df.fillna(0, inplace=True)

    # ── Save ──────────────────────────────────────────────────────────
    out_path = FINAL_DIR / "nyc_urban_features.csv"
    df.to_csv(out_path, index=False)
    log(f"✓ Final dataset saved → data/final/nyc_urban_features.csv")
    log(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} features")

    return df


# ═════════════════════════════════════════════════════════════════════
# CLEANING SUMMARY
# ═════════════════════════════════════════════════════════════════════
def print_cleaning_summary(final_df: pd.DataFrame):
    print("\n" + "═" * 60)
    print("  STEP 2 COMPLETE — DATA CLEANING SUMMARY")
    print("═" * 60)
    print(f"  Final dataset shape  : {final_df.shape[0]:,} rows × {final_df.shape[1]} features")
    print(f"  Target balance       : {final_df['high_development'].value_counts().to_dict()}")
    print(f"  Years covered        : {int(final_df['permit_year'].min())} – {int(final_df['permit_year'].max())}")
    print(f"  Boroughs             : {sorted(final_df['borough'].unique())}")
    print(f"  Remaining nulls      : {final_df.isnull().sum().sum()}")
    print(f"\n  Output → data/final/nyc_urban_features.csv")
    print(f"  Processed files → data/processed/")
    print("═" * 60)


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "═" * 60)
    print("  PROJECT STEP 2 — DATA CLEANING & PREPARATION")
    print("  NYC Urban Development Prediction")
    print("═" * 60)

    # Load raw files produced by Step 1
    permits_raw = load_raw("permits_raw.csv")
    pluto_raw   = load_raw("pluto_raw.csv")
    subway_raw  = load_raw("subway_stations_raw.csv")
    sales_raw   = load_raw("sales_raw.csv")

    # Clean each dataset
    permits_clean = clean_permits(permits_raw)
    pluto_clean   = clean_pluto(pluto_raw)
    subway_clean  = clean_subway(subway_raw)
    sales_clean   = clean_sales(sales_raw)

    # Compute spatial proximity feature
    pluto_with_subway = compute_subway_proximity(pluto_clean, subway_clean)

    # Aggregate to modeling grain
    permits_agg = aggregate_permits(permits_clean)
    sales_agg   = aggregate_sales(sales_clean)

    # Merge into final dataset
    final_df = build_final_dataset(permits_agg, pluto_with_subway, sales_agg)

    print_cleaning_summary(final_df)
    return final_df


if __name__ == "__main__":
    main()
