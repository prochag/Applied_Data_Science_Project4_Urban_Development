"""
╔══════════════════════════════════════════════════════════════════════╗
║         PROJECT STEP 1 — DATA COLLECTION                           ║
║         NYC Urban Development Prediction                            ║
╠══════════════════════════════════════════════════════════════════════╣
║  Purpose : Download all raw datasets from NYC Open Data APIs        ║
║  Output  : data/raw/*.csv  (raw, unmodified source files)           ║
║  Run     : python step1_data_collection.py                          ║
╚══════════════════════════════════════════════════════════════════════╝

Datasets collected:
  [1] DOB Building Permits     — NYC Dept of Buildings (2015–present)
  [2] PLUTO Land Use           — NYC Dept of City Planning
  [3] MTA Subway Stations      — MTA / NYC Open Data
  [4] Property Rolling Sales   — NYC Dept of Finance

All data is fetched from the NYC Open Data Socrata API.
No manual downloads are required.
"""

import warnings
import pandas as pd
from pathlib import Path
from sodapy import Socrata

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# DIRECTORY SETUP
# ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────
def log(msg: str):
    print(f"  {msg}")

def save_raw(df: pd.DataFrame, filename: str) -> Path:
    """Save a raw DataFrame to data/raw/ and report its size."""
    path = RAW_DIR / filename
    df.to_csv(path, index=False)
    log(f"✓ Saved → data/raw/{filename}  ({len(df):,} rows, {df.shape[1]} columns)")
    return path


# ═════════════════════════════════════════════════════════════════════
# DATASET 1 — DOB BUILDING PERMITS
# Source  : https://data.cityofnewyork.us/resource/ipu4-2q9a
# Why     : Permits are the primary signal of urban development activity.
#           We filter to New Building (NB) and Major Alteration (A1)
#           permit types from 2015 onward to capture the modern era.
# ═════════════════════════════════════════════════════════════════════
def collect_permits(limit: int = 300_000) -> pd.DataFrame:
    print("\n[DATASET 1] DOB Building Permits")
    print("  Source : NYC Open Data — DOB Permit Issuance (ipu4-2q9a)")
    print("  Filter : permit_type IN (NB, A1) AND filing_date >= 2015-01-01")

    client = Socrata("data.cityofnewyork.us", None)
    results = client.get(
        "ipu4-2q9a",
        where="filing_date >= '2015-01-01' AND permit_type IN ('NB', 'A1')",
        limit=limit,
        select=(
            "borough, bin__, block, lot, community_board, "
            "filing_date, issuance_date, "
            "permit_type, permit_subtype, job_type, "
            "latitude, longitude"
        )
    )
    df = pd.DataFrame.from_records(results)
    save_raw(df, "permits_raw.csv")
    return df


# ═════════════════════════════════════════════════════════════════════
# DATASET 2 — PLUTO (Primary Land Use Tax Lot Output)
# Source  : https://data.cityofnewyork.us/resource/64uk-42ks
# Why     : PLUTO describes every tax lot in NYC — its size, zoning,
#           land use category, building characteristics, and assessed
#           value. Essential for understanding the existing built
#           environment in each neighborhood.
# ═════════════════════════════════════════════════════════════════════
def collect_pluto(limit: int = 100_000) -> pd.DataFrame:
    print("\n[DATASET 2] PLUTO — Land Use & Tax Lot Data")
    print("  Source : NYC Open Data — MapPLUTO (64uk-42ks)")
    print("  Fields : bbl, zoning, land use, lot/building area, year built, assessed value")

    client = Socrata("data.cityofnewyork.us", None)
    results = client.get(
        "64uk-42ks",
        limit=limit,
        select=(
            "bbl, borough, block, lot, cd, "
            "zonedist1, landuse, "
            "lotarea, bldgarea, numfloors, yearbuilt, "
            "assesstot, latitude, longitude"
        )
    )
    df = pd.DataFrame.from_records(results)
    save_raw(df, "pluto_raw.csv")
    return df


# ═════════════════════════════════════════════════════════════════════
# DATASET 3 — MTA SUBWAY STATIONS
# Source  : https://data.cityofnewyork.us/resource/kk4q-3rt2
# Why     : Transit access is one of the strongest predictors of urban
#           development in NYC. Station locations are used to compute
#           proximity features (distance, within-800m flag) for each lot.
# ═════════════════════════════════════════════════════════════════════
def collect_subway_stations() -> pd.DataFrame:
    print("\n[DATASET 3] MTA Subway Stations")
    print("  Source : NYC Open Data — MTA Subway Stations (kk4q-3rt2)")
    print("  Fields : station name, line, borough, coordinates")

    client = Socrata("data.cityofnewyork.us", None)
    results = client.get(
        "kk4q-3rt2",
        limit=600,
        select="name, line, borough, latitude, longitude"
    )
    df = pd.DataFrame.from_records(results)
    save_raw(df, "subway_stations_raw.csv")
    return df


# ═════════════════════════════════════════════════════════════════════
# DATASET 4 — NYC PROPERTY ROLLING SALES
# Source  : https://data.cityofnewyork.us/resource/usep-8jbt
# Why     : Property prices are a leading economic indicator of where
#           developers are active. Rising prices signal demand that
#           attracts new construction. Filtered to arm's-length sales.
# ═════════════════════════════════════════════════════════════════════
def collect_sales(limit: int = 200_000) -> pd.DataFrame:
    print("\n[DATASET 4] NYC Property Rolling Sales")
    print("  Source : NYC Open Data — Rolling Sales (usep-8jbt)")
    print("  Fields : borough, neighborhood, building class, units, price, date")

    client = Socrata("data.cityofnewyork.us", None)
    results = client.get(
        "usep-8jbt",
        limit=limit,
        select=(
            "borough, neighborhood, building_class_category, "
            "tax_class_at_present, block, lot, zip_code, "
            "residential_units, commercial_units, total_units, "
            "land_square_feet, gross_square_feet, year_built, "
            "sale_price, sale_date"
        )
    )
    df = pd.DataFrame.from_records(results)
    save_raw(df, "sales_raw.csv")
    return df


# ═════════════════════════════════════════════════════════════════════
# COLLECTION SUMMARY
# ═════════════════════════════════════════════════════════════════════
def print_collection_summary(datasets: dict):
    print("\n" + "═" * 60)
    print("  STEP 1 COMPLETE — DATA COLLECTION SUMMARY")
    print("═" * 60)
    for name, df in datasets.items():
        print(f"  {name:<30} {len(df):>8,} rows  |  {df.shape[1]:>2} columns")
    print(f"\n  All raw files saved to: data/raw/")
    print("═" * 60)


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "═" * 60)
    print("  PROJECT STEP 1 — DATA COLLECTION")
    print("  NYC Urban Development Prediction")
    print("═" * 60)

    permits = collect_permits()
    pluto   = collect_pluto()
    subway  = collect_subway_stations()
    sales   = collect_sales()

    datasets = {
        "DOB Building Permits": permits,
        "PLUTO Land Use":       pluto,
        "MTA Subway Stations":  subway,
        "Property Sales":       sales,
    }

    print_collection_summary(datasets)
    return datasets


if __name__ == "__main__":
    main()
