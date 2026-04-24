# NYC Urban Development Prediction

An end-to-end machine learning project predicting which areas of New York City are likely to experience the most urban development.

---

## Project Structure

```
nyc_urban_development/
│
├── data/
│   ├── raw/                      ← Downloaded source files (auto-generated, git-ignored)
│   ├── processed/                ← Cleaned individual datasets (auto-generated)
│   └── final/                    ← Merged, model-ready dataset (auto-generated)
│
├── outputs/
│   └── eda/                      ← All EDA figures and tables (auto-generated)
│
├── step1_data_collection.py      ← PROJECT STEP 1: Download all raw datasets
├── step2_data_cleaning.py        ← PROJECT STEP 2: Clean, engineer, and merge
├── step3_eda.py                  ← PROJECT STEP 3: Exploratory Data Analysis
│
├── requirements.txt              ← Python dependencies
└── README.md
```

---

## How to Run (Step by Step)

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/nyc_urban_development.git
cd nyc_urban_development
```

### 2. Create and activate a virtual environment

**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt):**
```bash
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell):**
```bash
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run each project step in order

```bash
# Step 1 — Download all raw data from NYC Open Data APIs
python step1_data_collection.py

# Step 2 — Clean, engineer features, and merge into final dataset
python step2_data_cleaning.py

# Step 3 — Run full EDA (saves all figures to outputs/eda/)
python step3_eda.py
```

> Each script is self-contained and clearly labeled. Run them in order — each step depends on the output of the previous one.

---

## Data Sources

| Dataset | Source | Description |
|---|---|---|
| DOB Building Permits | NYC Open Data `ipu4-2q9a` | New building & alteration permits (2015–present) |
| PLUTO | NYC Open Data `64uk-42ks` | Tax lot land use, zoning, building characteristics |
| MTA Subway Stations | NYC Open Data `kk4q-3rt2` | Station locations for transit proximity features |
| Property Rolling Sales | NYC Open Data `usep-8jbt` | Property transaction prices by neighborhood |

---

## Project Steps

### Step 1 — Data Collection (`step1_data_collection.py`)
Downloads all raw datasets directly from the NYC Open Data Socrata API. No manual downloads needed. Saves raw CSVs to `data/raw/`.

### Step 2 — Data Cleaning (`step2_data_cleaning.py`)
Cleans each dataset individually, then merges into a final model-ready file:
- Parses and validates dates and coordinates
- Normalizes borough names
- Removes invalid/implausible records
- Engineers features (building age, price/sqft, vacancy flag, FAR proxy)
- Computes spatial features (subway proximity via GeoPandas)
- Aggregates to community board × year grain
- Defines binary target variable `high_development`

### Step 3 — Exploratory Data Analysis (`step3_eda.py`)
Runs 9 EDA sections and saves all figures to `outputs/eda/`:
- `3.1` Dataset overview & descriptive statistics
- `3.2` Target variable distribution
- `3.3` Permit trends over time
- `3.4` Borough-level comparisons
- `3.5` Land use & vacancy analysis
- `3.6` Transit proximity vs. development
- `3.7` Property price signals
- `3.8` Correlation heatmap
- `3.9` KMeans clustering (unsupervised learning)

---

## Output — Feature Description

The final dataset (`data/final/nyc_urban_features.csv`) contains one row per (borough, community board, year):

| Feature | Description |
|---|---|
| `total_permits` | Total building permits issued |
| `new_buildings` | Count of new building (NB) permits |
| `major_alterations` | Count of major alteration (A1) permits |
| `permit_growth_yoy` | Year-over-year permit growth rate |
| `permits_3yr_avg` | Rolling 3-year average permit count |
| `avg_lot_area` | Average lot area (sq ft) |
| `avg_building_age` | Average age of buildings |
| `pct_vacant` | Proportion of vacant lots |
| `avg_assessed_value` | Average assessed property value |
| `avg_dist_to_subway_m` | Average distance to nearest subway (meters) |
| `pct_within_800m_subway` | Proportion of lots within 800m of a subway station |
| `borough_median_price` | Borough-level median property sale price |
| `borough_price_appreciation` | Year-over-year price appreciation |
| **`high_development`** | **Target variable** — 1 if top-quartile permit activity |

---

## Team Contributions

| Member | Role |
|---|---|
| [Name] | Step 1 & 2 — Data Collection & Cleaning |
| [Name] | Step 3 — Exploratory Data Analysis |
| [Name] | Step 4 & 5 — Modeling & Evaluation |
| [Name] | Step 6 — Report & Presentation |

---

## Notes

- All data is fetched live from NYC Open Data — no manual downloads required.
- The `data/` and `outputs/` directories are git-ignored. Re-run the pipeline to regenerate.
- Requires Python 3.9+.
- Runtime: Step 1 ~5 min, Step 2 ~3 min, Step 3 ~2 min.
