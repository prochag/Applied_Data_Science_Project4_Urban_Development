# NYC Urban Development Prediction

An end-to-end machine learning project predicting which areas of New York City are likely to experience the most urban development activity, defined as top-quartile building permit issuance within a given year.

---

## Project Structure

```
nyc_urban_development/
│
├── data/
│   ├── raw/                      ← Raw source files (committed — real NYC Open Data)
│   ├── processed/                ← Cleaned individual datasets (committed)
│   └── final/                    ← Merged, model-ready dataset (committed)
│
├── outputs/
│   └── eda/                      ← All EDA figures and tables (committed)
│
├── step1_data_collection.py      ← PROJECT STEP 1: Download all raw datasets
├── step2_data_cleaning.py        ← PROJECT STEP 2: Clean, engineer, and merge
├── step3_eda.py                  ← PROJECT STEP 3: Exploratory Data Analysis
├── run_pipeline.sh               ← Runs Steps 1–3 end-to-end in one command
│
├── requirements.txt              ← Python dependencies
└── README.md
```

> **Data is pre-committed.** Teammates working on Steps 4–6 can use `data/final/nyc_urban_features.csv`
> directly without running anything. To regenerate from live APIs (e.g. to pull fresher data),
> run `./run_pipeline.sh` — see instructions below.

---

## How to Run

### Option A — One command (recommended)

```bash
git clone https://github.com/YOUR_USERNAME/nyc_urban_development.git
cd nyc_urban_development
chmod +x run_pipeline.sh
./run_pipeline.sh
```

This creates a virtual environment, installs dependencies, and runs Steps 1–3 in sequence.
At the end it prints the exact `git add / commit / push` commands to commit all generated files.
Total runtime: ~10 minutes. Requires internet access for Step 1.

---

### Option B — Step by step

#### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/nyc_urban_development.git
cd nyc_urban_development
```

#### 2. Create and activate a virtual environment

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

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### 4. Run each step in order

```bash
python step1_data_collection.py   # ~5 min — downloads raw data from NYC Open Data
python step2_data_cleaning.py     # ~3 min — cleans, engineers features, merges
python step3_eda.py               # ~2 min — produces all EDA figures
```

#### 5. Commit the generated files

```bash
git add data/ outputs/ .gitignore README.md
git commit -m "Add real processed data and EDA outputs"
git push
```

---

## Data Sources

| Dataset | API Endpoint | Description |
|---|---|---|
| DOB Building Permits | NYC Open Data `ipu4-2q9a` | New building & major alteration permits (2015–present) |
| PLUTO Land Use | NYC Open Data `64uk-42ks` | Tax lot land use, zoning, building characteristics — all ~870k NYC lots |
| MTA Subway Stations | NY State Open Data `39hk-dx4f` | GTFS stop locations used for transit proximity features |
| Property Rolling Sales | NYC Open Data `usep-8jbt` | Arm's-length property transaction prices by neighborhood |

All datasets are fetched live from public Socrata APIs — no credentials or manual downloads required.

---

## Project Steps

### Step 1 — Data Collection (`step1_data_collection.py`)

Downloads all four raw datasets from the NYC Open Data Socrata API and NY State Open Data.
Saves unmodified CSVs to `data/raw/`. No API key is required; the script includes a short
inter-request pause to stay within Socrata's unauthenticated rate limits.

**Data quality challenges present in the raw files** (addressed in Step 2):
- Borough names encoded inconsistently: full names, abbreviations, and all-caps mixed
- `yearbuilt` contains placeholder values (0, 1, pre-1800) indicating missing data
- Thousands of $0 and $1 property sales (inter-family transfers, not market prices)
- Lots with `lotarea = 0` — data entry errors that would corrupt ratio features
- Dates arrive as ISO strings requiring parsing; some are null or malformed
- Coordinates fall outside NYC bounds in a small fraction of PLUTO records
- Duplicate permit records for the same building × date × type

### Step 2 — Data Cleaning (`step2_data_cleaning.py`)

Cleans each dataset individually, engineers spatial and temporal features, then merges everything
into a single model-ready file at the **community board × year** grain.

**Cleaning operations:**
- Date parsing and validation for `filing_date`, `issuance_date`, `sale_date`
- Borough name normalization to a consistent 5-category standard
- Removal of implausible records (year_built < 1800, lot_area = 0, price ≤ $1)
- Coordinate bounding box filter (40.4°–41.0° N, −74.3°–−73.6° W)
- 99th-percentile outlier capping on assessed value, price/sqft, building area, FAR
- Deduplication of permits on (BIN, filing_date, permit_type)

**Feature engineering:**
- `building_age` = current year − year built
- `value_per_sqft` = assessed value ÷ lot area
- `far_proxy` = building area ÷ lot area (Floor Area Ratio approximation)
- `approval_lag_days` = issuance date − filing date (proxy for permitting demand)
- `is_vacant` = binary flag for land use code 11 (Vacant Land)
- `dist_to_subway_m` = Euclidean distance to nearest MTA station (GeoPandas, EPSG:2263)
- `within_800m_subway` = 1 if within the standard 800 m Transit-Oriented Development radius
- `permit_growth_yoy` = year-over-year permit volume change per district
- `permits_3yr_avg` = rolling 3-year average (smooths single-year spikes)
- `price_appreciation_yoy` = YoY median sale price change per neighborhood
- **`high_development`** (target) = 1 if community board's permit count ≥ 75th percentile for that year

### Step 3 — Exploratory Data Analysis (`step3_eda.py`)

Runs 9 EDA sections and saves all outputs to `outputs/eda/`:

| Section | Content |
|---|---|
| 3.1 | Dataset overview, shape, null counts, descriptive statistics |
| 3.2 | Target variable distribution — overall and by borough |
| 3.3 | Permit trends over time — monthly citywide volume and annual borough breakdown |
| 3.4 | Borough-level comparison of permits, vacancy, assessed value, transit access |
| 3.5 | Land use distribution and vacant lot analysis (from PLUTO) |
| 3.6 | Transit proximity vs. development — TOD zones and distance scatter |
| 3.7 | Property price signals — median price trends and price/sqft distributions |
| 3.8 | Correlation heatmap of all numeric features vs. target |
| 3.9 | KMeans clustering (K=4) — elbow/silhouette selection, PCA visualization, cluster profiles |

---

## Output — Feature Description

The final dataset (`data/final/nyc_urban_features.csv`) contains one row per
(borough, community board, year):

| Feature | Description |
|---|---|
| `total_permits` | Total building permits issued |
| `new_buildings` | Count of new building (NB) permits |
| `major_alterations` | Count of major alteration (A1) permits |
| `avg_approval_lag` | Mean days between filing and permit issuance |
| `permit_growth_yoy` | Year-over-year permit volume growth rate |
| `permits_3yr_avg` | Rolling 3-year average permit count |
| `avg_lot_area` | Mean lot area (sq ft) across all lots in district |
| `avg_bldg_area` | Mean building area (sq ft) |
| `avg_floors` | Mean number of floors |
| `avg_building_age` | Mean age of buildings (years) |
| `pct_vacant` | Proportion of lots classified as vacant land |
| `avg_assessed_value` | Mean assessed property value ($) |
| `avg_value_per_sqft` | Mean assessed value per sq ft ($) |
| `avg_far_proxy` | Mean Floor Area Ratio (building area ÷ lot area) |
| `avg_dist_to_subway_m` | Mean distance to nearest subway station (meters) |
| `pct_within_800m_subway` | Proportion of lots within 800 m of a subway station |
| `num_lots` | Number of PLUTO lots in district |
| `borough_median_price` | Borough-level median property sale price ($) |
| `borough_price_appreciation` | Borough-level YoY median price appreciation |
| `borough_num_sales` | Borough-level total arm's-length sales count |
| **`high_development`** | **Target** — 1 if top-quartile permit activity for that year |


## Team Contributions

| Member | Role |
|---|---|
| Pablo Rocha Gomez | Step 1 & 2 — Data Collection & Cleaning |
| Pablo Rocha Gomez | Step 3 — Exploratory Data Analysis |
| [Name] | Step 4 & 5 — Modeling & Evaluation |
| [Name] | Step 6 — Report & Presentation |

---

## Requirements

- Python 3.9+
- See `requirements.txt` for package versions
- Internet access required for Step 1 (NYC Open Data Socrata API)