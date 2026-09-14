# config.py
"""Central configuration: paths, rename map, episodes, parameters."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "uk_macro_integrated.csv"
EPISODE_METADATA_PATH = DATA_DIR / "episode_metadata.json"

DATE_COL = "month"

# Exact RENAME_MAP from Colab section 2
RENAME_MAP = {
    "RSI:Volume Seasonally Adjusted:All Retailers ex fuel:All Business Index": "retail",
    "cpih_index": "cpih",
    "petrol_price": "petrol",
    "diesel_price": "diesel",
    "unemployment_rate": "unemployment",
    "gva_yoy_growth": "gva_yoy",
    "gva_mom_growth": "gva_mom",
    "gva_index": "gva",
}

# Exact CRITICAL_COLS from Colab section 4
CRITICAL_COLS = [
    "cpih", "petrol", "diesel", "retail",
    "unemployment", "bank_rate", "awe_real_growth",
]

# Exact INTERP_COLS from Colab section 5
INTERP_COLS = [
    "cpih", "petrol", "diesel", "unemployment", "retail",
    "awe_real_growth", "gva", "gva_mom", "gva_yoy",
]

# Exact EPISODES from Colab
EPISODES = {
    "2008 Financial Crisis":     {"start": "2008-09", "end": "2009-06"},
    "2011-2012 Eurozone Crisis": {"start": "2011-06", "end": "2012-03"},
    "2015-2016 Oil Price Crash": {"start": "2014-10", "end": "2016-02"},
    "COVID-19 Pandemic":         {"start": "2020-03", "end": "2021-06"},
    "2022 Inflation Surge":      {"start": "2022-03", "end": "2023-08"},
}

# Exact VARIABLES dict from Colab
VARIABLES = {
    "CPIH":         {"column": "cpih",         "label": "CPIH Index",         "metric": "pct", "group": "Prices"},
    "Retail Sales": {"column": "retail",       "label": "Retail Sales Index", "metric": "pct", "group": "Real Economy"},
    "GVA Growth":   {"column": "gva_yoy",      "label": "GVA Growth",         "metric": "pp",  "group": "Real Economy"},
    "Unemployment": {"column": "unemployment", "label": "Unemployment Rate",  "metric": "pp",  "group": "Real Economy"},
    "Petrol":       {"column": "petrol",       "label": "Petrol Price",       "metric": "pct", "group": "Prices"},
    "Diesel":       {"column": "diesel",       "label": "Diesel Price",       "metric": "pct", "group": "Prices"},
    "Bank Rate":    {"column": "bank_rate",    "label": "Bank Rate",          "metric": "pp",  "group": "Monetary Policy"},
}

# VAR ordering from Colab: "most exogenous first"
VAR_VARS = ["petrol", "diesel", "cpih", "retail", "unemployment", "bank_rate"]

UNi_camp = { "lamba":0.5}

# ABM parameters from Colab Stages 1–3
ABM_PARAMS = {"lambda": 0.5, "theta": 0.8, "alpha": 0.15, "beta": 0.40}

# Chronological split (Colab predictive-modelling section)
TRAIN_END = "2020-12-31"
TEST_START = "2021-01-01"

# Feature lags (Colab final predictive pipeline)
ENERGY_LAGS = [1, 3, 6, 12]
TARGET_LAGS = [1, 2, 3]
BEHAVIOUR_LAGS = [1, 3, 6, 12]
CONTROL_LAGS = [1, 3, 6, 12]

LABELS = {
    "retail": "Retail Sales",
    "cpih": "CPIH",
    "gva_yoy": "GVA Growth",
    "unemployment": "Unemployment",
    "bank_rate": "Bank Rate",
    "petrol": "Petrol",
    "diesel": "Diesel",
    "total_consumption_cvm": "Consumption",
    "gva": "GVA Index",
    "gva_mom": "GVA MoM",
    "awe_real_growth": "AWE Real Growth",
}