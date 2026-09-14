# modules/data_loader.py
"""Cached data ingestion + cleaning — EXACT reproduction of Colab sections 1–8."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    CRITICAL_COLS, DATA_PATH, EPISODE_METADATA_PATH,
    INTERP_COLS, RENAME_MAP, DATE_COL,
)


@st.cache_data(show_spinner="Loading and cleaning UK macro data…")
def load_and_clean_data(filepath: Union[str, Path] = DATA_PATH) -> pd.DataFrame:
    """Full cleaning pipeline from Colab."""
    path = Path(filepath).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Could not find '{path}'.")

    # ---------- 1. FRESH LOAD ----------
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False, na=False)]
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).set_index(DATE_COL)

    # ---------- 2. RENAME ----------
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})
    if "year" in df.columns:
        df = df.drop(columns=["year"])

    # ---------- 3/4. TRIM USING FIRST/LAST VALID INDEX ----------
    present = [c for c in CRITICAL_COLS if c in df.columns]
    firsts = [df[c].first_valid_index() for c in present if df[c].first_valid_index() is not None]
    lasts = [df[c].last_valid_index() for c in present if df[c].last_valid_index() is not None]
    if firsts and lasts:
        df = df.loc[max(firsts):min(lasts)].copy()

    # ---------- 5. INTERPOLATE INTERIOR GAPS (time-aware) ----------
    for col in INTERP_COLS:
        if col in df.columns:
            df[col] = df[col].interpolate(method="time", limit_area="inside")

    # ---------- 6. BANK RATE FFILL/BFILL ----------
    if "bank_rate" in df.columns:
        df["bank_rate"] = df["bank_rate"].ffill().bfill()

    # ---------- 7. CONSUMPTION IQR OUTLIER + LINEAR INTERPOLATION ----------
    if "total_consumption_cvm" in df.columns:
        s = df["total_consumption_cvm"]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = (s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)
        df.loc[mask, "total_consumption_cvm"] = np.nan
        df["total_consumption_cvm"] = (
            df["total_consumption_cvm"]
            .interpolate(method="linear", limit_direction="both")
        )

    # ---------- 8. CPIH YoY IF STILL AN INDEX ----------
    # If cpih values are index-like (>20 and rising), create cpih_yoy but keep cpih.
    if "cpih" in df.columns:
        c = df["cpih"].dropna()
        looks_like_index = len(c) > 0 and c.max() > 20 and c.min() > 1
        if looks_like_index:
            df["cpih_yoy"] = c.pct_change(12, fill_method=None) * 100.0
            df["cpih_yoy"] = df["cpih_yoy"].replace([np.inf, -np.inf], np.nan).ffill().bfill()

    return df.reset_index()


@st.cache_data(show_spinner=False)
def load_episode_metadata(
    filepath: Union[str, Path] = EPISODE_METADATA_PATH,
) -> Dict[str, dict]:
    path = Path(filepath).expanduser().resolve()
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def get_monthly_index(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if DATE_COL in frame.columns:
        frame[DATE_COL] = pd.to_datetime(frame[DATE_COL])
        frame = frame.set_index(DATE_COL)
    frame.index = pd.DatetimeIndex(frame.index).to_period("M").to_timestamp()
    frame.index.name = DATE_COL
    return frame.sort_index()


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def available_indicators(df: pd.DataFrame) -> list:
    candidates = [
        "retail", "cpih", "cpih_yoy", "gva_yoy", "unemployment",
        "bank_rate", "petrol", "diesel", "total_consumption_cvm",
        "gva", "gva_mom", "awe_real_growth",
    ]
    return [c for c in candidates if c in df.columns]


def load_all() -> Tuple[pd.DataFrame, Dict[str, dict]]:
    return load_and_clean_data(), load_episode_metadata()