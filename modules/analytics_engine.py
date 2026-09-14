# modules/analytics_engine.py
"""Econometric framework — crisis impact, stress, energy shocks, volatility, VAR."""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import EPISODES, VARIABLES, VAR_VARS

try:
    from statsmodels.tsa.api import VAR
    _HAS_STATSMODELS = True
except Exception:
    _HAS_STATSMODELS = False


# =====================================================================
# Episode impact
# =====================================================================
def get_episode_data(data: pd.DataFrame, start, end) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = data.copy()
    data["month"] = pd.to_datetime(data["month"])
    ep_mask = (data["month"] >= start) & (data["month"] <= end)
    ep = data.loc[ep_mask].copy()
    pre_start = start - pd.DateOffset(months=12)
    pre_mask = (data["month"] >= pre_start) & (data["month"] < start)
    return ep, data.loc[pre_mask].copy()


def calculate_impact(ep_mean, pre_mean, metric: str):
    if pd.isna(ep_mean) or pd.isna(pre_mean):
        return np.nan
    if metric == "pp":
        return ep_mean - pre_mean
    if metric == "pct":
        if pre_mean == 0:
            return np.nan
        return (ep_mean - pre_mean) / pre_mean * 100
    return np.nan


def format_impact(value, metric: str) -> str:
    if pd.isna(value):
        return "N/A"
    if metric == "pp":
        return f"{value:+.2f} pp"
    if metric == "pct":
        return f"{value:+.2f}%"
    return f"{value:+.2f}"


def calculate_episode_impacts(df: pd.DataFrame, episodes: Mapping = EPISODES) -> pd.DataFrame:
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"])
    records = []
    for ep_name, ep_dates in episodes.items():
        start = pd.to_datetime(ep_dates["start"])
        end = pd.to_datetime(ep_dates["end"])
        ep_data, pre_data = get_episode_data(df, start, end)

        for name, cfg in VARIABLES.items():
            col = cfg["column"]
            metric = cfg["metric"]
            if col not in df.columns:
                continue
            ep_s = ep_data[col].dropna()
            pre_s = pre_data[col].dropna()
            ep_mean = ep_s.mean()
            pre_mean = pre_s.mean()
            impact = calculate_impact(ep_mean, pre_mean, metric)
            records.append({
                "Episode": ep_name, "Variable": name, "Group": cfg["group"],
                "Metric": metric, "Episode Mean": ep_mean, "Pre Mean": pre_mean,
                "Impact": impact, "Min": ep_s.min(), "Max": ep_s.max(),
                "N Episode": len(ep_s), "N Pre": len(pre_s),
            })
    return pd.DataFrame(records)


def calculate_bank_rate_response(df: pd.DataFrame, episodes: Mapping = EPISODES) -> pd.DataFrame:
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"])
    rows = []
    for ep_name, ep_dates in episodes.items():
        start = pd.to_datetime(ep_dates["start"])
        end = pd.to_datetime(ep_dates["end"])
        ep_data, pre_data = get_episode_data(df, start, end)
        if "bank_rate" not in df.columns:
            continue
        ep_s = ep_data["bank_rate"].dropna()
        pre_s = pre_data["bank_rate"].dropna()
        if len(ep_s) == 0:
            continue
        rows.append({
            "Episode": ep_name,
            "Start Rate (%)": ep_s.iloc[0],
            "End Rate (%)": ep_s.iloc[-1],
            "Episode Average (%)": ep_s.mean(),
            "Pre-Episode Average (%)": pre_s.mean(),
            "Average Change (pp)": ep_s.mean() - pre_s.mean(),
            "Start → End (pp)": ep_s.iloc[-1] - ep_s.iloc[0],
        })
    return pd.DataFrame(rows)


def calculate_stress_scores(
    df: pd.DataFrame, episodes: Mapping = EPISODES
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"])
    stress_vars = ["GVA Growth", "Unemployment", "Retail Sales"]
    records = []
    for ep_name, ep_dates in episodes.items():
        start = pd.to_datetime(ep_dates["start"])
        end = pd.to_datetime(ep_dates["end"])
        ep_data, pre_data = get_episode_data(df, start, end)
        scores = []
        for var_name in stress_vars:
            col = VARIABLES[var_name]["column"]
            ep_s = ep_data[col].dropna()
            pre_s = pre_data[col].dropna()
            if len(ep_s) == 0 or len(pre_s) < 3:
                continue
            ep_mean, pre_mean, pre_std = ep_s.mean(), pre_s.mean(), pre_s.std()
            if pd.isna(pre_std) or pre_std == 0:
                continue
            std_change = (ep_mean - pre_mean) / pre_std
            stress = -std_change if var_name in ["GVA Growth", "Retail Sales"] else std_change
            scores.append(stress)
            records.append({"Episode": ep_name, "Variable": var_name, "Standardized Stress": stress})
        avg = np.mean(np.abs(scores)) if scores else np.nan
        records.append({"Episode": ep_name, "Variable": "OVERALL", "Standardized Stress": avg})
    result = pd.DataFrame(records)
    ranking = (
        result[result["Variable"] == "OVERALL"]
        .sort_values("Standardized Stress", ascending=False)
        .reset_index(drop=True)
    )
    return result, ranking


# =====================================================================
# Energy shocks
# =====================================================================
def identify_energy_shocks(
    df: pd.DataFrame,
    fuel_col: str = "petrol",
    threshold_method: str = "percentile",
    threshold_value: float = 90,
) -> Tuple[pd.DataFrame, Dict]:
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"])
    df["fuel_price_mom_pct"] = df[fuel_col].pct_change() * 100
    df["fuel_price_mom_pct"] = df["fuel_price_mom_pct"].replace([np.inf, -np.inf], np.nan)

    if threshold_method == "percentile":
        threshold = df["fuel_price_mom_pct"].dropna().quantile(threshold_value / 100)
        df["fuel_price_shock"] = (df["fuel_price_mom_pct"] > threshold).astype(int)
        shock_label = f"Top {100 - threshold_value}% MoM increases"
        lower = df["fuel_price_mom_pct"].dropna().quantile((100 - threshold_value) / 100)
        df["fuel_price_drop_shock"] = (df["fuel_price_mom_pct"] < lower).astype(int)
    else:
        mean = df["fuel_price_mom_pct"].dropna().mean()
        std = df["fuel_price_mom_pct"].dropna().std()
        threshold = mean + threshold_value * std
        df["fuel_price_shock"] = (df["fuel_price_mom_pct"] > threshold).astype(int)
        shock_label = f">{threshold_value} std deviations above mean"

    shock_months = df[df["fuel_price_shock"] == 1]["month"].tolist()
    summary = {
        "shock_months": shock_months,
        "shock_count": len(shock_months),
        "total_months": len(df),
        "shock_percentage": len(shock_months) / len(df) * 100,
        "threshold": threshold,
        "shock_label": shock_label,
        "mean_shock_increase": df[df["fuel_price_shock"] == 1]["fuel_price_mom_pct"].mean(),
        "max_shock_increase": df[df["fuel_price_shock"] == 1]["fuel_price_mom_pct"].max(),
    }
    return df, summary


def energy_shock_ttests(df: pd.DataFrame, fuel_col: str = "petrol") -> pd.DataFrame:
    """Compare cpih/gva_yoy/unemployment/bank_rate across shock vs non-shock months."""
    df, _ = identify_energy_shocks(df, fuel_col=fuel_col)
    mask = df["fuel_price_shock"] == 1
    rows = []
    for v in ["cpih", "gva_yoy", "unemployment", "bank_rate"]:
        if v not in df.columns:
            continue
        sm = df.loc[mask, v].mean()
        nm = df.loc[~mask, v].mean()
        _, p = stats.ttest_ind(df.loc[mask, v].dropna(), df.loc[~mask, v].dropna())
        rows.append({"Variable": v, "Shock mean": sm, "Non-shock mean": nm, "p-value": p})
    return pd.DataFrame(rows)


# =====================================================================
# Volatility
# =====================================================================
def calculate_volatility_metrics(
    df: pd.DataFrame, windows: Sequence[int] = (3, 6, 12, 24)
) -> Tuple[pd.DataFrame, Dict]:
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"])
    vol_vars = {
        "CPIH":              {"column": "cpih",                   "type": "index"},
        "Retail Sales":      {"column": "retail",                 "type": "index"},
        "GVA Growth":        {"column": "gva_yoy",                "type": "rate"},
        "Unemployment":      {"column": "unemployment",           "type": "rate"},
        "Petrol":            {"column": "petrol",                 "type": "price"},
        "Diesel":            {"column": "diesel",                 "type": "price"},
        "Bank Rate":         {"column": "bank_rate",              "type": "rate"},
        "Consumer Spending": {"column": "total_consumption_cvm",  "type": "level"},
    }
    summary = {}
    for name, cfg in vol_vars.items():
        col, vtype = cfg["column"], cfg["type"]
        if col not in df.columns or df[col].isna().all():
            continue
        if vtype == "rate":
            df[f"{col}_change"] = df[col].diff()
            unit = "pp"
        else:
            df[f"{col}_change"] = df[col].pct_change() * 100
            unit = "%"
        for w in windows:
            mp = max(1, w // 2)
            df[f"{col}_vol_{w}m"] = df[f"{col}_change"].rolling(w, min_periods=mp).std()
        vc = f"{col}_vol_12m"
        if vc in df.columns and not df[vc].isna().all():
            vs = df[vc].dropna()
            summary[name] = {
                "mean_vol": vs.mean(), "max_vol": vs.max(),
                "min_vol": vs.min(),
                "current_vol": vs.iloc[-1] if len(vs) else np.nan,
                "units": unit, "type": vtype,
            }
    return df, summary


# =====================================================================
# Lagged cross-correlation
# =====================================================================
def compute_lagged_correlations(
    df: pd.DataFrame, target: str, variable: str, max_lag: int = 12
) -> Tuple[List[int], List[float]]:
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"])
    lags = list(range(-max_lag, max_lag + 1))
    corrs = []
    for lag in lags:
        if lag < 0:
            corrs.append(df[target].corr(df[variable].shift(-lag)))
        else:
            corrs.append(df[target].shift(lag).corr(df[variable]))
    return lags, corrs


# =====================================================================
# VAR
# =====================================================================
@st.cache_resource(show_spinner="Fitting VAR model…")
def fit_var_model(
    df_hash: str,
    var_vars: Tuple[str, ...] = tuple(VAR_VARS),
    maxlags: int = 12,
    data_json: Optional[str] = None,
) -> Dict:
    """Cache-safe wrapper. Pass `data_json=df.to_json()` from callers."""
    if not _HAS_STATSMODELS:
        raise ImportError("statsmodels required.")
    if data_json is None:
        raise ValueError("data_json is required.")

    import io
    df = pd.read_json(io.StringIO(data_json))
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month").set_index("month")

    var_vars = [v for v in var_vars if v in df.columns]
    if len(var_vars) < 2:
        raise ValueError(f"Need ≥2 VAR vars; have {var_vars}")

    data = df[var_vars].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    data = data.diff().dropna()
    if len(data) < 20:
        raise ValueError(f"Only {len(data)} rows after differencing.")

    try:
        data.index.freq = pd.infer_freq(data.index)
    except Exception:
        pass

    model = VAR(data)
    sel = model.select_order(maxlags=maxlags)
    p = sel.aic if sel.aic is not None and sel.aic >= 1 else 1
    p = int(max(1, min(p, maxlags, len(data) // (len(var_vars) + 1) - 1)))
    results = model.fit(p)
    return {"results": results, "lag_order": p, "variables": var_vars, "data": data, "n_obs": len(data)}


def compute_irf(var_result: Mapping, periods: int = 24, orth: bool = True) -> pd.DataFrame:
    res = var_result["results"]
    variables = var_result["variables"]
    shock_index = 0
    irf = res.irf(int(periods))
    arr = np.asarray(irf.orth_irfs if orth else irf.irfs, dtype=float)
    records = []
    for i, response in enumerate(variables):
        for h in range(int(periods) + 1):
            records.append({"horizon": h, "response": response, "value": float(arr[h, i, shock_index])})
    return pd.DataFrame(records)


def var_summary_df(var_result: Mapping) -> pd.DataFrame:
    res = var_result["results"]
    return pd.DataFrame([
        ("Observations", f"{int(res.nobs)}"),
        ("Variables", ", ".join(var_result["variables"])),
        ("Lag order (AIC)", str(var_result["lag_order"])),
        ("AIC", f"{res.aic:.3f}"),
        ("BIC", f"{res.bic:.3f}"),
        ("Log-likelihood", f"{res.llf:.2f}"),
    ], columns=["Property", "Value"])


def granger_causality(var_result: Mapping, cause: Sequence[str], effect: str) -> pd.DataFrame:
    res = var_result["results"]
    try:
        test = res.test_causality(effect, list(cause), kind="f")
        return pd.DataFrame([{
            "Effect": effect, "Causes": ", ".join(cause),
            "F-stat": float(test.test_statistic),
            "p-value": float(test.pvalue),
        }])
    except Exception as exc:
        return pd.DataFrame([{"Effect": effect, "Causes": ", ".join(cause), "error": str(exc)}])


# =====================================================================
# Growth-rate analysis
# =====================================================================
def calculate_growth_rates(df: pd.DataFrame, periods: Sequence[int] = (1, 3, 6, 12)) -> pd.DataFrame:
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"])
    vars_map = {
        "cpih": {"is_rate": False}, "retail": {"is_rate": False},
        "gva_yoy": {"is_rate": True}, "unemployment": {"is_rate": True},
        "total_consumption_cvm": {"is_rate": False}, "petrol": {"is_rate": False},
        "diesel": {"is_rate": False}, "bank_rate": {"is_rate": True},
    }
    for col, cfg in vars_map.items():
        if col not in df.columns:
            continue
        if cfg["is_rate"]:
            for p in periods:
                df[f"{col}_mom_pp" if p == 1 else f"{col}_{p}m_pp"] = (
                    df[col].diff() if p == 1 else df[col].diff(p)
                )
            df[f"{col}_mom_pct"] = df[col].pct_change() * 100
        else:
            for p in periods:
                df[f"{col}_mom_pct" if p == 1 else f"{col}_{p}m_pct"] = (
                    df[col].pct_change() * 100 if p == 1 else df[col].pct_change(p) * 100
                )
            df[f"{col}_abs_1m"] = df[col].diff()
    return df