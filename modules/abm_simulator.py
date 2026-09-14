from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ABM_PARAMS


def calculate_forecasting_performance(
    df: pd.DataFrame,
    fuel_col: str = "petrol",
    theta: float = ABM_PARAMS["theta"],
) -> pd.DataFrame:
    data = df.copy()

    data["month"] = pd.to_datetime(data["month"])
    data = data.set_index("month").sort_index()

    if fuel_col not in data.columns:
        raise ValueError(
            f"Fuel column '{fuel_col}' was not found in the dataset. "
            f"Available columns: {list(data.columns)}"
        )

    data["fuel_return"] = (
        np.log(data[fuel_col] / data[fuel_col].shift(1)) * 100
    )

    data["fundamentalist_forecast"] = (
        data[fuel_col]
        .rolling(window=12, min_periods=12)
        .mean()
        .shift(1)
    )

    data["momentum_trend"] = (
        data[fuel_col] - data[fuel_col].shift(3)
    )

    data["extrapolator_forecast"] = (
        data[fuel_col] + theta * data["momentum_trend"]
    ).shift(1)

    data["actual_price"] = data[fuel_col].shift(-1)

    data["fundamentalist_error"] = (
        data["actual_price"] - data["fundamentalist_forecast"]
    ) ** 2

    data["extrapolator_error"] = (
        data["actual_price"] - data["extrapolator_forecast"]
    ) ** 2

    data["fundamentalist_msfe_12m"] = (
        data["fundamentalist_error"]
        .rolling(12, min_periods=12)
        .mean()
    )

    data["extrapolator_msfe_12m"] = (
        data["extrapolator_error"]
        .rolling(12, min_periods=12)
        .mean()
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        data["msfe_ratio"] = (
            data["fundamentalist_msfe_12m"]
            / data["extrapolator_msfe_12m"]
        )

    if "petrol" in data.columns and "diesel" in data.columns:
        p = np.log(
            data["petrol"] / data["petrol"].shift(1)
        ) * 100

        d = np.log(
            data["diesel"] / data["diesel"].shift(1)
        ) * 100

        data["energy_shock"] = (p + d) / 2

    else:
        data["energy_shock"] = data["fuel_return"]

    return data.reset_index()


def calculate_agent_weights(
    df: pd.DataFrame,
    lambda_val: float = ABM_PARAMS["lambda"],
) -> pd.DataFrame:
    df = df.copy()
    msfe_f = df["fundamentalist_msfe_12m"]
    msfe_e = df["extrapolator_msfe_12m"]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log(msfe_f / msfe_e)
    score = lambda_val * log_ratio
    score = score.clip(lower=-50, upper=50)
    df["fundamentalist_weight"] = (
        1.0 / (1.0 + np.exp(score))
    )
    df["extrapolator_weight"] = (
        1.0 - df["fundamentalist_weight"]
    )
    invalid = (
        msfe_f.isna()
        | msfe_e.isna()
        | ~np.isfinite(log_ratio)
    )
    
    df.loc[invalid, "fundamentalist_weight"] = np.nan
    df.loc[invalid, "extrapolator_weight"] = np.nan

    df["dominant_strategy"] = np.where(
        df["fundamentalist_weight"] > 0.5,
        "Fundamentalist",
        np.where(
            df["extrapolator_weight"] > 0.5,
            "Extrapolator",
            "Equal",
        ),
    )
    df["dominance_strength"] = np.where(
        df["fundamentalist_weight"] > 0.5,
        df["fundamentalist_weight"],
        df["extrapolator_weight"],
    )
    return df




def calculate_behavioural_response(
    df: pd.DataFrame,
    alpha: float = ABM_PARAMS["alpha"],
    beta: float = ABM_PARAMS["beta"],
) -> pd.DataFrame:
    df = df.copy()

    shock = df["energy_shock"]
    w_f = df["fundamentalist_weight"]
    w_e = df["extrapolator_weight"]

    df["fundamentalist_response"] = -alpha * shock
    df["extrapolator_response"] = -beta * shock

    df["behavioural_response"] = (
        w_f * df["fundamentalist_response"]
        + w_e * df["extrapolator_response"]
    )

    return df


def test_lag_specifications(
    df: pd.DataFrame,
    candidate_lags: Tuple[int, ...] = (6, 12, 18),
) -> pd.DataFrame:
    rows = []

    for lag in candidate_lags:
        lagged = df["behavioural_response"].shift(lag)

        actual = (
            df["retail_sales_change"]
            if "retail_sales_change" in df.columns
            else df.get("retail_pct")
        )

        if actual is None:
            continue

        mask = lagged.notna() & actual.notna()

        if mask.sum() == 0:
            continue

        from sklearn.metrics import mean_squared_error

        rows.append({
            "lag_months": lag,
            "observations": int(mask.sum()),
            "correlation": lagged[mask].corr(actual[mask]),
            "rmse": float(
                np.sqrt(
                    mean_squared_error(
                        actual[mask],
                        lagged[mask],
                    )
                )
            ),
        })

    return pd.DataFrame(rows)


def compute_gva_transmission(
    abm_df: pd.DataFrame,
    df_original: pd.DataFrame,
    lags: Tuple[int, ...] = (6, 12, 18),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from scipy import stats
    from sklearn.metrics import mean_squared_error

    dfo = df_original.copy()

    dfo["month"] = pd.to_datetime(dfo["month"])
    dfo = dfo.set_index("month")

    if "gva_yoy" not in dfo.columns:
        return pd.DataFrame(), pd.DataFrame()

    gva = dfo["gva_yoy"].copy()
    delta_gva = gva.diff()

    abm_idx = abm_df.copy()

    abm_idx["month"] = pd.to_datetime(abm_idx["month"])
    abm_idx = abm_idx.set_index("month")

    common = abm_idx.index.intersection(delta_gva.index)

    merged = pd.DataFrame(index=common)

    merged["behavioural_response"] = (
        abm_idx.loc[common, "behavioural_response"]
    )

    merged["delta_gva"] = delta_gva.loc[common]

    rows = []

    for lag in lags:
        lagged = merged["behavioural_response"].shift(lag)

        valid = pd.DataFrame({
            "br": lagged,
            "gva": merged["delta_gva"],
        }).dropna()

        if len(valid) < 5:
            continue

        pr, pp = stats.pearsonr(
            valid["br"],
            valid["gva"],
        )

        sr, sp = stats.spearmanr(
            valid["br"],
            valid["gva"],
        )

        slope, intercept = np.polyfit(
            valid["br"],
            valid["gva"],
            1,
        )

        y_pred = slope * valid["br"] + intercept

        rmse = float(
            np.sqrt(
                mean_squared_error(
                    valid["gva"],
                    y_pred,
                )
            )
        )

        rows.append({
            "lag_months": lag,
            "observations": len(valid),
            "pearson_r": pr,
            "pearson_p": pp,
            "spearman_r": sr,
            "spearman_p": sp,
            "slope": slope,
            "intercept": intercept,
            "rmse": rmse,
        })

    return pd.DataFrame(rows), merged.reset_index()


def run_full_abm(
    df: pd.DataFrame,
    fuel_col: str = "petrol",
    lambda_val: float = ABM_PARAMS["lambda"],
    theta: float = ABM_PARAMS["theta"],
    alpha: float = ABM_PARAMS["alpha"],
    beta: float = ABM_PARAMS["beta"],
) -> Dict[str, pd.DataFrame]:

    perf = calculate_forecasting_performance(
        df,
        fuel_col=fuel_col,
        theta=theta,
    )

    perf = calculate_agent_weights(
        perf,
        lambda_val=lambda_val,
    )

    perf = calculate_behavioural_response(
        perf,
        alpha=alpha,
        beta=beta,
    )

    return {
        "performance": perf,
    }


def simulate_shock_response(
    df: pd.DataFrame,
    shock_bps: float = 100.0,
    lambda_val: float = ABM_PARAMS["lambda"],
    theta: float = ABM_PARAMS["theta"],
    alpha: float = ABM_PARAMS["alpha"],
    beta: float = ABM_PARAMS["beta"],
    fuel_col: str = "petrol",
    horizon: int = 24,
) -> pd.DataFrame:

    shock = shock_bps / 100.0

    months = np.arange(horizon)

    decay = np.exp(-months / 8.0)

    f_resp = -alpha * shock * decay
    e_resp = -beta * shock * decay

    perf = calculate_forecasting_performance(
        df,
        fuel_col=fuel_col,
        theta=theta,
    )

    perf = calculate_agent_weights(
        perf,
        lambda_val=lambda_val,
    )

    valid_weights = perf[
        ["fundamentalist_weight", "extrapolator_weight"]
    ].dropna()

    if len(valid_weights):
        w_f = float(
            valid_weights["fundamentalist_weight"].iloc[-1]
        )

        w_e = float(
            valid_weights["extrapolator_weight"].iloc[-1]
        )

    else:
        w_f = 0.5
        w_e = 0.5

    aggregate_response = (
        w_f * f_resp
        + w_e * e_resp
    )

    return pd.DataFrame({
        "horizon": months,
        "fundamentalist_response": f_resp,
        "extrapolator_response": e_resp,
        "aggregate_response": aggregate_response,
        "w_fundamentalist": w_f,
        "w_extrapolator": w_e,
    })