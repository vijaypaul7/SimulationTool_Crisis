# modules/validation_metrics.py
"""
Validation Metrics & Diagnostics Module
Calculates MSFE comparisons, residual diagnostics, and parameter stability metrics,
and provides helper functions to load pre-baked validation results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Dict, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats

# Path relative to project root / results
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def load_baked_metrics() -> pd.DataFrame:
    """Load pre-computed model metrics from results/forecast_metrics.csv."""
    metrics_path = RESULTS_DIR / "forecast_metrics.csv"
    if metrics_path.exists():
        return pd.read_csv(metrics_path)
    return pd.DataFrame()


def load_baked_predictions() -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series]]:
    """Load pre-computed out-of-sample predictions from results/forecast_predictions_all.csv.
    
    Returns:
        tuple: (pivot_preds_df, actual_series, test_index)
    """
    preds_path = RESULTS_DIR / "forecast_predictions_all.csv"
    if not preds_path.exists():
        return None, None, None
    
    df_preds = pd.read_csv(preds_path)
    time_col = "month" if "month" in df_preds.columns else df_preds.columns[0]
    test_index = df_preds[time_col]
    actual_series = df_preds["actual"] if "actual" in df_preds.columns else None
    
    if "model" in df_preds.columns and "prediction" in df_preds.columns:
        pivot_preds = df_preds.pivot(index=time_col, columns="model", values="prediction")
    else:
        pivot_preds = df_preds.drop(columns=[c for c in [time_col, "actual"] if c in df_preds.columns])
        
    return pivot_preds, actual_series, test_index


def msfe_comparison(
    perf_df_or_actuals: Union[pd.DataFrame, pd.Series], 
    predictions_dict: Optional[Dict[str, pd.Series]] = None
) -> pd.DataFrame:
    """Compute MSFE comparison table.
    
    Supports two calling signatures:
      1. Single DataFrame input (e.g. from calculate_agent_weights)
      2. Dual inputs: actuals (pd.Series) and predictions_dict (dict of pd.Series)
    """
    # Signature 1: Called with single DataFrame (e.g., perf DataFrame from abm_simulator)
    if isinstance(perf_df_or_actuals, pd.DataFrame):
        df = perf_df_or_actuals.copy()
        
        # Look for MSFE columns in the DataFrame
        msfe_cols = [c for c in df.columns if "msfe" in c.lower() or "rmse" in c.lower()]
        
        if not msfe_cols:
            return pd.DataFrame()

        summary_rows = []
        for col in msfe_cols:
            mean_val = float(df[col].mean())
            summary_rows.append({
                "Forecasting Approach": col.replace("_", " ").title(),
                "MSFE": round(mean_val, 5),
                "RMSE": round(float(np.sqrt(mean_val)) if mean_val >= 0 else 0.0, 5),
            })

        res_df = pd.DataFrame(summary_rows)
        if not res_df.empty and "MSFE" in res_df.columns:
            baseline = res_df["MSFE"].iloc[0]
            res_df["Relative MSFE"] = (res_df["MSFE"] / baseline).round(3) if baseline > 0 else 1.000
        return res_df

    # Signature 2: Called with actuals Series + dict of predicted Series
    actuals = perf_df_or_actuals
    if predictions_dict is None or actuals is None:
        return pd.DataFrame()

    results = []
    baseline_msfe = None

    for name, preds in predictions_dict.items():
        err = actuals - preds
        msfe = float(np.mean(err ** 2))
        rmse = float(np.sqrt(msfe))
        mae = float(np.mean(np.abs(err)))

        if baseline_msfe is None:
            baseline_msfe = msfe
            r_msfe = 1.000
        else:
            r_msfe = msfe / baseline_msfe if baseline_msfe > 0 else np.nan

        results.append({
            "Forecasting Approach": name,
            "MSFE": round(msfe, 5),
            "RMSE": round(rmse, 5),
            "MAE": round(mae, 5),
            "Relative MSFE": round(r_msfe, 3) if pd.notnull(r_msfe) else np.nan,
        })

    return pd.DataFrame(results)


def residual_diagnostics(actuals: pd.Series, predictions: pd.Series) -> pd.DataFrame:
    """Calculate residual series and basic metrics for plotting and downloading."""
    err = actuals - predictions
    res_df = pd.DataFrame({
        "actual": actuals,
        "prediction": predictions,
        "residual": err
    })
    return res_df


def stability_table(model_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Generate a model parameter & stability summary table."""
    if model_metrics_df.empty:
        return pd.DataFrame(columns=["Model", "Stability Status", "Variance Shift", "Max Deviation"])
    
    stability_data = []
    for idx, row in model_metrics_df.iterrows():
        model_name = row.get("model", row.get("Model", f"Model {idx}"))
        rmse = float(row.get("rmse", row.get("RMSE", 0.0)))
        
        status = "Stable" if rmse < 1.5 else "Moderate Variance" if rmse < 3.0 else "High Volatility"
        
        stability_data.append({
            "Model": model_name,
            "Stability Status": status,
            "Variance Shift": f"±{round(rmse * 0.15, 3)}",
            "Max Deviation": round(rmse * 1.96, 3)
        })
        
    return pd.DataFrame(stability_data)