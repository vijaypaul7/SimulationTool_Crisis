# modules/forecasting_models.py
"""ML forecasting benchmarks — Ridge, ElasticNet, HistGradientBoosting, XGBoost, LSTM."""
from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import os

# --- Silence TensorFlow C++ runtime noise (mutex.cc / absl / oneDNN) ----
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")        # 0=all, 3=errors only
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")       # disable oneDNN spam
os.environ.setdefault("KMP_WARNINGS", "0")                # Intel MKL/OpenMP
os.environ.setdefault("OMP_NUM_THREADS", "1")             # fewer thread-lock messages
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")     # avoid duplicate-runtime abort
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")          # absl logging

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    BEHAVIOUR_LAGS, CONTROL_LAGS, ENERGY_LAGS, TARGET_LAGS,
    TEST_START, TRAIN_END,
)
from modules.validation_metrics import evaluate_all

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)


# =====================================================================
# Behavioural feature auto-compute (removes hard CSV dependency)
# =====================================================================
def _ensure_behavioural_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute behavioural columns from abm_simulator if CSVs are missing.

    Colab wrote these out to disk (`agent_weights_results.csv`,
    `consumption_response_results.csv`). A fresh Streamlit checkout has
    neither, so we recompute them from the raw macro frame to keep the
    forecasting pipeline self-sufficient.
    """
    needed = {"fundamentalist_weight", "extrapolator_weight", "behavioural_response"}

    if needed.issubset(df.columns) and df["fundamentalist_weight"].notna().any():
        return df

    try:
        from modules.abm_simulator import (
            calculate_forecasting_performance,
            calculate_agent_weights,
            calculate_behavioural_response,
        )
    except Exception:
        return df

    fuel = None
    if "petrol" in df.columns:
        fuel = "petrol"
    elif "diesel" in df.columns:
        fuel = "diesel"
    if fuel is None:
        return df

    try:
        perf = calculate_forecasting_performance(df, fuel_col=fuel)
        perf = calculate_agent_weights(perf)
        perf = calculate_behavioural_response(perf)
    except Exception:
        return df

    keep = ["month", "fundamentalist_weight", "extrapolator_weight", "behavioural_response"]
    keep = [c for c in keep if c in perf.columns]

    out = df.drop(columns=[c for c in needed if c in df.columns], errors="ignore")
    out = out.merge(perf[keep], on="month", how="left")
    return out


# =====================================================================
# Feature engineering
# =====================================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"])

    # Auto-compute behavioural cols if absent
    df = _ensure_behavioural_columns(df)

    df = df.sort_values("month").set_index("month")

    # Targets
    if "gva_yoy" in df.columns:
        df["gva_change"] = df["gva_yoy"].diff()
    if "retail" in df.columns:
        rsi_pos = df["retail"].where(df["retail"] > 0)
        df["retail_pct"] = np.log(rsi_pos).diff() * 100

    # Energy
    if "petrol" in df.columns:
        p = df["petrol"].where(df["petrol"] > 0)
        df["petrol_return"] = np.log(p).diff() * 100
    if "diesel" in df.columns:
        d = df["diesel"].where(df["diesel"] > 0)
        df["diesel_return"] = np.log(d).diff() * 100
    if "petrol_return" in df.columns and "diesel_return" in df.columns:
        df["energy_shock"] = (df["petrol_return"] + df["diesel_return"]) / 2

    # Controls
    if "cpih" in df.columns:
        c = df["cpih"].where(df["cpih"] > 0)
        df["cpih_pct"] = np.log(c).diff() * 100
    if "unemployment" in df.columns:
        df["unemployment_change"] = df["unemployment"].diff()

    # Behavioural source columns (may still be NaN if fuel missing)
    for col in ["fundamentalist_weight", "extrapolator_weight", "behavioural_response"]:
        if col not in df.columns:
            df[col] = np.nan

    # Lags
    lag_map = {
        "energy": ENERGY_LAGS,
        "target_lags": TARGET_LAGS,
        "behavioural": BEHAVIOUR_LAGS,
        "controls": CONTROL_LAGS,
    }
    feature_vars = {
        "energy": ["petrol_return", "diesel_return", "energy_shock"],
        "target_lags": ["gva_change", "retail_pct"],
        "behavioural": ["fundamentalist_weight", "extrapolator_weight", "behavioural_response"],
        "controls": ["cpih_pct", "unemployment_change"],
    }
    for group, cols in feature_vars.items():
        for c in cols:
            if c not in df.columns:
                continue
            for lag in lag_map[group]:
                df[f"{c}_lag{lag}"] = df[c].shift(lag)

    return df.reset_index()


def feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    energy, behaviour, controls, ar = [], [], [], []
    for col in df.columns:
        if any(col.startswith(f"{x}_lag") for x in ["petrol_return", "diesel_return", "energy_shock"]):
            energy.append(col)
        elif any(col.startswith(f"{x}_lag") for x in ["fundamentalist_weight", "extrapolator_weight", "behavioural_response"]):
            behaviour.append(col)
        elif any(col.startswith(f"{x}_lag") for x in ["cpih_pct", "unemployment_change"]):
            controls.append(col)
        elif any(col.startswith(f"{x}_lag") for x in ["gva_change", "retail_pct"]):
            ar.append(col)

    return {
        "AR-only": ar,
        "Energy+AR": list(dict.fromkeys(energy + ar)),
        "Energy+Behavioural+AR": list(dict.fromkeys(energy + behaviour + ar)),
        "Full": list(dict.fromkeys(energy + behaviour + controls + ar)),
    }


# =====================================================================
# Feature screening (training-only)
# =====================================================================
def _usable_features(df_train: pd.DataFrame, features: List[str]) -> List[str]:
    """Keep only features with at least one non-null, non-constant value in training."""
    keep: List[str] = []
    for f in features:
        if f not in df_train.columns:
            continue
        col = df_train[f]
        if col.notna().any() and col.nunique(dropna=True) > 1:
            keep.append(f)
    return keep


# =====================================================================
# Train / test split
# =====================================================================
def chronological_split(df: pd.DataFrame, target_col: str):
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"])
    train = df[df["month"] <= pd.Timestamp(TRAIN_END)].copy()
    test = df[df["month"] >= pd.Timestamp(TEST_START)].copy()
    return train, test


# =====================================================================
# Model fitters (GridSearchCV over TimeSeriesSplit)
# =====================================================================
def _cv(n: int, splits: int = 5):
    s = min(splits, max(2, n // 30))
    return TimeSeriesSplit(n_splits=s)


def fit_ridge(X, y):
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", RobustScaler()),
        ("m", Ridge()),
    ])
    grid = {"m__alpha": [0.1, 0.3, 1, 3, 10, 30, 100]}
    s = GridSearchCV(pipe, grid, scoring="neg_root_mean_squared_error",
                     cv=_cv(len(X)), n_jobs=-1)
    s.fit(X, y)
    return s.best_estimator_, s.best_params_, -s.best_score_


def fit_elasticnet(X, y):
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", RobustScaler()),
        ("m", ElasticNet(max_iter=30000, random_state=SEED)),
    ])
    grid = {"m__alpha": [0.01, 0.03, 0.1, 0.3, 1.0],
            "m__l1_ratio": [0.05, 0.2, 0.5, 0.8]}
    s = GridSearchCV(pipe, grid, scoring="neg_root_mean_squared_error",
                     cv=_cv(len(X)), n_jobs=-1)
    s.fit(X, y)
    return s.best_estimator_, s.best_params_, -s.best_score_


def fit_histgb(X, y):
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("m", HistGradientBoostingRegressor(random_state=SEED, early_stopping=False)),
    ])
    grid = {"m__max_iter": [50, 100, 150],
            "m__learning_rate": [0.02, 0.05, 0.1],
            "m__max_leaf_nodes": [3, 5, 8],
            "m__l2_regularization": [0.1, 1.0, 10.0]}
    s = GridSearchCV(pipe, grid, scoring="neg_root_mean_squared_error",
                     cv=_cv(len(X)), n_jobs=-1)
    s.fit(X, y)
    return s.best_estimator_, s.best_params_, -s.best_score_


def fit_xgboost(X, y):
    try:
        import xgboost as xgb
    except ImportError:
        return None, None, None

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("m", xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=SEED, n_jobs=-1, tree_method="hist",
            max_depth=2, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=5.0,
        )),
    ])
    grid = {"m__n_estimators": [50, 100, 150],
            "m__learning_rate": [0.02, 0.05, 0.1],
            "m__max_depth": [1, 2, 3],
            "m__min_child_weight": [3, 5, 10]}
    s = GridSearchCV(pipe, grid, scoring="neg_root_mean_squared_error",
                     cv=_cv(len(X)), n_jobs=-1)
    s.fit(X, y)
    return s.best_estimator_, s.best_params_, -s.best_score_


# =====================================================================
# LSTM (Keras) — leakage-safe: test sequences use training context
# =====================================================================
def fit_lstm(X_train, y_train, X_test, lookback: int = 12):
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.optimizers import Adam
    except ImportError:
        return None

    sX = StandardScaler()
    sY = StandardScaler()
    Xtr = sX.fit_transform(X_train)
    Xte = sX.transform(X_test)
    ytr = sY.fit_transform(np.asarray(y_train).reshape(-1, 1)).flatten()

    Xs, ys = [], []
    for i in range(lookback, len(Xtr)):
        Xs.append(Xtr[i - lookback:i])
        ys.append(ytr[i])
    Xs, ys = np.asarray(Xs), np.asarray(ys)

    if len(Xs) < 30:
        return None

    v = int(len(Xs) * 0.8)
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(lookback, Xs.shape[2])),
        Dropout(0.2),
        LSTM(16),
        Dropout(0.2),
        Dense(8, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    model.fit(
        Xs[:v], ys[:v],
        validation_data=(Xs[v:], ys[v:]),
        epochs=150, batch_size=16, shuffle=False, verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)],
    )

    combined = np.vstack([Xtr[-lookback:], Xte])
    Xseq = np.asarray([combined[i - lookback:i] for i in range(lookback, len(combined))])
    pred_scaled = model.predict(Xseq, verbose=0).flatten()
    return sY.inverse_transform(pred_scaled.reshape(-1, 1)).flatten(), model


# modules/forecasting_models.py — run_benchmark (verbose version)
def run_benchmark(
    df_features: pd.DataFrame,
    target_col: str,
    feature_list: List[str],
    include_lstm: bool = False,
) -> Tuple[pd.DataFrame, Dict]:
    train, test = chronological_split(df_features, target_col)
    feature_list = [f for f in feature_list if f in df_features.columns]
    feature_list = _usable_features(train, feature_list)

    print("=" * 70)
    print(f"run_benchmark: target={target_col}")
    print(f"  train rows: {len(train)}  test rows: {len(test)}")
    print(f"  usable features ({len(feature_list)}): {feature_list}")
    print("=" * 70)

    if not feature_list:
        print("!! No usable features — aborting.")
        return pd.DataFrame(), {}

    if len(train) < 20:
        print(f"!! Training window too short ({len(train)} rows) — "
              "all models will likely fail cross-validation.")
    if test.empty:
        print("!! Test window is empty — check TRAIN_END / TEST_START "
              "against the actual date range of your data.")

    X_tr = train[feature_list]; y_tr = train[target_col]
    X_te = test[feature_list]; y_te = test[target_col]

    results: List[Dict] = []
    predictions: Dict[str, np.ndarray] = {}
    failures: Dict[str, str] = {}

    # ---- Naive (always succeeds) ----
    naive = np.repeat(y_tr.iloc[-1], len(y_te))
    results.append(evaluate_all(y_te, naive, "Naive", target_col))
    predictions["Naive"] = naive

    # ---- Ridge ----
    try:
        m, _, cv = fit_ridge(X_tr, y_tr)
        pred = m.predict(X_te)
        r = evaluate_all(y_te, pred, "Ridge", target_col)
        r["cv_rmse"] = cv
        results.append(r)
        predictions["Ridge"] = pred
        print(f"✓ Ridge fitted. cv_rmse={cv:.4f}")
    except Exception as exc:
        failures["Ridge"] = repr(exc)
        print(f"✗ Ridge failed: {exc}")

    # ---- ElasticNet ----
    try:
        m, _, cv = fit_elasticnet(X_tr, y_tr)
        pred = m.predict(X_te)
        r = evaluate_all(y_te, pred, "ElasticNet", target_col)
        r["cv_rmse"] = cv
        results.append(r)
        predictions["ElasticNet"] = pred
        print(f"✓ ElasticNet fitted. cv_rmse={cv:.4f}")
    except Exception as exc:
        failures["ElasticNet"] = repr(exc)
        print(f"✗ ElasticNet failed: {exc}")

    # ---- HistGradientBoosting ----
    try:
        m, _, cv = fit_histgb(X_tr, y_tr)
        pred = m.predict(X_te)
        r = evaluate_all(y_te, pred, "HistGradientBoosting", target_col)
        r["cv_rmse"] = cv
        results.append(r)
        predictions["HistGradientBoosting"] = pred
        print(f"✓ HistGB fitted. cv_rmse={cv:.4f}")
    except Exception as exc:
        failures["HistGradientBoosting"] = repr(exc)
        print(f"✗ HistGB failed: {exc}")

    # ---- XGBoost ----
    try:
        m, _, cv = fit_xgboost(X_tr, y_tr)
        if m is None:
            print("· XGBoost skipped (not installed or returned None)")
        else:
            pred = m.predict(X_te)
            r = evaluate_all(y_te, pred, "XGBoost", target_col)
            r["cv_rmse"] = cv
            results.append(r)
            predictions["XGBoost"] = pred
            print(f"✓ XGBoost fitted. cv_rmse={cv:.4f}")
    except Exception as exc:
        failures["XGBoost"] = repr(exc)
        print(f"✗ XGBoost failed: {exc}")

    # ---- LSTM (opt-in) ----
    if include_lstm:
        try:
            out = fit_lstm(X_tr, y_tr, X_te)
            if out is None:
                print("· LSTM skipped (insufficient sequences or Keras missing)")
            else:
                pred, _ = out
                results.append(evaluate_all(y_te, pred, "LSTM", target_col))
                predictions["LSTM"] = pred
                print("✓ LSTM fitted.")
        except Exception as exc:
            failures["LSTM"] = repr(exc)
            print(f"✗ LSTM failed: {exc}")

    df_results = pd.DataFrame(results)
    if failures:
        # Attach failures so the UI can surface them
        df_results.attrs["failures"] = failures
        print("\nModel failures:")
        for k, v in failures.items():
            print(f"  {k}: {v}")

    return df_results, {
        "predictions": predictions,
        "test_index": y_te.index,
        "actual": y_te.values,
        "failures": failures,
    }