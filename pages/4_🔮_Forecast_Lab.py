# pages/4_🔮_Forecast_Lab.py
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.data_loader import to_csv_bytes


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Forecast Lab",
    page_icon="🔮",
    layout="wide",
)

st.title("🔮 Forecast Lab")
st.caption(
    "Compare machine-learning models and see how well they forecast UK economic indicators."
)

RESULTS_DIR = PROJECT_ROOT / "results"


# ============================================================
# LOAD RESULTS
# ============================================================

@st.cache_data(show_spinner=False)
def load_baked_results():

    out = {}

    manifest_path = RESULTS_DIR / "manifest.json"
    out["manifest"] = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else None
    )

    metrics_path = RESULTS_DIR / "forecast_metrics.csv"
    out["metrics"] = (
        pd.read_csv(metrics_path)
        if metrics_path.exists()
        else pd.DataFrame()
    )

    predictions_path = RESULTS_DIR / "forecast_predictions_all.csv"

    if predictions_path.exists():
        out["predictions"] = pd.read_csv(
            predictions_path,
            parse_dates=["month"],
        )
    else:
        out["predictions"] = pd.DataFrame()

    failures_path = RESULTS_DIR / "forecast_failures.json"

    out["failures"] = (
        json.loads(failures_path.read_text())
        if failures_path.exists()
        else {}
    )

    return out


baked = load_baked_results()


# ============================================================
# CHECK RESULTS
# ============================================================

if baked["manifest"] is None or baked["metrics"].empty:

    st.error(
        "No forecast results found.\n\n"
        "Run `colab_export_results.py` in Colab and place the "
        "`results/` folder inside the project root.",
        icon="🚫",
    )

    st.stop()


manifest = baked["manifest"]
metrics = baked["metrics"]
predictions_all = baked["predictions"]
failures = baked["failures"]


# ============================================================
# CONFIGURATION
# ============================================================

st.markdown("### Forecast setup")

c1, c2 = st.columns(2)

target_name = c1.selectbox(
    "What do you want to forecast?",
    manifest["targets"],
)

feature_sets_available = sorted(
    metrics["feature_set"].dropna().unique().tolist()
)

feature_set = c2.selectbox(
    "Information used by the models",
    feature_sets_available,
)


# ============================================================
# FILTER
# ============================================================

subset = metrics[
    (metrics["target"] == target_name)
    & (metrics["feature_set"] == feature_set)
].copy()

preds = predictions_all[
    (predictions_all["target"] == target_name)
    & (predictions_all["feature_set"] == feature_set)
].copy()


# ============================================================
# BEST MODEL
# ============================================================

if subset.empty:

    st.warning("No forecast results are available for this selection.")

else:

    subset = subset.sort_values("rmse")

    best = subset.iloc[0]

    st.divider()

    # --------------------------------------------------------
    # TOP METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Best model",
        str(best["model"]),
    )

    c2.metric(
        "RMSE",
        f"{best['rmse']:.3f}",
    )

    if "mae" in best:
        c3.metric(
            "MAE",
            f"{best['mae']:.3f}",
        )

    if "directional_accuracy" in best:
        c4.metric(
            "Direction accuracy",
            f"{best['directional_accuracy'] * 100:.1f}%",
        )

    st.caption(
        "Lower RMSE and MAE are better. Direction accuracy measures how often "
        "the model correctly predicts whether the indicator will rise or fall."
    )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    st.divider()

    st.subheader("🏆 Model comparison")

    comparison = subset.copy()

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=comparison["model"],
            y=comparison["rmse"],
            text=comparison["rmse"].round(3),
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "RMSE: %{y:.4f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        height=400,
        template="plotly_white",
        showlegend=False,
        xaxis_title="Model",
        yaxis_title="RMSE — lower is better",
        margin=dict(t=40, b=60),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False},
    )

    st.caption(
        "RMSE measures the typical size of forecast errors. "
        "The model with the lowest bar performs best on the out-of-sample test."
    )

    # ========================================================
    # ACTUAL VS FORECAST
    # ========================================================

    if not preds.empty:

        st.divider()

        st.subheader("📈 Actual vs forecast")

        st.caption(
            "Hover over the chart to see exact values. Drag across the chart "
            "to zoom into a particular period."
        )

        pivot = (
            preds
            .pivot_table(
                index="month",
                columns="model",
                values="predicted",
            )
            .sort_index()
        )

        actual = (
            preds
            .drop_duplicates("month")
            .set_index("month")["actual"]
            .sort_index()
        )

        fig = go.Figure()

        # Actual
        fig.add_trace(
            go.Scatter(
                x=actual.index,
                y=actual.values,
                mode="lines",
                name="Actual",
                line=dict(
                    width=3,
                ),
                hovertemplate=(
                    "<b>%{x|%b %Y}</b><br>"
                    "Actual: %{y:.3f}"
                    "<extra></extra>"
                ),
            )
        )

        # Forecasts
        for model_name in pivot.columns:

            is_best = model_name == best["model"]

            fig.add_trace(
                go.Scatter(
                    x=pivot.index,
                    y=pivot[model_name],
                    mode="lines",
                    name=model_name,
                    line=dict(
                        width=2.5 if is_best else 1.5,
                        dash="dash",
                    ),
                    opacity=1 if is_best else 0.65,
                    hovertemplate=(
                        f"<b>{model_name}</b><br>"
                        "%{x|%b %Y}<br>"
                        "Forecast: %{y:.3f}"
                        "<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            height=500,
            template="plotly_white",
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title=target_name,
            legend_title="Series",
            margin=dict(t=40, b=60),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displaylogo": False,
                "scrollZoom": True,
            },
        )

        st.caption(
            "The solid line shows the actual outcome. Dashed lines show model "
            "forecasts. The highlighted dashed line is the best-performing model."
        )

        # ====================================================
        # ERROR CHART
        # ====================================================

        st.divider()

        st.subheader("🎯 Forecast errors")

        st.caption(
            "An error is the difference between what actually happened and "
            "what the model predicted. Smaller errors mean a closer forecast."
        )

        error_model = st.selectbox(
            "Model",
            pivot.columns.tolist(),
            index=list(pivot.columns).index(best["model"])
            if best["model"] in pivot.columns
            else 0,
        )

        forecast = pivot[error_model]

        error = actual - forecast

        error = error.dropna()

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=error.index,
                y=error.values,
                name="Forecast error",
                hovertemplate=(
                    "<b>%{x|%b %Y}</b><br>"
                    "Error: %{y:.3f}"
                    "<extra></extra>"
                ),
            )
        )

        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
        )

        fig.update_layout(
            height=400,
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title="Actual − Forecast",
            showlegend=False,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displaylogo": False},
        )

        st.caption(
            "Bars above zero mean the model underestimated the actual value. "
            "Bars below zero mean it overestimated the actual value."
        )

        # ====================================================
        # PERFORMANCE TABLE
        # ====================================================

        st.divider()

        st.subheader("Performance by model")

        display_cols = [
            c
            for c in [
                "model",
                "n",
                "mae",
                "rmse",
                "mape",
                "r2",
                "directional_accuracy",
                "cv_rmse",
            ]
            if c in subset.columns
        ]

        display = subset[display_cols].copy()

        if "directional_accuracy" in display.columns:
            display["directional_accuracy"] = (
                display["directional_accuracy"] * 100
            )

        st.dataframe(
            display.round(3),
            use_container_width=True,
            hide_index=True,
        )

        # ====================================================
        # DOWNLOADS
        # ====================================================

        with st.expander("⬇️ Download results"):

            st.download_button(
                "Download predictions",
                to_csv_bytes(preds),
                file_name=(
                    f"predictions_"
                    f"{target_name.lower()}_"
                    f"{feature_set.lower()}.csv"
                ),
                mime="text/csv",
                key="fl_dl_preds",
            )

            st.download_button(
                "Download full metrics",
                to_csv_bytes(metrics),
                file_name="forecast_metrics.csv",
                mime="text/csv",
                key="fl_dl_metrics",
            )


# ============================================================
# FAILURES / MANIFEST
# ============================================================

if failures:

    with st.expander(
        f"⚠️ {len(failures)} forecast combination(s) failed"
    ):

        for key, errs in failures.items():

            st.markdown(f"**{key}**")

            for model_name, err in errs.items():
                st.markdown(
                    f"- `{model_name}`: {err}"
                )


with st.expander("Technical details"):

    st.json(manifest)