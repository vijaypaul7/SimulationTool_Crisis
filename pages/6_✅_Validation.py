# pages/6_✅_Validation.py
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.data_loader import load_and_clean_data, to_csv_bytes
from modules.abm_simulator import (
    calculate_forecasting_performance,
    calculate_agent_weights,
)
from modules.validation_metrics import (
    load_baked_metrics,
    load_baked_predictions,
    msfe_comparison,
    residual_diagnostics,
    stability_table,
)

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------

st.set_page_config(
    page_title="Validation",
    page_icon="✅",
    layout="wide",
)

st.title("✅ Validation")
st.caption(
    "Check whether the models are accurate, stable, and producing sensible errors."
)


# ---------------------------------------------------------
# DATA
# ---------------------------------------------------------

df = st.session_state.get("df")

if df is None:
    df = load_and_clean_data()
    st.session_state["df"] = df


# ---------------------------------------------------------
# 1. ABM VALIDATION
# ---------------------------------------------------------

st.subheader("1. ABM forecast accuracy")

st.caption(
    "MSFE measures how large the forecasting errors are. "
    "Lower values mean the model is making smaller forecast errors."
)

perf = calculate_forecasting_performance(
    df,
    fuel_col="petrol",
)

perf = calculate_agent_weights(
    perf,
    lambda_val=0.5,
)

cmp = msfe_comparison(perf)

if not cmp.empty:
    numeric_cols = cmp.select_dtypes(include="number").columns.tolist()

    # Display table
    st.dataframe(
        cmp.round(4),
        use_container_width=True,
        hide_index=True,
    )

    # -----------------------------------------------------
    # ABM MSFE CHART
    # -----------------------------------------------------

    if len(cmp) >= 2:
        fig_cmp = go.Figure()

        for column in numeric_cols:
            if column.lower() not in {"rank", "n"}:
                fig_cmp.add_trace(
                    go.Bar(
                        x=cmp.index.astype(str),
                        y=cmp[column],
                        name=column.replace("_", " ").title(),
                        hovertemplate=(
                            "<b>%{x}</b><br>"
                            "MSFE: %{y:.4f}"
                            "<extra></extra>"
                        ),
                    )
                )

        fig_cmp.update_layout(
            height=380,
            template="plotly_white",
            barmode="group",
            xaxis_title="Forecasting approach",
            yaxis_title="MSFE",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
        )

        st.plotly_chart(
            fig_cmp,
            use_container_width=True,
            config={"displaylogo": False},
        )


# ---------------------------------------------------------
# 2. ROLLING MSFE
# ---------------------------------------------------------

st.divider()

st.subheader("2. Does forecast accuracy remain stable?")

st.caption(
    "Rolling 12-month MSFE shows whether forecasting errors become larger or "
    "smaller over time. Spikes indicate periods when forecasting became more difficult."
)

fig_msfe = go.Figure()

fig_msfe.add_trace(
    go.Scatter(
        x=perf["month"],
        y=perf["fundamentalist_msfe_12m"],
        mode="lines",
        name="Fundamentalist",
        line=dict(width=2.5),
        hovertemplate=(
            "<b>%{x|%b %Y}</b><br>"
            "MSFE: %{y:.4f}"
            "<extra></extra>"
        ),
    )
)

fig_msfe.add_trace(
    go.Scatter(
        x=perf["month"],
        y=perf["extrapolator_msfe_12m"],
        mode="lines",
        name="Extrapolator",
        line=dict(width=2.5),
        hovertemplate=(
            "<b>%{x|%b %Y}</b><br>"
            "MSFE: %{y:.4f}"
            "<extra></extra>"
        ),
    )
)

fig_msfe.update_layout(
    height=430,
    template="plotly_white",
    hovermode="x unified",
    xaxis_title="Month",
    yaxis_title="Rolling 12-month MSFE",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
)

st.plotly_chart(
    fig_msfe,
    use_container_width=True,
    config={
        "displaylogo": False,
        "scrollZoom": True,
    },
)


# ---------------------------------------------------------
# 3. MACHINE-LEARNING BENCHMARK
# ---------------------------------------------------------

st.divider()

st.subheader("3. Machine-learning benchmark")

st.caption(
    "These results compare out-of-sample performance across ML specifications. "
    "Lower RMSE and MAE indicate higher forecast accuracy."
)

# Fetch from session state or fall back to baked metrics
results = st.session_state.get("forecast_results")

if results is None or (isinstance(results, pd.DataFrame) and results.empty):
    results = load_baked_metrics()

if results is None or results.empty:
    st.info(
        "No benchmark results are currently available. "
        "Run the Forecast Lab or populate `results/forecast_metrics.csv`."
    )
else:
    results = results.copy()

    # -----------------------------------------------------
    # TARGET SELECTION (Prevents scale-mismatch bug)
    # -----------------------------------------------------

    if "target" in results.columns:
        available_targets = sorted(results["target"].dropna().unique().tolist())
        
        # Default target selection to GVA if available
        gva_candidates = [t for t in available_targets if "gva" in str(t).lower()]
        default_idx = available_targets.index(gva_candidates[0]) if gva_candidates else 0
        
        selected_target = st.selectbox(
            "Select target variable to validate:",
            options=available_targets,
            index=default_idx,
            help="Comparing models across different target variables is invalid due to unit/scale differences."
        )
        
        target_results = results[results["target"] == selected_target].copy()
    else:
        target_results = results.copy()

    if target_results.empty:
        st.warning(f"No results found for target: {selected_target}")
    else:
        # -------------------------------------------------
        # BEST MODEL METRICS
        # -------------------------------------------------

        if "rmse" in target_results.columns:
            best = target_results.sort_values("rmse").iloc[0]

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Best model",
                str(best.get("model", "N/A")),
            )

            c2.metric(
                "Lowest RMSE",
                f"{best['rmse']:.3f}",
            )

            if "mae" in target_results.columns and pd.notnull(best["mae"]):
                c3.metric(
                    "MAE",
                    f"{best['mae']:.3f}",
                )

        # -------------------------------------------------
        # MODEL COMPARISON CHART (Proportional Height Fix)
        # -------------------------------------------------

        if "rmse" in target_results.columns:
            chart_results = target_results.sort_values("rmse", ascending=True)

            fig_models = go.Figure()

            fig_models.add_trace(
                go.Bar(
                    x=chart_results["rmse"],
                    y=chart_results["model"].astype(str),
                    orientation="h",
                    text=chart_results["rmse"].round(3),
                    textposition="outside",
                    marker=dict(color="#2b5c8f"),
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "RMSE: %{x:.3f}"
                        "<extra></extra>"
                    ),
                )
            )

            # Keep height balanced so bars don't stretch excessively wide/tall
            calculated_height = max(280, min(600, len(chart_results) * 35))

            fig_models.update_layout(
                height=calculated_height,
                template="plotly_white",
                xaxis_title="RMSE — lower is better",
                yaxis_title="",
                margin=dict(l=20, r=40, t=20, b=40),
            )

            st.plotly_chart(
                fig_models,
                use_container_width=True,
                config={"displaylogo": False},
            )

        # -------------------------------------------------
        # RESULTS TABLE
        # -------------------------------------------------

        with st.expander("View detailed model results"):
            st.dataframe(
                target_results.round(4),
                use_container_width=True,
                hide_index=True,
            )

        # -------------------------------------------------
        # STABILITY ANALYSIS
        # -------------------------------------------------

        st.subheader("Model stability")

        st.caption(
            "Stability checks whether model performance remains reasonably "
            "consistent across splits rather than overfitting one specific time period."
        )

        stability = stability_table(target_results)

        if stability is not None and not stability.empty:
            st.dataframe(
                stability,
                use_container_width=True,
                hide_index=True,
            )

        # -------------------------------------------------
        # RESIDUAL DIAGNOSTICS
        # -------------------------------------------------

        store = st.session_state.get("forecast_store", {})
        
        # Load baked predictions fallback if store is empty
        if not store or "predictions" not in store:
            pivot_preds, actual_series, test_idx = load_baked_predictions()
            if pivot_preds is not None and not pivot_preds.empty:
                store = {
                    "predictions": pivot_preds.to_dict(orient="series"),
                    "actual": actual_series,
                    "test_index": test_idx,
                }

        if (
            "predictions" in store
            and not target_results.empty
            and "rmse" in target_results.columns
        ):
            best_model_name = str(
                target_results.sort_values("rmse").iloc[0]["model"]
            )

            if best_model_name in store["predictions"]:
                st.subheader(f"Forecast errors — {best_model_name}")

                st.caption(
                    "Residuals are the difference between actual values and "
                    "forecasts. Ideally, they fluctuate around zero with no structured patterns."
                )

                resid = residual_diagnostics(
                    store["actual"],
                    store["predictions"][best_model_name],
                )

                col_res1, col_res2 = st.columns(2)

                # -----------------------------------------
                # Residual timeline
                # -----------------------------------------
                with col_res1:
                    fig_resid = go.Figure()

                    fig_resid.add_trace(
                        go.Scatter(
                            x=store["test_index"],
                            y=resid["residual"],
                            mode="lines+markers",
                            name="Residual",
                            line=dict(width=1.8, color="#d9534f"),
                            hovertemplate=(
                                "<b>%{x}</b><br>"
                                "Error: %{y:.3f}"
                                "<extra></extra>"
                            ),
                        )
                    )

                    fig_resid.add_hline(
                        y=0,
                        line_dash="dash",
                        line_width=1,
                    )

                    fig_resid.update_layout(
                        height=350,
                        template="plotly_white",
                        xaxis_title="Test period",
                        yaxis_title="Forecast error",
                        showlegend=False,
                    )

                    st.plotly_chart(
                        fig_resid,
                        use_container_width=True,
                        config={"displaylogo": False},
                    )

                # -----------------------------------------
                # Residual distribution
                # -----------------------------------------
                with col_res2:
                    fig_hist = go.Figure()

                    fig_hist.add_trace(
                        go.Histogram(
                            x=resid["residual"],
                            nbinsx=20,
                            marker=dict(color="#428bca"),
                            hovertemplate=(
                                "Error range: %{x}<br>"
                                "Observations: %{y}"
                                "<extra></extra>"
                            ),
                        )
                    )

                    fig_hist.add_vline(
                        x=0,
                        line_dash="dash",
                        line_width=1,
                    )

                    fig_hist.update_layout(
                        height=350,
                        template="plotly_white",
                        xaxis_title="Forecast error",
                        yaxis_title="Frequency",
                        showlegend=False,
                    )

                    st.plotly_chart(
                        fig_hist,
                        use_container_width=True,
                        config={"displaylogo": False},
                    )

                # -----------------------------------------
                # Residual download
                # -----------------------------------------
                st.download_button(
                    "⬇️ Download residuals",
                    to_csv_bytes(resid.reset_index()),
                    file_name="residuals.csv",
                    mime="text/csv",
                )