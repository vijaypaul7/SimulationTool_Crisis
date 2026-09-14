# pages/5_🧪_Simulation_Studio.py
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from modules.data_loader import load_and_clean_data, to_csv_bytes
from modules.abm_simulator import (
    calculate_forecasting_performance,
    calculate_agent_weights,
    calculate_behavioural_response,
    simulate_shock_response,
    compute_gva_transmission,
)
from config import ABM_PARAMS


# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------

st.set_page_config(
    page_title="Simulation Studio",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 Simulation Studio")
st.caption(
    "Explore how different types of consumers respond to changes in fuel prices "
    "and interest rates."
)


# ---------------------------------------------------------
# DATA
# ---------------------------------------------------------

df = st.session_state.get("df")

if df is None:
    df = load_and_clean_data()
    st.session_state["df"] = df


# ---------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------

st.subheader("Simulation settings")
st.caption("Adjust the assumptions below and the charts will update automatically.")

c1, c2, c3, c4, c5 = st.columns(5)

lambda_val = c1.slider(
    "λ — selection",
    0.0,
    2.0,
    float(ABM_PARAMS["lambda"]),
    0.05,
    help="Controls how strongly agents favour forecasting strategies that performed well.",
)

theta = c2.slider(
    "θ — extrapolation",
    0.0,
    1.5,
    float(ABM_PARAMS["theta"]),
    0.05,
    help="Controls how strongly agents extrapolate recent price movements.",
)

alpha = c3.slider(
    "α — fundamentalist",
    0.0,
    0.6,
    float(ABM_PARAMS["alpha"]),
    0.01,
    help="Strength of the fundamentalist response to prices moving away from their expected level.",
)

beta = c4.slider(
    "β — extrapolator",
    0.0,
    1.0,
    float(ABM_PARAMS["beta"]),
    0.01,
    help="Strength of the extrapolator response to recent price movements.",
)

shock_bps = c5.slider(
    "Interest-rate shock",
    -300,
    500,
    100,
    25,
    help="A positive value represents an increase in interest rates.",
)

fuel = st.selectbox(
    "Fuel series",
    ["petrol", "diesel"],
)


# ---------------------------------------------------------
# MODEL CALCULATIONS
# ---------------------------------------------------------

perf = calculate_forecasting_performance(
    df,
    fuel_col=fuel,
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


# ---------------------------------------------------------
# LATEST MODEL STATE
# ---------------------------------------------------------

valid_weights = perf[
    [
        "fundamentalist_weight",
        "extrapolator_weight",
    ]
].dropna()

if not valid_weights.empty:
    latest_f = float(valid_weights["fundamentalist_weight"].iloc[-1])
    latest_e = float(valid_weights["extrapolator_weight"].iloc[-1])
    
    st.divider()
    st.subheader("Current model state")
    m1, m2, m3 = st.columns(3)
    
    m1.metric("Fundamentalist weight", f"{latest_f:.1%}")
    m2.metric("Extrapolator weight", f"{latest_e:.1%}")
    
    if latest_f > latest_e:
        dominant = "Fundamentalists"
    elif latest_e > latest_f:
        dominant = "Extrapolators"
    else:
        dominant = "Equal"
        
    m3.metric("Dominant behaviour", dominant)


# ---------------------------------------------------------
# 1. AGENT WEIGHTS
# ---------------------------------------------------------

st.divider()
st.subheader("1. How consumer behaviour changes")
st.caption(
    "The model has two types of agents. Fundamentalists respond to prices "
    "relative to their expected level, while extrapolators respond to recent trends. "
    "The lines show how their importance changes over time."
)

fig_weights = go.Figure()

fig_weights.add_trace(
    go.Scatter(
        x=perf["month"],
        y=perf["fundamentalist_weight"],
        mode="lines",
        name="Fundamentalist",
        line=dict(width=2.5),
        hovertemplate=(
            "<b>%{x|%b %Y}</b><br>"
            "Fundamentalist: %{y:.1%}"
            "<extra></extra>"
        ),
    )
)

fig_weights.add_trace(
    go.Scatter(
        x=perf["month"],
        y=perf["extrapolator_weight"],
        mode="lines",
        name="Extrapolator",
        line=dict(width=2.5),
        hovertemplate=(
            "<b>%{x|%b %Y}</b><br>"
            "Extrapolator: %{y:.1%}"
            "<extra></extra>"
        ),
    )
)

fig_weights.add_hline(
    y=0.5,
    line_dash="dash",
    line_width=1,
    annotation_text="50 / 50",
    annotation_position="top left",
)

fig_weights.update_layout(
    height=430,
    template="plotly_white",
    hovermode="x unified",
    yaxis=dict(
        title="Agent weight",
        tickformat=".0%",
        range=[0, 1],
    ),
    xaxis=dict(title=""),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
)

st.plotly_chart(
    fig_weights,
    use_container_width=True,
    config={"displaylogo": False, "scrollZoom": True},
)


# ---------------------------------------------------------
# 2. SIMULATED VS ACTUAL
# ---------------------------------------------------------

st.divider()
st.subheader("2. Simulated behaviour vs actual retail changes")
st.caption(
    "This compares the model's behavioural response with the observed "
    "month-to-month change in retail activity. A closer fit suggests that "
    "the simulated behaviour is broadly consistent with what happened in the data."
)

fig_actual = go.Figure()

if "retail" in df.columns:
    retail_pct = np.log(df["retail"] / df["retail"].shift(1)) * 100

    fig_actual.add_trace(
        go.Scatter(
            x=df["month"],
            y=retail_pct,
            mode="lines",
            name="Actual retail change",
            line=dict(width=2),
            hovertemplate=(
                "<b>%{x|%b %Y}</b><br>"
                "Actual: %{y:.2f}%"
                "<extra></extra>"
            ),
        )
    )

fig_actual.add_trace(
    go.Scatter(
        x=perf["month"],
        y=perf["behavioural_response"],
        mode="lines",
        name="Model response",
        line=dict(width=2.5, dash="dash"),
        hovertemplate=(
            "<b>%{x|%b %Y}</b><br>"
            "Model response: %{y:.2f} pp"
            "<extra></extra>"
        ),
    )
)

fig_actual.add_hline(
    y=0,
    line_width=1,
)

fig_actual.update_layout(
    height=430,
    template="plotly_white",
    hovermode="x unified",
    yaxis_title="Monthly change / response",
    xaxis_title="",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
)

st.plotly_chart(
    fig_actual,
    use_container_width=True,
    config={"displaylogo": False, "scrollZoom": True},
)


# ---------------------------------------------------------
# 3. POLICY SHOCK
# ---------------------------------------------------------

st.divider()
st.subheader("3. What happens after an interest-rate shock?")

shock_direction = (
    "increase"
    if shock_bps > 0
    else "decrease"
    if shock_bps < 0
    else "no change"
)

st.caption(
    f"You are simulating a **{abs(shock_bps):.0f} basis-point "
    f"{shock_direction}** in interest rates. "
    "The chart shows how consumption is predicted to respond over the following 24 months."
)

sim = simulate_shock_response(
    df,
    shock_bps=shock_bps,
    lambda_val=lambda_val,
    theta=theta,
    alpha=alpha,
    beta=beta,
    fuel_col=fuel,
    horizon=24,
)

fig_shock = go.Figure()

fig_shock.add_trace(
    go.Scatter(
        x=sim["horizon"],
        y=sim["fundamentalist_response"],
        mode="lines+markers",
        name="Fundamentalist",
        line=dict(width=2),
        hovertemplate=(
            "Month %{x}<br>"
            "Response: %{y:.3f} pp"
            "<extra></extra>"
        ),
    )
)

fig_shock.add_trace(
    go.Scatter(
        x=sim["horizon"],
        y=sim["extrapolator_response"],
        mode="lines+markers",
        name="Extrapolator",
        line=dict(width=2),
        hovertemplate=(
            "Month %{x}<br>"
            "Response: %{y:.3f} pp"
            "<extra></extra>"
        ),
    )
)

fig_shock.add_trace(
    go.Scatter(
        x=sim["horizon"],
        y=sim["aggregate_response"],
        mode="lines+markers",
        name="Overall response",
        line=dict(width=3),
        hovertemplate=(
            "Month %{x}<br>"
            "Aggregate response: %{y:.3f} pp"
            "<extra></extra>"
        ),
    )
)

fig_shock.add_hline(
    y=0,
    line_width=1,
)

fig_shock.update_layout(
    height=450,
    template="plotly_white",
    hovermode="x unified",
    xaxis_title="Months after shock",
    yaxis_title="Consumption response (pp)",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
)

st.plotly_chart(
    fig_shock,
    use_container_width=True,
    config={"displaylogo": False, "scrollZoom": True},
)


# ---------------------------------------------------------
# SIMPLE INTERPRETATION
# ---------------------------------------------------------

if not sim.empty and "aggregate_response" in sim.columns:
    final_response = sim["aggregate_response"].iloc[-1]
    peak_response = sim["aggregate_response"].abs().max()

    st.info(
        f"**How to read this simulation:** "
        f"the model predicts a final consumption response of "
        f"**{final_response:.3f} percentage points** after 24 months. "
        f"The largest simulated response is approximately "
        f"**{peak_response:.3f} percentage points**."
    )


# ---------------------------------------------------------
# 4. GVA TRANSMISSION ANALYSIS
# ---------------------------------------------------------

st.divider()
st.subheader("4. GVA transmission analysis")
st.caption(
    "Examines how strongly simulated agent behavioral responses ($B_{t-k}$) "
    "translate into subsequent changes in GVA YoY growth ($\Delta Y_t$) across different time lags."
)

trans_metrics, trans_merged = compute_gva_transmission(
    abm_df=perf,
    df_original=df,
    lags=(6, 12, 18),
)

if not trans_metrics.empty:
    best_lag_row = trans_metrics.loc[trans_metrics["pearson_r"].abs().idxmax()]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Optimal Lag", f"{int(best_lag_row['lag_months'])} months")
    m2.metric("Pearson Correlation (r)", f"{best_lag_row['pearson_r']:.3f}")
    m3.metric("p-value", f"{best_lag_row['pearson_p']:.4f}")
    m4.metric("Regression Slope", f"{best_lag_row['slope']:.4f}")

    st.markdown("##### Transmission across lags")

    formatted_trans = trans_metrics.copy()
    formatted_trans.columns = [
        "Lag (Months)",
        "Observations",
        "Pearson r",
        "Pearson p-val",
        "Spearman r",
        "Spearman p-val",
        "Slope",
        "Intercept",
        "RMSE",
    ]

    st.dataframe(
        formatted_trans.round(4),
        use_container_width=True,
        hide_index=True,
    )

    best_lag = int(best_lag_row["lag_months"])
    scatter_df = trans_merged.copy()
    scatter_df["lagged_br"] = scatter_df["behavioural_response"].shift(best_lag)
    scatter_df = scatter_df.dropna(subset=["lagged_br", "delta_gva"])

    if not scatter_df.empty:
        fig_trans = go.Figure()

        fig_trans.add_trace(
            go.Scatter(
                x=scatter_df["lagged_br"],
                y=scatter_df["delta_gva"],
                mode="markers",
                name="Observations",
                marker=dict(size=7, color="#2b5c8f", opacity=0.7),
                hovertemplate=(
                    "Behavioral Response (t-" + str(best_lag) + "): %{x:.3f}<br>"
                    "Δ GVA Growth (t): %{y:.3f}<extra></extra>"
                ),
            )
        )

        x_range = np.linspace(
            scatter_df["lagged_br"].min(),
            scatter_df["lagged_br"].max(),
            50,
        )
        y_line = best_lag_row["slope"] * x_range + best_lag_row["intercept"]

        fig_trans.add_trace(
            go.Scatter(
                x=x_range,
                y=y_line,
                mode="lines",
                name="Linear Fit",
                line=dict(color="#d9534f", width=2),
            )
        )

        fig_trans.update_layout(
            height=380,
            template="plotly_white",
            xaxis_title=f"Agent Behavioral Response (t-{best_lag}m)",
            yaxis_title="Change in GVA YoY Growth (Δ Y_t)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
        )

        st.plotly_chart(
            fig_trans,
            use_container_width=True,
            config={"displaylogo": False, "scrollZoom": True},
        )
else:
    st.info("Insufficient overlapping observations to compute GVA transmission statistics.")


# ---------------------------------------------------------
# DOWNLOAD
# ---------------------------------------------------------

st.divider()
st.subheader("Download results")

st.download_button(
    "⬇️ Download simulation results",
    to_csv_bytes(
        perf[
            [
                "month",
                "fundamentalist_weight",
                "extrapolator_weight",
                "behavioural_response",
            ]
        ]
    ),
    file_name="simulation_results.csv",
    mime="text/csv",
)