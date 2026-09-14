# app.py
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="UK Economic Simulator",
    page_icon="🇬🇧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Warm the data cache once
# ---------------------------------------------------------------------
df = st.session_state.get("df")
if df is None:
    try:
        from modules.data_loader import load_and_clean_data
        df = load_and_clean_data()
        st.session_state["df"] = df
    except Exception as exc:
        st.warning(f"Data file not yet loaded — `{exc}`")
        st.caption("Place `uk_macro_integrated.csv` inside the `data/` folder.")
        df = None


# ---------------------------------------------------------------------
# Shared card renderer
# ---------------------------------------------------------------------
def nav_card(
    col,
    icon: str,
    title: str,
    description: str,
    page_path: str,
    tags: list[str],
    key: str,
):
    """A clickable navigation card with a heading, description and button."""
    with col:
        with st.container(border=True):
            st.markdown(
                f"<h3 style='margin-bottom: 0.2rem;'>{icon} {title}</h3>",
                unsafe_allow_html=True,
            )
            st.caption(description)
            if tags:
                st.markdown(
                    " ".join(
                        f"<span style='background:#EEF2FF;color:#3730A3;"
                        f"padding:2px 8px;border-radius:10px;font-size:0.75rem;"
                        f"margin-right:4px;'>{t}</span>"
                        for t in tags
                    ),
                    unsafe_allow_html=True,
                )
            if st.button("Open", key=key, use_container_width=True):
                st.switch_page(page_path)


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("🇬🇧 UK Economic Simulator")
st.caption(
    "Behavioural-agent macro model · econometric VAR · ML predictive suite · "
    "interactive scenario tools."
)

# ---------------------------------------------------------------------
# Snapshot strip (only if data loaded)
# ---------------------------------------------------------------------
if df is not None and len(df) > 0:
    latest = df.iloc[-1]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Sample", f"{df['month'].min():%Y-%m} → {df['month'].max():%Y-%m}")

    def _fmt(col, unit=""):
        if col in df.columns:
            return f"{latest[col]:.2f}{unit}"
        return "—"

    c3.metric("Bank Rate", _fmt("bank_rate", " %"))
    c4.metric("CPIH", _fmt("cpih"))
    c5.metric("Unemployment", _fmt("unemployment", " %"))

st.divider()

# ---------------------------------------------------------------------
# Navigation grid
# ---------------------------------------------------------------------
st.subheader("Explore the app")

row1 = st.columns(3)
nav_card(
    row1[0], "📈", "Macro Dashboard",
    "4×2 grid of core macro series with latest-value cards.",
    "pages/1_📈_Macro_Dashboard.py",
    ["Descriptive", "Time series"],
    key="nav_macro",
)
nav_card(
    row1[1], "📊", "Data Explorer",
    "Date filters, correlation heatmap, scatter, PCA and lagged CCF.",
    "pages/2_📊_Data_Explorer.py",
    ["EDA", "Correlations"],
    key="nav_explorer",
)
nav_card(
    row1[2], "📐", "Analytical Framework",
    "Mathematical specification of the ABM, VAR and shock identification.",
    "pages/2_📐_Analytical_Framework.py",
    ["Theory"],
    key="nav_framework",
)

row2 = st.columns(3)
nav_card(
    row2[0], "🔍", "Analytics Engine",
    "Crisis impact, energy-shock t-tests, rolling volatility and VAR IRFs.",
    "pages/3_🔍_Analytics_Engine.py",
    ["Econometrics", "VAR"],
    key="nav_engine",
)
nav_card(
    row2[1], "🔮", "Forecast Lab",
    "Ridge · ElasticNet · HistGB · XGBoost · LSTM — pre-computed in Colab.",
    "pages/4_🔮_Forecast_Lab.py",
    ["ML", "Benchmark"],
    key="nav_forecast",
)
nav_card(
    row2[2], "🧪", "Simulation Studio",
    "Interactive ABM: λ, θ, α, β and policy-shock sliders.",
    "pages/5_🧪_Simulation_Studio.py",
    ["ABM", "Interactive"],
    key="nav_sim",
)

row3 = st.columns(3)
nav_card(
    row3[0], "✅", "Validation",
    "MAE / RMSE / MAPE / MDA / MSFE tables and residual diagnostics.",
    "pages/6_✅_Validation.py",
    ["Metrics", "Diagnostics"],
    key="nav_validation",
)
nav_card(
    row3[1], "🕹️", "Scenario Explorer",
    "Compose stagflation / tightening scenarios and compare them.",
    "pages/7_🕹️_Scenario_Explorer.py",
    ["Scenarios"],
    key="nav_scenarios",
)
nav_card(
    row3[2], "🏛️", "Stakeholder Gateway",
    "Executive summary with crisis rankings and policy insights.",
    "pages/8_🏛️_Stakeholder_Gateway.py",
    ["Executive"],
    key="nav_stakeholder",
)

st.divider()

# ---------------------------------------------------------------------
# Preview chart (only if data loaded)
# ---------------------------------------------------------------------
if df is not None and len(df) > 0 and {"bank_rate", "cpih"}.issubset(df.columns):
    st.subheader("At a glance")
    st.caption("Bank Rate and CPIH over the full sample.")

    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(12, 3.5))

    # Left axis — Bank Rate
    ax1.plot(
        df["month"],
        df["bank_rate"],
        color="#1f77b4",
        linewidth=1.8,
        label="Bank Rate (%)"
    )
    ax1.set_ylabel("Bank Rate (%)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(0, 6.5)

    # Right axis — CPIH
    ax2 = ax1.twinx()
    ax2.plot(
        df["month"],
        df["cpih"],
        color="#d62728",
        linewidth=1.8,
        alpha=0.85,
        label="CPIH Index"
    )
    ax2.set_ylabel("CPIH Index", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    ax1.set_xlabel("")
    ax1.grid(True, alpha=0.25)

    # Remove top spines
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        ncol=2,
        frameon=False
    )

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------
# Deliverables footer
# ---------------------------------------------------------------------
st.subheader("Project deliverables")
cols = st.columns(5)
deliverables = [
    ("1", "Data pipeline", "Cleaning, trimming, interpolation, exports"),
    ("2", "Econometric core", "Crisis stress, energy shocks, VAR, IRFs"),
    ("3", "ML benchmark", "Ridge, ElasticNet, HistGB, XGBoost, LSTM"),
    ("4", "ABM", "Fundamentalist / Extrapolator, softmax weights"),
    ("5", "Validation & scenarios", "Metrics, diagnostics, sandbox"),
]
for c, (num, title, body) in zip(cols, deliverables):
    with c:
        with st.container(border=True):
            st.markdown(f"**{num}. {title}**")
            st.caption(body)

st.caption(
    "Educational and analytical tool. Model-implied results only — "
    "not forecasts, not investment advice."
)