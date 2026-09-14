# pages/1_📈_Macro_Dashboard.py
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.data_loader import load_and_clean_data

st.set_page_config(
    page_title="Macro Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 UK Macro Dashboard")
st.caption("Interactive UK economic indicators dashboard.")

# -------------------------------------------------------------------
# DATA
# -------------------------------------------------------------------

df = st.session_state.get("df")

if df is None:
    df = load_and_clean_data()
    st.session_state["df"] = df

df = df.copy()
df["month"] = pd.to_datetime(df["month"])
df = df.sort_values("month")

# -------------------------------------------------------------------
# LATEST VALUES
# -------------------------------------------------------------------

latest = df.iloc[-1]

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Bank Rate",
    f"{latest.get('bank_rate', np.nan):.2f} %"
)

c2.metric(
    "CPIH",
    f"{latest.get('cpih', np.nan):.2f}"
)

c3.metric(
    "Unemployment",
    f"{latest.get('unemployment', np.nan):.2f} %"
)

c4.metric(
    "GVA YoY",
    f"{latest.get('gva_yoy', np.nan):.2f} %"
)

st.divider()

# -------------------------------------------------------------------
# CHART SETTINGS
# -------------------------------------------------------------------

colors = {
    "retail": "#2E86AB",
    "cpih": "#A23B72",
    "gva_yoy": "#F18F01",
    "unemployment": "#C73E1D",
    "bank_rate": "#3B8C5E",
    "petrol": "#6A4C93",
    "diesel": "#D4A373",
}

panels = [
    ("retail", "Retail Sales Volume Index", "Index"),
    ("cpih", "CPIH Index", "Index"),
    ("gva_yoy", "GVA Year-on-Year Growth", "Growth (%)"),
    ("unemployment", "Unemployment Rate", "Rate (%)"),
    ("bank_rate", "Bank of England Base Rate", "Rate (%)"),
    ("petrol", "Petrol Price", "p/litre"),
    ("diesel", "Diesel Price", "p/litre"),
]

# -------------------------------------------------------------------
# INTERACTIVE 4 × 2 DASHBOARD
# -------------------------------------------------------------------

fig = make_subplots(
    rows=4,
    cols=2,
    subplot_titles=[
        title for _, title, _ in panels
    ] + ["Summary Statistics"],
    vertical_spacing=0.07,
    horizontal_spacing=0.08,
)

for i, (col, title, ylabel) in enumerate(panels):

    row = (i // 2) + 1
    column = (i % 2) + 1

    if col not in df.columns:
        continue

    # Bank Rate as a step chart
    if col == "bank_rate":
        fig.add_trace(
            go.Scatter(
                x=df["month"],
                y=df[col],
                mode="lines",
                name="Bank Rate",
                line=dict(
                    color=colors[col],
                    width=2.5,
                    shape="hv"
                ),
                hovertemplate=(
                    "<b>%{x|%b %Y}</b><br>"
                    "Bank Rate: %{y:.2f}%"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=column,
        )

    else:
        fig.add_trace(
            go.Scatter(
                x=df["month"],
                y=df[col],
                mode="lines",
                name=title,
                line=dict(
                    color=colors[col],
                    width=2.2
                ),
                fill="tozeroy",
                fillcolor=colors[col].replace(")", ", 0.12)") 
                    if colors[col].startswith("rgba")
                    else None,
                hovertemplate=(
                    "<b>%{x|%b %Y}</b><br>"
                    f"{title}: %{{y:.2f}}"
                    f"<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=column,
        )

    # Zero line for GVA
    if col == "gva_yoy":
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_width=1,
            line_color="gray",
            row=row,
            col=column,
        )

    # 100 baseline for retail
    if col == "retail":
        fig.add_hline(
            y=100,
            line_dash="dash",
            line_width=1,
            line_color="gray",
            row=row,
            col=column,
        )

    fig.update_yaxes(
        title_text=ylabel,
        row=row,
        col=column,
        showgrid=True,
        gridcolor="rgba(128,128,128,0.2)",
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(128,128,128,0.15)",
        row=row,
        col=column,
    )


# -------------------------------------------------------------------
# LAYOUT
# -------------------------------------------------------------------

fig.update_layout(
    height=1500,
    title=dict(
        text="UK Economic Dashboard",
        font=dict(size=24),
    ),
    hovermode="x unified",
    template="plotly_white",
    margin=dict(
        l=60,
        r=40,
        t=90,
        b=60,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.01,
        xanchor="left",
        x=0,
    ),
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displayModeBar": True,
        "displaylogo": False,
        "scrollZoom": True,
    },
)

st.divider()

# -------------------------------------------------------------------
# BANK RATE VS CPIH — DUAL AXIS
# -------------------------------------------------------------------

st.subheader("Bank Rate vs CPIH")
st.caption(
    "Both indicators use their own y-axis so the Bank Rate is not visually compressed."
)

fig2 = go.Figure()

# Bank Rate — LEFT AXIS
fig2.add_trace(
    go.Scatter(
        x=df["month"],
        y=df["bank_rate"],
        mode="lines",
        name="Bank Rate",
        line=dict(
            color="#1f77b4",
            width=2.5,
            shape="hv",
        ),
        hovertemplate=(
            "<b>%{x|%b %Y}</b><br>"
            "Bank Rate: %{y:.2f}%"
            "<extra></extra>"
        ),
    )
)

# CPIH — RIGHT AXIS
fig2.add_trace(
    go.Scatter(
        x=df["month"],
        y=df["cpih"],
        mode="lines",
        name="CPIH",
        line=dict(
            color="#d62728",
            width=2.5,
        ),
        yaxis="y2",
        hovertemplate=(
            "<b>%{x|%b %Y}</b><br>"
            "CPIH: %{y:.2f}"
            "<extra></extra>"
        ),
    )
)

fig2.update_layout(
    height=500,
    template="plotly_white",
    hovermode="x unified",

    xaxis=dict(
        title="Month",
        showgrid=True,
        gridcolor="rgba(128,128,128,0.2)",
    ),

    # LEFT AXIS
    yaxis=dict(
        title="Bank Rate (%)",
        range=[0, 6.5],
        showgrid=True,
        gridcolor="rgba(128,128,128,0.15)",
    ),

    # RIGHT AXIS
    yaxis2=dict(
        title="CPIH Index",
        overlaying="y",
        side="right",
        showgrid=False,
    ),

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),

    margin=dict(
        l=70,
        r=70,
        t=60,
        b=60,
    ),
)

st.plotly_chart(
    fig2,
    use_container_width=True,
    config={
        "displayModeBar": True,
        "displaylogo": False,
        "scrollZoom": True,
    },
)