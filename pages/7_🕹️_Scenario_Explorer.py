# pages/7_🕹️_Scenario_Explorer.py
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
from modules.abm_simulator import simulate_shock_response
from config import ABM_PARAMS


# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------

st.set_page_config(
    page_title="Scenario Explorer",
    page_icon="🕹️",
    layout="wide",
)

st.title("🕹️ Scenario Explorer")
st.caption(
    "Compare how different policy and economic scenarios could affect consumption."
)


# ---------------------------------------------------------
# DATA
# ---------------------------------------------------------

df = st.session_state.get("df")

if df is None:
    df = load_and_clean_data()
    st.session_state["df"] = df


# ---------------------------------------------------------
# SCENARIOS
# ---------------------------------------------------------

SCENARIOS = {
    "Monetary Tightening": {
        "shock_bps": 200,
        "lambda": 0.5,
        "theta": 0.8,
    },
    "Stagflation Shock": {
        "shock_bps": 300,
        "lambda": 0.7,
        "theta": 1.0,
    },
    "Policy Easing": {
        "shock_bps": -150,
        "lambda": 0.4,
        "theta": 0.6,
    },
    "COVID-style Collapse": {
        "shock_bps": -250,
        "lambda": 0.9,
        "theta": 1.2,
    },
    "Custom": {
        "shock_bps": 100,
        "lambda": ABM_PARAMS["lambda"],
        "theta": ABM_PARAMS["theta"],
    },
}


# ---------------------------------------------------------
# CONTROLS
# ---------------------------------------------------------

st.subheader("Choose scenarios")

selected = st.multiselect(
    "Scenarios to compare",
    list(SCENARIOS.keys()),
    default=[
        "Monetary Tightening",
        "Policy Easing",
    ],
    help="Select one or more scenarios to compare on the same chart.",
)


if "Custom" in selected:

    st.markdown("#### Custom scenario")

    c1, c2, c3 = st.columns(3)

    SCENARIOS["Custom"]["shock_bps"] = c1.slider(
        "Interest-rate shock (bps)",
        -400,
        600,
        100,
        25,
    )

    SCENARIOS["Custom"]["lambda"] = c2.slider(
        "λ — strategy selection",
        0.0,
        2.0,
        0.5,
        0.05,
    )

    SCENARIOS["Custom"]["theta"] = c3.slider(
        "θ — extrapolation",
        0.0,
        1.5,
        0.8,
        0.05,
    )


horizon = st.slider(
    "Simulation horizon",
    6,
    48,
    24,
    help="Number of months simulated after the initial shock.",
)


# ---------------------------------------------------------
# NO SCENARIOS
# ---------------------------------------------------------

if not selected:

    st.info(
        "Select at least one scenario above to start the simulation."
    )

    st.stop()


# ---------------------------------------------------------
# RUN SIMULATIONS
# ---------------------------------------------------------

results = {}

for name in selected:

    params = SCENARIOS[name]

    sim = simulate_shock_response(
        df,
        shock_bps=params["shock_bps"],
        lambda_val=params["lambda"],
        alpha=ABM_PARAMS["alpha"],
        beta=ABM_PARAMS["beta"],
        horizon=horizon,
    )

    results[name] = sim


# ---------------------------------------------------------
# MAIN COMPARISON
# ---------------------------------------------------------

st.divider()

st.subheader("1. Scenario comparison")

st.caption(
    "The lines show the simulated change in consumption after each shock. "
    "Values below zero indicate weaker consumption, while values above zero "
    "indicate stronger consumption."
)

fig = go.Figure()

for name, sim in results.items():

    fig.add_trace(
        go.Scatter(
            x=sim["horizon"],
            y=sim["aggregate_response"],
            mode="lines+markers",
            name=name,
            line=dict(width=2.5),
            hovertemplate=(
                f"<b>{name}</b><br>"
                "Month: %{x}<br>"
                "Consumption response: %{y:.3f} pp"
                "<extra></extra>"
            ),
        )
    )


fig.add_hline(
    y=0,
    line_dash="dash",
    line_width=1,
)


fig.update_layout(
    height=500,
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
    fig,
    use_container_width=True,
    config={
        "displaylogo": False,
        "scrollZoom": True,
    },
)


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

st.subheader("2. Scenario summary")

st.caption(
    "Use the summary below to compare the size of the initial impact "
    "and the response at the end of the simulation."
)


summary_rows = []

for name, sim in results.items():

    aggregate = sim["aggregate_response"]

    min_value = float(aggregate.min())
    max_value = float(aggregate.max())
    final_value = float(aggregate.iloc[-1])

    if abs(min_value) >= abs(max_value):
        peak_value = min_value
    else:
        peak_value = max_value

    params = SCENARIOS[name]

    summary_rows.append(
        {
            "Scenario": name,
            "Shock (bps)": params["shock_bps"],
            "Peak response (pp)": peak_value,
            "Final response (pp)": final_value,
        }
    )


summary = pd.DataFrame(summary_rows)


# ---------------------------------------------------------
# KEY METRICS
# ---------------------------------------------------------

if not summary.empty:

    strongest = summary.iloc[
        summary["Peak response (pp)"].abs().argmax()
    ]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Largest simulated impact",
        f"{strongest['Peak response (pp)']:.3f} pp",
    )

    c2.metric(
        "Scenario",
        strongest["Scenario"],
    )

    c3.metric(
        "Final response",
        f"{strongest['Final response (pp)']:.3f} pp",
    )


st.dataframe(
    summary.round(3),
    width="stretch",
    hide_index=True,
)


# ---------------------------------------------------------
# AGENT DECOMPOSITION
# ---------------------------------------------------------

st.divider()

st.subheader("3. What drives each scenario?")

st.caption(
    "Each scenario combines two behavioural responses. "
    "Fundamentalists respond to economic conditions relative to expectations, "
    "while extrapolators respond more strongly to recent movements."
)


for name, sim in results.items():

    with st.expander(name):

        fig_agent = go.Figure()

        fig_agent.add_trace(
            go.Scatter(
                x=sim["horizon"],
                y=sim["fundamentalist_response"],
                mode="lines",
                name="Fundamentalist",
                line=dict(width=2),
                hovertemplate=(
                    "Month: %{x}<br>"
                    "Response: %{y:.3f} pp"
                    "<extra></extra>"
                ),
            )
        )

        fig_agent.add_trace(
            go.Scatter(
                x=sim["horizon"],
                y=sim["extrapolator_response"],
                mode="lines",
                name="Extrapolator",
                line=dict(width=2),
                hovertemplate=(
                    "Month: %{x}<br>"
                    "Response: %{y:.3f} pp"
                    "<extra></extra>"
                ),
            )
        )

        fig_agent.add_trace(
            go.Scatter(
                x=sim["horizon"],
                y=sim["aggregate_response"],
                mode="lines",
                name="Overall",
                line=dict(width=3),
                hovertemplate=(
                    "Month: %{x}<br>"
                    "Overall response: %{y:.3f} pp"
                    "<extra></extra>"
                ),
            )
        )

        fig_agent.add_hline(
            y=0,
            line_dash="dash",
            line_width=1,
        )

        fig_agent.update_layout(
            height=380,
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
            fig_agent,
            use_container_width=True,
            config={"displaylogo": False},
        )


# ---------------------------------------------------------
# HOW TO READ IT
# ---------------------------------------------------------

st.divider()

st.subheader("How to interpret the scenarios")

st.info(
    "**More negative response:** consumption falls more strongly after the shock.  \n"
    "**More positive response:** consumption increases relative to the baseline.  \n"
    "**Steeper line:** the effect happens more quickly.  \n"
    "**Long-lasting effect:** the response remains away from zero for longer."
)


# ---------------------------------------------------------
# DOWNLOAD
# ---------------------------------------------------------

st.divider()

st.download_button(
    "⬇️ Download all scenario results",
    to_csv_bytes(
        pd.concat(
            [
                sim.assign(scenario=name)
                for name, sim in results.items()
            ],
            ignore_index=True,
        )
    ),
    file_name="scenario_simulations.csv",
    mime="text/csv",
)