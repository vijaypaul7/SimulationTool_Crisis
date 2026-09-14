# pages/8_🏛️_Stakeholder_Gateway.py
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.data_loader import (
    load_and_clean_data,
    load_episode_metadata,
)
from modules.analytics_engine import (
    EPISODES,
    calculate_bank_rate_response,
    calculate_stress_scores,
)
from modules.abm_simulator import (
    calculate_forecasting_performance,
    calculate_agent_weights,
)


# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------

st.set_page_config(
    page_title="Stakeholder Gateway",
    page_icon="🏛️",
    layout="wide",
)

st.title("🏛️ Stakeholder Gateway")
st.caption(
    "Executive overview of economic stress, monetary policy response, "
    "and current behavioural conditions."
)


# ---------------------------------------------------------
# DATA
# ---------------------------------------------------------

df = st.session_state.get("df")

if df is None:
    df = load_and_clean_data()
    st.session_state["df"] = df

meta = load_episode_metadata()

latest = df.iloc[-1]


# ---------------------------------------------------------
# 1. CURRENT ECONOMY
# ---------------------------------------------------------

st.subheader("Current economic conditions")

st.caption(
    "The latest available observations provide a quick snapshot of the economy."
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Bank Rate",
    f"{latest['bank_rate']:.2f}%",
)

c2.metric(
    "CPIH",
    f"{latest['cpih']:.1f}",
)

c3.metric(
    "Unemployment",
    f"{latest['unemployment']:.2f}%",
)

c4.metric(
    "GVA growth",
    f"{latest['gva_yoy']:.2f}%",
)


# ---------------------------------------------------------
# 2. STRESS RANKING
# ---------------------------------------------------------

st.divider()

st.subheader("1. Which crisis created the most economic stress?")

st.caption(
    "The stress score combines changes in GVA growth, unemployment and "
    "retail activity relative to the 12 months before each crisis. "
    "A higher score means a larger disruption."
)

try:

    _, stress_ranking = calculate_stress_scores(
        df,
        EPISODES,
    )

except Exception as exc:

    stress_ranking = pd.DataFrame()

    st.warning(
        f"Stress ranking could not be calculated: {exc}"
    )


if not stress_ranking.empty:

    sr = stress_ranking.copy()

    # Sort so the largest appears at the top
    sr = sr.sort_values(
        "Standardized Stress",
        ascending=True,
    )

    fig_stress = go.Figure()

    fig_stress.add_trace(
        go.Bar(
            x=sr["Standardized Stress"],
            y=sr["Episode"],
            orientation="h",
            text=sr["Standardized Stress"].round(2),
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Stress score: %{x:.2f}σ"
                "<extra></extra>"
            ),
        )
    )

    fig_stress.update_layout(
        height=max(350, len(sr) * 65),
        template="plotly_white",
        xaxis_title="Standardised stress score",
        yaxis_title="",
        showlegend=False,
    )

    st.plotly_chart(
        fig_stress,
        use_container_width=True,
        config={"displaylogo": False},
    )

    top = stress_ranking.sort_values(
        "Standardized Stress",
        ascending=False,
    ).iloc[0]

    st.success(
        f"**Highest-stress episode:** {top['Episode']} "
        f"with a score of **{top['Standardized Stress']:.2f}σ**.",
        icon="🏆",
    )

    with st.expander("View stress ranking data"):

        st.dataframe(
            stress_ranking.round(3),
            width="stretch",
            hide_index=True,
        )


# ---------------------------------------------------------
# 3. MONETARY POLICY
# ---------------------------------------------------------

st.divider()

st.subheader("2. How did monetary policy respond?")

st.caption(
    "This compares the Bank Rate at the beginning and end of each crisis episode. "
    "It helps show whether monetary policy tightened or eased during periods of stress."
)

try:

    br = calculate_bank_rate_response(
        df,
        EPISODES,
    )

    if br is not None and not br.empty:

        fig_rate = go.Figure()

        fig_rate.add_trace(
            go.Bar(
                x=br["Episode"],
                y=br["Start Rate (%)"],
                name="Start of crisis",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Start: %{y:.2f}%"
                    "<extra></extra>"
                ),
            )
        )

        fig_rate.add_trace(
            go.Bar(
                x=br["Episode"],
                y=br["End Rate (%)"],
                name="End of crisis",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "End: %{y:.2f}%"
                    "<extra></extra>"
                ),
            )
        )

        fig_rate.update_layout(
            height=430,
            template="plotly_white",
            barmode="group",
            yaxis_title="Bank Rate (%)",
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
            fig_rate,
            use_container_width=True,
            config={"displaylogo": False},
        )

        with st.expander("View monetary policy data"):

            st.dataframe(
                br.round(2),
                width="stretch",
                hide_index=True,
            )

except Exception as exc:

    st.warning(
        f"Bank Rate response could not be calculated: {exc}"
    )


# ---------------------------------------------------------
# 4. CURRENT AGENT REGIME
# ---------------------------------------------------------

st.divider()

st.subheader("3. What type of behaviour dominates today?")

st.caption(
    "The ABM separates agents into two behavioural groups: "
    "fundamentalists, who focus on underlying conditions, and "
    "extrapolators, who place more weight on recent trends."
)

try:

    perf = calculate_forecasting_performance(
        df,
        fuel_col="petrol",
    )

    perf = calculate_agent_weights(
        perf,
        lambda_val=0.5,
    )

    valid = perf.dropna(
        subset=[
            "fundamentalist_weight",
            "extrapolator_weight",
        ]
    )

    if not valid.empty:

        last = valid.iloc[-1]

        fundamentalist = float(
            last["fundamentalist_weight"]
        )

        extrapolator = float(
            last["extrapolator_weight"]
        )

        if fundamentalist >= extrapolator:
            dominant = "Fundamentalist"
            dominant_weight = fundamentalist
        else:
            dominant = "Extrapolator"
            dominant_weight = extrapolator

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Fundamentalist",
            f"{fundamentalist:.1%}",
        )

        c2.metric(
            "Extrapolator",
            f"{extrapolator:.1%}",
        )

        c3.metric(
            "Dominant behaviour",
            dominant,
        )


        # ---------------------------------------------
        # Agent regime chart
        # ---------------------------------------------

        fig_agents = go.Figure()

        fig_agents.add_trace(
            go.Scatter(
                x=valid["month"],
                y=valid["fundamentalist_weight"],
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

        fig_agents.add_trace(
            go.Scatter(
                x=valid["month"],
                y=valid["extrapolator_weight"],
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

        fig_agents.add_hline(
            y=0.5,
            line_dash="dash",
            line_width=1,
            annotation_text="50 / 50",
            annotation_position="top left",
        )

        fig_agents.update_layout(
            height=420,
            template="plotly_white",
            hovermode="x unified",
            yaxis=dict(
                title="Agent weight",
                range=[0, 1],
                tickformat=".0%",
            ),
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
            fig_agents,
            use_container_width=True,
            config={
                "displaylogo": False,
                "scrollZoom": True,
            },
        )

        st.info(
            f"The current model is **{dominant.lower()}-weighted**, "
            f"with a simulated weight of **{dominant_weight:.1%}**."
        )

except Exception as exc:

    st.warning(
        f"Agent regime could not be calculated: {exc}"
    )


# ---------------------------------------------------------
# 5. EXECUTIVE TAKEAWAYS
# ---------------------------------------------------------

st.divider()

st.subheader("Key takeaways")

takeaways = []

if not stress_ranking.empty:

    top = stress_ranking.sort_values(
        "Standardized Stress",
        ascending=False,
    ).iloc[0]

    takeaways.append(
        f"**Highest economic stress:** {top['Episode']} "
        f"({top['Standardized Stress']:.2f}σ)."
    )


try:

    if not br.empty:

        br_copy = br.copy()

        br_copy["Change"] = (
            br_copy["End Rate (%)"]
            - br_copy["Start Rate (%)"]
        )

        largest_change = br_copy.loc[
            br_copy["Change"].abs().idxmax()
        ]

        direction = (
            "increased"
            if largest_change["Change"] > 0
            else "decreased"
        )

        takeaways.append(
            f"**Largest Bank Rate movement:** during "
            f"{largest_change['Episode']}, the Bank Rate "
            f"{direction} by "
            f"{abs(largest_change['Change']):.2f} percentage points."
        )

except Exception:
    pass


try:

    takeaways.append(
        f"**Current behavioural regime:** "
        f"{dominant} ({dominant_weight:.1%})."
    )

except Exception:
    pass


if takeaways:

    for item in takeaways:
        st.markdown(f"- {item}")


# ---------------------------------------------------------
# NEXT STEPS
# ---------------------------------------------------------

st.divider()

st.subheader("Explore the analysis")

c1, c2, c3 = st.columns(3)

with c1:

    st.markdown("### 🧪 Simulation Studio")

    st.caption(
        "Change model assumptions and simulate the effect of an individual shock."
    )


with c2:

    st.markdown("### 🕹️ Scenario Explorer")

    st.caption(
        "Compare multiple policy scenarios side-by-side."
    )


with c3:

    st.markdown("### 🔮 Forecast Lab")

    st.caption(
        "Compare machine-learning forecasts and model performance."
    )