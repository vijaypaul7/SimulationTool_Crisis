# pages/3_🔍_Analytics_Engine.py
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.data_loader import load_and_clean_data, to_csv_bytes
from modules.analytics_engine import (
    EPISODES,
    VARIABLES,
    calculate_episode_impacts,
    calculate_bank_rate_response,
    calculate_stress_scores,
    identify_energy_shocks,
    energy_shock_ttests,
    calculate_volatility_metrics,
    fit_var_model,
    compute_irf,
    var_summary_df,
    granger_causality,
)


st.set_page_config(
    page_title="Analytics Engine",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Analytics Engine")
st.caption(
    "Explore how major UK economic shocks affected prices, growth, employment "
    "and monetary policy."
)


# ============================================================
# DATA
# ============================================================

df = st.session_state.get("df")

if df is None:
    df = load_and_clean_data()
    st.session_state["df"] = df

df = df.copy()
df["month"] = pd.to_datetime(df["month"])
df = df.sort_values("month")


# ============================================================
# PRECOMPUTE
# ============================================================

try:
    impacts_df = calculate_episode_impacts(df, EPISODES)
except Exception:
    impacts_df = pd.DataFrame()

try:
    _, stress_ranking = calculate_stress_scores(df, EPISODES)
except Exception:
    stress_ranking = pd.DataFrame()


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Crisis Impact",
        "🔥 Energy Shocks",
        "🌊 Volatility",
        "📈 VAR IRFs",
    ]
)


# ============================================================
# TAB 1 — CRISIS IMPACT
# ============================================================

with tab1:

    st.subheader("📊 How did major crises affect the UK economy?")

    st.info(
        """
        **How to read this section:** Each crisis is compared with the period
        immediately before it. Positive values mean the indicator increased;
        negative values mean it decreased.
        """
    )

    if impacts_df.empty:
        st.warning("Impact analysis unavailable.")

    else:

        # --------------------------------------------------------
        # IMPACT SUMMARY
        # --------------------------------------------------------

        pct = impacts_df[
            impacts_df["Metric"] == "pct"
        ].copy()

        pp = impacts_df[
            impacts_df["Metric"] == "pp"
        ].copy()

        if not pct.empty:

            st.markdown("### Crisis impact by indicator")

            pivot = pct.pivot(
                index="Episode",
                columns="Variable",
                values="Impact",
            )

            fig = go.Figure(
                data=go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns.tolist(),
                    y=pivot.index.tolist(),
                    colorscale="RdBu_r",
                    zmid=0,
                    text=np.round(pivot.values, 1),
                    texttemplate="%{text}",
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "%{x}: %{z:.2f}%"
                        "<extra></extra>"
                    ),
                    colorbar=dict(title="% change"),
                )
            )

            fig.update_layout(
                height=max(350, len(pivot) * 70),
                template="plotly_white",
                xaxis_title="Economic indicator",
                yaxis_title="Crisis episode",
                margin=dict(l=20, r=20, t=30, b=50),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displaylogo": False},
            )

            st.caption(
                "🔴 Red indicates a fall relative to the pre-crisis period. "
                "🔵 Blue indicates an increase. Darker colours represent larger changes."
            )

        # --------------------------------------------------------
        # IMPACT BAR CHART
        # --------------------------------------------------------

        st.markdown("### Largest economic changes")

        if not pct.empty:

            chart_df = pct.copy()
            chart_df["Label"] = (
                chart_df["Episode"].astype(str)
                + " — "
                + chart_df["Variable"].astype(str)
            )

            chart_df = chart_df.sort_values("Impact")

            fig = go.Figure(
                go.Bar(
                    x=chart_df["Impact"],
                    y=chart_df["Label"],
                    orientation="h",
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Impact: %{x:.2f}%"
                        "<extra></extra>"
                    ),
                )
            )

            fig.add_vline(
                x=0,
                line_dash="dash",
                line_color="gray",
            )

            fig.update_layout(
                height=max(450, len(chart_df) * 30),
                template="plotly_white",
                xaxis_title="Change from pre-crisis period (%)",
                yaxis_title="",
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=50),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displaylogo": False},
            )

            st.caption(
                "This chart ranks the size of the measured economic response. "
                "Bars extending left represent declines; bars extending right represent increases."
            )

        # --------------------------------------------------------
        # BANK RATE RESPONSE
        # --------------------------------------------------------

        st.divider()

        st.subheader("🏦 Bank Rate response")

        st.info(
            """
            **What this shows:** How the Bank of England changed interest rates
            around each crisis. Large positive changes indicate monetary
            tightening; negative changes indicate rate cuts.
            """
        )

        try:

            br = calculate_bank_rate_response(
                df,
                EPISODES,
            )

            if not br.empty:

                st.dataframe(
                    br.round(2),
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as exc:
            st.warning(f"Bank Rate response failed — {exc}")

        # --------------------------------------------------------
        # STRESS RANKING
        # --------------------------------------------------------

        st.divider()

        st.subheader("🚨 Which crisis produced the most stress?")

        st.info(
            """
            **Stress score:** A combined measure of how unusual the economy's
            movements were during each episode. A higher score means the crisis
            produced a larger overall disruption across the indicators.
            """
        )

        if not stress_ranking.empty:

            sr = stress_ranking.sort_values(
                "Standardized Stress",
                ascending=True,
            )

            fig = go.Figure(
                go.Bar(
                    x=sr["Standardized Stress"],
                    y=sr["Episode"],
                    orientation="h",
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Stress score: %{x:.2f}"
                        "<extra></extra>"
                    ),
                )
            )

            fig.update_layout(
                height=max(350, len(sr) * 60),
                template="plotly_white",
                xaxis_title="Standardised stress score",
                yaxis_title="",
                showlegend=False,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displaylogo": False},
            )

            st.caption(
                "Higher bars indicate periods when economic conditions deviated "
                "more strongly from normal conditions."
            )

            with st.expander("View underlying stress data"):
                st.dataframe(
                    stress_ranking.round(3),
                    use_container_width=True,
                    hide_index=True,
                )

            st.download_button(
                "⬇️ Download stress ranking",
                to_csv_bytes(stress_ranking),
                file_name="stress_ranking.csv",
                mime="text/csv",
            )


# ============================================================
# TAB 2 — ENERGY SHOCKS
# ============================================================

with tab2:

    st.subheader("🔥 Energy price shocks")

    st.info(
        """
        **What is an energy shock?** A month where fuel prices rise unusually
        quickly. This section tests whether unusually large fuel price increases
        are associated with changes in retail activity.
        """
    )

    available_fuels = [
        fuel
        for fuel in ["petrol", "diesel"]
        if fuel in df.columns
    ]

    if not available_fuels:

        st.warning("Petrol or diesel data is required.")

    else:

        c1, c2 = st.columns(2)

        fuel = c1.selectbox(
            "Fuel",
            available_fuels,
        )

        pct_thr = c2.slider(
            "Shock threshold percentile",
            75,
            99,
            90,
            help=(
                "A 90th percentile threshold means only the largest "
                "10% of monthly price increases are classified as shocks."
            ),
        )

        shock_df, summary = identify_energy_shocks(
            df,
            fuel_col=fuel,
            threshold_method="percentile",
            threshold_value=pct_thr,
        )

        # --------------------------------------------------------
        # METRICS
        # --------------------------------------------------------

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Energy shock months",
            summary["shock_count"],
        )

        m2.metric(
            "Share of sample",
            f"{summary['shock_percentage']:.1f}%",
        )

        m3.metric(
            "Shock threshold",
            f"{summary['threshold']:.2f}%",
        )

        # --------------------------------------------------------
        # PRICE CHART
        # --------------------------------------------------------

        st.markdown("### When did energy shocks occur?")

        mask = shock_df["fuel_price_shock"] == 1

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=shock_df["month"],
                y=shock_df[fuel],
                mode="lines",
                name=fuel.title(),
                line=dict(width=2),
                hovertemplate=(
                    "<b>%{x|%b %Y}</b><br>"
                    f"{fuel.title()}: %{{y:.1f}}"
                    "<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=shock_df.loc[mask, "month"],
                y=shock_df.loc[mask, fuel],
                mode="markers",
                name="Energy shock",
                marker=dict(
                    size=9,
                    symbol="circle",
                ),
                hovertemplate=(
                    "<b>%{x|%b %Y}</b><br>"
                    f"{fuel.title()}: %{{y:.1f}}<br>"
                    "<b>Shock month</b>"
                    "<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            height=450,
            template="plotly_white",
            hovermode="x unified",
            xaxis_title="Month",
            yaxis_title=f"{fuel.title()} price",
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
            "The highlighted points are months where fuel prices increased "
            "more sharply than the selected percentile threshold."
        )

        # --------------------------------------------------------
        # DISTRIBUTION
        # --------------------------------------------------------

        st.markdown("### How unusual were the price increases?")

        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=shock_df["fuel_price_mom_pct"].dropna(),
                nbinsx=40,
                name="Monthly changes",
                hovertemplate=(
                    "Monthly change: %{x:.2f}%<br>"
                    "Months: %{y}"
                    "<extra></extra>"
                ),
            )
        )

        fig.add_vline(
            x=summary["threshold"],
            line_dash="dash",
            line_width=2,
            annotation_text="Shock threshold",
        )

        fig.update_layout(
            height=400,
            template="plotly_white",
            xaxis_title="Month-on-month fuel price change (%)",
            yaxis_title="Number of months",
            showlegend=False,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.caption(
            "Most months cluster around normal price movements. The far-right "
            "tail contains unusually large increases; these are classified as shocks."
        )

        # --------------------------------------------------------
        # SHOCK VS NON-SHOCK
        # --------------------------------------------------------

        if "retail" in shock_df.columns:

            st.markdown("### Retail sales: shock vs normal months")

            normal = shock_df.loc[
                ~mask,
                "retail",
            ].dropna()

            shock = shock_df.loc[
                mask,
                "retail",
            ].dropna()

            fig = go.Figure()

            fig.add_trace(
                go.Box(
                    y=normal,
                    name="Normal months",
                    boxmean=True,
                    hovertemplate="Retail index: %{y:.2f}<extra></extra>",
                )
            )

            fig.add_trace(
                go.Box(
                    y=shock,
                    name="Energy shock months",
                    boxmean=True,
                    hovertemplate="Retail index: %{y:.2f}<extra></extra>",
                )
            )

            fig.update_layout(
                height=450,
                template="plotly_white",
                yaxis_title="Retail sales index",
                showlegend=False,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.caption(
                "The boxes show the distribution of retail sales during normal "
                "months versus energy shock months. A large difference suggests "
                "that fuel shocks may coincide with changes in consumer activity."
            )

        # --------------------------------------------------------
        # T TEST
        # --------------------------------------------------------

        st.divider()

        st.subheader("🧪 Statistical test")

        st.caption(
            "The t-test asks whether the average difference between shock and "
            "non-shock months is large enough to be unlikely due to random variation."
        )

        try:

            tt = energy_shock_ttests(
                df,
                fuel_col=fuel,
            )

            st.dataframe(
                tt.round(3),
                use_container_width=True,
                hide_index=True,
            )

        except Exception as exc:
            st.warning(f"Statistical test failed — {exc}")


# ============================================================
# TAB 3 — VOLATILITY
# ============================================================

with tab3:

    st.subheader("🌊 Economic volatility")

    st.info(
        """
        **What is volatility?** Volatility measures how much an indicator has
        been moving recently. The charts below use a rolling 12-month window,
        so spikes identify periods of unusually unstable economic conditions.
        """
    )

    vol_df, vol_summary = calculate_volatility_metrics(df)

    indicators = [
        ("CPIH", "cpih"),
        ("Retail", "retail"),
        ("GVA YoY", "gva_yoy"),
        ("Unemployment", "unemployment"),
        ("Bank Rate", "bank_rate"),
        ("Petrol", "petrol"),
        ("Diesel", "diesel"),
    ]

    available = [
        (name, col)
        for name, col in indicators
        if f"{col}_vol_12m" in vol_df.columns
    ]

    # --------------------------------------------------------
    # SELECT INDICATOR
    # --------------------------------------------------------

    selected_name = st.selectbox(
        "Choose an indicator",
        [name for name, _ in available],
    )

    selected_col = dict(available)[selected_name]
    vol_col = f"{selected_col}_vol_12m"

    series = vol_df[vol_col]

    # --------------------------------------------------------
    # VOLATILITY CHART
    # --------------------------------------------------------

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=vol_df["month"],
            y=series,
            mode="lines",
            name="12-month volatility",
            line=dict(width=2.5),
            fill="tozeroy",
            hovertemplate=(
                "<b>%{x|%b %Y}</b><br>"
                "12m volatility: %{y:.3f}"
                "<extra></extra>"
            ),
        )
    )

    average = series.mean()

    fig.add_hline(
        y=average,
        line_dash="dash",
        line_color="gray",
        annotation_text="Long-run average",
    )

    fig.update_layout(
        height=450,
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Month",
        yaxis_title="12-month volatility",
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
        f"Interpretation: higher values mean {selected_name} has been changing "
        "more strongly or unpredictably over the recent 12 months."
    )

    # --------------------------------------------------------
    # SUMMARY CARDS
    # --------------------------------------------------------

    valid = series.dropna()

    if len(valid):

        current = valid.iloc[-1]
        maximum = valid.max()

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Current volatility",
            f"{current:.3f}",
        )

        c2.metric(
            "Average volatility",
            f"{valid.mean():.3f}",
        )

        c3.metric(
            "Maximum volatility",
            f"{maximum:.3f}",
        )

    # --------------------------------------------------------
    # ALL INDICATORS COMPARISON
    # --------------------------------------------------------

    st.divider()

    st.markdown("### Volatility across the economy")

    fig = go.Figure()

    for name, col in available:

        vc = f"{col}_vol_12m"

        fig.add_trace(
            go.Scatter(
                x=vol_df["month"],
                y=vol_df[vc],
                mode="lines",
                name=name,
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "%{x|%b %Y}<br>"
                    "Volatility: %{y:.3f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=500,
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Month",
        yaxis_title="12-month volatility",
        legend_title="Indicator",
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
        "Use the legend to hide or show individual indicators. "
        "This helps identify periods when volatility increased across several "
        "parts of the economy at the same time."
    )

    if vol_summary:

        with st.expander("View volatility statistics"):

            st.dataframe(
                pd.DataFrame(vol_summary).T.round(3),
                use_container_width=True,
            )


# ============================================================
# TAB 4 — VAR / IMPULSE RESPONSE
# ============================================================

with tab4:

    st.subheader("📈 VAR impulse responses")

    st.info(
        """
        **What is an impulse response?** It estimates how an unexpected shock
        to one economic variable is followed by changes in other variables
        over the following months.

        For example, a positive petrol-price shock can be used to estimate
        how CPIH subsequently responds.
        """
    )

    st.caption(
        "Model: differenced data → AIC lag selection → Cholesky identification."
    )

    @st.cache_resource(show_spinner="Fitting VAR model…")
    def _fit(data_json, maxlags):
        return fit_var_model(
            "hash",
            maxlags=maxlags,
            data_json=data_json,
        )

    try:

        var_result = _fit(
            df.to_json(),
            12,
        )

    except Exception as exc:

        var_result = None

        st.warning(
            f"VAR estimation failed — {exc}"
        )

    if var_result is not None:

        variables = var_result["variables"]

        # --------------------------------------------------------
        # MODEL SUMMARY
        # --------------------------------------------------------

        st.markdown("### Model variables")

        cols = st.columns(len(variables))

        for i, variable in enumerate(variables):

            cols[i].metric(
                variable.upper(),
                "Included",
            )

        with st.expander("View VAR model summary"):

            st.dataframe(
                var_summary_df(var_result),
                use_container_width=True,
                hide_index=True,
            )

        # --------------------------------------------------------
        # CONTROLS
        # --------------------------------------------------------

        st.divider()

        c1, c2, c3 = st.columns(3)

        periods = c1.slider(
            "Months after shock",
            6,
            36,
            24,
        )

        orth = c2.checkbox(
            "Orthogonalised responses",
            value=True,
            help=(
                "Separates shocks using a Cholesky decomposition so that "
                "the responses can be interpreted as distinct innovations."
            ),
        )

        shock_variable = c3.selectbox(
            "Shock variable",
            variables,
        )

        # --------------------------------------------------------
        # IRF
        # --------------------------------------------------------

        try:

            irf_df = compute_irf(
                var_result,
                periods=periods,
                orth=orth,
            )

            # Filter responses to selected shock
            if "shock" in irf_df.columns:

                selected_irf = irf_df[
                    irf_df["shock"] == shock_variable
                ].copy()

            else:

                selected_irf = irf_df.copy()

            # If implementation doesn't expose shock column,
            # retain original behaviour.
            if selected_irf.empty:

                selected_irf = irf_df.copy()

            fig = go.Figure()

            for response in variables:

                sub = selected_irf[
                    selected_irf["response"] == response
                ].sort_values("horizon")

                if sub.empty:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=sub["horizon"],
                        y=sub["value"],
                        mode="lines+markers",
                        name=response,
                        marker=dict(size=5),
                        hovertemplate=(
                            f"<b>{response}</b><br>"
                            "Month: %{x}<br>"
                            "Response: %{y:.4f}"
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
                height=550,
                template="plotly_white",
                hovermode="x unified",
                title=f"Response to a shock in {shock_variable}",
                xaxis_title="Months after shock",
                yaxis_title="Estimated response",
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
                "Lines above zero indicate a positive response to the shock; "
                "lines below zero indicate a negative response. The further a "
                "line moves from zero, the stronger the estimated response."
            )

            # ----------------------------------------------------
            # IRF DATA
            # ----------------------------------------------------

            with st.expander("View IRF data"):

                st.dataframe(
                    irf_df.round(4),
                    use_container_width=True,
                    hide_index=True,
                )

            st.download_button(
                "⬇️ Download IRF matrix",
                to_csv_bytes(irf_df),
                file_name="var_irfs.csv",
                mime="text/csv",
            )

            # ----------------------------------------------------
            # GRANGER CAUSALITY
            # ----------------------------------------------------

            st.divider()

            st.subheader("🔎 Granger causality")

            st.info(
                """
                **What does this test ask?** Whether past values of one
                variable contain useful information for predicting another
                variable.

                A low p-value provides evidence that the first variable
                improves prediction of the second. It does **not** prove
                that the first variable causes the second in a strict
                economic or philosophical sense.
                """
            )

            try:

                gc = granger_causality(
                    var_result,
                    ["petrol", "diesel"],
                    "cpih",
                )

                st.dataframe(
                    gc,
                    use_container_width=True,
                    hide_index=True,
                )

            except Exception as exc:

                st.warning(
                    f"Granger causality calculation failed — {exc}"
                )

        except Exception as exc:

            st.error(
                f"IRF computation failed — {exc}"
            )