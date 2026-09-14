# pages/2_📐_Analytical_Framework.py
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from config import ABM_PARAMS, EPISODES, VAR_VARS

st.set_page_config(page_title="Analytical Framework", page_icon="📐", layout="wide")
st.title("📐 Analytical Framework")
st.caption("Mathematical specification of the ABM + VAR + energy-shock pipeline.")

st.markdown(
    """
    ### 1 · Behavioural Agent Model (ABM)

    **Dual expectation formation.** At each month $t$:

    - Fundamentalist forecast:
      $$F_t = \\frac{1}{12} \\sum_{k=1}^{12} P_{t-k}$$
    - Extrapolator forecast:
      $$E_t = P_{t-1} + \\theta \\,(P_{t-1} - P_{t-4})$$
      with $\\theta$ = extrapolation strength (default 0.8).

    **Adaptive weights (softmax over −λ·MSFE).** With rolling 12-month MSFE on
    squared errors $\\varepsilon^2$:

    $$w_i = \\frac{e^{-\\lambda \\cdot \\text{MSFE}_i}}
                   {\\sum_j e^{-\\lambda \\cdot \\text{MSFE}_j}}$$

    **Consumption response.** For a shock $s_t$ (energy log-return),
    $$\\Delta C_t = w_F \\cdot (-\\alpha \\, s_t) + w_E \\cdot (-\\beta \\, s_t)$$
    with $\\alpha$ = fundamentalist sensitivity (0.15) and
    $\\beta$ = extrapolator sensitivity (0.40).

    **Transmission to output.** $\\Delta \\text{GVA}_t$ is regressed on
    $\\Delta C_{t-k}$ for $k \\in \\{6, 12, 18\\}$ months.
    """
)

st.latex(r"w_i = \frac{e^{-\lambda \cdot \mathrm{MSFE}_i}}{\sum_j e^{-\lambda \cdot \mathrm{MSFE}_j}}")

st.markdown(
    f"""
    ### 2 · VAR Specification

    Variable ordering (most exogenous first):
    `{' → '.join(VAR_VARS)}`

    - All series are **first-differenced** to reach stationarity.
    - Lag order **p** is chosen by AIC (`select_order(maxlags=12)`), with
      automatic fallback to 1 if selection fails.
    - Impulse responses computed with Cholesky orthogonalisation over 24 months.

    ### 3 · Crisis Episodes
    """
)

import pandas as pd
st.dataframe(pd.DataFrame.from_dict(EPISODES, orient="index").reset_index()
             .rename(columns={"index": "Episode", "start": "Start", "end": "End"}),
             use_container_width=True, hide_index=True)

st.markdown(
    """
    ### 4 · Real-Economy Stress Score

    For each episode, stress is measured as the **sign-aligned, standardised**
    change relative to the 12-month pre-window:

    $$z_i = \\frac{\\bar{x}^{\\text{crisis}}_i - \\bar{x}^{\\text{pre}}_i}
                  {\\sigma^{\\text{pre}}_i}$$

    Sign convention:
    - GVA growth ↓ = stress  → stress$_\\text{GVA}$ = $-z_\\text{GVA}$
    - Unemployment ↑ = stress → stress$_\\text{UNP}$ = $+z_\\text{UNP}$
    - Retail sales ↓ = stress → stress$_\\text{RET}$ = $-z_\\text{RET}$

    Composite: $\\text{Stress} = \\overline{|\\text{stress}_i|}$.

    ### 5 · Energy Shock Identification

    Shock months are defined as those whose month-on-month fuel price change
    exceeds the **90th percentile** of the full-sample MoM distribution.
    Impact is tested via Welch's t-test against non-shock months.
    """
)

with st.expander("Parameters in use"):
    st.json(ABM_PARAMS)