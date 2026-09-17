# pages/2_📊_Data_Explorer.py
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from modules.data_loader import load_and_clean_data, to_csv_bytes, available_indicators
from modules.analytics_engine import compute_lagged_correlations
from config import LABELS

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")
st.title("📊 Data Explorer")
st.caption("Filters, correlations, scatter, PCA, lagged CCF — Colab analysis sections.")

df = st.session_state.get("df")
if df is None:
    df = load_and_clean_data(); st.session_state["df"] = df
df = df.copy()
df["month"] = pd.to_datetime(df["month"])

raw_cols = available_indicators(df)
if not raw_cols:
    st.error("No recognised indicators."); st.stop()

# Date slider
min_d = df["month"].min().to_pydatetime(); max_d = df["month"].max().to_pydatetime()
dr = st.slider("Date range", min_value=min_d, max_value=max_d, value=(min_d, max_d), format="YYYY-MM")
start, end = pd.Timestamp(dr[0]), pd.Timestamp(dr[1])
filtered = df[(df["month"] >= start) & (df["month"] <= end)].copy()
if filtered.empty:
    st.warning("No data in range."); st.stop()

raw_df = filtered[raw_cols].copy().ffill().bfill()
display_df = raw_df.rename(columns=LABELS)


def label(c): return LABELS.get(c, c)


tab1, tab2, tab3, tab4 = st.tabs(["🔥 Correlation", "🔀 Scatter", "🧠 PCA", "⏱️ Lagged CCF"])

with tab1:
    st.subheader("Correlation matrix")
    corr = display_df.corr()
    fig, ax = plt.subplots(figsize=(11, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=1, ax=ax,
                cbar_kws={"shrink": 0.8, "label": "Correlation"})
    ax.set_title("Correlation Matrix", fontweight="bold")
    plt.tight_layout(); st.pyplot(fig, use_container_width=True)

    if "Retail Sales" in corr.columns:
        st.markdown("#### Correlation with Retail Sales")
        tc = corr["Retail Sales"].drop("Retail Sales").sort_values()
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.barh(tc.index, tc.values,
                 color=["#E74C3C" if v < 0 else "#2ECC71" for v in tc.values],
                 edgecolor="black", linewidth=0.5)
        ax2.axvline(0, color="black", linewidth=0.8); ax2.set_xlim(-1, 1)
        ax2.set_xlabel("Correlation with Retail Sales"); ax2.grid(True, alpha=0.3, axis="x")
        plt.tight_layout(); st.pyplot(fig2, use_container_width=True)

    st.divider()
    st.dataframe(filtered, use_container_width=True, height=350)
    st.download_button("⬇️ Download filtered CSV", to_csv_bytes(filtered),
                       file_name="uk_macro_filtered.csv", mime="text/csv")

with tab2:
    st.subheader("Scatter")
    c1, c2 = st.columns(2)
    x_var = c1.selectbox("X-axis", raw_cols, index=raw_cols.index("cpih") if "cpih" in raw_cols else 0, format_func=label)
    y_var = c2.selectbox("Y-axis", raw_cols, index=raw_cols.index("retail") if "retail" in raw_cols else 1, format_func=label)
    color_var = st.selectbox("Colour by", ["None"] + raw_cols, format_func=lambda c: "None" if c == "None" else label(c))

    fig, ax = plt.subplots(figsize=(11, 6))
    if color_var != "None":
        sc = ax.scatter(raw_df[x_var], raw_df[y_var], c=raw_df[color_var],
                        cmap="RdYlBu_r", s=60, alpha=0.7, edgecolors="black", linewidth=0.5)
        plt.colorbar(sc, ax=ax, label=label(color_var))
    else:
        ax.scatter(raw_df[x_var], raw_df[y_var], s=60, alpha=0.7, edgecolors="black", linewidth=0.5)

    pair = raw_df[[x_var, y_var]].dropna()
    if len(pair) > 2:
        z = np.polyfit(pair[x_var], pair[y_var], 1); p = np.poly1d(z)
        xs = np.sort(pair[x_var].values)
        ax.plot(xs, p(xs), "--", color="black", linewidth=1.5, alpha=0.7,
                label=f"r = {pair[x_var].corr(pair[y_var]):.2f}")
        ax.legend()

    ax.set_xlabel(label(x_var)); ax.set_ylabel(label(y_var))
    ax.set_title(f"{label(y_var)} vs {label(x_var)}", fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True)

with tab3:
    st.subheader("Principal Component Analysis")
    pca_data = display_df.dropna()
    if len(pca_data) < 10 or pca_data.shape[1] < 2:
        st.warning("Not enough rows/columns for PCA.")
    else:
        scaled = StandardScaler().fit_transform(pca_data)
        pca = PCA().fit(scaled)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
        a1.bar(range(1, len(pca.explained_variance_ratio_) + 1),
               pca.explained_variance_ratio_ * 100, color="#3498DB", edgecolor="black")
        a1.set_xlabel("PC"); a1.set_ylabel("Variance %")
        a1.set_title("Scree", fontweight="bold"); a1.grid(True, alpha=0.3)

        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f"PC{i+1}" for i in range(len(pca.components_))],
            index=pca_data.columns,
        )
        
        a2.scatter(
            loadings["PC1"], loadings["PC2"],
            s=100, color="#2ECC71",
            edgecolors="black", linewidth=1,
        )
        
        for i, v in enumerate(pca_data.columns):
            a2.annotate(
                v,
                (loadings["PC1"].iloc[i], loadings["PC2"].iloc[i]),   # ← .iloc here
                fontsize=10, ha="center", va="bottom",
            )
        a2.axhline(0, color="gray", linewidth=0.8, alpha=0.5)
        a2.axvline(0, color="gray", linewidth=0.8, alpha=0.5)
        a2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        a2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        a2.set_title("Loadings", fontweight="bold"); a2.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True)
        st.metric("Cumulative variance (PC1+PC2)",
                  f"{pca.explained_variance_ratio_[:2].sum()*100:.1f}%")

with tab4:
    st.subheader("Lagged correlations with Retail Sales")
    if "retail" not in filtered.columns:
        st.warning("Retail missing.")
    else:
        work = filtered[["month", "retail"] +
                        [c for c in ["cpih", "bank_rate", "petrol"] if c in filtered.columns]].copy()
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, var in zip(axes, ["cpih", "bank_rate", "petrol"]):
            if var not in work.columns:
                ax.axis("off"); continue
            lags, corrs = compute_lagged_correlations(work, "retail", var, max_lag=12)
            ax.bar(lags, corrs,
                   color=["#E74C3C" if v < 0 else "#2ECC71" for v in corrs],
                   edgecolor="black", linewidth=0.5)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.axvline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Current")
            ax.set_xlabel("Lag (months)"); ax.set_ylabel("Corr with Retail")
            ax.set_title(f"Retail vs {label(var)}", fontweight="bold")
            ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True)
