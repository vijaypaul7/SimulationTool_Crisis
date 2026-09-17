```markdown
# 🇬🇧 UK Economic Crisis Simulator

**Behavioural-agent macro model · econometric VAR · ML predictive suite · interactive scenario tools**

**Client:** Lambda BI Ltd  
**Project:** Simulation Tool for Economic Crisis Prediction 2/2


## What This Is

An interactive Streamlit dashboard that simulates how energy shocks and monetary policy transmit through consumer behaviour to affect UK output (GVA) and retail spending during crisis periods.

It combines:

- A cleaned 263-month UK macro dataset (June 2003 – April 2025)
- An evolutionary Agent-Based Model (ABM) with dual heuristics
- Econometric VAR with impulse response functions
- Machine learning forecasting benchmarks (Ridge, ElasticNet, HistGB, XGBoost)
- An 8-page interactive scenario sandbox

## Research Question

> How do energy price shocks and monetary policy transmit through consumer behavioural responses to influence UK macroeconomic outcomes during crisis periods?

---

## Project Structure

```

## File Structure
.
├── app.py                     # Entry point + navigation
├── config.py                  # Parameters, episodes, rename map
├── requirements.txt           # Dependencies
├── data/                      # Integrated macro dataset + crisis metadata
├── modules/                   # Core logic
│   ├── data_loader.py         # Cleaning, trimming, interpolation
│   ├── analytics_engine.py    # VAR, IRFs, stress scoring, Granger
│   ├── abm_simulator.py       # Agent expectations, MSFE, softmax switching
│   ├── forecasting_models.py  # ML pipeline (Ridge, ElasticNet, HistGB, XGBoost)
│   └── validation_metrics.py  # RMSE, MAE, R², MDA, residual diagnostics
├── pages/                     # 8 Streamlit pages
│   ├── 1_📈_Macros_Dashboard.py
│   ├── 2_📊_Data_Explorer.py
│   ├── 3_🔬_Analytics_Engine.py
│   ├── 4_🔮_Forecast_Lab.py
│   ├── 5_🧪_Simulation_Studio.py
│   ├── 6_✅_Validation.py
│   ├── 7_🕹️_Scenario_Explorer.py
│   └── 8_🏛️_Stakeholder_Gateway.py
└── results/                   # Baked outputs consumed by the UI
    ├── manifest.json
    ├── forecast_metrics.csv
    ├── forecast_predictions_all.csv
    ├── energy_vs_behavioural.csv
    └── agent_weights_results.csv
```

---

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires **Python 3.10+**.

---

## Dashboard Pages

| Page | Purpose |
|---|---|
| **Macro Dashboard** | Overview of core UK macro series |
| **Data Explorer** | EDA: correlation, scatter, PCA, lagged CCF |
| **Analytical Framework** | Mathematical specification of ABM, VAR, shocks |
| **Analytics Engine** | Crisis stress, energy shocks, VAR impulse responses |
| **Forecast Lab** | Out-of-sample ML benchmark results |
| **Simulation Studio** | Real-time ABM parameter sandbox |
| **Validation** | MSFE stability, residuals, model comparison |
| **Scenario Explorer** | Compare pre-configured crisis scenarios |
| **Stakeholder Gateway** | Executive summary for non-technical users |


---

## Data Sources

| Source | Series |
|---|---|
| **ONS** | Retail Sales Index, CPIH, GVA, Unemployment, AWE, Household Consumption (CVM) |
| **DESNZ / BEIS** | Petrol and diesel road fuel prices |
| **Bank of England** | Official Bank Rate |

**Frequency:** Monthly  
**Range:** June 2003 – April 2025  
**Observations:** 263

---

## Limitations
- 263 monthly observations — small sample for high-capacity models
- Mild ABM look-ahead leakage (documented, partially mitigated)
- Consumption variable units not formally verified
---

## License

Academic project — Lambda BI Ltd.
```
