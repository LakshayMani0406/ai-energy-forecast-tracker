# US AI Datacenter Energy Reference Model: Methodology

**Version:** 1.0  
**Date:** May 2026  
**Author:** Lakshay Mani, MS Analytics, Northeastern University  
**Repository:** github.com/LakshayMani0406/ai-energy-forecast-tracker

---

## Abstract

Published estimates of US datacenter energy consumption and CO₂ emissions diverge by a factor of two to three, creating confusion for policymakers, investors, and infrastructure planners. This document describes a reproducible, open-source reference model that reconciles those estimates through a hierarchical Bayesian fusion of public data sources, generates probabilistic forecasts to 2030 using four time-series models, and grades every major institutional forecast against the resulting actuals. The model is not a research paper; it is an operational baseline — updated automatically when new EIA monthly data is published — whose primary contribution is transparency about *why* the divergences exist and which methodological choices drive them.

---

## 1. Motivation and Problem Statement

The energy footprint of AI infrastructure has become a material concern in energy planning. In 2025, the International Energy Agency estimated US datacenters consumed 183 TWh of electricity (IT load) and emitted 105 Mt of CO₂ in 2024. Earlier analyses disagreed sharply:

- **LBNL (Shehabi et al., 2016)** projected 73 TWh for 2020 under a base scenario.
- **Masanet et al. (2020)** argued workload-adjusted efficiency improvements would keep energy consumption near 91 TWh and flat through 2030.
- **Goldman Sachs (2024)** projected 310 TWh by 2030 — a 160% increase from 2023.

These estimates differ not because one group has better data, but because they use different definitions of:
1. **Scope**: IT load only vs. total facility (IT + cooling + power distribution)
2. **Emission factors**: DC-weighted regional rates vs. national average
3. **Workload mix**: pre-AI efficiency trajectories vs. post-2023 AI compute explosion

This model builds a transparent chain from raw data to estimates, makes every methodological choice explicit, and provides a living benchmark against which future estimates can be positioned.

---

## 2. Data Sources

### 2.1 EIA Commercial Sector Monthly (Primary)

The primary input is the US Energy Information Administration's monthly commercial sector electricity consumption series (EIA Form 861 derivative), covering January 2001 through February 2026 (302 months). This series captures total commercial building energy use, from which datacenter energy is inferred as a fraction.

**Key properties:**
- Monthly resolution, national aggregate
- Covers the full AI adoption era (2015–2026)
- Publicly available via EIA API; updated monthly

**Limitation:** The commercial sector includes all non-industrial, non-residential uses — offices, retail, hospitals — not just datacenters. The datacenter share is a latent variable inferred by the Bayesian fusion model (§4).

### 2.2 EPA eGRID State-Level CO₂ Emission Factors

EPA's Emissions & Generation Resource Integrated Database (eGRID) provides state-level CO₂ emission rates (lb/MWh) and net generation by state for 2020–2023. These are used to:
1. Compute a DC-weighted national emission factor (weighted by the state distribution of datacenter capacity)
2. Compare against the national average emission factor used by IEA

**Key finding from this comparison:** The DC-weighted rate (2023: ~317 g/kWh) is materially lower than the national average (366 g/kWh) because 35% of US datacenter capacity is concentrated in Virginia (Dominion Energy service territory), which has a relatively clean nuclear/gas mix. This explains approximately 13% of the gap between Guidi et al.'s (2024) CO₂ estimate and IEA's.

**Conversion:** lb/MWh × 0.4536 = g/kWh

### 2.3 FERC Interconnection Queues (Partial)

All seven ISO interconnection queues (PJM, MISO, CAISO, ERCOT, NYISO, ISO-NE, SPP) were identified as potential sources for forward-looking capacity data. As of May 2026, all seven ISOs require manual portal access for bulk queue downloads; automated harvesting was blocked by URL changes and authentication requirements. The `ferc_interconnection_datacenter` table is currently empty. Forward-looking capacity estimates instead use the extrapolation of historical trends from fusion_posterior.

---

## 3. Architecture

```
EIA API ─────────────────────────┐
eGRID xlsx (2020–2023) ──────────┤──► DuckDB warehouse ──► Bayesian fusion
FERC queues (manual, partial) ───┘    (warehouse.duckdb)     (PyMC 5)
                                                                   │
                                       ┌───────────────────────────┘
                                       ▼
                                 fusion_posterior table
                                 (302 months × 18 variables)
                                       │
                           ┌───────────┴───────────┐
                           ▼                       ▼
                    Four forecast models     Benchmark scorer
                    (Prophet, SARIMA,        (benchmark_scores)
                     OLS, naive_seasonal)
                    → model_forecasts             │
                                       ┌──────────┘
                                       ▼
                               Streamlit dashboard
                               FastAPI public API
```

All state is persisted in a local DuckDB file (`data/warehouse.duckdb`). The model can be fully reproduced by cloning the repository and running:

```bash
python src/ingest/eia.py
python src/ingest/epa_egrid.py
python src/fusion/bayesian_model.py
python src/forecast/sarima_model.py   # + prophet, ols, naive_seasonal
python src/forecast/evaluate.py
python src/benchmarks/forecasts.py
```

---

## 4. Hierarchical Bayesian Fusion Model

### 4.1 Design Rationale

The fusion model converts EIA commercial sector monthly electricity (GWh) into datacenter-specific energy and CO₂ estimates by inferring three latent variables:

- **dc_share**: fraction of commercial sector energy consumed by datacenters
- **PUE** (Power Usage Effectiveness): ratio of total facility energy to IT load
- **ai_frac**: fraction of DC energy attributable to AI workloads

These cannot be observed directly from EIA data; they are inferred by anchoring the model to three external point estimates:
- IEA Energy and AI 2025: 183 TWh IT load (2024)
- IEA Energy and AI 2025: 105 Mt CO₂ (2024)
- Guidi et al. 2024: 31.5 Mt CO₂ (2018)

### 4.2 Model Specification (PyMC 5)

```python
# dc_share prior: ~4% in 2024, slow linear growth
dc_share_2024  = pm.Beta("dc_share_2024", alpha=35, beta=965)        # mean 3.5%
dc_share_slope = pm.Normal("dc_share_slope", mu=0.001, sigma=0.0005)
dc_share = clip(dc_share_2024 + dc_share_slope × (month_year - 2024), 0.005, 0.20)

# PUE prior: typical datacenter PUE range
pue = pm.TruncatedNormal("pue", mu=1.5, sigma=0.15, lower=1.1, upper=2.2)

# AI fraction: logistic structural break at 2023
ai_frac_base   = pm.Beta("ai_frac_base", alpha=5, beta=95)           # ~5% pre-2023
ai_frac_growth = pm.HalfNormal("ai_frac_growth", sigma=0.4)
ai_frac = ai_frac_base + (1 - ai_frac_base) × (1 - exp(-ai_frac_growth × max(0, year - 2023)))

# Deterministic quantities
dc_gwh             = commercial_gwh × dc_share
dc_it_gwh          = dc_gwh / pue
dc_co2_mt_monthly  = dc_gwh × co2_rate_series × 1e-6

# Observations (benchmark anchors)
obs_energy_2024 ~ Normal(dc_it_gwh[2024-mid] / 1000 × 12, sigma_energy)  observed=183
obs_co2_2024    ~ Normal(dc_co2_mt_monthly[2024-mid] × 12, sigma_co2)    observed=105
obs_co2_2018    ~ Normal(dc_co2_mt_monthly[2018-mid] × 12, sigma_co2)    observed=31.5
```

**CO₂ rate series:** National average g/kWh from eGRID (2020–2023), linearly extrapolated outside eGRID coverage years. The national average (not DC-weighted) is used because: (a) IEA's 105 Mt benchmark uses national average methodology; (b) the DC-weighted rate systematically understates CO₂ due to Virginia's clean grid.

### 4.3 Inference

Sampling: NUTS via numpyro backend (JAX JIT-compiled), 2 chains × 1000 draws, target_accept=0.95. Trace saved to `data/exports/fusion_trace.nc`.

**Posterior summary (2024):**

| Parameter | Posterior mean | 95% CI |
|-----------|---------------|--------|
| dc_share_2024 | 4.2% | 3.8–4.6% |
| PUE | 1.34 | 1.12–1.57 |
| AI fraction (2024) | 37% | 28–47% |
| Total facility energy (2024) | 187.6 TWh | — |
| CO₂ (2024) | 67.7 Mt | — |

Convergence: 14 divergences (acceptable), R-hat ≈ 1.0, ESS > 800 for key parameters.

### 4.4 The IEA CO₂ Gap: An Honest Account

The model produces 67.7 Mt CO₂ for 2024 against IEA's 105 Mt — a gap of 37 Mt (+55%). This is not a rounding error; it reflects a genuine methodological difference that this model cannot fully reconcile.

**Root causes:**

1. **PUE scope**: IEA implicitly uses PUE ~1.58 (facility overhead = 58% above IT load) when converting their 183 TWh IT load to a CO₂ figure. Our posterior PUE = 1.34 gives total facility = 183 × 1.34 = 245 TWh (vs IEA's implied ~290 TWh). This alone accounts for ~18 Mt of the gap.

2. **dc_share vs. IEA bottom-up**: IEA uses a direct facility inventory methodology (counting real buildings from industry reports). Our model infers dc_share from EIA commercial sector data, which may undercount certain facility types or allocate some datacenter energy to industrial or other sectors. The commercial sector boundary is imprecise.

3. **Emission factor application**: Both methodologies use the national average, but the specific grid mix applied to each year may differ.

**Interpretation:** The 37 Mt gap represents the upper bound of what is unknown about US datacenter scope and methodology. Neither estimate is definitively "correct." IEA's bottom-up approach is arguably more direct; our model's Bayesian approach is arguably more transparent and reproducible. The gap is documented everywhere it appears in this project — in the dashboard, in the API, and in this document.

**For users of the CO₂ estimates:** When comparing to IEA, note that our 67.7 Mt is a lower-bound estimate using the EIA commercial sector allocation methodology. The true value likely lies between our estimate and IEA's, depending on how facility scope is defined.

---

## 5. Forecast Models

### 5.1 Target Variable

All four models forecast `dc_co2_mt_monthly` (monthly CO₂ in Megatons) from the `fusion_posterior` table. The last 12 months (the holdout window) are withheld for evaluation; models are trained on the preceding data and forecast through December 2030.

### 5.2 Model Descriptions

**SARIMA(1,1,1)(1,1,1)[12]** (winner, MAE 0.056 Mt/mo):
Seasonal ARIMA with first-order differencing and seasonal differencing, MA(1) and seasonal MA(1) components. Implemented via statsmodels SARIMAX. Captures the strong 12-month periodicity in commercial sector energy with high precision. Wins on holdout MAE by a factor of 2× over Prophet.

**Prophet** (MAE 0.121 Mt/mo):
Facebook/Meta's additive decomposition model with automatic changepoint detection. Also produces state-level energy forecasts (13 states) and is the only model with registered probability intervals from a distributional model. Registered in MLflow model registry.

**OLS regression** (MAE 0.391 Mt/mo):
Annual aggregate regression: CO₂ ~ DC_Energy_TWh + Grid_CO₂_Rate_g_kWh. Ported from the original repo's `phase2_regression.rmd`. Predictors are projected forward using their own linear trends. R² = 0.97 on training data but ranks last on monthly holdout MAE because the annual→monthly flat distribution strips all intra-year variation.

**Seasonal naive** (MAE 0.179 Mt/mo):
Last-year-same-month baseline. Any useful model must beat this; both SARIMA and Prophet do. OLS does not.

### 5.3 2030 Projections

| Model | 2030 CO₂ (Mt/yr) | 95% CI |
|-------|-----------------|--------|
| SARIMA | 80.8 | 69–93 |
| Prophet | 79.0 | 76–82 |
| OLS | 79.2 | 67–91 |
| Naive | 68.3 | 61–75 |

The three probabilistic models converge near 79–81 Mt/yr. The naive baseline produces a lower value because it repeats 2025 seasonal patterns with no trend component.

---

## 6. Institutional Forecast Benchmarking

### 6.1 Methodology

Each published forecast is characterized by:
- Source, report, report year
- Target year
- Variable: total facility energy (TWh), IT load (TWh), or CO₂ (Mt)
- Point estimate + published range (lo/hi where available)

For target years within the fusion_posterior data range (2001–2025), the posterior mean is used as the "actual." For future targets (2026–2030), status is "pending."

Error% = (forecast_mid − actual) / actual × 100

Grade thresholds: A < 5%, B < 15%, C < 30%, D < 50%, F ≥ 50%

### 6.2 Results Summary

| Source | Target | Variable | Forecast | Actual | Error% | Grade |
|--------|--------|----------|----------|--------|--------|-------|
| LBNL (2016) | 2020 | energy_twh | 73 | 146 | −50% | F |
| Masanet (2020) | 2018 | energy_twh | 90 | 146 | −39% | D |
| Guidi (2024) | 2018 | co2_mt | 31.5 | 60.0 | −48% | D |
| IEA (2025) | 2024 | energy_twh_it | 183 | 140* | +31% | D |
| IEA (2025) | 2024 | co2_mt | 105 | 67.7 | +55% | F |

*140 TWh = our model's IT load (total 187.6 / PUE 1.34). IEA scope difference accounts for the apparent overestimate.

### 6.3 The Workload-Efficiency Narrative Failed

The central finding from the benchmark scoreboard is that LBNL 2016 and Masanet 2020 both severely underestimated energy consumption. The Masanet paper explicitly argued that efficiency improvements — hyperscale migration, server consolidation, improved cooling — would offset growing compute workloads, keeping energy flat through 2030. This narrative was technically defensible through 2022.

The AI compute explosion of 2023 invalidated it. GPU-based AI training is inherently less efficient per unit of useful compute than inference or traditional workloads. The arrival of large-scale foundation model training (GPT-4, Gemini, Claude) introduced a step-change in energy intensity that no efficiency curve could absorb on a 5-year horizon.

The 2023 structural break is directly visible in the fusion_posterior AI fraction parameter, which jumps from ~5% (pre-2023) to ~37% (2024) via the logistic growth term.

### 6.4 IEA "Over" Bias

IEA appears to overestimate relative to our model. This is an artifact of the scope/methodology difference described in §4.4, not evidence that IEA is wrong. Users comparing against IEA should use our `energy_twh_it` column (IT load = total / 1.34) rather than `energy_twh` (total facility).

---

## 7. State Decomposition

State-level DC energy is computed as: national `dc_gwh` × state weight, where weights are fixed Dirichlet priors derived from 2024 CBRE/JLL datacenter market reports:

| State | Weight | 2024 TWh | 2030 TWh (Prophet) |
|-------|--------|----------|-------------------|
| VA | 35.0% | 65.7 | 83.2 |
| TX | 8.5% | 15.9 | 20.2 |
| CA | 7.0% | 13.1 | 16.6 |
| GA | 5.5% | 10.3 | 13.1 |
| OH | 4.0% | 7.5 | 9.5 |
| IL | 3.8% | 7.1 | 9.0 |
| AZ | 3.2% | 6.0 | 7.6 |
| WA | 2.8% | 5.3 | 6.7 |
| OR | 2.2% | 4.1 | 5.2 |
| NY | 2.0% | 3.8 | 4.8 |
| NJ | 1.5% | 2.8 | 3.6 |
| NC | 1.2% | 2.3 | 2.9 |
| FL | 1.0% | 1.9 | 2.4 |

Virginia dominates (35%) because of its established hyperscale cluster in Loudoun County ("Data Center Alley"), proximity to major internet exchange points, and favorable regulatory environment. The weights are static; the model does not track the ongoing geographic diversification as operators build in Texas, Arizona, and Georgia.

**Limitation:** Fixed state weights mean state-level forecasts have identical growth trajectories as the national forecast. A more rigorous state model would use state-specific FERC interconnection queue data to estimate differential growth rates. This is a planned improvement (see §9).

---

## 8. MLOps Pipeline

The model runs on a GitHub Actions monthly schedule:

```
EIA ingestion → Bayesian fusion → Four forecasts → Evaluation → Dashboard
```

Each step:
1. Logs parameters and metrics to MLflow tracking server (local file store)
2. Writes results to DuckDB tables (persistent)
3. The winning model is promoted to Production in the MLflow model registry

Current Production model: **SARIMA v2** (pyfunc-wrapped, MAE 0.056 Mt/mo)

Promotion logic: `evaluate.py` computes holdout MAE for all four models from `model_forecasts`, identifies the winner, finds the corresponding MLflow registry version by matching the `model` parameter, and transitions it to Production.

---

## 9. Limitations and Known Issues

**Model-level:**
1. EIA commercial sector boundary imprecision — some DC energy may be classified as industrial
2. Fixed PUE at posterior mean (1.34) for state decomposition and IT-load estimates
3. Static state weights don't capture geographic shift toward TX/AZ/GA post-2023
4. AI fraction inferred from benchmarks, not facility-level operational data

**Data gaps:**
5. FERC interconnection queue table empty — requires manual download from 7 ISOs
6. No hyperscaler capex or GPU shipment data (SEC EDGAR integration not yet built)
7. Guidi 2018 baseline uses DC-weighted emission factor — our national-avg methodology gives systematically higher CO₂

**Forecast limitations:**
8. SARIMA extrapolates historical trend/seasonality; will not capture discontinuities (new nuclear capacity, grid decarbonization, next AI paradigm shift)
9. OLS ranks last because annual→monthly conversion destroys seasonality

**Infrastructure:**
10. MLflow file-based tracking backend deprecated in 2.9+; migration to SQLite planned

---

## 10. References

1. Shehabi, A. et al. (2016). *United States Data Center Energy Usage Report*. LBNL-1005323.
2. Masanet, E. et al. (2020). Recalibrating global data center energy-use estimates. *Science* 367(6481). https://doi.org/10.1126/science.aba3758
3. Guidi, G. et al. (2024). Measuring the carbon intensity of AI in cloud instances. *ASPLOS 2024*. https://doi.org/10.1145/3620666.3651329
4. IEA (2025). *Energy and AI*. International Energy Agency. https://www.iea.org/reports/energy-and-ai
5. Goldman Sachs (2024). *AI is poised to drive 160% increase in power demand*. GS Global Investment Research.
6. EPRI (2024). *Powering Intelligence: Analyzing Artificial Intelligence and Data Center Energy Consumption*. EPRI 3002028905.
7. McKinsey & Company (2024). *Investing in the rising data center economy*.
8. Salvatier, J., Wiecki, T. V., & Fonnesbeck, C. (2016). Probabilistic programming in Python using PyMC3. *PeerJ Computer Science*. https://doi.org/10.7717/peerj-cs.55
9. Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1). https://doi.org/10.1080/00031305.2017.1380080

---

*This document is maintained in the repository at `blog/methodology.md`. The model code, data pipeline, and this writeup are updated together. All figures referenced here are reproduced programmatically from the live DuckDB warehouse.*
