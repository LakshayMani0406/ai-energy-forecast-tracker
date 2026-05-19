# ⚡ US AI Datacenter Energy Reference Model

> A reproducible, open-source reference model for US AI datacenter energy consumption and CO₂ emissions. Bayesian fusion of EIA + eGRID data, four forecast models to 2030, and a living scoreboard that grades every major published institutional forecast against measured actuals.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.13-orange.svg)](https://mlflow.org)
[![DuckDB](https://img.shields.io/badge/DuckDB-warehouse-yellow.svg)](https://duckdb.org)

---

## Key Findings

1. **Pre-2023 forecasts underestimated by 38–50%** — LBNL 2016 predicted 73 TWh for 2020; actual was 146 TWh. The workload-efficiency narrative assumed AI wouldn't happen.
2. **AI fraction jumped from 5% to 37%** between 2022 and 2024 — a structural break, not a trend.
3. **SARIMA beats Prophet** on 12-month holdout (MAE 0.056 vs 0.121 Mt/mo).
4. **IEA's 105 Mt CO₂ estimate vs our 68 Mt** — explained by PUE scope (1.34 vs 1.58) and facility boundary definition, documented in full.

---

## Architecture

```
EIA commercial monthly ──────────────────┐
EPA eGRID state CO₂ rates ───────────────┤──► DuckDB warehouse
FERC interconnection queues (partial) ───┘         │
                                                    ▼
                                     PyMC 5 Hierarchical Bayesian Fusion
                                     (infers dc_share, PUE, AI fraction)
                                                    │
                                         fusion_posterior table
                                         (302 months × 18 variables)
                                                    │
                              ┌─────────────────────┤──────────────────────┐
                              ▼                     ▼                      ▼
                       4 Forecast Models    Benchmark Scorer         Streamlit Dashboard
                       (SARIMA ⭐ winner)   10 institutions          FastAPI public API
                       → model_forecasts    → benchmark_scores
```

---

## Institutional Forecast Scoreboard

| Source | Target Year | Forecast | Actual (model) | Error% | Grade |
|--------|-------------|----------|----------------|--------|-------|
| LBNL 2016 | 2020 energy | 73 TWh | 146 TWh | −50% | **F** |
| Masanet 2020 | 2018 energy | 90 TWh | 146 TWh | −39% | **D** |
| Guidi 2024 | 2018 CO₂ | 31.5 Mt | 60 Mt | −48% | **D** |
| IEA 2025 | 2024 IT load | 183 TWh | 140 TWh* | +31% | **D** |
| IEA 2025 | 2024 CO₂ | 105 Mt | 67.7 Mt | +55% | **F** |

Goldman Sachs / EPRI / McKinsey / BloombergNEF 2030 targets: **pending** — graded automatically when actuals arrive.

*IEA IT-load scope: our total (187.6 TWh) ÷ PUE 1.34 = 140 TWh IT load

---

## Quick Start

```bash
git clone https://github.com/LakshayMani0406/ai-energy-forecast-tracker
cd ai-energy-forecast-tracker
pip install -r requirements.txt

# Add EIA API key
echo "EIA_API_KEY=your_key_here" > .env  # free at eia.gov/opendata

# Run the full pipeline
python src/ingest/eia.py
python src/ingest/epa_egrid.py
python src/fusion/bayesian_model.py       # ~10 min (NUTS sampling)
python src/forecast/sarima_model.py
python src/forecast/prophet_model.py
python src/forecast/ols_model.py
python src/forecast/naive_seasonal.py
python src/forecast/evaluate.py
python src/benchmarks/forecasts.py

# Dashboard
streamlit run src/dashboard/app.py

# API (local)
uvicorn src.api.main:app --port 8000
# then: http://localhost:8000/docs
```

---

## Project Structure

```
src/
  ingest/        EIA, eGRID, FERC data ingestion → DuckDB
  fusion/        Hierarchical Bayesian model (PyMC 5)
  forecast/      4 time-series models + holdout evaluator
  benchmarks/    Institutional forecast grader
  dashboard/     Streamlit app (4 tabs)
  api/           FastAPI (7 endpoints)
data/
  warehouse.duckdb   (gitignored — regenerate from pipeline)
blog/
  methodology.md     Full technical writeup (~2700 words)
  post.md            Accessible version (~1350 words)
  figures/           PNG exports of all charts
.github/workflows/
  retrain.yml        Monthly GitHub Actions schedule
```

---

## API Reference

```
GET /health                          liveness check
GET /meta                            series metadata
GET /actuals?start=2020&end=2024     annual DC energy + CO₂ from fusion
GET /forecast?model=sarima           monthly forecasts through 2030
GET /forecast/states/VA              state DC energy forecast
GET /models                          holdout MAE leaderboard
GET /benchmarks                      institutional forecast grades
```

Docs at `http://localhost:8000/docs` when running locally.

---

## Data Sources

| Source | Coverage | Access |
|--------|----------|--------|
| EIA Form 861 (commercial monthly) | 2001–present | Free API key at eia.gov |
| EPA eGRID | 2020–2023 | Public Excel files |
| FERC interconnection queues | 7 ISOs | Manual download (all ISOs require portal) |
| IEA Energy and AI 2025 | 2024 benchmarks | Hardcoded in fusion model |

---

## Methodology

Full writeup: [blog/methodology.md](blog/methodology.md)  
Accessible post: [blog/post.md](blog/post.md)

Key design choices:
- **National avg emission factor** (not DC-weighted) — IEA parity; Virginia's nuclear grid would otherwise understate CO₂
- **PUE as latent variable** — jointly inferred from IEA IT-load and CO₂ benchmarks
- **AI fraction structural break at 2023** — logistic growth from 5% baseline

---

## License

MIT. Data sources have their own terms (EIA: public domain; eGRID: public domain; IEA: cited, not redistributed).
