"""
main.py — FastAPI public reference API for the US AI Datacenter Energy model.

All data served from DuckDB warehouse (no live recomputation).

Endpoints
  GET /health                              → liveness
  GET /meta                               → series metadata
  GET /forecast                           → national CO₂ forecast (one or all models)
  GET /forecast/states/{state}            → state DC energy forecast
  GET /models                             → model leaderboard (holdout MAE)
  GET /benchmarks                         → institutional forecast scoreboard
  GET /actuals                            → fusion_posterior annual actuals

Run locally:
  uvicorn src.api.main:app --reload --port 8000

Or via Docker:
  docker build -t ai-energy-api .
  docker run -p 8000:8000 ai-energy-api
"""
import sys
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "ingest"))
from db import get_conn

app = FastAPI(
    title="US AI Datacenter Energy Reference API",
    description=(
        "Open reference data for US AI datacenter energy consumption and CO₂ emissions. "
        "Powered by a hierarchical Bayesian fusion of EIA commercial sector data, "
        "eGRID emission factors, and four time-series forecast models."
    ),
    version="1.0.0",
    contact={"name": "Lakshay Mani", "email": "mani.l@northeastern.edu"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

VALID_MODELS = {"sarima", "prophet", "ols", "naive_seasonal"}
TOP_STATES   = {"VA","TX","CA","GA","OH","IL","AZ","WA","OR","NY","NJ","NC","FL"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _exec(sql: str, params=None) -> list[dict]:
    with get_conn() as conn:
        if params:
            result = conn.execute(sql, params)
        else:
            result = conn.execute(sql)
        cols = [d[0] for d in result.description]
        return [dict(zip(cols, row)) for row in result.fetchall()]


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
def health():
    """Liveness check."""
    return {"status": "ok"}


@app.get("/meta", tags=["meta"])
def meta():
    """Dataset metadata: date range, variable list, model list."""
    rows = _exec("""
        SELECT variable, count(*) AS n_rows, min(ds) AS first, max(ds) AS last
        FROM fusion_posterior
        GROUP BY variable ORDER BY variable
    """)
    models = _exec("""
        SELECT model, count(*) AS n_rows, min(ds) AS first, max(ds) AS last
        FROM model_forecasts GROUP BY model ORDER BY model
    """)
    return {
        "fusion_posterior_variables": rows,
        "forecast_models":            models,
        "data_source":                "EIA commercial monthly + EPA eGRID 2020–2023",
        "methodology":                "Hierarchical Bayesian fusion (PyMC 5 / NUTS)",
    }


@app.get("/actuals", tags=["data"])
def actuals(
    start: Optional[str] = Query(None, description="Start year, e.g. '2010'"),
    end:   Optional[str] = Query(None, description="End year, e.g. '2025'"),
):
    """
    Annual DC energy and CO₂ actuals from fusion_posterior (posterior mean).

    Returns one row per full calendar year. 2026 partial — excluded.
    """
    start_yr = int(start) if start else 2001
    end_yr   = int(end)   if end   else 2025

    rows = _exec("""
        SELECT year(ds) AS year,
               round(sum(CASE WHEN variable='dc_gwh'            THEN mean ELSE 0 END)/1000, 2) AS total_facility_twh,
               round(sum(CASE WHEN variable='dc_co2_mt_monthly' THEN mean ELSE 0 END), 3)       AS co2_mt,
               round(sum(CASE WHEN variable='ai_gwh'            THEN mean ELSE 0 END)/1000, 2)  AS ai_twh,
               round(sum(CASE WHEN variable='non_ai_gwh'        THEN mean ELSE 0 END)/1000, 2)  AS non_ai_twh
        FROM fusion_posterior
        WHERE variable IN ('dc_gwh','dc_co2_mt_monthly','ai_gwh','non_ai_gwh')
          AND year(ds) BETWEEN ? AND ?
        GROUP BY year(ds)
        HAVING count(DISTINCT ds) = 12
        ORDER BY year(ds)
    """, [start_yr, end_yr])
    return {"years": rows, "unit_energy": "TWh/yr", "unit_co2": "Mt/yr"}


@app.get("/forecast", tags=["forecast"])
def forecast(
    model:    Optional[str] = Query(None, description="sarima | prophet | ols | naive_seasonal | all"),
    variable: str            = Query("dc_co2_mt_monthly", description="Forecast variable"),
    start:    Optional[str]  = Query(None, description="Start date YYYY-MM (inclusive)"),
    end:      Optional[str]  = Query("2030-12", description="End date YYYY-MM (inclusive)"),
):
    """
    Monthly model forecasts from model_forecasts table.

    - **model**: one of sarima, prophet, ols, naive_seasonal, or 'all' (default)
    - **variable**: dc_co2_mt_monthly (default), dc_gwh, or state_dc_gwh_XX
    - **start / end**: date range filter (YYYY-MM)
    """
    if model and model != "all" and model not in VALID_MODELS:
        raise HTTPException(400, f"model must be one of {sorted(VALID_MODELS)} or 'all'")

    filters = ["variable = ?"]
    params  = [variable]

    if model and model != "all":
        filters.append("model = ?")
        params.append(model)
    if start:
        filters.append("ds >= ?::DATE")
        params.append(f"{start}-01")
    if end:
        filters.append("ds <= ?::DATE")
        params.append(f"{end}-01")

    where = " AND ".join(filters)
    rows = _exec(f"""
        SELECT ds::VARCHAR AS ds, model, variable,
               round(yhat, 4)       AS yhat,
               round(yhat_lower, 4) AS yhat_lower,
               round(yhat_upper, 4) AS yhat_upper,
               is_holdout
        FROM model_forecasts
        WHERE {where}
        ORDER BY model, ds
    """, params)
    return {"n": len(rows), "variable": variable, "rows": rows}


@app.get("/forecast/states/{state}", tags=["forecast"])
def forecast_state(
    state: str,
    start: Optional[str] = Query(None),
    end:   Optional[str] = Query("2030-12"),
):
    """
    Prophet monthly energy forecast for a single US state.

    State codes: VA, TX, CA, GA, OH, IL, AZ, WA, OR, NY, NJ, NC, FL
    """
    state = state.upper()
    if state not in TOP_STATES:
        raise HTTPException(404, f"No forecast available for state '{state}'. "
                                 f"Available: {sorted(TOP_STATES)}")
    variable = f"state_dc_gwh_{state}"
    params   = [variable]
    filters  = ["variable = ?", "model = 'prophet'"]
    if start:
        filters.append("ds >= ?::DATE"); params.append(f"{start}-01")
    if end:
        filters.append("ds <= ?::DATE"); params.append(f"{end}-01")
    where = " AND ".join(filters)

    rows = _exec(f"""
        SELECT ds::VARCHAR AS ds,
               round(yhat, 2)       AS dc_gwh_forecast,
               round(yhat_lower, 2) AS dc_gwh_lower,
               round(yhat_upper, 2) AS dc_gwh_upper
        FROM model_forecasts
        WHERE {where}
        ORDER BY ds
    """, params)
    return {"state": state, "model": "prophet", "unit": "GWh/month", "rows": rows}


@app.get("/models", tags=["evaluation"])
def model_leaderboard():
    """
    12-month holdout MAE leaderboard for all four forecast models on dc_co2_mt_monthly.
    """
    rows = _exec("""
        SELECT mf.model,
               round(avg(abs(mf.yhat - fp.mean)), 4)      AS mae_mt_per_month,
               round(avg(abs(mf.yhat - fp.mean)/fp.mean)*100, 2) AS mape_pct,
               count(*) AS n_holdout
        FROM model_forecasts mf
        JOIN fusion_posterior fp
          ON mf.ds = fp.ds AND fp.variable = mf.variable
        WHERE mf.variable = 'dc_co2_mt_monthly'
          AND mf.is_holdout = TRUE
        GROUP BY mf.model
        ORDER BY mae_mt_per_month
    """)
    return {
        "winner":   rows[0]["model"] if rows else None,
        "variable": "dc_co2_mt_monthly",
        "holdout_months": 12,
        "leaderboard": rows,
    }


@app.get("/benchmarks", tags=["evaluation"])
def benchmarks(graded_only: bool = Query(False, description="Return only graded entries")):
    """
    Institutional forecast scoreboard: LBNL, Masanet, IEA, Goldman Sachs, EPRI, McKinsey, BNEF.

    Error% = (forecast_mid - actual) / actual × 100.
    Grade: A < 5%, B < 15%, C < 30%, D < 50%, F ≥ 50%, pending = no actuals yet.
    """
    filter_sql = "WHERE grade != 'pending'" if graded_only else ""
    rows = _exec(f"""
        SELECT source, report, report_year, forecast_year, variable,
               forecast_lo, forecast_mid, forecast_hi,
               round(actual_value, 3) AS actual_value,
               round(error_pct, 2)    AS error_pct,
               bias, grade, notes, url
        FROM benchmark_scores
        {filter_sql}
        ORDER BY forecast_year, source
    """)
    return {
        "n_total":  len(rows),
        "n_graded": sum(1 for r in rows if r["grade"] != "pending"),
        "benchmarks": rows,
    }
