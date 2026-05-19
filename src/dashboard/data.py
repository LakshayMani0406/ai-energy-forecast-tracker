"""Shared cached data-loading layer for all dashboard tabs."""
import sys
import pandas as pd
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "ingest"))
from db import get_conn

MODEL_COLORS = {
    "sarima":         "#00b4d8",
    "prophet":        "#f77f00",
    "ols":            "#7209b7",
    "naive_seasonal": "#606c38",
}
MODEL_LABELS = {
    "sarima":         "SARIMA(1,1,1)(1,1,1)[12] ⭐",
    "prophet":        "Prophet",
    "ols":            "OLS regression",
    "naive_seasonal": "Seasonal naive",
}
GRADE_COLORS = {
    "A": "#2dc653", "B": "#7bc67e", "C": "#f0c040",
    "D": "#e07800", "F": "#c0392b", "pending": "#aaaaaa",
}


@st.cache_data(ttl=300)
def load_co2_history() -> pd.DataFrame:
    with get_conn() as conn:
        df = conn.execute("""
            SELECT ds, mean, p2_5, p97_5
            FROM fusion_posterior
            WHERE variable = 'dc_co2_mt_monthly'
            ORDER BY ds
        """).df()
    df["ds"] = pd.to_datetime(df["ds"])
    return df


@st.cache_data(ttl=300)
def load_energy_history() -> pd.DataFrame:
    with get_conn() as conn:
        df = conn.execute("""
            SELECT year(ds) as yr, sum(mean)/1000.0 as twh
            FROM fusion_posterior
            WHERE variable = 'dc_gwh'
              AND year(ds) <= 2025
            GROUP BY yr
            HAVING count(*) = 12
            ORDER BY yr
        """).df()
    return df


@st.cache_data(ttl=300)
def load_model_forecasts(variable: str = "dc_co2_mt_monthly") -> pd.DataFrame:
    with get_conn() as conn:
        df = conn.execute("""
            SELECT ds, model, yhat, yhat_lower, yhat_upper, is_holdout
            FROM model_forecasts
            WHERE variable = ?
            ORDER BY model, ds
        """, [variable]).df()
    df["ds"] = pd.to_datetime(df["ds"])
    return df


@st.cache_data(ttl=300)
def load_holdout_comparison() -> pd.DataFrame:
    with get_conn() as conn:
        df = conn.execute("""
            SELECT mf.model, mf.ds, mf.yhat, fp.mean AS y_actual
            FROM model_forecasts mf
            JOIN fusion_posterior fp
              ON mf.ds = fp.ds AND fp.variable = mf.variable
            WHERE mf.variable = 'dc_co2_mt_monthly'
              AND mf.is_holdout = TRUE
            ORDER BY mf.model, mf.ds
        """).df()
    df["ds"] = pd.to_datetime(df["ds"])
    return df


@st.cache_data(ttl=300)
def load_leaderboard() -> pd.DataFrame:
    from sklearn.metrics import mean_absolute_error
    cmp = load_holdout_comparison()
    rows = []
    for model, grp in cmp.groupby("model"):
        mae = mean_absolute_error(grp["y_actual"], grp["yhat"])
        rows.append({"Model": MODEL_LABELS.get(model, model), "model_key": model,
                     "Holdout MAE (Mt/mo)": round(mae, 4), "N holdout": len(grp)})
    lb = (pd.DataFrame(rows)
            .sort_values("Holdout MAE (Mt/mo)")
            .reset_index(drop=True))
    lb.insert(0, "Rank", lb.index + 1)
    return lb


@st.cache_data(ttl=300)
def load_2030_projections() -> pd.DataFrame:
    with get_conn() as conn:
        df = conn.execute("""
            SELECT model,
                   sum(yhat)          AS co2_mt_2030,
                   sum(yhat_lower)    AS co2_lower,
                   sum(yhat_upper)    AS co2_upper
            FROM model_forecasts
            WHERE variable = 'dc_co2_mt_monthly'
              AND year(ds) = 2030
            GROUP BY model
        """).df()
    df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    return df.sort_values("co2_mt_2030")


@st.cache_data(ttl=300)
def load_benchmark_scores() -> pd.DataFrame:
    with get_conn() as conn:
        df = conn.execute("""
            SELECT source, report, report_year, forecast_year,
                   variable, forecast_lo, forecast_mid, forecast_hi,
                   actual_value, error_pct, bias, grade, notes, url
            FROM benchmark_scores
            ORDER BY forecast_year, source
        """).df()
    return df


@st.cache_data(ttl=300)
def load_state_2024() -> pd.DataFrame:
    with get_conn() as conn:
        df = conn.execute("""
            SELECT variable, sum(mean)/1000.0 AS twh_2024
            FROM fusion_posterior
            WHERE variable LIKE 'state_dc_gwh_%'
              AND year(ds) = 2024
            GROUP BY variable
        """).df()
    df["state"] = df["variable"].str.replace("state_dc_gwh_", "", regex=False)
    return df[["state", "twh_2024"]].sort_values("twh_2024", ascending=False)


@st.cache_data(ttl=300)
def load_state_2030() -> pd.DataFrame:
    with get_conn() as conn:
        df = conn.execute("""
            SELECT variable, sum(yhat)/1000.0 AS twh_2030
            FROM model_forecasts
            WHERE variable LIKE 'state_dc_gwh_%'
              AND model = 'prophet'
              AND year(ds) = 2030
            GROUP BY variable
        """).df()
    df["state"] = df["variable"].str.replace("state_dc_gwh_", "", regex=False)
    return df[["state", "twh_2030"]].sort_values("twh_2030", ascending=False)
