"""Shared cached data-loading layer for all dashboard tabs."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "ingest"))
sys.path.insert(0, str(ROOT / "src"))
from db import get_conn  # noqa: E402

MODEL_COLORS = {
    "sarima":         "#00b4d8",
    "prophet":        "#f97316",
    "ols":            "#a855f7",
    "naive_seasonal": "#606c38",
}
MODEL_LABELS = {
    "sarima":         "SARIMA(1,1,1)(1,1,1)[12] ⭐",
    "prophet":        "Prophet",
    "ols":            "OLS regression",
    "naive_seasonal": "Seasonal naive",
}
GRADE_COLORS = {
    "A": "#22c55e", "B": "#86efac", "C": "#fbbf24",
    "D": "#f97316", "F": "#ef4444", "pending": "#475569",
}

# ── Existing loaders (unchanged) ─────────────────────────────────────────────

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
            WHERE variable = 'dc_gwh' AND year(ds) <= 2025
            GROUP BY yr HAVING count(*) = 12
            ORDER BY yr
        """).df()
    return df


@st.cache_data(ttl=300)
def load_model_forecasts(variable: str = "dc_co2_mt_monthly") -> pd.DataFrame:
    with get_conn() as conn:
        df = conn.execute("""
            SELECT ds, model, yhat, yhat_lower, yhat_upper, is_holdout
            FROM model_forecasts WHERE variable = ?
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
            JOIN fusion_posterior fp ON mf.ds = fp.ds AND fp.variable = mf.variable
            WHERE mf.variable = 'dc_co2_mt_monthly' AND mf.is_holdout = TRUE
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
    lb = (pd.DataFrame(rows).sort_values("Holdout MAE (Mt/mo)").reset_index(drop=True))
    lb.insert(0, "Rank", lb.index + 1)
    return lb


@st.cache_data(ttl=300)
def load_2030_projections() -> pd.DataFrame:
    with get_conn() as conn:
        df = conn.execute("""
            SELECT model,
                   sum(yhat)       AS co2_mt_2030,
                   sum(yhat_lower) AS co2_lower,
                   sum(yhat_upper) AS co2_upper
            FROM model_forecasts
            WHERE variable = 'dc_co2_mt_monthly' AND year(ds) = 2030
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
            FROM benchmark_scores ORDER BY forecast_year, source
        """).df()
    return df


@st.cache_data(ttl=300)
def load_state_2024() -> pd.DataFrame:
    with get_conn() as conn:
        df = conn.execute("""
            SELECT variable, sum(mean)/1000.0 AS twh_2024
            FROM fusion_posterior
            WHERE variable LIKE 'state_dc_gwh_%' AND year(ds) = 2024
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
              AND model = 'prophet' AND year(ds) = 2030
            GROUP BY variable
        """).df()
    df["state"] = df["variable"].str.replace("state_dc_gwh_", "", regex=False)
    return df[["state", "twh_2030"]].sort_values("twh_2030", ascending=False)


# ── Futures Engine loaders ────────────────────────────────────────────────────

_CACHE_DIR = ROOT / "data" / "simulation_outputs" / "cache"


@st.cache_data(ttl=600)
def load_simulation_summary() -> pd.DataFrame | None:
    """
    Load all scenario summaries from real simulation cache dirs.
    Falls back to legacy data/simulations/summary.parquet if cache is empty.

    Columns: scenario, year, variable, p5, p25, p50, p75, p95, mean, std,
             variance, cvar_95, prob_exceed_iea, prob_exceed_2x_anchor,
             prob_exceed_4x_anchor, n_trajectories
    """
    import json

    if _CACHE_DIR.exists():
        frames = []
        for run_dir in sorted(_CACHE_DIR.iterdir()):
            manifest_p = run_dir / "simulation_manifest.json"
            summary_p = run_dir / "summary_metrics.parquet"
            if not manifest_p.exists() or not summary_p.exists():
                continue
            scenario_name = json.loads(manifest_p.read_text())["scenario"]
            df = pd.read_parquet(summary_p)
            df["scenario"] = scenario_name
            frames.append(df)
        if frames:
            return pd.concat(frames, ignore_index=True)

    # Legacy fallback
    try:
        from simulation_engine.trajectories import load_summary
        return load_summary()
    except FileNotFoundError:
        return None


@st.cache_data(ttl=600)
def load_run_manifest(scenario: str) -> dict | None:
    """Return simulation_manifest.json for the most recent run of a scenario."""
    import json

    if not _CACHE_DIR.exists():
        return None
    for run_dir in sorted(_CACHE_DIR.iterdir()):
        manifest_p = run_dir / "simulation_manifest.json"
        if not manifest_p.exists():
            continue
        m = json.loads(manifest_p.read_text())
        if m.get("scenario") == scenario:
            return m
    return None


@st.cache_data(ttl=600)
def load_scenario_report(scenario: str) -> str | None:
    """Return the auto-generated scenario_report.md for a given scenario."""
    if not _CACHE_DIR.exists():
        return None
    for run_dir in sorted(_CACHE_DIR.iterdir()):
        manifest_p = run_dir / "simulation_manifest.json"
        report_p = run_dir / "scenario_report.md"
        if not manifest_p.exists() or not report_p.exists():
            continue
        import json
        if json.loads(manifest_p.read_text()).get("scenario") == scenario:
            return report_p.read_text()
    return None


_SENS_PATH = ROOT / "data" / "simulations" / "sensitivity.parquet"


@st.cache_data(ttl=600)
def load_sensitivity() -> pd.DataFrame | None:
    """
    Spearman sensitivity of driver parameters vs CO₂ at 2030 and 2040.
    Reads from per-run cache dirs first; falls back to committed sensitivity.parquet.

    Columns: scenario, year, parameter, display_name, spearman_r, r_squared_pct
    """
    import json

    if _CACHE_DIR.exists():
        frames = []
        for run_dir in sorted(_CACHE_DIR.iterdir()):
            sens_p = run_dir / "sensitivity_metrics.parquet"
            if sens_p.exists():
                frames.append(pd.read_parquet(sens_p))
        if frames:
            return pd.concat(frames, ignore_index=True)

    if _SENS_PATH.exists():
        return pd.read_parquet(_SENS_PATH)
    return None


@st.cache_data(ttl=300)
def load_co2_annual_history() -> pd.DataFrame:
    """Annual CO₂ totals for fan-chart historical segment."""
    with get_conn() as conn:
        df = conn.execute("""
            SELECT year(ds) AS year, sum(mean) AS co2_mt
            FROM fusion_posterior
            WHERE variable = 'dc_co2_mt_monthly'
            GROUP BY year(ds) HAVING count(*) = 12
            ORDER BY year(ds)
        """).df()
    return df


@st.cache_data(ttl=300)
def load_energy_annual_history() -> pd.DataFrame:
    with get_conn() as conn:
        df = conn.execute("""
            SELECT year(ds) AS year, sum(mean)/1000.0 AS dc_twh
            FROM fusion_posterior
            WHERE variable = 'dc_gwh'
            GROUP BY year(ds) HAVING count(*) = 12
            ORDER BY year(ds)
        """).df()
    return df


# ── Forecast Memory loaders ───────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_forecast_memory() -> pd.DataFrame:
    try:
        from forecast_memory.store import load_all
        return load_all()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_org_credibility() -> pd.DataFrame:
    try:
        from forecast_memory.decay import org_credibility
        return org_credibility()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_decay_curve() -> pd.DataFrame:
    try:
        from forecast_memory.decay import decay_curve
        return decay_curve()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_assumption_autopsy() -> pd.DataFrame:
    try:
        from forecast_memory.decay import assumption_autopsy
        return assumption_autopsy()
    except Exception:
        return pd.DataFrame()


# ── Agent loaders ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def load_gpu_demand_agent() -> pd.DataFrame:
    try:
        from agents.gpu_demand import GPUDemandAgent
        out = GPUDemandAgent().run()
        return out.to_dataframe()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def load_emissions_agent() -> pd.DataFrame:
    try:
        from agents.emissions import EmissionsAgent
        out = EmissionsAgent().run()
        return out.to_dataframe()
    except Exception:
        return pd.DataFrame()
