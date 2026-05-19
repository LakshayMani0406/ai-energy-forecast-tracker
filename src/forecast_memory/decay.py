"""
Forecast Decay Analysis.

Questions answered:
  - Which forecasts failed fastest?
  - Which assumptions collapsed first?
  - Which organizations systematically underestimated?
  - How does error grow as the target year approaches?
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from forecast_memory.store import load_all, load_graded


def org_credibility() -> pd.DataFrame:
    """Rank organizations by historical forecast accuracy on graded entries."""
    df = load_graded()
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby("source")
        .agg(
            n_forecasts=("forecast_id", "count"),
            mean_abs_error=("error_pct", lambda x: x.abs().mean()),
            bias=("error_pct", "mean"),
            worst_error=("error_pct", lambda x: x.abs().max()),
        )
        .reset_index()
    )
    summary["confidence_score"] = (1 - summary["mean_abs_error"] / 100).clip(0, 1).round(3)
    summary["bias_direction"] = summary["bias"].apply(
        lambda b: "under-estimated" if b < -5 else ("over-estimated" if b > 5 else "calibrated")
    )
    return summary.sort_values("confidence_score", ascending=False)


def assumption_autopsy() -> pd.DataFrame:
    """
    Parse assumptions JSON across all graded forecasts.
    Returns failure rates by assumption category.
    """
    df = load_graded()
    if df.empty:
        return pd.DataFrame()

    df["failed"] = df["error_pct"].abs() > 30

    rows: list[dict] = []
    for _, row in df.iterrows():
        try:
            assumptions = json.loads(row["assumptions"]) if row["assumptions"] else {}
        except Exception:
            assumptions = {}
        for key, val in assumptions.items():
            rows.append({
                "source":           row["source"],
                "target_year":      row["target_year"],
                "error_pct":        row["error_pct"],
                "failed":           row["failed"],
                "assumption_key":   key,
                "assumption_value": str(val),
            })

    if not rows:
        return pd.DataFrame()

    autopsy = pd.DataFrame(rows)
    summary = (
        autopsy.groupby("assumption_key")
        .agg(
            n_forecasts=("source", "count"),
            n_failed=("failed", "sum"),
            avg_error=("error_pct", lambda x: x.abs().mean()),
        )
        .reset_index()
    )
    summary["failure_rate"] = (summary["n_failed"] / summary["n_forecasts"]).round(3)
    return summary.sort_values("failure_rate", ascending=False)


def decay_curve() -> pd.DataFrame:
    """
    Build error-vs-horizon data.

    For each graded forecast, compute how many years elapsed between
    publication and target year (forecast horizon). Returns a DataFrame
    suitable for plotting error% vs horizon.
    """
    df = load_graded()
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["pub_year"] = pd.to_datetime(df["published_date"]).dt.year
    df["horizon_years"] = df["target_year"] - df["pub_year"]
    df["abs_error"] = df["error_pct"].abs()
    return df[["source", "forecast_id", "variable", "horizon_years",
               "error_pct", "abs_error", "target_year", "pub_year", "notes"]].dropna(
        subset=["horizon_years", "error_pct"]
    )
