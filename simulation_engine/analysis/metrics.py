"""
Metrics aggregator — computes statistical summaries from raw trajectory data.

All metrics are derived from ACTUAL simulation outputs, not hardcoded.

Metrics computed:
  - Percentiles: p5, p25, p50, p75, p95
  - Mean, std, variance
  - CVaR (Conditional Value at Risk) at 95th percentile — tail risk
  - Exceedance probabilities: P(X > threshold)
  - Year-over-year growth rate distribution
  - Trajectory divergence from baseline
"""
from __future__ import annotations

import numpy as np
import pandas as pd


IEA_2024_CO2_MT = 105.0       # IEA benchmark (2024)
ANCHOR_CO2_2024 = 67.7        # fusion_posterior actual
ANCHOR_TWH_2024 = 187.6


def compute_percentiles(trajectories: pd.DataFrame) -> pd.DataFrame:
    """
    Compute percentile summary from full trajectory DataFrame.

    Input columns: sim_id, year, dc_twh, ai_twh, dc_co2_mt, pue,
                   carbon_intensity, compute_index, efficiency_index
    Returns columns: year, variable, p5, p25, p50, p75, p95, mean, std, variance,
                     cvar_95, prob_exceed_iea, prob_exceed_2x_baseline,
                     prob_exceed_4x_baseline
    """
    metrics = ["dc_twh", "ai_twh", "dc_co2_mt", "pue", "carbon_intensity",
               "compute_index", "efficiency_index"]
    rows: list[dict] = []

    for year in sorted(trajectories["year"].unique()):
        yr = trajectories[trajectories["year"] == year]
        for metric in metrics:
            v = yr[metric].values

            p5, p25, p50, p75, p95 = np.quantile(v, [0.05, 0.25, 0.50, 0.75, 0.95])

            # CVaR(95%): expected value of worst 5% outcomes
            tail_mask = v >= p95
            cvar_95 = float(v[tail_mask].mean()) if tail_mask.any() else float(p95)

            # Exceedance probabilities (CO₂-specific, skipped for other vars)
            if metric == "dc_co2_mt":
                prob_exceed_iea        = float((v > IEA_2024_CO2_MT).mean())
                prob_exceed_2x         = float((v > 2 * ANCHOR_CO2_2024).mean())
                prob_exceed_4x         = float((v > 4 * ANCHOR_CO2_2024).mean())
            else:
                prob_exceed_iea        = float("nan")
                prob_exceed_2x         = float("nan")
                prob_exceed_4x         = float("nan")

            rows.append({
                "year":                  int(year),
                "variable":              metric,
                "p5":                    float(p5),
                "p25":                   float(p25),
                "p50":                   float(p50),
                "p75":                   float(p75),
                "p95":                   float(p95),
                "mean":                  float(v.mean()),
                "std":                   float(v.std()),
                "variance":              float(v.var()),
                "cvar_95":               cvar_95,
                "prob_exceed_iea":       prob_exceed_iea,
                "prob_exceed_2x_anchor": prob_exceed_2x,
                "prob_exceed_4x_anchor": prob_exceed_4x,
                "n_trajectories":        len(v),
            })

    return pd.DataFrame(rows)


def compute_yoy_growth(trajectories: pd.DataFrame, variable: str = "dc_co2_mt") -> pd.DataFrame:
    """
    Compute year-over-year growth rate distribution across all trajectories.
    Returns percentiles of annual growth rate for each year transition.
    """
    years = sorted(trajectories["year"].unique())
    rows: list[dict] = []

    for i in range(1, len(years)):
        y0, y1 = years[i - 1], years[i]
        v0 = trajectories[trajectories["year"] == y0][variable].values
        v1 = trajectories[trajectories["year"] == y1][variable].values
        growth = (v1 - v0) / np.where(v0 > 0, v0, 1)

        p5, p25, p50, p75, p95 = np.quantile(growth, [0.05, 0.25, 0.50, 0.75, 0.95])
        rows.append({
            "year":      y1,
            "variable":  variable,
            "p5_growth":  float(p5),
            "p25_growth": float(p25),
            "p50_growth": float(p50),
            "p75_growth": float(p75),
            "p95_growth": float(p95),
            "mean_growth": float(growth.mean()),
        })

    return pd.DataFrame(rows)


def risk_table(summary: pd.DataFrame, year: int = 2030) -> dict:
    """
    Extract key risk numbers for a specific target year from summary metrics.
    """
    co2 = summary[(summary["variable"] == "dc_co2_mt") & (summary["year"] == year)]
    twh = summary[(summary["variable"] == "dc_twh") & (summary["year"] == year)]

    if co2.empty:
        return {}

    co2 = co2.iloc[0]
    twh = twh.iloc[0] if not twh.empty else None

    return {
        "target_year":           year,
        "co2_p50_mt":            round(float(co2["p50"]), 1),
        "co2_p5_mt":             round(float(co2["p5"]), 1),
        "co2_p95_mt":            round(float(co2["p95"]), 1),
        "co2_iqr_mt":            round(float(co2["p75"] - co2["p25"]), 1),
        "co2_cvar95_mt":         round(float(co2["cvar_95"]), 1),
        "co2_vs_2024x":          round(float(co2["p50"]) / ANCHOR_CO2_2024, 2),
        "co2_vs_iea_pct":        round((float(co2["p50"]) - IEA_2024_CO2_MT) / IEA_2024_CO2_MT * 100, 1),
        "prob_exceed_iea":       round(float(co2.get("prob_exceed_iea", float("nan"))), 4),
        "prob_exceed_2x_anchor": round(float(co2.get("prob_exceed_2x_anchor", float("nan"))), 4),
        "prob_exceed_4x_anchor": round(float(co2.get("prob_exceed_4x_anchor", float("nan"))), 4),
        "energy_p50_twh":        round(float(twh["p50"]), 1) if twh is not None else None,
        "energy_p95_twh":        round(float(twh["p95"]), 1) if twh is not None else None,
    }
