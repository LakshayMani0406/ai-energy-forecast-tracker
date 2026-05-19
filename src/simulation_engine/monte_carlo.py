"""
Monte Carlo simulation engine for AI infrastructure futures.

Generates N_SIMS stochastic trajectories per scenario from 2025–2040.
All computation is vectorized over (N_SIMS, N_YEARS) arrays.

Anchor: 2024 fusion_posterior actuals
  dc_twh=187.6, ai_twh=69.4, dc_co2_mt=67.7
  pue=1.34, carbon_intensity=361 g/kWh, ai_fraction=0.37

Model equations (per step t):
  compute_index[t] = compute_index[t-1] × (1 + growth_rate[t])
  efficiency_index[t] = efficiency_index[t-1] × (1 - efficiency_gain[t])
  ai_twh[t] = ANCHOR_ai_twh × compute_index[t] × efficiency_index[t]
  dc_twh[t] = ai_twh[t] / ANCHOR_ai_fraction × (pue[t] / ANCHOR_pue)
  dc_co2_mt[t] = dc_twh[t] × carbon_intensity[t] / 1000
    [1 TWh = 1e9 kWh; CO₂ Mt = TWh × g/kWh × 1e9 / 1e12 = TWh × g/kWh / 1000]
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from simulation_engine.scenarios import SCENARIOS, ScenarioParams

log = logging.getLogger(__name__)

ANCHOR = {
    "dc_twh": 187.6,
    "ai_twh": 69.4,
    "dc_co2_mt": 67.7,
    "pue": 1.34,
    "carbon_intensity": 361.0,
    "ai_fraction": 0.37,
}

YEARS = list(range(2025, 2041))
N_YEARS = len(YEARS)
N_SIMS_DEFAULT = 10_000


def run_scenario(
    params: ScenarioParams,
    n_sims: int = N_SIMS_DEFAULT,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Vectorized Monte Carlo for one scenario.

    Returns DataFrame columns:
      scenario, sim_id, year, dc_twh, ai_twh, dc_co2_mt,
      pue, carbon_intensity, compute_index, efficiency_index
    Shape: (n_sims × N_YEARS) rows
    """
    rng = np.random.default_rng(seed)
    N = n_sims

    # (N, N_YEARS) annual growth and efficiency draws
    raw_growth = rng.normal(params.compute_growth[0], params.compute_growth[1], (N, N_YEARS))
    compute_growth = raw_growth.clip(min=-0.50)

    # Apply growth breakpoint: post-break years scale growth by multiplier
    if params.growth_break is not None:
        break_year, mult = params.growth_break
        if break_year in YEARS:
            idx = YEARS.index(break_year)
            compute_growth[:, idx:] *= mult

    efficiency_gain = rng.normal(
        params.efficiency_gain[0], params.efficiency_gain[1], (N, N_YEARS)
    ).clip(0.0, 0.85)

    # PUE: linear path + per-step noise
    pue_start, pue_target, pue_std = params.pue
    pue_path = np.linspace(pue_start, pue_target, N_YEARS)        # (N_YEARS,)
    pue = (pue_path + rng.normal(0, pue_std, (N, N_YEARS))).clip(1.05, 2.0)

    # Carbon intensity: linear path + noise
    ci_start, ci_target, ci_std = params.carbon_intensity
    ci_path = np.linspace(ci_start, ci_target, N_YEARS)
    carbon_intensity = (ci_path + rng.normal(0, ci_std, (N, N_YEARS))).clip(20.0, 900.0)

    # Cumulative indices
    compute_index = np.cumprod(1.0 + compute_growth, axis=1)       # (N, N_YEARS)
    efficiency_index = np.cumprod(1.0 - efficiency_gain, axis=1)   # (N, N_YEARS), ≤1

    # Energy and emissions
    ai_twh = ANCHOR["ai_twh"] * compute_index * efficiency_index
    dc_twh = ai_twh / ANCHOR["ai_fraction"] * (pue / ANCHOR["pue"])
    dc_co2_mt = dc_twh * carbon_intensity / 1_000.0

    # Flatten to DataFrame (row-major: sim varies fast inside each year block)
    n_total = N * N_YEARS
    return pd.DataFrame({
        "scenario":         np.full(n_total, params.name),
        "sim_id":           np.tile(np.arange(N), N_YEARS),
        "year":             np.repeat(YEARS, N),
        "dc_twh":           dc_twh.T.ravel(),
        "ai_twh":           ai_twh.T.ravel(),
        "dc_co2_mt":        dc_co2_mt.T.ravel(),
        "pue":              pue.T.ravel(),
        "carbon_intensity": carbon_intensity.T.ravel(),
        "compute_index":    compute_index.T.ravel(),
        "efficiency_index": efficiency_index.T.ravel(),
    })


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute percentile summary from full trajectories.

    Returns columns: scenario, year, variable, p5, p25, p50, p75, p95, mean, std
    """
    metrics = ["dc_twh", "ai_twh", "dc_co2_mt", "pue", "carbon_intensity", "compute_index"]
    rows: list[dict] = []

    for scenario in df["scenario"].unique():
        sc = df[df["scenario"] == scenario]
        for year in sorted(sc["year"].unique()):
            yr = sc[sc["year"] == year]
            for metric in metrics:
                v = yr[metric].values
                rows.append({
                    "scenario": scenario,
                    "year":     int(year),
                    "variable": metric,
                    "p5":       float(np.quantile(v, 0.05)),
                    "p25":      float(np.quantile(v, 0.25)),
                    "p50":      float(np.quantile(v, 0.50)),
                    "p75":      float(np.quantile(v, 0.75)),
                    "p95":      float(np.quantile(v, 0.95)),
                    "mean":     float(v.mean()),
                    "std":      float(v.std()),
                })

    return pd.DataFrame(rows)


def run_all_scenarios(n_sims: int = N_SIMS_DEFAULT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all scenarios. Returns (trajectories_df, summary_df).
    trajectories_df is large; summary_df is what the dashboard uses.
    """
    frames: list[pd.DataFrame] = []
    for i, sc in enumerate(SCENARIOS):
        log.info("Scenario %d/%d: %s", i + 1, len(SCENARIOS), sc.name)
        frames.append(run_scenario(sc, n_sims=n_sims, seed=42 + i))

    trajectories = pd.concat(frames, ignore_index=True)
    log.info("Computing summary statistics...")
    summary = compute_summary(trajectories)
    return trajectories, summary
