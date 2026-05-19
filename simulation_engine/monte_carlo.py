"""
Monte Carlo simulation engine — persistent pipeline version.

Returns both trajectory DataFrame AND parameter draws DataFrame for
full reproducibility storage.

Anchor (2024 fusion_posterior actuals):
  dc_twh=187.6, ai_twh=69.4, dc_co2_mt=67.7
  pue=1.34, carbon_intensity=361 g/kWh, ai_fraction=0.37

Model per step t:
  compute_index[t]   = compute_index[t-1] × (1 + growth_rate[t])
  efficiency_index[t]= efficiency_index[t-1] × (1 - efficiency_gain[t])
  ai_twh[t]          = ANCHOR_ai_twh × compute_index[t] × efficiency_index[t]
  dc_twh[t]          = ai_twh[t] / ANCHOR_ai_fraction × (pue[t] / ANCHOR_pue)
  dc_co2_mt[t]       = dc_twh[t] × carbon_intensity[t] / 1000
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from simulation_engine.scenarios import ScenarioParams

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
    """Trajectories only (backward-compatible)."""
    trajectories, _ = run_scenario_with_draws(params, n_sims, seed)
    return trajectories


def run_scenario_with_draws(
    params: ScenarioParams,
    n_sims: int = N_SIMS_DEFAULT,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full simulation returning (trajectories, param_draws).

    param_draws columns: sim_id, year, compute_growth_draw, efficiency_gain_draw,
                          pue_draw, carbon_intensity_draw
    Shape of both: (n_sims × N_YEARS) rows
    """
    rng = np.random.default_rng(seed)
    N = n_sims

    # ── Stochastic draws ──────────────────────────────────────────────────────
    raw_growth = rng.normal(params.compute_growth[0], params.compute_growth[1], (N, N_YEARS))
    compute_growth = raw_growth.clip(min=-0.50)

    if params.growth_break is not None:
        break_year, mult = params.growth_break
        if break_year in YEARS:
            idx = YEARS.index(break_year)
            compute_growth[:, idx:] *= mult

    efficiency_gain = rng.normal(
        params.efficiency_gain[0], params.efficiency_gain[1], (N, N_YEARS)
    ).clip(0.0, 0.85)

    pue_start, pue_target, pue_std = params.pue
    pue_path = np.linspace(pue_start, pue_target, N_YEARS)
    pue = (pue_path + rng.normal(0, pue_std, (N, N_YEARS))).clip(1.05, 2.0)

    ci_start, ci_target, ci_std = params.carbon_intensity
    ci_path = np.linspace(ci_start, ci_target, N_YEARS)
    carbon_intensity = (ci_path + rng.normal(0, ci_std, (N, N_YEARS))).clip(20.0, 900.0)

    # ── Derived physics ───────────────────────────────────────────────────────
    compute_index = np.cumprod(1.0 + compute_growth, axis=1)
    efficiency_index = np.cumprod(1.0 - efficiency_gain, axis=1)

    ai_twh = ANCHOR["ai_twh"] * compute_index * efficiency_index
    dc_twh = ai_twh / ANCHOR["ai_fraction"] * (pue / ANCHOR["pue"])
    dc_co2_mt = dc_twh * carbon_intensity / 1_000.0

    # ── Flatten ───────────────────────────────────────────────────────────────
    n_total = N * N_YEARS
    sim_ids = np.tile(np.arange(N), N_YEARS)
    years_col = np.repeat(YEARS, N)

    trajectories = pd.DataFrame({
        "scenario":         [params.name] * n_total,
        "sim_id":           sim_ids,
        "year":             years_col,
        "dc_twh":           dc_twh.T.ravel(),
        "ai_twh":           ai_twh.T.ravel(),
        "dc_co2_mt":        dc_co2_mt.T.ravel(),
        "pue":              pue.T.ravel(),
        "carbon_intensity": carbon_intensity.T.ravel(),
        "compute_index":    compute_index.T.ravel(),
        "efficiency_index": efficiency_index.T.ravel(),
    })

    param_draws = pd.DataFrame({
        "scenario":              [params.name] * n_total,
        "sim_id":                sim_ids,
        "year":                  years_col,
        "compute_growth_draw":   compute_growth.T.ravel(),
        "efficiency_gain_draw":  efficiency_gain.T.ravel(),
        "pue_draw":              pue.T.ravel(),
        "carbon_intensity_draw": carbon_intensity.T.ravel(),
    })

    return trajectories, param_draws
