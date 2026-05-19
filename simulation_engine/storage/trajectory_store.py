"""
Trajectory store — writes all simulation artifacts to a run directory.

For each run, writes:
  {output_dir}/
    simulation_manifest.json    — config, params, seeds, runtime, checksums
    trajectories.parquet        — all n_sims × n_years rows
    parameter_draws.parquet     — raw stochastic draws (reproducibility)
    summary_metrics.parquet     — percentiles + tail risk from actual trajectories
    scenario_report.md          — auto-generated narrative (written by report_generator)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def save_run(
    output_dir: Path,
    run_id: str,
    scenario: str,
    scenario_label: str,
    timestamp: datetime,
    n_sims: int,
    seed: int,
    params: dict[str, Any],
    runtime_seconds: float,
    cache_hash: str,
    trajectories: pd.DataFrame,
    param_draws: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Trajectories ───────────────────────────────────────────────────────
    traj_path = output_dir / "trajectories.parquet"
    trajectories.to_parquet(traj_path, index=False, compression="snappy")

    # ── 2. Parameter draws ────────────────────────────────────────────────────
    draws_path = output_dir / "parameter_draws.parquet"
    param_draws.to_parquet(draws_path, index=False, compression="snappy")

    # ── 3. Summary metrics ────────────────────────────────────────────────────
    summary_path = output_dir / "summary_metrics.parquet"
    summary.to_parquet(summary_path, index=False, compression="snappy")

    # ── 4. Manifest ───────────────────────────────────────────────────────────
    manifest = {
        "run_id":           run_id,
        "scenario":         scenario,
        "scenario_label":   scenario_label,
        "timestamp":        timestamp.isoformat(),
        "n_sims":           n_sims,
        "n_years":          int(trajectories["year"].nunique()),
        "years":            sorted(trajectories["year"].unique().tolist()),
        "seed":             seed,
        "params":           params,
        "anchor_year":      2024,
        "anchor_values": {
            "dc_twh":    187.6,
            "ai_twh":    69.4,
            "dc_co2_mt": 67.7,
            "pue":       1.34,
            "carbon_intensity_g_kwh": 361.0,
            "ai_fraction": 0.37,
        },
        "runtime_seconds":  round(runtime_seconds, 3),
        "cache_hash":       cache_hash,
        "output_files": [
            "trajectories.parquet",
            "parameter_draws.parquet",
            "summary_metrics.parquet",
            "scenario_report.md",
        ],
        "trajectory_rows":  len(trajectories),
        "trajectory_cols":  list(trajectories.columns),
        "summary_rows":     len(summary),
    }
    (output_dir / "simulation_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )


def load_run(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "simulation_manifest.json").read_text())
    return {
        "manifest":    manifest,
        "summary":     pd.read_parquet(run_dir / "summary_metrics.parquet"),
        "trajectories": pd.read_parquet(run_dir / "trajectories.parquet"),
        "param_draws": pd.read_parquet(run_dir / "parameter_draws.parquet"),
    }


def load_manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "simulation_manifest.json").read_text())
