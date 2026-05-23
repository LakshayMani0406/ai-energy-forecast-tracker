"""
End-to-end scenario pipeline.

run_scenario_full():
  1. Compute cache hash for (scenario, params, n_sims, seed)
  2. Return cached run if hash already exists on disk
  3. Register run in experiment registry (status=running)
  4. Run vectorized Monte Carlo → (trajectories, param_draws)
  5. Compute statistical metrics from real trajectories
  6. Write all artifacts to disk (parquet + manifest + report)
  7. Update experiment registry (status=completed, key metrics)
  8. Clear any checkpoint file

Usage:
    from simulation_engine.orchestration.runner import run_scenario_full
    from simulation_engine.scenarios import SCENARIO_MAP
    result = run_scenario_full(SCENARIO_MAP['agi_explosion'])
"""
from __future__ import annotations

import dataclasses
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from simulation_engine.monte_carlo import run_scenario_with_draws
from simulation_engine.scenarios import ScenarioParams
from simulation_engine.storage import cache as cache_mod
from simulation_engine.storage import trajectory_store
from simulation_engine.storage.experiment import ExperimentRun, register, update_status
from simulation_engine.analysis import metrics as metrics_mod
from simulation_engine.analysis import report_generator
from simulation_engine.orchestration.checkpoint import clear_checkpoint

log = logging.getLogger(__name__)


def _params_to_dict(params: ScenarioParams) -> dict:
    return dataclasses.asdict(params)


def run_scenario_full(
    params: ScenarioParams,
    n_sims: int = 10_000,
    seed: int = 42,
) -> dict:
    """
    Full persistent pipeline for one scenario.

    Returns dict:
      run_id, cache_hash, output_dir (Path), cached (bool),
      runtime_seconds, p50_co2_2030, p95_co2_2030
    """
    params_dict = _params_to_dict(params)
    h = cache_mod.compute_hash(params.name, params_dict, n_sims, seed)
    output_dir = cache_mod.cache_path(h)

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if cache_mod.is_cached(h):
        log.info("Cache hit: %s  hash=%s", params.name, h)
        summary = cache_mod.read_cached_summary(h)
        r2030 = metrics_mod.risk_table(summary, year=2030)

        # Backfill sensitivity if not yet computed for this run
        sens_path = output_dir / "sensitivity_metrics.parquet"
        if not sens_path.exists():
            log.info("Backfilling sensitivity for cached run %s", h)
            _traj = cache_mod.read_cached_trajectories(h)
            sens = metrics_mod.compute_sensitivity(_traj)
            sens.to_parquet(sens_path, index=False, compression="snappy")

        return {
            "run_id":           h,
            "cache_hash":       h,
            "output_dir":       output_dir,
            "cached":           True,
            "runtime_seconds":  0.0,
            "p50_co2_2030":     r2030.get("co2_p50_mt"),
            "p95_co2_2030":     r2030.get("co2_cvar95_mt"),
        }

    # ── Fresh run ─────────────────────────────────────────────────────────────
    timestamp = datetime.now(timezone.utc)
    run_id = f"{params.name}_{timestamp.strftime('%Y%m%dT%H%M%S')}_{h[:8]}"

    run = ExperimentRun(
        run_id=run_id,
        scenario=params.name,
        scenario_label=params.label,
        timestamp=timestamp,
        n_sims=n_sims,
        seed=seed,
        cache_hash=h,
        output_dir=output_dir,
        params=params_dict,
        status="running",
    )
    register(run)
    log.info("Run registered: %s", run_id)

    try:
        t0 = time.perf_counter()

        # ── Monte Carlo ───────────────────────────────────────────────────────
        log.info("Running %d trajectories × %d years  scenario=%s",
                 n_sims, 16, params.name)
        trajectories, param_draws = run_scenario_with_draws(params, n_sims, seed)
        log.info("Monte Carlo complete: %d rows", len(trajectories))

        # ── Metrics ───────────────────────────────────────────────────────────
        log.info("Computing percentiles and tail risk...")
        summary = metrics_mod.compute_percentiles(trajectories)
        log.info("Computing sensitivity...")
        sensitivity = metrics_mod.compute_sensitivity(trajectories)

        runtime = time.perf_counter() - t0

        # ── Persist artifacts ─────────────────────────────────────────────────
        log.info("Writing artifacts to %s", output_dir)
        trajectory_store.save_run(
            output_dir=output_dir,
            run_id=run_id,
            scenario=params.name,
            scenario_label=params.label,
            timestamp=timestamp,
            n_sims=n_sims,
            seed=seed,
            params=params_dict,
            runtime_seconds=runtime,
            cache_hash=h,
            trajectories=trajectories,
            param_draws=param_draws,
            summary=summary,
        )

        # ── Sensitivity ───────────────────────────────────────────────────────
        sensitivity.to_parquet(output_dir / "sensitivity_metrics.parquet",
                               index=False, compression="snappy")

        # ── Scenario report ───────────────────────────────────────────────────
        report_path = report_generator.generate(
            run_id=run_id,
            scenario=params.name,
            scenario_label=params.label,
            timestamp=timestamp,
            n_sims=n_sims,
            seed=seed,
            params=params_dict,
            runtime_seconds=runtime,
            summary=summary,
            trajectories=trajectories,
            output_dir=output_dir,
        )
        log.info("Report written → %s", report_path)

        # ── Update registry ───────────────────────────────────────────────────
        r2030 = metrics_mod.risk_table(summary, year=2030)
        run.runtime_seconds = runtime
        run.status = "completed"
        run.n_trajectories = len(trajectories)
        run.p50_co2_2030 = r2030.get("co2_p50_mt")
        run.p95_co2_2030 = r2030.get("co2_cvar95_mt")
        register(run)

        clear_checkpoint(output_dir)

        log.info(
            "DONE  run_id=%s  runtime=%.1fs  p50_co2_2030=%.1f Mt",
            run_id, runtime, r2030.get("co2_p50_mt", 0),
        )

        return {
            "run_id":           run_id,
            "cache_hash":       h,
            "output_dir":       output_dir,
            "cached":           False,
            "runtime_seconds":  runtime,
            "p50_co2_2030":     r2030.get("co2_p50_mt"),
            "p95_co2_2030":     r2030.get("co2_cvar95_mt"),
        }

    except Exception as exc:
        update_status(run_id, "failed")
        log.error("Run failed: %s  run_id=%s", exc, run_id)
        raise
