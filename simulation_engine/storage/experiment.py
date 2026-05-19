"""
Experiment registry — persistent DuckDB record of every simulation run.

Every call to run_scenario_full() registers a row here, enabling:
  - run history and lineage
  - reproducibility queries (find run by hash)
  - status tracking (running / completed / failed)
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb

ROOT = Path(__file__).resolve().parent.parent.parent
REGISTRY_PATH = ROOT / "data" / "simulation_outputs" / "experiment_registry.duckdb"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id          VARCHAR PRIMARY KEY,
    scenario        VARCHAR NOT NULL,
    scenario_label  VARCHAR,
    timestamp       TIMESTAMP NOT NULL,
    n_sims          INTEGER NOT NULL,
    seed            INTEGER NOT NULL,
    runtime_seconds DOUBLE,
    cache_hash      VARCHAR NOT NULL,
    output_dir      VARCHAR NOT NULL,
    params_json     JSON,
    status          VARCHAR DEFAULT 'completed',
    n_trajectories  INTEGER,
    p50_co2_2030    DOUBLE,
    p95_co2_2030    DOUBLE
);
"""


@dataclass
class ExperimentRun:
    run_id:          str
    scenario:        str
    scenario_label:  str
    timestamp:       datetime
    n_sims:          int
    seed:            int
    cache_hash:      str
    output_dir:      Path
    params:          dict
    runtime_seconds: float = 0.0
    status:          str   = "running"
    n_trajectories:  int   = 0
    p50_co2_2030:    Optional[float] = None
    p95_co2_2030:    Optional[float] = None

    def to_registry_row(self) -> dict:
        return {
            "run_id":          self.run_id,
            "scenario":        self.scenario,
            "scenario_label":  self.scenario_label,
            "timestamp":       self.timestamp,
            "n_sims":          self.n_sims,
            "seed":            self.seed,
            "runtime_seconds": self.runtime_seconds,
            "cache_hash":      self.cache_hash,
            "output_dir":      str(self.output_dir),
            "params_json":     json.dumps(self.params),
            "status":          self.status,
            "n_trajectories":  self.n_trajectories,
            "p50_co2_2030":    self.p50_co2_2030,
            "p95_co2_2030":    self.p95_co2_2030,
        }


def _conn() -> duckdb.DuckDBPyConnection:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(REGISTRY_PATH))
    conn.execute(_SCHEMA)
    return conn


def register(run: ExperimentRun) -> None:
    row = run.to_registry_row()
    with _conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO experiment_runs VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, [
            row["run_id"], row["scenario"], row["scenario_label"],
            row["timestamp"], row["n_sims"], row["seed"],
            row["runtime_seconds"], row["cache_hash"], row["output_dir"],
            row["params_json"], row["status"], row["n_trajectories"],
            row["p50_co2_2030"], row["p95_co2_2030"],
        ])


def update_status(run_id: str, status: str, runtime_seconds: float | None = None) -> None:
    with _conn() as conn:
        if runtime_seconds is not None:
            conn.execute(
                "UPDATE experiment_runs SET status=?, runtime_seconds=? WHERE run_id=?",
                [status, runtime_seconds, run_id]
            )
        else:
            conn.execute(
                "UPDATE experiment_runs SET status=? WHERE run_id=?",
                [status, run_id]
            )


def load_history(scenario: str | None = None, limit: int = 50):
    import pandas as pd
    with _conn() as conn:
        if scenario:
            return conn.execute(
                "SELECT * FROM experiment_runs WHERE scenario=? ORDER BY timestamp DESC LIMIT ?",
                [scenario, limit]
            ).df()
        return conn.execute(
            "SELECT * FROM experiment_runs ORDER BY timestamp DESC LIMIT ?",
            [limit]
        ).df()


def find_by_hash(cache_hash: str):
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM experiment_runs WHERE cache_hash=? AND status='completed' ORDER BY timestamp DESC LIMIT 1",
            [cache_hash]
        ).df()
    return rows.iloc[0].to_dict() if len(rows) > 0 else None
