"""
Hash-based simulation cache.

Cache key = SHA256 of (scenario_name + params + n_sims + seed).
If the key exists in the cache directory, the simulation is skipped and
stored results are returned instead.

This means identical re-runs are instant (reads parquet) and any change
to parameters, seed, or simulation count invalidates the cache automatically.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = ROOT / "data" / "simulation_outputs" / "cache"


def compute_hash(scenario_name: str, params: dict, n_sims: int, seed: int) -> str:
    payload = json.dumps(
        {"scenario": scenario_name, "params": params, "n_sims": n_sims, "seed": seed},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def cache_path(h: str) -> Path:
    return CACHE_DIR / h


def is_cached(h: str) -> bool:
    p = cache_path(h)
    return (
        (p / "summary_metrics.parquet").exists()
        and (p / "trajectories.parquet").exists()
        and (p / "simulation_manifest.json").exists()
    )


def read_cached_summary(h: str) -> pd.DataFrame:
    return pd.read_parquet(cache_path(h) / "summary_metrics.parquet")


def read_cached_trajectories(h: str) -> pd.DataFrame:
    return pd.read_parquet(cache_path(h) / "trajectories.parquet")
