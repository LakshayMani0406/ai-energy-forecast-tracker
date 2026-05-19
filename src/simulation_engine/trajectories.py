"""Parquet-backed storage for simulation summary statistics."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
SIM_DIR = ROOT / "data" / "simulations"


def save_summary(df: pd.DataFrame) -> Path:
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    path = SIM_DIR / "summary.parquet"
    df.to_parquet(path, index=False, compression="snappy")
    return path


def load_summary() -> pd.DataFrame:
    path = SIM_DIR / "summary.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Simulation summary not found at {path}. "
            "Run: python src/simulation_engine/run.py"
        )
    return pd.read_parquet(path)


def summary_exists() -> bool:
    return (SIM_DIR / "summary.parquet").exists()
