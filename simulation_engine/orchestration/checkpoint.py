"""
Checkpoint manager — saves and loads in-progress simulation state.

Enables resume-on-failure for long runs by persisting completed
trajectory batches to disk incrementally.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "_checkpoint.json"


def save_checkpoint(
    output_dir: Path,
    run_id: str,
    completed_sims: int,
    total_sims: int,
    partial_trajectories: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_trajectories.to_parquet(
        output_dir / "_partial_trajectories.parquet", index=False, compression="snappy"
    )
    checkpoint_path(output_dir).write_text(json.dumps({
        "run_id":         run_id,
        "completed_sims": completed_sims,
        "total_sims":     total_sims,
    }))


def load_checkpoint(output_dir: Path) -> tuple[dict, pd.DataFrame] | None:
    cp = checkpoint_path(output_dir)
    partial = output_dir / "_partial_trajectories.parquet"
    if not cp.exists() or not partial.exists():
        return None
    meta = json.loads(cp.read_text())
    return meta, pd.read_parquet(partial)


def clear_checkpoint(output_dir: Path) -> None:
    for name in ("_checkpoint.json", "_partial_trajectories.parquet"):
        p = output_dir / name
        if p.exists():
            p.unlink()
