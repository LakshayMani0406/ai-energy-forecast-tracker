"""
Runner: generate Monte Carlo simulation summary and seed forecast memory.

Usage:
    python src/simulation_engine/run.py [--sims N]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from simulation_engine.monte_carlo import run_all_scenarios, N_SIMS_DEFAULT
from simulation_engine.trajectories import save_summary
from forecast_memory.seed import seed_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main(n_sims: int = N_SIMS_DEFAULT) -> None:
    log.info("Running %d Monte Carlo simulations × 8 scenarios...", n_sims)
    _, summary = run_all_scenarios(n_sims=n_sims)
    path = save_summary(summary)
    log.info("Summary saved → %s  (%d rows)", path, len(summary))

    log.info("Seeding forecast memory...")
    seed_all()
    log.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=N_SIMS_DEFAULT)
    args = parser.parse_args()
    main(args.sims)
