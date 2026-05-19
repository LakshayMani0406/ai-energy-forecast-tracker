"""
CLI entry point for the persistent simulation pipeline.

Usage:
    python run_pipeline.py --scenario agi_explosion
    python run_pipeline.py --scenario baseline --sims 10000 --seed 42
    python run_pipeline.py --all
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AI energy scenario simulation")
    parser.add_argument("--scenario", default="agi_explosion",
                        help="Scenario name (default: agi_explosion)")
    parser.add_argument("--sims", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true", help="Run all 8 scenarios")
    args = parser.parse_args()

    from simulation_engine.orchestration.runner import run_scenario_full
    from simulation_engine.scenarios import SCENARIO_MAP, SCENARIOS

    targets = SCENARIOS if args.all else [SCENARIO_MAP[args.scenario]]

    for sc in targets:
        result = run_scenario_full(sc, n_sims=args.sims, seed=args.seed)
        cached_tag = " [CACHED]" if result["cached"] else ""
        log.info(
            "%-25s  p50_CO2_2030=%s Mt  runtime=%.1fs%s",
            sc.name,
            result["p50_co2_2030"],
            result["runtime_seconds"],
            cached_tag,
        )
        log.info("  Output dir: %s", result["output_dir"])


if __name__ == "__main__":
    main()
