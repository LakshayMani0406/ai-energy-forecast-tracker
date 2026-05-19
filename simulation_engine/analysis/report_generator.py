"""
Automatic scenario report generator.

Produces a Markdown report from real simulation metrics:
  - parameter distributions used
  - key statistical findings at 2030 and 2040
  - tail risk analysis
  - year-over-year growth distribution
  - comparison to IEA benchmark and 2024 actuals
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from simulation_engine.analysis.metrics import risk_table, compute_yoy_growth


def generate(
    run_id: str,
    scenario: str,
    scenario_label: str,
    timestamp: datetime,
    n_sims: int,
    seed: int,
    params: dict,
    runtime_seconds: float,
    summary: pd.DataFrame,
    trajectories: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Write scenario_report.md and return its path."""

    r2030 = risk_table(summary, year=2030)
    r2040 = risk_table(summary, year=2040)
    yoy = compute_yoy_growth(trajectories, "dc_co2_mt")

    lines: list[str] = [
        f"# Scenario Report: {scenario_label}",
        "",
        f"**Run ID:** `{run_id}`  ",
        f"**Generated:** {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        f"**Scenario:** `{scenario}`  ",
        f"**Simulations:** {n_sims:,} trajectories × {summary['year'].nunique()} years  ",
        f"**Seed:** {seed}  ",
        f"**Runtime:** {runtime_seconds:.2f}s  ",
        "",
        "---",
        "",
        "## Parameter Distributions",
        "",
        f"| Parameter | Mean | Std |",
        f"|-----------|------|-----|",
        f"| Annual compute growth | {params.get('compute_growth', ['-','-'])[0]:.0%} | ±{params.get('compute_growth', ['-','-'])[1]:.0%} |",
        f"| Annual efficiency gain (energy/FLOP) | {params.get('efficiency_gain', ['-','-'])[0]:.0%} | ±{params.get('efficiency_gain', ['-','-'])[1]:.0%} |",
        f"| PUE 2025 start | {params.get('pue', ['-','-','-'])[0]} | ±{params.get('pue', ['-','-','-'])[2]} |",
        f"| PUE 2030 target | {params.get('pue', ['-','-','-'])[1]} | — |",
        f"| Grid carbon intensity 2025 (g/kWh) | {params.get('carbon_intensity', ['-','-','-'])[0]} | ±{params.get('carbon_intensity', ['-','-','-'])[2]} |",
        f"| Grid carbon intensity 2030 target (g/kWh) | {params.get('carbon_intensity', ['-','-','-'])[1]} | — |",
    ]

    if params.get("growth_break"):
        yr, mult = params["growth_break"]
        lines.append(f"| Growth break | After {yr}: ×{mult} of baseline rate | — |")

    lines += [
        "",
        "---",
        "",
        "## Key Findings — 2030",
        "",
    ]

    if r2030:
        lines += [
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| CO₂ median | **{r2030['co2_p50_mt']} Mt/yr** |",
            f"| CO₂ 5th–95th pct | {r2030['co2_p5_mt']} – {r2030['co2_p95_mt']} Mt/yr |",
            f"| CO₂ interquartile range | {r2030['co2_iqr_mt']} Mt/yr |",
            f"| CO₂ CVaR(95%) | {r2030['co2_cvar95_mt']} Mt/yr (expected worst-5% outcome) |",
            f"| CO₂ vs 2024 actual | {r2030['co2_vs_2024x']}× 2024 levels |",
            f"| CO₂ vs IEA 2024 benchmark (105 Mt) | {r2030['co2_vs_iea_pct']:+.1f}% |",
            f"| P(CO₂ > IEA 105 Mt) | {r2030['prob_exceed_iea']:.1%} |",
            f"| P(CO₂ > 2× 2024) | {r2030['prob_exceed_2x_anchor']:.1%} |",
            f"| P(CO₂ > 4× 2024) | {r2030['prob_exceed_4x_anchor']:.1%} |",
            f"| Total DC energy median | {r2030['energy_p50_twh']} TWh/yr |",
            f"| Total DC energy CVaR(95%) | {r2030['energy_p95_twh']} TWh/yr |",
        ]

    lines += ["", "## Key Findings — 2040", ""]

    if r2040:
        lines += [
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| CO₂ median | **{r2040['co2_p50_mt']} Mt/yr** |",
            f"| CO₂ 5th–95th pct | {r2040['co2_p5_mt']} – {r2040['co2_p95_mt']} Mt/yr |",
            f"| CO₂ CVaR(95%) | {r2040['co2_cvar95_mt']} Mt/yr |",
            f"| CO₂ vs 2024 actual | {r2040['co2_vs_2024x']}× 2024 levels |",
            f"| P(CO₂ > IEA 105 Mt) | {r2040['prob_exceed_iea']:.1%} |",
        ]

    if not yoy.empty:
        lines += [
            "",
            "## Year-over-Year CO₂ Growth Distribution",
            "",
            "| Year | p5 | p25 | Median | p75 | p95 |",
            "|------|-----|-----|--------|-----|-----|",
        ]
        for _, row in yoy.iterrows():
            lines.append(
                f"| {int(row['year'])} "
                f"| {row['p5_growth']:+.1%} "
                f"| {row['p25_growth']:+.1%} "
                f"| {row['p50_growth']:+.1%} "
                f"| {row['p75_growth']:+.1%} "
                f"| {row['p95_growth']:+.1%} |"
            )

    lines += [
        "",
        "---",
        "",
        "## Reproducibility",
        "",
        f"To reproduce this exact run:",
        "```python",
        f"from simulation_engine.orchestration.runner import run_scenario_full",
        f"from simulation_engine.scenarios import SCENARIO_MAP",
        f"run_scenario_full(SCENARIO_MAP['{scenario}'], n_sims={n_sims}, seed={seed})",
        "```",
        "",
        f"Cache hash: `{_extract_hash(output_dir)}`",
        "",
        "_Report generated automatically from real simulation outputs._",
    ]

    report_path = output_dir / "scenario_report.md"
    report_path.write_text("\n".join(lines))
    return report_path


def _extract_hash(output_dir: Path) -> str:
    return output_dir.name
