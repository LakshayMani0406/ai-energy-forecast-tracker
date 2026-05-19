"""
Named scenario parameter distributions for the AI Infrastructure Futures Engine.
Canonical copy used by the persistent simulation pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ScenarioParams:
    name: str
    label: str
    color: str
    description: str
    compute_growth: Tuple[float, float]
    efficiency_gain: Tuple[float, float]
    pue: Tuple[float, float, float]
    carbon_intensity: Tuple[float, float, float]
    growth_break: Optional[Tuple[int, float]] = None


SCENARIOS: list[ScenarioParams] = [
    ScenarioParams(
        name="baseline",
        label="Baseline",
        color="#94a3b8",
        description="Current trends continue. Moderate compute growth with steady efficiency gains and gradual grid decarbonization.",
        compute_growth=(0.55, 0.10),
        efficiency_gain=(0.22, 0.05),
        pue=(1.34, 1.22, 0.02),
        carbon_intensity=(361, 310, 15),
    ),
    ScenarioParams(
        name="agi_explosion",
        label="AGI-Scale Explosion",
        color="#ef4444",
        description="AGI-level inference demand triggers aggressive compute scaling. Energy surges faster than grid can respond.",
        compute_growth=(0.65, 0.12),
        efficiency_gain=(0.12, 0.04),
        pue=(1.38, 1.36, 0.03),
        carbon_intensity=(361, 345, 20),
    ),
    ScenarioParams(
        name="blackwell_surge",
        label="Blackwell Surge",
        color="#f97316",
        description="Near-term Blackwell/GB200 deployment explosion 2025–2027, then normalization.",
        compute_growth=(0.80, 0.15),
        efficiency_gain=(0.35, 0.07),
        pue=(1.34, 1.24, 0.02),
        carbon_intensity=(361, 320, 12),
        growth_break=(2027, 0.32),
    ),
    ScenarioParams(
        name="nuclear_expansion",
        label="Nuclear Expansion",
        color="#8b5cf6",
        description="Microsoft/Amazon/Google nuclear deals deliver at scale. Carbon collapses while compute scales.",
        compute_growth=(0.65, 0.12),
        efficiency_gain=(0.25, 0.05),
        pue=(1.30, 1.18, 0.02),
        carbon_intensity=(361, 130, 25),
    ),
    ScenarioParams(
        name="efficiency_breakthrough",
        label="Efficiency Breakthrough",
        color="#22c55e",
        description="Architectural breakthrough cuts energy/FLOP by 10× by 2030. Demand flattens.",
        compute_growth=(0.60, 0.12),
        efficiency_gain=(0.52, 0.08),
        pue=(1.28, 1.15, 0.02),
        carbon_intensity=(361, 295, 12),
    ),
    ScenarioParams(
        name="regulation_crackdown",
        label="Regulation Crackdown",
        color="#f59e0b",
        description="Hard power caps and carbon mandates constrain datacenter buildout post-2026.",
        compute_growth=(0.55, 0.10),
        efficiency_gain=(0.30, 0.06),
        pue=(1.30, 1.16, 0.02),
        carbon_intensity=(361, 260, 15),
        growth_break=(2026, 0.28),
    ),
    ScenarioParams(
        name="power_shortage",
        label="Power Shortage",
        color="#dc2626",
        description="Grid capacity constraints choke AI expansion after 2027.",
        compute_growth=(0.70, 0.15),
        efficiency_gain=(0.18, 0.05),
        pue=(1.38, 1.32, 0.03),
        carbon_intensity=(361, 350, 20),
        growth_break=(2027, 0.18),
    ),
    ScenarioParams(
        name="renewable_stall",
        label="Renewable Stall",
        color="#b45309",
        description="Renewable buildout plateaus at ~40%. Grid stays fossil-heavy as AI demand surges.",
        compute_growth=(0.60, 0.12),
        efficiency_gain=(0.20, 0.05),
        pue=(1.34, 1.25, 0.02),
        carbon_intensity=(361, 355, 10),
    ),
]

SCENARIO_MAP: dict[str, ScenarioParams] = {s.name: s for s in SCENARIOS}
