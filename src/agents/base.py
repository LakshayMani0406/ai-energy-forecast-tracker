"""Abstract base class for AI infrastructure agents."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class AgentOutput:
    agent_name: str
    variable: str
    unit: str
    years: list[int]
    baseline: list[float]
    lo: list[float]
    hi: list[float]
    assumptions: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "agent":    self.agent_name,
            "variable": self.variable,
            "year":     self.years,
            "baseline": self.baseline,
            "lo":       self.lo,
            "hi":       self.hi,
        })


class BaseAgent(ABC):
    name: str
    description: str

    @abstractmethod
    def run(self, **kwargs) -> AgentOutput:
        """Execute agent analysis."""
        ...
