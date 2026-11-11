"""Geometric Asian option contract."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeometricAsianOption:
    """European-style option on a geometric average."""

    strike: float
    is_call: bool

    def payoff(self, average: float | np.ndarray) -> float | np.ndarray:
        """Payoff as a function of the geometric average."""
        if self.is_call:
            return np.maximum(average - self.strike, 0.0)
        return np.maximum(self.strike - average, 0.0)
