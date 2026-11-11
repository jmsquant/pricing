"""Initial-condition adapter for PDE engines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InitialCondition:
    """Product-like wrapper around a heat-equation initial condition."""

    function: Callable[[np.ndarray], np.ndarray]

    def values(self, x_grid: np.ndarray) -> np.ndarray:
        """Return f(x) on the supplied spatial grid as a one-dimensional array."""
        raw_values = np.asarray(self.function(x_grid), dtype=float)
        if raw_values.shape == ():
            return np.full_like(x_grid, float(raw_values), dtype=float)
        if raw_values.shape != x_grid.shape:
            raise ValueError("initial_condition must return an array shaped like x_grid")
        return raw_values.astype(float, copy=True)
