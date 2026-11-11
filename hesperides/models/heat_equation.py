"""Heat-equation model and boundary data."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


BoundaryInput = float | Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class HeatEquationModel:
    """One-dimensional heat equation on a uniform rectangular grid."""

    kappa: float
    domain_size: float
    maturity: float
    n_x: int
    n_t: int
    left_boundary: BoundaryInput = 0.0
    right_boundary: BoundaryInput = 0.0
    boundary_type: str = "dirichlet"

    @property
    def dx(self) -> float:
        """Spatial grid spacing."""
        return self.domain_size / self.n_x

    @property
    def dt(self) -> float:
        """Time grid spacing."""
        return self.maturity / self.n_t

    @property
    def stability_ratio(self) -> float:
        """Dimensionless heat-equation ratio kappa * dt / dx^2."""
        return self.kappa * self.dt / (self.dx * self.dx)

    def spatial_grid(self) -> np.ndarray:
        """Uniform spatial grid on [0, M]."""
        return np.linspace(0.0, self.domain_size, self.n_x + 1, dtype=float)

    def time_grid(self) -> np.ndarray:
        """Uniform time grid on [0, T]."""
        return np.linspace(0.0, self.maturity, self.n_t + 1, dtype=float)

    def boundary_values(self) -> tuple[np.ndarray, np.ndarray]:
        """Return left and right boundary values on the time grid."""
        time_grid = self.time_grid()
        return (
            self._boundary_to_array(self.left_boundary, time_grid),
            self._boundary_to_array(self.right_boundary, time_grid),
        )

    def _boundary_to_array(
        self,
        boundary: BoundaryInput,
        time_grid: np.ndarray,
    ) -> np.ndarray:
        if callable(boundary):
            try:
                values = np.asarray(boundary(time_grid), dtype=float)
            except Exception:
                values = np.asarray([boundary(t) for t in time_grid], dtype=float)
        else:
            values = np.asarray(boundary, dtype=float)

        if values.shape == ():
            return np.full_like(time_grid, float(values), dtype=float)
        if values.shape != time_grid.shape:
            raise ValueError("boundary values must be scalar or shaped like the time grid")
        return values.astype(float, copy=True)
