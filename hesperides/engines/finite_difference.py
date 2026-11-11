"""Finite-difference engines for one-dimensional parabolic PDEs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hesperides.contracts.initial_condition import InitialCondition
from hesperides.models.heat_equation import HeatEquationModel


@dataclass(frozen=True)
class HeatEquationFiniteDifferenceEngine:
    """FTCS/BTCS finite-difference engine for the heat equation."""

    def solve(
        self,
        initial_condition: InitialCondition,
        model: HeatEquationModel,
        scheme: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the spatial grid and terminal heat-equation solution."""
        x_grid = model.spatial_grid()
        left_values, right_values = model.boundary_values()
        values = initial_condition.values(x_grid)
        self._apply_boundary(values, model, left_values[0], right_values[0])

        if scheme == "explicit":
            terminal_values = self._solve_explicit(
                values=values,
                model=model,
                left_values=left_values,
                right_values=right_values,
            )
        else:
            terminal_values = self._solve_implicit(
                values=values,
                model=model,
                left_values=left_values,
                right_values=right_values,
            )

        return x_grid, terminal_values

    def _solve_explicit(
        self,
        values: np.ndarray,
        model: HeatEquationModel,
        left_values: np.ndarray,
        right_values: np.ndarray,
    ) -> np.ndarray:
        ratio = model.stability_ratio
        current = values.astype(float, copy=True)

        for step in range(model.n_t):
            updated = current.copy()
            updated[1:-1] = (
                ratio * current[:-2]
                + (1.0 - 2.0 * ratio) * current[1:-1]
                + ratio * current[2:]
            )
            self._apply_boundary(
                updated,
                model,
                left_values[step + 1],
                right_values[step + 1],
            )
            current = updated

        return current

    def _solve_implicit(
        self,
        values: np.ndarray,
        model: HeatEquationModel,
        left_values: np.ndarray,
        right_values: np.ndarray,
    ) -> np.ndarray:
        if model.boundary_type == "neumann":
            return self._solve_implicit_neumann(
                values=values,
                model=model,
                left_values=left_values,
                right_values=right_values,
            )

        interior_size = model.n_x - 1
        current = values.astype(float, copy=True)
        if interior_size <= 0:
            for step in range(model.n_t):
                self._apply_boundary(
                    current,
                    model,
                    left_values[step + 1],
                    right_values[step + 1],
                )
            return current

        ratio = model.stability_ratio
        solver = self._inverse_tridiagonal_matrix(interior_size, ratio)

        for step in range(model.n_t):
            rhs = current[1:-1].copy()
            rhs[0] += ratio * left_values[step + 1]
            rhs[-1] += ratio * right_values[step + 1]
            current[1:-1] = solver @ rhs
            self._apply_boundary(
                current,
                model,
                left_values[step + 1],
                right_values[step + 1],
            )

        return current

    def _solve_implicit_neumann(
        self,
        values: np.ndarray,
        model: HeatEquationModel,
        left_values: np.ndarray,
        right_values: np.ndarray,
    ) -> np.ndarray:
        if model.n_x == 1:
            return self._solve_explicit(
                values=values,
                model=model,
                left_values=left_values,
                right_values=right_values,
            )

        ratio = model.stability_ratio
        dx = model.dx
        size = model.n_x + 1
        system = np.eye(size, dtype=float) * (1.0 + 2.0 * ratio)
        diagonal = np.arange(1, size - 1)
        system[diagonal, diagonal - 1] = -ratio
        system[diagonal, diagonal + 1] = -ratio
        system[0, :] = 0.0
        system[0, 0] = -1.0
        system[0, 1] = 1.0
        system[-1, :] = 0.0
        system[-1, -2] = -1.0
        system[-1, -1] = 1.0
        solver = np.linalg.inv(system)

        current = values.astype(float, copy=True)
        for step in range(model.n_t):
            rhs = current.copy()
            rhs[0] = dx * left_values[step + 1]
            rhs[-1] = dx * right_values[step + 1]
            current = solver @ rhs

        return current

    def _inverse_tridiagonal_matrix(
        self,
        interior_size: int,
        ratio: float,
    ) -> np.ndarray:
        diagonal = np.full(interior_size, 1.0 + 2.0 * ratio, dtype=float)
        off_diagonal = np.full(interior_size - 1, -ratio, dtype=float)
        matrix = np.diag(diagonal)
        if interior_size > 1:
            matrix += np.diag(off_diagonal, k=-1)
            matrix += np.diag(off_diagonal, k=1)
        return np.linalg.inv(matrix)

    def _apply_boundary(
        self,
        values: np.ndarray,
        model: HeatEquationModel,
        left_value: float,
        right_value: float,
    ) -> None:
        if model.boundary_type == "dirichlet":
            values[0] = left_value
            values[-1] = right_value
            return

        if values.size == 1:
            values[0] = 0.0
            return

        values[0] = values[1] - model.dx * left_value
        values[-1] = values[-2] + model.dx * right_value
