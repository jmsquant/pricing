"""Simulation services for stochastic models."""

from __future__ import annotations

from math import log

import numpy as np

from hesperides.models.protocols import LognormalSimulationDynamics


class BlackScholesPathSimulator:
    """Vectorised path simulator for Black-Scholes dynamics."""

    def simulate_terminal_spots(
        self,
        model: LognormalSimulationDynamics,
        spot: float,
        maturity: float,
        n_paths: int,
        seed: int | None,
    ) -> np.ndarray:
        """Simulate terminal spots at maturity."""
        rng = np.random.default_rng(seed)
        z = rng.normal(size=n_paths)
        return spot * np.exp(
            model.integrated_log_drift(maturity)
            + model.volatility * np.sqrt(maturity) * z
        )

    def simulate_log_paths(
        self,
        model: LognormalSimulationDynamics,
        spot: float,
        maturity: float,
        n_paths: int,
        n_steps: int,
        seed: int | None,
    ) -> np.ndarray:
        """Simulate log paths on a uniform grid excluding t=0."""
        rng = np.random.default_rng(seed)
        dt = maturity / n_steps
        z = rng.normal(size=(n_paths, n_steps))
        times = np.arange(1, n_steps + 1) * dt
        drift = np.array([model.integrated_log_drift(float(t)) for t in times])
        return log(spot) + drift + np.cumsum(model.volatility * np.sqrt(dt) * z, axis=1)


BLACK_SCHOLES_PATH_SIMULATOR = BlackScholesPathSimulator()
