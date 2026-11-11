"""Separate terminal and path simulation dependencies."""
from __future__ import annotations
from typing import Protocol
import numpy as np
from hesperides.models.protocols import LognormalSimulationDynamics

class TerminalSimulation(Protocol):

    def simulate_terminal_spots(
        self,
        model: LognormalSimulationDynamics,
        spot: float,
        maturity: float,
        n_paths: int,
        seed: int | None,
    ) -> np.ndarray:
        ...

class PathSimulation(Protocol):

    def simulate_log_paths(
        self,
        model: LognormalSimulationDynamics,
        spot: float,
        maturity: float,
        n_paths: int,
        n_steps: int,
        seed: int | None,
    ) -> np.ndarray:
        ...
