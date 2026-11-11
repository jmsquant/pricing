"""Contract abstractions used by pricing engines."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class OptionContract(Protocol):
    """Minimal option contract interface required by pricing engines."""

    strike: float
    is_call: bool

    def payoff(self, underlying: float | np.ndarray) -> float | np.ndarray:
        """Return the contract payoff for a scalar or vector underlying."""
