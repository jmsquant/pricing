"""European option contract."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EuropeanOption:
    """Vanilla European option defined by strike and type (call/put)."""

    strike: float
    is_call: bool

    def payoff(self, spot: float | np.ndarray) -> float | np.ndarray:
        """Terminal payoff at expiry.

        Works with both scalar spots and NumPy arrays.
        """
        if self.is_call:
            return np.maximum(spot - self.strike, 0.0)
        return np.maximum(self.strike - spot, 0.0)
