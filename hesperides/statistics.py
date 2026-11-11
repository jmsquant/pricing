"""Statistical helpers used by analytical pricing formulas."""

from __future__ import annotations

from math import erf, sqrt


class StandardNormalDistribution:
    """Standard normal distribution utility."""

    def cdf(self, x: float) -> float:
        """Return Phi(x), the standard normal cumulative probability."""
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))


STANDARD_NORMAL = StandardNormalDistribution()
