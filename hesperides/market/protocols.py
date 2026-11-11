"""Market-data abstractions used by pricing engines."""

from __future__ import annotations

from typing import Protocol


class DiscountCurve(Protocol):
    """Minimal discount curve interface required by pricing engines."""

    def discount(self, time: float) -> float:
        """Return the discount factor for the requested time."""
