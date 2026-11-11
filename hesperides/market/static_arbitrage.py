"""Carr–Madan (2005) static-arbitrage spread grids on call surfaces.

Implements the numerical checks for vertical, butterfly, and calendar
spread conditions on a discrete call-price surface C_{i,j} (strikes × maturities).

References
----------
Carr, P. and Madan, D. (2005). A note on sufficient conditions for no
arbitrage. *Finance Research Letters*, 2(3–4), 125–130.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np


# ── Data Object ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CallSurface:
    """Discrete call-price surface C_{i,j} (strikes × maturities).

    Attributes
    ----------
    surface : ndarray, shape (nK, nT)
        Call prices by strike (row) and expiry (column).
    strikes : ndarray or None
        Strictly increasing strikes.  Required for vertical and
        butterfly quantities; ignored for calendar spreads.
    """

    surface: np.ndarray
    strikes: np.ndarray | None = None


# ── Abstract Strategy ────────────────────────────────────────────────────

class SpreadQuantity(ABC):
    """Abstract base for Carr–Madan spread-quantity computations.

    Each concrete subclass encapsulates a single spread type,
    satisfying the Single-Responsibility and Open/Closed principles.
    """

    @abstractmethod
    def compute(self, call_surface: CallSurface) -> np.ndarray:
        """Return the spread grid for the given call surface."""


# ── Concrete Strategies ──────────────────────────────────────────────────

class VerticalSpread(SpreadQuantity):
    """Normalized vertical (call-spread) values Q_{i,j}.

    Q_{i,j} = (C_{i-1,j} - C_{i,j}) / (K_i - K_{i-1})

    for i = 1, …, nK-1.
    """

    def compute(self, call_surface: CallSurface) -> np.ndarray:
        """Return ndarray, shape (nK-1, nT)."""
        if call_surface.strikes is None:
            raise ValueError("strikes is required for quantity='vertical'")

        surface = call_surface.surface
        strikes = call_surface.strikes

        # Numerator: C_{i-1,j} - C_{i,j}  → shape (nK-1, nT)
        delta_c = np.diff(surface, axis=0)  # C_{i,j} - C_{i-1,j}
        delta_c = -delta_c                   # C_{i-1,j} - C_{i,j}

        # Denominator: K_i - K_{i-1}  → shape (nK-1,)
        delta_k = np.diff(strikes)

        # Broadcasting: (nK-1, nT) / (nK-1, 1)
        return delta_c / delta_k[:, np.newaxis]


@dataclass(frozen=True)
class ButterflySpread(SpreadQuantity):
    """Interior butterfly spread values BS_{i,j}.

    BS_{i,j} = Q_{i,j} − Q_{i+1,j}

    where Q are the normalized vertical spreads. Requires at least three
    strikes (nK >= 3).
    """

    vertical: SpreadQuantity = field(default_factory=VerticalSpread)

    def compute(self, call_surface: CallSurface) -> np.ndarray:
        """Return ndarray, shape (nK-2, nT)."""
        if call_surface.strikes is None:
            raise ValueError("strikes is required for quantity='butterfly'")

        q = self.vertical.compute(call_surface)
        # BS_{i,j} = Q_{i,j} - Q_{i+1,j}  → shape (nK-2, nT)
        return -np.diff(q, axis=0)


class CalendarSpread(SpreadQuantity):
    """Calendar spread values CS_{i,j} = C_{i,j+1} − C_{i,j}."""

    def compute(self, call_surface: CallSurface) -> np.ndarray:
        """Return ndarray, shape (nK, nT-1)."""
        return np.diff(call_surface.surface, axis=1)


# ── Registry (Strategy look-up) ─────────────────────────────────────────

class SpreadDefaults:
    """Default spread strategies."""

    @classmethod
    def _default_spread_strategies(cls) -> dict[str, SpreadQuantity]:
        return {
            "vertical": VerticalSpread(),
            "butterfly": ButterflySpread(),
            "calendar": CalendarSpread(),
        }


# Compatibility names refer directly to class methods.
_default_spread_strategies = SpreadDefaults._default_spread_strategies


@dataclass(frozen=True)
class SpreadQuantityRegistry:
    """Lookup service for Carr-Madan spread strategies."""

    strategies: Mapping[str, SpreadQuantity] = field(
        default_factory=_default_spread_strategies
    )

    def get(self, quantity: str) -> SpreadQuantity:
        """Return the strategy for quantity or raise the legacy ValueError."""
        strategy = self.strategies.get(quantity)
        if strategy is None:
            raise ValueError(
                f"Unknown quantity {quantity!r}. "
                f"Must be 'vertical', 'butterfly', or 'calendar'."
            )
        return strategy


@dataclass(frozen=True)
class StaticArbitrageCalculator:
    """Application service for Carr-Madan static-arbitrage quantities."""

    registry: SpreadQuantityRegistry = field(default_factory=SpreadQuantityRegistry)

    def compute(
        self,
        surface: np.ndarray,
        strikes: np.ndarray | None = None,
        quantity: str = "vertical",
    ) -> np.ndarray:
        """Compute the selected spread quantity for a call surface."""
        call_surface = CallSurface(surface=surface, strikes=strikes)
        return self.registry.get(quantity).compute(call_surface)


_DEFAULT_CALCULATOR = StaticArbitrageCalculator()


# ── Public façade (signature preserved for backward compatibility) ───────

compute_static_arbitrage_quantity = _DEFAULT_CALCULATOR.compute
