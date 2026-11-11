"""Vectorised binomial pricing engine for European options."""

from __future__ import annotations

import numpy as np
from math import isfinite

from hesperides.contracts.protocols import OptionContract
from hesperides.market.protocols import DiscountCurve
from hesperides.models.protocols import BinomialTreeModel


class BinomialEuropeanEngine:
    """Prices European options on a recombining binomial tree.

    The backward induction is fully vectorised over tree nodes;
    only the time-step loop is in Python.
    """

    @staticmethod
    def price(
        spot: float,
        option: OptionContract,
        maturity_steps: int,
        model: BinomialTreeModel,
        discount_curve: DiscountCurve,
    ) -> float:
        """Return the present value of (option) under (model).

        PARAMETERS

        spot : float
            Current underlying price.
        option : OptionContract
            The contract to price.
        maturity_steps : int
            Number of binomial time steps remaining until expiry.
        model : BinomialTreeModel
            Up/down factors and risk-free rate.
        discount_curve : DiscountCurve
            Used for one-period discounting during backward induction.
        """
        if maturity_steps < 0:
            raise ValueError("maturity_steps must be >= 0")

        # Edge case (we are already at expiry)
        if maturity_steps == 0:
            return float(option.payoff(spot))

        T = maturity_steps

        # Terminal stock prices: S * u^j * d^(T-j), j = 0,..., T
        j = np.arange(T + 1, dtype=np.float64)
        terminal_spots = spot * (model.up ** j) * (model.down ** (T - j))

        # Terminal payoffs (vectorised)
        values = option.payoff(terminal_spots).astype(np.float64)

        # Backward induction – loop over time, vectorise over nodes
        for step in reversed(range(T)):
            start = discount_curve.discount(step)
            end = discount_curve.discount(step + 1)
            if not isfinite(start) or not isfinite(end) or start <= 0 or end <= 0:
                raise ValueError("Discount factors must be positive and finite")
            df = end / start
            q = (1 / df - model.down) / (model.up - model.down)
            if not 0 <= q <= 1:
                raise ValueError(f"Binomial arbitrage at step {step}: require down <= P(t)/P(t+dt) <= up")
            values = df * (q * values[1:] + (1.0 - q) * values[:-1])

        return float(values[0])
