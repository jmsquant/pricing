"""Pricing-engine abstractions for pricer services."""

from __future__ import annotations

from typing import Protocol

from hesperides.contracts.protocols import OptionContract
from hesperides.market.protocols import DiscountCurve
from hesperides.models.protocols import (
    EuropeanDynamics,
    GeometricAsianDynamics,
    LognormalSimulationDynamics,
    BinomialTreeModel,
)


class BinomialEuropeanPricingEngine(Protocol):
    """Engine interface for binomial European option pricing."""

    def price(
        self,
        spot: float,
        option: OptionContract,
        maturity_steps: int,
        model: BinomialTreeModel,
        discount_curve: DiscountCurve,
    ) -> float:
        """Return the present value of the option."""


class BlackScholesAnalyticalPricingEngine(Protocol):
    """Engine interface for analytical Black-Scholes pricing."""

    def price(
        self,
        spot: float,
        option: OptionContract,
        maturity: float,
        model: EuropeanDynamics,
        discount_curve: DiscountCurve,
    ) -> float:
        """Return the present value of the option."""


class BlackScholesMonteCarloPricingEngine(Protocol):
    """Engine interface for Black-Scholes Monte Carlo pricing."""

    def price(
        self,
        spot: float,
        option: OptionContract,
        maturity: float,
        model: LognormalSimulationDynamics,
        discount_curve: DiscountCurve,
        n_paths: int,
        seed: int | None,
    ) -> float:
        """Return the Monte Carlo present value of the option."""


class BlackScholesPathMonteCarloPricingEngine(Protocol):
    """Engine interface for path-dependent Black-Scholes Monte Carlo pricing."""

    def price(
        self,
        spot: float,
        option: OptionContract,
        maturity: float,
        model: LognormalSimulationDynamics,
        discount_curve: DiscountCurve,
        n_paths: int,
        n_steps: int,
        seed: int | None,
    ) -> float:
        """Return the path-dependent Monte Carlo present value."""


class GeometricAsianAnalyticalPricingEngine(Protocol):
    def price(self, spot: float, option: OptionContract, maturity: float,
              model: GeometricAsianDynamics, discount_curve: DiscountCurve) -> float: ...
