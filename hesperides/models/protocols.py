"""Model abstractions used by pricing engines."""

from __future__ import annotations

from typing import Protocol
from hesperides.market.curves import Rate


class BinomialTreeModel(Protocol):
    """Interface for recombining binomial tree dynamics."""

    up: float
    down: float
    rate: float

    def risk_neutral_prob(self) -> float:
        """Return the one-step risk-neutral up probability."""


class LognormalSimulationDynamics(Protocol):
    """Only volatility and cumulative drift are needed to simulate paths."""

    @property
    def volatility(self) -> float: ...

    def integrated_log_drift(self, time: float) -> float: ...


class EuropeanDynamics(Protocol):
    @property
    def rate(self) -> Rate: ...

    @property
    def volatility(self) -> float: ...

    @property
    def variance(self) -> float:
        """Return instantaneous variance."""

    def discounted_spot(self, spot: float, maturity: float) -> float:
        """Return the spot adjusted by the continuous yield."""

    def deterministic_terminal_spot(self, spot: float, maturity: float) -> float:
        """Return terminal spot when volatility is zero."""


class GeometricAsianDynamics(Protocol):
    @property
    def volatility(self) -> float: ...

    def continuous_geometric_log_distribution(
        self,
        spot: float,
        maturity: float,
    ) -> tuple[float, float]:
        """Return mean and variance of log continuous geometric average."""

    def deterministic_continuous_geometric_average(
        self,
        spot: float,
        maturity: float,
    ) -> float:
        """Return continuous geometric average when volatility is zero."""


class BlackScholesDynamics(EuropeanDynamics, GeometricAsianDynamics, LognormalSimulationDynamics, Protocol):
    """Compatibility contract for models implementing all three capabilities."""

    @property
    def dividend_yield(self) -> Rate: ...

    @property
    def cost_of_carry(self) -> float: ...

    def income_discount(self, maturity: float) -> float: ...

    def risk_neutral_log_drift(self) -> float: ...
