"""Black-Scholes model under the risk-neutral measure."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

import numpy as np
from scipy.integrate import quad

from hesperides.simulation import BLACK_SCHOLES_PATH_SIMULATOR
from hesperides.market.curves import Rate, integrated_rate


@dataclass(frozen=True)
class BlackScholesModel:
    """Black-Scholes cost-of-carry dynamics.

    Under the pricing measure, dS_t = (r - q) S_t dt + sigma S_t dW_t.
    The discount rate remains in the discount curve; q only affects the
    model dynamics.
    """

    rate: Rate
    volatility: float
    dividend_yield: Rate = 0.0

    @property
    def variance(self) -> float:
        """Instantaneous variance sigma squared."""
        return self.volatility * self.volatility

    @property
    def cost_of_carry(self) -> float:
        """Risk-neutral spot drift r - q."""
        return self.rate - self.dividend_yield

    def income_discount(self, maturity: float) -> float:
        """Present-value factor for continuous yield q over maturity."""
        return exp(-integrated_rate(self.dividend_yield, maturity))

    def integrated_carry(self, time: float) -> float:
        return integrated_rate(self.rate, time) - integrated_rate(self.dividend_yield, time)

    def integrated_log_drift(self, time: float) -> float:
        return self.integrated_carry(time) - 0.5 * self.variance * time

    def average_integrated_carry(self, maturity: float) -> float:
        if maturity == 0:
            return 0.0
        # Integrate the supplied curve: Asian payoffs depend on its full shape.
        value, error = quad(self.integrated_carry, 0, maturity,
                            epsabs=1e-10, epsrel=1e-10, limit=500)
        if not np.isfinite(value) or error > 1e-8 * max(1, abs(value)):
            raise ValueError("Curve integral did not converge")
        return value / maturity

    def discounted_spot(self, spot: float, maturity: float) -> float:
        """Spot adjusted by the continuous yield over maturity."""
        return spot * self.income_discount(maturity)

    def risk_neutral_log_drift(self) -> float:
        """Drift of log(S_t) under the risk-neutral measure."""
        return self.cost_of_carry - 0.5 * self.variance

    def deterministic_terminal_spot(self, spot: float, maturity: float) -> float:
        """Terminal spot when volatility is zero."""
        return spot * exp(self.integrated_carry(maturity))

    def continuous_geometric_log_distribution(
        self,
        spot: float,
        maturity: float,
    ) -> tuple[float, float]:
        """Mean and variance of log(G_T) for the continuous geometric average."""
        mean = log(spot) + self.average_integrated_carry(maturity) - self.variance * maturity / 4.0
        variance = self.variance * maturity / 3.0
        return mean, variance

    def deterministic_continuous_geometric_average(
        self,
        spot: float,
        maturity: float,
    ) -> float:
        """Continuous geometric average when volatility is zero."""
        return exp(log(spot) + self.average_integrated_carry(maturity))

    def simulate_terminal_spots(
        self,
        spot: float,
        maturity: float,
        n_paths: int,
        seed: int | None,
    ) -> np.ndarray:
        """Vectorised terminal spot simulation."""
        return BLACK_SCHOLES_PATH_SIMULATOR.simulate_terminal_spots(
            model=self,
            spot=spot,
            maturity=maturity,
            n_paths=n_paths,
            seed=seed,
        )

    def simulate_log_paths(
        self,
        spot: float,
        maturity: float,
        n_paths: int,
        n_steps: int,
        seed: int | None,
    ) -> np.ndarray:
        """Vectorised log-path simulation on a uniform grid excluding t=0."""
        return BLACK_SCHOLES_PATH_SIMULATOR.simulate_log_paths(
            model=self,
            spot=spot,
            maturity=maturity,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
        )
