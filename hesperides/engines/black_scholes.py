"""Black-Scholes pricing engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log, sqrt

import numpy as np

from hesperides.contracts.protocols import OptionContract
from hesperides.market.protocols import DiscountCurve
from hesperides.market.curves import integrated_rate
from hesperides.models.protocols import EuropeanDynamics, GeometricAsianDynamics, LognormalSimulationDynamics
from hesperides.simulation import BlackScholesPathSimulator
from hesperides.simulation_protocols import TerminalSimulation, PathSimulation
from hesperides.statistics import STANDARD_NORMAL


class BlackScholesEuropeanAnalyticalEngine:
    """Closed-form Black-Scholes engine for European options."""

    def price(
        self,
        spot: float,
        option: OptionContract,
        maturity: float,
        model: EuropeanDynamics,
        discount_curve: DiscountCurve,
    ) -> float:
        if maturity == 0.0:
            return float(option.payoff(spot))

        if model.volatility == 0.0:
            terminal_spot = model.deterministic_terminal_spot(spot, maturity)
            return float(
                discount_curve.discount(maturity) * option.payoff(terminal_spot)
            )

        sqrt_t = sqrt(maturity)
        discounted_spot = model.discounted_spot(spot, maturity)
        d1 = (
            log(discounted_spot / option.strike)
            + integrated_rate(model.rate, maturity) + 0.5 * model.variance * maturity
        ) / (model.volatility * sqrt_t)
        d2 = d1 - model.volatility * sqrt_t
        discount = discount_curve.discount(maturity)

        if option.is_call:
            price = (
                discounted_spot * STANDARD_NORMAL.cdf(d1)
                - option.strike * discount * STANDARD_NORMAL.cdf(d2)
            )
        else:
            price = (
                option.strike * discount * STANDARD_NORMAL.cdf(-d2)
                - discounted_spot * STANDARD_NORMAL.cdf(-d1)
            )
        return float(price)


@dataclass(frozen=True)
class BlackScholesEuropeanMonteCarloEngine:
    """Vectorised Monte Carlo Black-Scholes engine for European options."""

    simulator: TerminalSimulation = field(
        default_factory=BlackScholesPathSimulator
    )

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
        if maturity == 0.0:
            return float(option.payoff(spot))

        terminal_spots = self.simulator.simulate_terminal_spots(
            model=model,
            spot=spot,
            maturity=maturity,
            n_paths=n_paths,
            seed=seed,
        )
        payoffs = option.payoff(terminal_spots)
        return float(discount_curve.discount(maturity) * np.mean(payoffs))


class BlackScholesGeometricAsianAnalyticalEngine:
    """Closed-form engine for continuous geometric Asian options."""

    def price(
        self,
        spot: float,
        option: OptionContract,
        maturity: float,
        model: GeometricAsianDynamics,
        discount_curve: DiscountCurve,
    ) -> float:
        if maturity == 0.0:
            return float(option.payoff(spot))

        discount = discount_curve.discount(maturity)
        if model.volatility == 0.0:
            average = model.deterministic_continuous_geometric_average(spot, maturity)
            return float(discount * option.payoff(average))

        mean, variance = model.continuous_geometric_log_distribution(spot, maturity)
        std = sqrt(variance)
        expected_average = np.exp(mean + 0.5 * variance)
        d1 = (mean - log(option.strike) + variance) / std
        d2 = d1 - std

        if option.is_call:
            price = discount * (
                expected_average * STANDARD_NORMAL.cdf(d1)
                - option.strike * STANDARD_NORMAL.cdf(d2)
            )
        else:
            price = discount * (
                option.strike * STANDARD_NORMAL.cdf(-d2)
                - expected_average * STANDARD_NORMAL.cdf(-d1)
            )
        return float(price)


@dataclass(frozen=True)
class BlackScholesGeometricAsianMonteCarloEngine:
    """Vectorised Monte Carlo engine for discrete geometric Asian approximation."""

    simulator: PathSimulation = field(
        default_factory=BlackScholesPathSimulator
    )

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
        if maturity == 0.0:
            return float(option.payoff(spot))

        log_paths = self.simulator.simulate_log_paths(
            model=model,
            spot=spot,
            maturity=maturity,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
        )
        geometric_averages = np.exp(np.mean(log_paths, axis=1))
        payoffs = option.payoff(geometric_averages)
        return float(discount_curve.discount(maturity) * np.mean(payoffs))
