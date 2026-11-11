"""Heat-equation pricer services and Black-Scholes heat transform."""

from __future__ import annotations

from hesperides.validation_protocols import HeatValidation

from hesperides.engines.solver_protocols import HeatEquationEngine

from hesperides.pricers.protocols import HeatSolving

from collections.abc import Callable
from dataclasses import dataclass, field
from math import exp, log, sqrt

import numpy as np

from hesperides.contracts.initial_condition import InitialCondition
from hesperides.engines.finite_difference import HeatEquationFiniteDifferenceEngine
from hesperides.models.heat_equation import BoundaryInput, HeatEquationModel
from hesperides.validation import HeatEquationInputValidator
from hesperides.market.curves import Rate, equivalent_rate


@dataclass(frozen=True)
class HeatPayoff:
    """Terminal option payoff in the transformed spatial coordinate."""

    left_y: float
    volatility: float
    strike: float
    is_call: bool

    def __call__(self, z_grid: np.ndarray) -> np.ndarray:
        y_grid = self.left_y + z_grid
        terminal_spots = np.exp(self.volatility * y_grid)
        if self.is_call:
            return np.maximum(terminal_spots - self.strike, 0.0)
        return np.maximum(self.strike - terminal_spots, 0.0)


@dataclass(frozen=True)
class HeatBoundary:
    """Time-dependent asymptotic value at one spatial boundary."""

    volatility: float
    coordinate: float
    variance_half: float
    strike: float
    is_call: bool

    def __call__(self, time_grid: np.ndarray) -> np.ndarray:
        underlying = np.exp(self.volatility * self.coordinate + self.variance_half * time_grid)
        return np.maximum(underlying - self.strike if self.is_call else self.strike - underlying, 0.0)


@dataclass(frozen=True)
class HeatEquationSolver:
    """Application service for finite-difference heat-equation solves."""

    validator: HeatValidation = field(
        default_factory=HeatEquationInputValidator
    )
    engine: HeatEquationEngine = field(
        default_factory=HeatEquationFiniteDifferenceEngine
    )

    def solve(
        self,
        initial_condition: Callable[[np.ndarray], np.ndarray],
        kappa: float,
        domain_size: float,
        maturity: float,
        n_x: int,
        n_t: int,
        scheme: str = "explicit",
        left_boundary: BoundaryInput = 0.0,
        right_boundary: BoundaryInput = 0.0,
        boundary_type: str = "dirichlet",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve v_t = kappa v_xx on [0, M] up to T."""
        self.validator.validate_scheme(scheme)
        self.validator.validate_boundary_type(boundary_type)
        self.validator.validate_heat_inputs(
            kappa=kappa,
            domain_size=domain_size,
            maturity=maturity,
            n_x=n_x,
            n_t=n_t,
        )
        model = HeatEquationModel(
            kappa=kappa,
            domain_size=domain_size,
            maturity=maturity,
            n_x=int(n_x),
            n_t=int(n_t),
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            boundary_type=boundary_type,
        )
        adapter = InitialCondition(initial_condition)
        return self.engine.solve(
            initial_condition=adapter,
            model=model,
            scheme=scheme,
        )


@dataclass(frozen=True)
class BlackScholesHeatPricer:
    """Price European options through the Black-Scholes heat transform."""

    validator: HeatValidation = field(
        default_factory=HeatEquationInputValidator
    )
    heat_solver: HeatSolving = field(default_factory=HeatEquationSolver)

    def price(
        self,
        spot: float,
        strike: float,
        maturity: float,
        rate: Rate,
        volatility: float,
        is_call: bool,
        n_x: int = 400,
        n_t: int = 400,
        scheme: str = "implicit",
    ) -> float:
        """Price a European option by solving the transformed heat equation."""
        self.validator.validate_black_scholes_heat_inputs(
            spot=spot,
            strike=strike,
            maturity=maturity,
            volatility=volatility,
            n_x=n_x,
            n_t=n_t,
            scheme=scheme,
        )

        rate = equivalent_rate(rate, maturity)
        domain = self._heat_domain(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=volatility,
            n_x=int(n_x),
            n_t=int(n_t),
        )
        left_y, right_y, y0 = domain
        domain_size = right_y - left_y

        payoff_in_heat_space = HeatPayoff(left_y, volatility, strike, is_call)

        left_boundary, right_boundary = self._asymptotic_boundaries(
            strike=strike,
            volatility=volatility,
            is_call=is_call,
            left_y=left_y,
            right_y=right_y,
        )
        x_grid, heat_values = self.heat_solver.solve(
            initial_condition=payoff_in_heat_space,
            kappa=0.5,
            domain_size=domain_size,
            maturity=maturity,
            n_x=int(n_x),
            n_t=int(n_t),
            scheme=scheme,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            boundary_type="dirichlet",
        )
        transformed_value = np.interp(y0 - left_y, x_grid, heat_values)
        return float(exp(-rate * maturity) * transformed_value)

    def _heat_domain(
        self,
        spot: float,
        strike: float,
        maturity: float,
        rate: Rate,
        volatility: float,
        n_x: int,
        n_t: int,
    ) -> tuple[float, float, float]:
        target_y = (
            log(spot) - (0.5 * volatility * volatility - rate) * maturity
        ) / volatility
        strike_y = log(strike) / volatility
        stability_half_width = 0.5 * n_x * sqrt(maturity / n_t)
        diffusion_half_width = 8.0 * sqrt(maturity)
        half_width = max(stability_half_width, diffusion_half_width)
        left_y = min(target_y, strike_y) - half_width
        right_y = max(target_y, strike_y) + half_width
        return left_y, right_y, target_y

    def _asymptotic_boundaries(
        self,
        strike: float,
        volatility: float,
        is_call: bool,
        left_y: float,
        right_y: float,
    ) -> tuple[BoundaryInput, BoundaryInput]:
        variance_half = 0.5 * volatility * volatility

        if is_call:
            return (
                0.0,
                HeatBoundary(volatility, right_y, variance_half, strike, True),
            )

        return (
            HeatBoundary(volatility, left_y, variance_half, strike, False),
            0.0,
        )


_DEFAULT_HEAT_SOLVER = HeatEquationSolver()
_DEFAULT_BLACK_SCHOLES_HEAT_PRICER = BlackScholesHeatPricer()


solve_heat_equation = _DEFAULT_HEAT_SOLVER.solve


price_bs_european_heat = _DEFAULT_BLACK_SCHOLES_HEAT_PRICER.price
