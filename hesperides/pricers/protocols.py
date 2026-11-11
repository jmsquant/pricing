"""Narrow application contracts; default wiring stays in concrete services."""
from __future__ import annotations
from collections.abc import Callable
from typing import Protocol, TYPE_CHECKING
import numpy as np
from hesperides.market.curves import Rate
from hesperides.models.heat_equation import BoundaryInput
if TYPE_CHECKING:
    from hesperides.pricers.european_option_pricer import BinomialEuropeanComponents
    from hesperides.pricers.black_scholes_pricer import BlackScholesComponents

class BinomialPricing(Protocol):

    def price(
        self,
        spot: float,
        strike: float,
        is_call: bool,
        maturity_steps: int,
        up: float,
        down: float,
        rate: Rate,
    ) -> float:
        ...

class EuropeanPricing(Protocol):

    def price(
        self,
        spot: float,
        strike: float,
        maturity: float,
        rate: Rate,
        volatility: float,
        is_call: bool,
        engine: str = 'analytical',
        n_paths: int | None = None,
        seed: int | None = None,
        dividend_yield: Rate = 0.0,
        require_positive_volatility: bool = False,
        spot_name: str = 'St',
    ) -> float:
        ...

class AsianPricing(Protocol):

    def price(
        self,
        spot: float,
        strike: float,
        maturity: float,
        rate: Rate,
        volatility: float,
        is_call: bool,
        engine: str = 'analytical',
        n_paths: int | None = None,
        n_steps: int | None = None,
        seed: int | None = None,
    ) -> float:
        ...

class ProductPricing(Protocol):

    def price_bs_european(
        self,
        spot: float,
        strike: float,
        maturity: float,
        rate: Rate,
        volatility: float,
        is_call: bool,
        engine: str = 'analytical',
        n_paths: int | None = None,
        seed: int | None = None,
        dividend_yield: Rate = 0.0,
    ) -> float:
        ...

    def price_bs_european_dividend(
        self,
        spot: float,
        strike: float,
        maturity: float,
        rate: Rate,
        volatility: float,
        is_call: bool,
        dividend_yield: Rate = 0.0,
        engine: str = 'analytical',
        n_paths: int | None = None,
        seed: int | None = None,
        spot_name: str = 'St',
    ) -> float:
        ...

    def price_fx_option(
        self,
        spot: float,
        strike: float,
        maturity: float,
        domestic_rate: Rate,
        foreign_rate: Rate,
        volatility: float,
        is_call: bool,
        engine: str = 'analytical',
        n_paths: int | None = None,
        seed: int | None = None,
    ) -> float:
        ...

    def price_future_option(
        self,
        future: float,
        strike: float,
        maturity: float,
        rate: Rate,
        volatility: float,
        is_call: bool,
        engine: str = 'analytical',
        n_paths: int | None = None,
        seed: int | None = None,
    ) -> float:
        ...

    def price_bs_geometric_asian(
        self,
        spot: float,
        strike: float,
        maturity: float,
        rate: Rate,
        volatility: float,
        is_call: bool,
        engine: str = 'analytical',
        n_paths: int | None = None,
        n_steps: int | None = None,
        seed: int | None = None,
    ) -> float:
        ...

class GreekPricing(Protocol):

    def greek(
        self,
        spot: float,
        strike: float,
        maturity: float,
        rate: Rate,
        volatility: float,
        is_call: bool,
        greek: str,
        engine: str = 'analytical',
        greek_engine: str = 'analytical',
        fd_scheme: str = 'central',
        h: float | None = None,
        n_paths: int | None = None,
        seed: int | None = None,
    ) -> float:
        ...

class HeatSolving(Protocol):

    def solve(
        self,
        initial_condition: Callable[[np.ndarray], np.ndarray],
        kappa: float,
        domain_size: float,
        maturity: float,
        n_x: int,
        n_t: int,
        scheme: str = 'explicit',
        left_boundary: BoundaryInput = 0.0,
        right_boundary: BoundaryInput = 0.0,
        boundary_type: str = 'dirichlet',
    ) -> tuple[np.ndarray, np.ndarray]:
        ...

class HeatPricing(Protocol):

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
        scheme: str = 'implicit',
    ) -> float:
        ...

class ArbitrageCalculation(Protocol):

    def compute(
        self,
        surface: np.ndarray,
        strikes: np.ndarray | None = None,
        quantity: str = 'vertical',
    ) -> np.ndarray:
        ...

class BinomialComponentFactory(Protocol):

    def create(
        self,
        strike: float,
        is_call: bool,
        up: float,
        down: float,
        rate: Rate,
    ) -> BinomialEuropeanComponents:
        ...

class BlackScholesFactory(Protocol):

    def create(self, rate: Rate, volatility: float, dividend_yield: Rate=0.0) -> BlackScholesComponents:
        ...
