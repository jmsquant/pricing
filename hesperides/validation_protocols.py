"""Input validation contracts consumed by pricing services."""
from __future__ import annotations
from typing import Protocol

class BlackScholesValidation(Protocol):

    def validate_engine(self, engine: str) -> None:
        ...

    def validate_market_inputs(
        self,
        spot: float,
        strike: float,
        maturity: float,
        volatility: float,
        spot_name: str = 'St',
        require_positive_volatility: bool = False,
    ) -> None:
        ...

    def require_positive_integer(self, value: int | None, name: str) -> int:
        ...

class GreekValidation(BlackScholesValidation):

    def validate_greek(self, greek: str) -> None:
        ...

    def validate_greek_engine(self, greek_engine: str) -> None:
        ...

    def validate_fd_scheme(self, fd_scheme: str) -> None:
        ...

    def validate_greek_market_inputs(
        self,
        spot: float,
        strike: float,
        maturity: float,
        volatility: float,
    ) -> None:
        ...

    def validate_positive_bump(self, bump: float | None) -> None:
        ...

    def require_seed(self, seed: int | None) -> int:
        ...

class HeatValidation(Protocol):

    def validate_scheme(self, scheme: str) -> None:
        ...

    def validate_boundary_type(self, boundary_type: str) -> None:
        ...

    def validate_heat_inputs(
        self,
        kappa: float,
        domain_size: float,
        maturity: float,
        n_x: int,
        n_t: int,
    ) -> None:
        ...

    def require_grid_integer(self, value: int, name: str) -> int:
        ...

    def validate_black_scholes_heat_inputs(
        self,
        spot: float,
        strike: float,
        maturity: float,
        volatility: float,
        n_x: int,
        n_t: int,
        scheme: str,
    ) -> None:
        ...
