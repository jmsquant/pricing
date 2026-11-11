"""Input validation services shared by pricers and models."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class BinomialParameterValidator:
    """Validate binomial model parameters."""

    def validate(self, up: float, down: float, rate: float) -> None:
        """Raise ValueError when binomial parameters are inconsistent."""
        if up <= down:
            raise ValueError(f"up ({up}) must be strictly greater than down ({down})")
        if 1.0 + rate == 0.0:
            raise ValueError("1 + rate must be non-zero (infinite discount)")


@dataclass(frozen=True)
class BlackScholesInputValidator:
    """Validate public Black-Scholes pricer inputs."""

    valid_engines: tuple[str, ...] = ("analytical", "mc")

    def validate_engine(self, engine: str) -> None:
        """Validate pricing engine selector."""
        if engine not in self.valid_engines:
            raise ValueError("engine must be 'analytical' or 'mc'")

    def validate_market_inputs(
        self,
        spot: float,
        strike: float,
        maturity: float,
        volatility: float,
        spot_name: str = "St",
        require_positive_volatility: bool = False,
    ) -> None:
        """Validate market and contract inputs used by Black-Scholes pricers."""
        if spot <= 0.0:
            raise ValueError(f"{spot_name} must be > 0")
        if strike <= 0.0:
            raise ValueError("K must be > 0")
        if maturity < 0.0:
            raise ValueError("T must be >= 0")
        if require_positive_volatility and volatility <= 0.0:
            raise ValueError("sigma must be > 0")
        if not require_positive_volatility and volatility < 0.0:
            raise ValueError("sigma must be >= 0")

    def require_positive_integer(self, value: int | None, name: str) -> int:
        """Return value as int when it is a positive non-bool integer."""
        if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)


@dataclass(frozen=True)
class BlackScholesGreekInputValidator(BlackScholesInputValidator):
    """Validate public Black-Scholes Greek inputs."""

    valid_greeks: tuple[str, ...] = ("delta", "gamma", "vega", "rho")
    valid_greek_engines: tuple[str, ...] = ("analytical", "fd")
    valid_fd_schemes: tuple[str, ...] = ("forward", "central")

    def validate_greek(self, greek: str) -> None:
        """Validate Greek selector."""
        if greek not in self.valid_greeks:
            raise ValueError("greek must be 'delta', 'gamma', 'vega', or 'rho'")

    def validate_greek_engine(self, greek_engine: str) -> None:
        """Validate Greek engine selector."""
        if greek_engine not in self.valid_greek_engines:
            raise ValueError("greek_engine must be 'analytical' or 'fd'")

    def validate_fd_scheme(self, fd_scheme: str) -> None:
        """Validate first-derivative finite-difference scheme selector."""
        if fd_scheme not in self.valid_fd_schemes:
            raise ValueError("fd_scheme must be 'forward' or 'central'")

    def validate_greek_market_inputs(
        self,
        spot: float,
        strike: float,
        maturity: float,
        volatility: float,
    ) -> None:
        """Validate Black-Scholes inputs for Greek calculations."""
        if spot <= 0.0:
            raise ValueError("St must be > 0")
        if strike <= 0.0:
            raise ValueError("K must be > 0")
        if maturity <= 0.0:
            raise ValueError("T must be > 0")
        if volatility <= 0.0:
            raise ValueError("sigma must be > 0")

    def validate_positive_bump(self, bump: float | None) -> None:
        """Validate an optional finite-difference bump."""
        if bump is not None and bump <= 0.0:
            raise ValueError("h must be > 0")

    def require_seed(self, seed: int | None) -> int:
        """Return seed when present for common-random-number MC repricing."""
        if seed is None:
            raise ValueError("seed is required when greek_engine='fd' and engine='mc'")
        return seed


@dataclass(frozen=True)
class HeatEquationInputValidator:
    """Validate public heat-equation finite-difference inputs."""

    valid_schemes: tuple[str, ...] = ("explicit", "implicit")
    valid_boundary_types: tuple[str, ...] = ("dirichlet", "neumann")

    def validate_scheme(self, scheme: str) -> None:
        """Validate finite-difference scheme selector."""
        if scheme not in self.valid_schemes:
            raise ValueError("scheme must be 'explicit' or 'implicit'")

    def validate_boundary_type(self, boundary_type: str) -> None:
        """Validate boundary-condition selector."""
        if boundary_type not in self.valid_boundary_types:
            raise ValueError("boundary_type must be 'dirichlet' or 'neumann'")

    def validate_heat_inputs(
        self,
        kappa: float,
        domain_size: float,
        maturity: float,
        n_x: int,
        n_t: int,
    ) -> None:
        """Validate heat-equation model and grid inputs."""
        if kappa <= 0.0:
            raise ValueError("kappa must be > 0")
        if domain_size <= 0.0:
            raise ValueError("M must be > 0")
        if maturity <= 0.0:
            raise ValueError("T must be > 0")
        self.require_grid_integer(n_x, "n_x")
        self.require_grid_integer(n_t, "n_t")

    def require_grid_integer(self, value: int, name: str) -> int:
        """Return value as int when it is a positive non-bool integer."""
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be >= 1")
        return int(value)

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
        """Validate inputs for Black-Scholes pricing through the heat equation."""
        self.validate_scheme(scheme)
        if spot <= 0.0:
            raise ValueError("St must be > 0")
        if strike <= 0.0:
            raise ValueError("K must be > 0")
        if maturity <= 0.0:
            raise ValueError("T must be > 0")
        if volatility <= 0.0:
            raise ValueError("sigma must be > 0")
        self.require_grid_integer(n_x, "n_x")
        self.require_grid_integer(n_t, "n_t")
