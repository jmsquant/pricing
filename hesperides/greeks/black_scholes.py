"""Black-Scholes Greek engines for European options."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import exp, log, pi, sqrt

from hesperides.contracts.protocols import OptionContract
from hesperides.greeks.bumps import (
    BumpedBlackScholesInputs,
    bump_rate,
    bump_spot,
    bump_volatility,
)
from hesperides.market.curves import FlatDiscountCurve
from hesperides.models.black_scholes import BlackScholesModel
from hesperides.statistics import STANDARD_NORMAL


PricingFunction = Callable[[float, BlackScholesModel, FlatDiscountCurve], float]

DEFAULT_RELATIVE_SPOT_BUMP = 1.0e-3
DEFAULT_MIN_SPOT_BUMP = 1.0e-4
DEFAULT_VOLATILITY_BUMP = 1.0e-4
DEFAULT_RATE_BUMP = 1.0e-4


class BlackScholesGreekFormula:
    """Shared analytical Greek terms."""

    @classmethod
    def _normal_pdf(cls, x: float) -> float:
        return exp(-0.5 * x * x) / sqrt(2.0 * pi)

    @classmethod
    def _d1(cls,
        spot: float,
        strike: float,
        maturity: float,
        model: BlackScholesModel,
    ) -> float:
        sqrt_t = sqrt(maturity)
        return (
            log(spot / strike) + (model.rate + 0.5 * model.variance) * maturity
        ) / (model.volatility * sqrt_t)


# Compatibility names refer directly to class methods.
_normal_pdf = BlackScholesGreekFormula._normal_pdf
_d1 = BlackScholesGreekFormula._d1


class BlackScholesEuropeanAnalyticalGreekEngine:
    """Closed-form Black-Scholes Greeks for vanilla European options."""

    def greek(
        self,
        spot: float,
        option: OptionContract,
        maturity: float,
        model: BlackScholesModel,
        discount_curve: FlatDiscountCurve,
        greek: str,
    ) -> float:
        """Return the requested closed-form Greek."""
        d1 = _d1(
            spot=spot,
            strike=option.strike,
            maturity=maturity,
            model=model,
        )
        d2 = d1 - model.volatility * sqrt(maturity)

        if greek == "delta":
            if option.is_call:
                return float(STANDARD_NORMAL.cdf(d1))
            return float(STANDARD_NORMAL.cdf(d1) - 1.0)

        if greek == "gamma":
            return float(_normal_pdf(d1) / (spot * model.volatility * sqrt(maturity)))

        if greek == "vega":
            return float(spot * _normal_pdf(d1) * sqrt(maturity))

        discount = discount_curve.discount(maturity)
        if option.is_call:
            return float(option.strike * maturity * discount * STANDARD_NORMAL.cdf(d2))
        return float(-option.strike * maturity * discount * STANDARD_NORMAL.cdf(-d2))


@dataclass(frozen=True)
class BlackScholesEuropeanFiniteDifferenceGreekEngine:
    """Finite-difference bump-and-reprice engine for European Greeks.

    Bumps are additive in the natural units of the target parameter. Defaults
    follow practical orders of magnitude: a small relative spot bump and
    one-basis-point-style volatility/rate bumps.
    """

    def greek(
        self,
        spot: float,
        option: OptionContract,
        maturity: float,
        model: BlackScholesModel,
        discount_curve: FlatDiscountCurve,
        greek: str,
        fd_scheme: str,
        h: float | None,
        price: PricingFunction,
    ) -> float:
        """Return a finite-difference Greek by repricing bumped inputs."""
        bump = self._default_bump(greek, spot, model) if h is None else h

        if greek == "gamma":
            self._ensure_lower_bump_is_valid(greek, spot, model, bump)
            up = self._price_state(
                self._bumped_state("gamma", spot, model, discount_curve, bump),
                price,
            )
            base = self._price_state(
                self._bumped_state("gamma", spot, model, discount_curve, 0.0),
                price,
            )
            down = self._price_state(
                self._bumped_state("gamma", spot, model, discount_curve, -bump),
                price,
            )
            return float((up - 2.0 * base + down) / (bump * bump))

        if fd_scheme == "central":
            self._ensure_lower_bump_is_valid(greek, spot, model, bump)
            up = self._price_state(
                self._bumped_state(greek, spot, model, discount_curve, bump),
                price,
            )
            down = self._price_state(
                self._bumped_state(greek, spot, model, discount_curve, -bump),
                price,
            )
            return float((up - down) / (2.0 * bump))

        up = self._price_state(
            self._bumped_state(greek, spot, model, discount_curve, bump),
            price,
        )
        base = self._price_state(
            self._bumped_state(greek, spot, model, discount_curve, 0.0),
            price,
        )
        return float((up - base) / bump)

    def _default_bump(
        self,
        greek: str,
        spot: float,
        model: BlackScholesModel,
    ) -> float:
        if greek in {"delta", "gamma"}:
            return max(DEFAULT_RELATIVE_SPOT_BUMP * spot, DEFAULT_MIN_SPOT_BUMP)
        if greek == "vega":
            return min(DEFAULT_VOLATILITY_BUMP, 0.5 * model.volatility)
        return DEFAULT_RATE_BUMP

    def _bumped_state(
        self,
        greek: str,
        spot: float,
        model: BlackScholesModel,
        discount_curve: FlatDiscountCurve,
        shift: float,
    ) -> BumpedBlackScholesInputs:
        if greek in {"delta", "gamma"}:
            return bump_spot(spot, model, discount_curve, shift)
        if greek == "vega":
            return bump_volatility(spot, model, discount_curve, shift)
        return bump_rate(spot, model, discount_curve, shift)

    def _ensure_lower_bump_is_valid(
        self,
        greek: str,
        spot: float,
        model: BlackScholesModel,
        bump: float,
    ) -> None:
        if greek in {"delta", "gamma"} and spot - bump <= 0.0:
            raise ValueError("h is too large: bumped St must remain > 0")
        if greek == "vega" and model.volatility - bump <= 0.0:
            raise ValueError("h is too large: bumped sigma must remain > 0")

    def _price_state(
        self,
        state: BumpedBlackScholesInputs,
        price: PricingFunction,
    ) -> float:
        return price(state.spot, state.model, state.discount_curve)
