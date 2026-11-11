"""Bump helpers for Black-Scholes finite-difference Greeks."""

from __future__ import annotations

from dataclasses import dataclass

from hesperides.market.curves import FlatDiscountCurve
from hesperides.models.black_scholes import BlackScholesModel


@dataclass(frozen=True)
class BumpedBlackScholesInputs:
    """Black-Scholes pricing inputs after one additive bump."""

    spot: float
    model: BlackScholesModel
    discount_curve: FlatDiscountCurve


class BlackScholesBumps:
    """Create independent market states for bump-and-reprice."""

    @classmethod
    def _copy_model(cls, model: BlackScholesModel) -> BlackScholesModel:
        return BlackScholesModel(
            rate=model.rate,
            volatility=model.volatility,
            dividend_yield=model.dividend_yield,
        )

    @classmethod
    def _copy_curve(cls, discount_curve: FlatDiscountCurve) -> FlatDiscountCurve:
        return FlatDiscountCurve(
            rate=discount_curve.rate,
            compounding=discount_curve.compounding,
        )

    @classmethod
    def bump_spot(cls,
        spot: float,
        model: BlackScholesModel,
        discount_curve: FlatDiscountCurve,
        shift: float,
    ) -> BumpedBlackScholesInputs:
        """Return fresh inputs with the spot shifted additively."""
        return BumpedBlackScholesInputs(
            spot=spot + shift,
            model=cls._copy_model(model),
            discount_curve=cls._copy_curve(discount_curve),
        )

    @classmethod
    def bump_volatility(cls,
        spot: float,
        model: BlackScholesModel,
        discount_curve: FlatDiscountCurve,
        shift: float,
    ) -> BumpedBlackScholesInputs:
        """Return fresh inputs with the Black-Scholes volatility shifted."""
        return BumpedBlackScholesInputs(
            spot=spot,
            model=BlackScholesModel(
                rate=model.rate,
                volatility=model.volatility + shift,
                dividend_yield=model.dividend_yield,
            ),
            discount_curve=cls._copy_curve(discount_curve),
        )

    @classmethod
    def bump_rate(cls,
        spot: float,
        model: BlackScholesModel,
        discount_curve: FlatDiscountCurve,
        shift: float,
    ) -> BumpedBlackScholesInputs:
        """Return fresh inputs with both model drift and discount rate shifted."""
        return BumpedBlackScholesInputs(
            spot=spot,
            model=BlackScholesModel(
                rate=model.rate + shift,
                volatility=model.volatility,
                dividend_yield=model.dividend_yield,
            ),
            discount_curve=FlatDiscountCurve(
                rate=discount_curve.rate + shift,
                compounding=discount_curve.compounding,
            ),
        )


# Compatibility names refer directly to class methods.
_copy_model = BlackScholesBumps._copy_model
_copy_curve = BlackScholesBumps._copy_curve
bump_spot = BlackScholesBumps.bump_spot
bump_volatility = BlackScholesBumps.bump_volatility
bump_rate = BlackScholesBumps.bump_rate
