"""Discount curves for deterministic rate models."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log
from numbers import Real

from hesperides.market.protocols import DiscountCurve

Rate = float | DiscountCurve


class RateAdapter:
    """Adapt scalar rates and deterministic discount curves."""

    @classmethod
    def as_discount_curve(cls, rate: Rate, compounding: str = "continuous") -> DiscountCurve:
        """Accept a scalar rate or any curve exposing discount(time)."""
        if isinstance(rate, Real):
            if not isfinite(rate):
                raise ValueError("Rate must be finite")
            if compounding == "discrete" and rate <= -1:
                raise ValueError("Discrete rate must be greater than -1")
            return FlatDiscountCurve(float(rate), compounding)
        if not callable(getattr(rate, "discount", None)):
            raise TypeError("Expected a rate or a curve with discount(time)")
        if rate.discount(0.0) != 1.0:
            raise ValueError("Curve must have P(0)=1")
        return rate

    @classmethod
    def integrated_rate(cls, rate: Rate, time: float) -> float:
        """Integral of deterministic rates: -log(P(0,t))."""
        if not isfinite(time) or time < 0:
            raise ValueError("Time must be finite and nonnegative")
        if isinstance(rate, Real):
            if not isfinite(rate):
                raise ValueError("Rate must be finite")
            return float(rate) * time
        factor = cls.as_discount_curve(rate).discount(time)
        if not isfinite(factor) or factor <= 0:
            raise ValueError("Discount factors must be positive and finite")
        return -log(factor)

    @classmethod
    def equivalent_rate(cls, rate: Rate, time: float) -> float:
        """Exact terminal equivalent for European payoffs under deterministic rates."""
        integral = cls.integrated_rate(rate, time)
        return integral / time if time else 0.0


# Compatibility names refer directly to class methods.
as_discount_curve = RateAdapter.as_discount_curve
integrated_rate = RateAdapter.integrated_rate
equivalent_rate = RateAdapter.equivalent_rate


@dataclass(frozen=True)
class TimeScaledCurve:
    """Map binomial steps to the time unit used by the supplied curve."""

    curve: DiscountCurve
    step_years: float

    def __post_init__(self):
        if not isfinite(self.step_years) or self.step_years <= 0:
            raise ValueError("step_years must be positive and finite")

    def discount(self, time: float) -> float:
        return self.curve.discount(time * self.step_years)


@dataclass(frozen=True)
class FlatDiscountCurve:
    """Flat (constant) interest-rate discount curve."""

    rate: float
    compounding: str = "continuous"

    def __post_init__(self) -> None:
        if self.compounding not in {"continuous", "discrete"}:
            raise ValueError("compounding must be 'continuous' or 'discrete'")

    def discount(self, time: float) -> float:
        """Discount factor for the given time or compounding periods.

        Continuous compounding uses P(0,T) = exp(-r*T).  Discrete
        compounding is kept for the earlier binomial assignment.
        """
        if self.compounding == "continuous":
            return exp(-self.rate * time)
        return (1.0 + self.rate) ** (-time)
