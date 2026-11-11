"""Orchestrator that wires contracts, market data, model, and engine."""

from __future__ import annotations

from hesperides.pricers.protocols import BinomialComponentFactory

from dataclasses import dataclass, field

from hesperides.contracts.european_option import EuropeanOption
from hesperides.engines.binomial_european import BinomialEuropeanEngine
from hesperides.engines.protocols import BinomialEuropeanPricingEngine
from hesperides.market.curves import Rate, as_discount_curve
from hesperides.market.protocols import DiscountCurve
from hesperides.models.binomial import BinomialModel


@dataclass(frozen=True)
class BinomialEuropeanComponents:
    """Objects required to price a European option with a binomial model."""

    option: EuropeanOption
    model: BinomialModel
    discount_curve: DiscountCurve


@dataclass(frozen=True)
class BinomialEuropeanComponentFactory:
    """Create binomial pricing components from public input values."""

    def create(
        self,
        strike: float,
        is_call: bool,
        up: float,
        down: float,
        rate: Rate,
    ) -> BinomialEuropeanComponents:
        curve = as_discount_curve(rate, compounding="discrete")
        model = BinomialModel(up=up, down=down, rate=1 / curve.discount(1) - 1)
        model.validate()
        return BinomialEuropeanComponents(
            option=EuropeanOption(strike=strike, is_call=is_call),
            model=model,
            discount_curve=curve,
        )


@dataclass(frozen=True)
class BinomialEuropeanPricer:
    """Application service that orchestrates binomial European pricing."""

    engine: BinomialEuropeanPricingEngine = field(
        default_factory=BinomialEuropeanEngine
    )
    component_factory: BinomialComponentFactory = field(
        default_factory=BinomialEuropeanComponentFactory
    )

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
        """Build internal objects and return the European option price."""
        if maturity_steps == 0:
            return float(EuropeanOption(strike=strike, is_call=is_call).payoff(spot))
        components = self.component_factory.create(
            strike=strike,
            is_call=is_call,
            up=up,
            down=down,
            rate=rate,
        )
        return self.engine.price(
            spot=spot,
            option=components.option,
            maturity_steps=maturity_steps,
            model=components.model,
            discount_curve=components.discount_curve,
        )


_DEFAULT_PRICER = BinomialEuropeanPricer()


price_european_option = _DEFAULT_PRICER.price
