"""Pricer service for Black-Scholes European option Greeks."""

from __future__ import annotations

from hesperides.validation_protocols import GreekValidation

from hesperides.engines.solver_protocols import AnalyticalGreekEngine, FiniteDifferenceGreekEngine

from hesperides.pricers.protocols import BlackScholesFactory

from dataclasses import dataclass, field

from hesperides.contracts.european_option import EuropeanOption
from hesperides.engines.black_scholes import (
    BlackScholesEuropeanAnalyticalEngine,
    BlackScholesEuropeanMonteCarloEngine,
)
from hesperides.greeks.black_scholes import (
    BlackScholesEuropeanAnalyticalGreekEngine,
    BlackScholesEuropeanFiniteDifferenceGreekEngine,
)
from hesperides.market.curves import FlatDiscountCurve, Rate, equivalent_rate
from hesperides.models.black_scholes import BlackScholesModel
from hesperides.pricers.black_scholes_pricer import BlackScholesComponentFactory
from hesperides.validation import BlackScholesGreekInputValidator
from hesperides.engines.protocols import (
    BlackScholesAnalyticalPricingEngine,
    BlackScholesMonteCarloPricingEngine,
)


@dataclass(frozen=True)
class AnalyticalRepricing:
    engine: BlackScholesAnalyticalPricingEngine
    option: EuropeanOption
    maturity: float

    def __call__(self, spot, model, discount_curve) -> float:
        return self.engine.price(spot=spot, option=self.option, maturity=self.maturity,
                                 model=model, discount_curve=discount_curve)


@dataclass(frozen=True)
class MonteCarloRepricing:
    engine: BlackScholesMonteCarloPricingEngine
    option: EuropeanOption
    maturity: float
    n_paths: int
    seed: int

    def __call__(self, spot, model, discount_curve) -> float:
        return self.engine.price(spot=spot, option=self.option, maturity=self.maturity,
                                 model=model, discount_curve=discount_curve,
                                 n_paths=self.n_paths, seed=self.seed)


@dataclass(frozen=True)
class BlackScholesEuropeanGreekPricer:
    """Application service for European option Greeks under Black-Scholes."""

    validator: GreekValidation = field(
        default_factory=BlackScholesGreekInputValidator
    )
    component_factory: BlackScholesFactory = field(
        default_factory=BlackScholesComponentFactory
    )
    analytical_pricing_engine: BlackScholesAnalyticalPricingEngine = field(
        default_factory=BlackScholesEuropeanAnalyticalEngine
    )
    monte_carlo_pricing_engine: BlackScholesMonteCarloPricingEngine = field(
        default_factory=BlackScholesEuropeanMonteCarloEngine
    )
    analytical_greek_engine: AnalyticalGreekEngine = field(
        default_factory=BlackScholesEuropeanAnalyticalGreekEngine
    )
    finite_difference_greek_engine: FiniteDifferenceGreekEngine = (
        field(default_factory=BlackScholesEuropeanFiniteDifferenceGreekEngine)
    )

    def greek(
        self,
        spot: float,
        strike: float,
        maturity: float,
        rate: Rate,
        volatility: float,
        is_call: bool,
        greek: str,
        engine: str = "analytical",
        greek_engine: str = "analytical",
        fd_scheme: str = "central",
        h: float | None = None,
        n_paths: int | None = None,
        seed: int | None = None,
    ) -> float:
        """Compute a European option Greek under Black-Scholes."""
        self.validator.validate_greek(greek)
        self.validator.validate_engine(engine)
        self.validator.validate_greek_engine(greek_engine)
        self.validator.validate_greek_market_inputs(
            spot=spot,
            strike=strike,
            maturity=maturity,
            volatility=volatility,
        )

        option = EuropeanOption(strike=strike, is_call=is_call)
        # European Greeks depend only on P(0,T). Rho is a parallel continuous-rate shift.
        components = self.component_factory.create(rate=equivalent_rate(rate, maturity), volatility=volatility)

        if greek_engine == "analytical":
            return self.analytical_greek_engine.greek(
                spot=spot,
                option=option,
                maturity=maturity,
                model=components.model,
                discount_curve=components.discount_curve,
                greek=greek,
            )

        self.validator.validate_fd_scheme(fd_scheme)
        self.validator.validate_positive_bump(h)

        if engine == "analytical":
            return self.finite_difference_greek_engine.greek(
                spot=spot,
                option=option,
                maturity=maturity,
                model=components.model,
                discount_curve=components.discount_curve,
                greek=greek,
                fd_scheme=fd_scheme,
                h=h,
                price=AnalyticalRepricing(self.analytical_pricing_engine, option, maturity),
            )

        required_n_paths = self.validator.require_positive_integer(
            n_paths,
            "n_paths",
        )
        required_seed = self.validator.require_seed(seed)
        return self.finite_difference_greek_engine.greek(
            spot=spot,
            option=option,
            maturity=maturity,
            model=components.model,
            discount_curve=components.discount_curve,
            greek=greek,
            fd_scheme=fd_scheme,
            h=h,
            price=MonteCarloRepricing(self.monte_carlo_pricing_engine, option, maturity,
                                     required_n_paths, required_seed),
        )

_DEFAULT_EUROPEAN_GREEK_PRICER = BlackScholesEuropeanGreekPricer()


greek_bs_european = _DEFAULT_EUROPEAN_GREEK_PRICER.greek
