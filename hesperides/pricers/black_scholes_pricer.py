"""Pricer services that wire Black-Scholes contracts, market data, and engines."""

from __future__ import annotations

from hesperides.validation_protocols import BlackScholesValidation

from hesperides.pricers.protocols import EuropeanPricing, AsianPricing, BlackScholesFactory

from dataclasses import dataclass, field

from hesperides.contracts.european_option import EuropeanOption
from hesperides.contracts.geometric_asian_option import GeometricAsianOption
from hesperides.engines.black_scholes import (
    BlackScholesEuropeanAnalyticalEngine,
    BlackScholesEuropeanMonteCarloEngine,
    BlackScholesGeometricAsianAnalyticalEngine,
    BlackScholesGeometricAsianMonteCarloEngine,
)
from hesperides.engines.protocols import (
    BlackScholesAnalyticalPricingEngine,
    GeometricAsianAnalyticalPricingEngine,
    BlackScholesMonteCarloPricingEngine,
    BlackScholesPathMonteCarloPricingEngine,
)
from hesperides.market.curves import Rate, as_discount_curve
from hesperides.market.protocols import DiscountCurve
from hesperides.models.black_scholes import BlackScholesModel
from hesperides.validation import BlackScholesInputValidator


@dataclass(frozen=True)
class BlackScholesComponents:
    """Reusable Black-Scholes pricing components for one market snapshot."""

    model: BlackScholesModel
    discount_curve: DiscountCurve


@dataclass(frozen=True)
class BlackScholesComponentFactory:
    """Create model and market objects from public Black-Scholes inputs."""

    def create(
        self,
        rate: Rate,
        volatility: float,
        dividend_yield: Rate = 0.0,
    ) -> BlackScholesComponents:
        """Return components shared by all Black-Scholes engines."""
        for name in ("valuation_date", "basis"):
            domestic = getattr(rate, name, None)
            foreign = getattr(dividend_yield, name, None)
            if domestic is not None and foreign is not None and domestic != foreign:
                raise ValueError(f"Curves must share {name}")
        return BlackScholesComponents(
            model=BlackScholesModel(
                rate=rate,
                volatility=volatility,
                dividend_yield=dividend_yield,
            ),
            discount_curve=as_discount_curve(rate),
        )


@dataclass(frozen=True)
class BlackScholesEuropeanPricer:
    """Application service for European option pricing under Black-Scholes."""

    validator: BlackScholesValidation = field(
        default_factory=BlackScholesInputValidator
    )
    component_factory: BlackScholesFactory = field(
        default_factory=BlackScholesComponentFactory
    )
    analytical_engine: BlackScholesAnalyticalPricingEngine = field(
        default_factory=BlackScholesEuropeanAnalyticalEngine
    )
    monte_carlo_engine: BlackScholesMonteCarloPricingEngine = field(
        default_factory=BlackScholesEuropeanMonteCarloEngine
    )

    def price(
        self,
        spot: float,
        strike: float,
        maturity: float,
        rate: Rate,
        volatility: float,
        is_call: bool,
        engine: str = "analytical",
        n_paths: int | None = None,
        seed: int | None = None,
        dividend_yield: Rate = 0.0,
        require_positive_volatility: bool = False,
        spot_name: str = "St",
    ) -> float:
        """Price a European option under Black-Scholes."""
        self.validator.validate_engine(engine)
        self.validator.validate_market_inputs(
            spot,
            strike,
            maturity,
            volatility,
            spot_name=spot_name,
            require_positive_volatility=require_positive_volatility,
        )

        option = EuropeanOption(strike=strike, is_call=is_call)
        components = self.component_factory.create(
            rate=rate,
            volatility=volatility,
            dividend_yield=dividend_yield,
        )

        if engine == "analytical":
            return self.analytical_engine.price(
                spot=spot,
                option=option,
                maturity=maturity,
                model=components.model,
                discount_curve=components.discount_curve,
            )

        required_n_paths = self.validator.require_positive_integer(
            n_paths,
            "n_paths",
        )
        return self.monte_carlo_engine.price(
            spot=spot,
            option=option,
            maturity=maturity,
            model=components.model,
            discount_curve=components.discount_curve,
            n_paths=required_n_paths,
            seed=seed,
        )


@dataclass(frozen=True)
class BlackScholesGeometricAsianPricer:
    """Application service for geometric Asian pricing under Black-Scholes."""

    validator: BlackScholesValidation = field(
        default_factory=BlackScholesInputValidator
    )
    component_factory: BlackScholesFactory = field(
        default_factory=BlackScholesComponentFactory
    )
    analytical_engine: GeometricAsianAnalyticalPricingEngine = field(
        default_factory=BlackScholesGeometricAsianAnalyticalEngine
    )
    monte_carlo_engine: BlackScholesPathMonteCarloPricingEngine = field(
        default_factory=BlackScholesGeometricAsianMonteCarloEngine
    )

    def price(
        self,
        spot: float,
        strike: float,
        maturity: float,
        rate: Rate,
        volatility: float,
        is_call: bool,
        engine: str = "analytical",
        n_paths: int | None = None,
        n_steps: int | None = None,
        seed: int | None = None,
    ) -> float:
        """Price a geometric Asian option under Black-Scholes."""
        self.validator.validate_engine(engine)
        self.validator.validate_market_inputs(spot, strike, maturity, volatility)

        option = GeometricAsianOption(strike=strike, is_call=is_call)
        components = self.component_factory.create(rate=rate, volatility=volatility)

        if engine == "analytical":
            return self.analytical_engine.price(
                spot=spot,
                option=option,
                maturity=maturity,
                model=components.model,
                discount_curve=components.discount_curve,
            )

        required_n_paths = self.validator.require_positive_integer(
            n_paths,
            "n_paths",
        )
        required_n_steps = self.validator.require_positive_integer(
            n_steps,
            "n_steps",
        )
        return self.monte_carlo_engine.price(
            spot=spot,
            option=option,
            maturity=maturity,
            model=components.model,
            discount_curve=components.discount_curve,
            n_paths=required_n_paths,
            n_steps=required_n_steps,
            seed=seed,
        )


@dataclass(frozen=True)
class BlackScholesPricingService:
    """Adapt product inputs to the injected Black-Scholes pricers."""

    european: EuropeanPricing = field(default_factory=BlackScholesEuropeanPricer)
    asian: AsianPricing = field(default_factory=BlackScholesGeometricAsianPricer)

    def price_bs_european(self,
        spot: float,
        strike: float,
        maturity: float,
        rate: Rate,
        volatility: float,
        is_call: bool,
        engine: str = "analytical",
        n_paths: int | None = None,
        seed: int | None = None,
        dividend_yield: Rate = 0.0,
    ) -> float:
        """Price a European option under Black-Scholes."""
        return self.european.price(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=volatility,
            is_call=is_call,
            engine=engine,
            n_paths=n_paths,
            seed=seed,
            dividend_yield=dividend_yield,
        )

    def price_bs_european_dividend(self,
        spot: float,
        strike: float,
        maturity: float,
        rate: Rate,
        volatility: float,
        is_call: bool,
        dividend_yield: Rate = 0.0,
        engine: str = "analytical",
        n_paths: int | None = None,
        seed: int | None = None,
        spot_name: str = "St",
    ) -> float:
        """Price a European option with continuous yield q under Black-Scholes."""
        return self.european.price(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=volatility,
            is_call=is_call,
            engine=engine,
            n_paths=n_paths,
            seed=seed,
            dividend_yield=dividend_yield,
            require_positive_volatility=True,
            spot_name=spot_name,
        )

    def price_fx_option(self,
        spot: float,
        strike: float,
        maturity: float,
        domestic_rate: Rate,
        foreign_rate: Rate,
        volatility: float,
        is_call: bool,
        engine: str = "analytical",
        n_paths: int | None = None,
        seed: int | None = None,
    ) -> float:
        """Price an FX option via the cost-of-carry Black-Scholes model."""
        return self.price_bs_european_dividend(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=domestic_rate,
            volatility=volatility,
            is_call=is_call,
            dividend_yield=foreign_rate,
            engine=engine,
            n_paths=n_paths,
            seed=seed,
        )

    def price_future_option(self,
        future: float,
        strike: float,
        maturity: float,
        rate: Rate,
        volatility: float,
        is_call: bool,
        engine: str = "analytical",
        n_paths: int | None = None,
        seed: int | None = None,
    ) -> float:
        """Price a futures option via Black-Scholes with q = r."""
        return self.price_bs_european_dividend(
            spot=future,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=volatility,
            is_call=is_call,
            dividend_yield=rate,
            engine=engine,
            n_paths=n_paths,
            seed=seed,
            spot_name="F0",
        )

    def price_bs_geometric_asian(self,
        spot: float,
        strike: float,
        maturity: float,
        rate: Rate,
        volatility: float,
        is_call: bool,
        engine: str = "analytical",
        n_paths: int | None = None,
        n_steps: int | None = None,
        seed: int | None = None,
    ) -> float:
        """Price a geometric Asian option under Black-Scholes."""
        return self.asian.price(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=volatility,
            is_call=is_call,
            engine=engine,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
        )


# Compatibility names refer directly to bound methods.
_DEFAULT_SERVICE = BlackScholesPricingService()
price_bs_european = _DEFAULT_SERVICE.price_bs_european
price_bs_european_dividend = _DEFAULT_SERVICE.price_bs_european_dividend
price_fx_option = _DEFAULT_SERVICE.price_fx_option
price_future_option = _DEFAULT_SERVICE.price_future_option
price_bs_geometric_asian = _DEFAULT_SERVICE.price_bs_geometric_asian
