"""Public API facade for the Hesperides pricing library.

All external consumers (tests, notebooks, grading scripts) import from
here.  No financial logic lives in this module – it only delegates.
"""

from __future__ import annotations

from hesperides.pricers.protocols import (
    BinomialPricing,
    ProductPricing,
    GreekPricing,
    HeatSolving,
    HeatPricing,
    ArbitrageCalculation,
)

from dataclasses import dataclass, field
import numpy as np
from hesperides.pricers.european_option_pricer import BinomialEuropeanPricer
from hesperides.pricers.black_scholes_pricer import BlackScholesPricingService
from hesperides.pricers.black_scholes_greek_pricer import BlackScholesEuropeanGreekPricer
from hesperides.pricers.heat_equation_pricer import HeatEquationSolver, BlackScholesHeatPricer
from hesperides.market.static_arbitrage import StaticArbitrageCalculator
from hesperides.market.curves import Rate, TimeScaledCurve, RateAdapter


@dataclass(frozen=True)
class HesperidesAPI:
    """Public valuation facade preserving legacy argument names."""

    binomial: BinomialPricing = field(default_factory=BinomialEuropeanPricer)
    black_scholes: ProductPricing = field(default_factory=BlackScholesPricingService)
    greeks: GreekPricing = field(default_factory=BlackScholesEuropeanGreekPricer)
    heat_solver: HeatSolving = field(default_factory=HeatEquationSolver)
    heat_pricer: HeatPricing = field(default_factory=BlackScholesHeatPricer)
    arbitrage: ArbitrageCalculation = field(default_factory=StaticArbitrageCalculator)


    def get_price_binomial_european(self,
        St: float,
        K: float,
        T: int,
        R: Rate,
        u: float,
        d: float,
        call: bool,
        *,
        step_years: float = 1.0,
    ) -> float:
        """Price a European call or put option using the binomial model.
    PARAMETERS:
        St : float   Current spot price of the underlying.
        K  : float   Strike price.
        T  : int     Number of remaining time steps to maturity.
        R  : float   Per-period risk-free rate.
        u  : float   Up factor per step.
        d  : float   Down factor per step.
        call : bool  True for a call, False for a put.

        Returns
        -------
        float
            Present value of the option.
        """
        return float(
            self.binomial.price(
                spot=St,
                strike=K,
                is_call=call,
                maturity_steps=T,
                up=u,
                down=d,
                rate=TimeScaledCurve(RateAdapter.as_discount_curve(R, "discrete"), step_years),
            )
        )

    def get_price_bs_european(self,
        St: float,
        K: float,
        T: float,
        r: Rate,
        sigma: float,
        call: bool,
        engine: str = "analytical",
        n_paths: int | None = None,
        seed: int | None = None,
    ) -> float:
        """
        Price a European option (call or put) under Black-Scholes.
        """
        return float(
            self.black_scholes.price_bs_european(
                spot=St,
                strike=K,
                maturity=T,
                rate=r,
                volatility=sigma,
                is_call=call,
                engine=engine,
                n_paths=n_paths,
                seed=seed,
            )
        )

    def get_price_bs_european_dividend(self,
        St: float,
        K: float,
        T: float,
        r: Rate,
        sigma: float,
        call: bool,
        q: Rate = 0.0,
        engine: str = "analytical",
        n_paths: int | None = None,
        seed: int | None = None,
    ) -> float:
        """
        Price a European option on a stock paying a continuous dividend yield q.

        Risk-neutral dynamics dS = (r - q) S dt + sigma S dW; discounting still
        uses r. With q = 0 this recovers Assignment 3.
        """
        return float(
            self.black_scholes.price_bs_european_dividend(
                spot=St,
                strike=K,
                maturity=T,
                rate=r,
                volatility=sigma,
                is_call=call,
                dividend_yield=q,
                engine=engine,
                n_paths=n_paths,
                seed=seed,
            )
        )

    def get_price_fx_option(self,
        St: float,
        K: float,
        T: float,
        r_d: Rate,
        r_f: Rate,
        sigma: float,
        call: bool,
        engine: str = "analytical",
        n_paths: int | None = None,
        seed: int | None = None,
    ) -> float:
        """
        Price a European FX option under the Garman-Kohlhagen model.

        St is domestic per unit foreign. Internally this is the cost-of-carry
        Black-Scholes model with q = r_f and r = r_d.
        """
        return float(
            self.black_scholes.price_fx_option(
                spot=St,
                strike=K,
                maturity=T,
                domestic_rate=r_d,
                foreign_rate=r_f,
                volatility=sigma,
                is_call=call,
                engine=engine,
                n_paths=n_paths,
                seed=seed,
            )
        )

    def get_price_future_option(self,
        F0: float,
        K: float,
        T: float,
        r: Rate,
        sigma: float,
        call: bool,
        engine: str = "analytical",
        n_paths: int | None = None,
        seed: int | None = None,
    ) -> float:
        """
        Price a European option on a future under the Black-76 model.

        The future is driftless under Q, so this reuses Black-Scholes with q = r;
        discounting still uses r.
        """
        return float(
            self.black_scholes.price_future_option(
                future=F0,
                strike=K,
                maturity=T,
                rate=r,
                volatility=sigma,
                is_call=call,
                engine=engine,
                n_paths=n_paths,
                seed=seed,
            )
        )

    def solve_heat_equation(self,
        initial_condition,
        kappa: float,
        M: float,
        T: float,
        n_x: int,
        n_t: int,
        scheme: str = "explicit",
        left_boundary=0.0,
        right_boundary=0.0,
        boundary_type: str = "dirichlet",
    ):
        """
        Solve v_t = kappa * v_xx on [0, M] x (0, T] by finite differences.

        Returns the spatial grid and the numerical solution at time T.
        """
        return self.heat_solver.solve(
            initial_condition=initial_condition,
            kappa=kappa,
            domain_size=M,
            maturity=T,
            n_x=n_x,
            n_t=n_t,
            scheme=scheme,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            boundary_type=boundary_type,
        )

    def get_price_bs_european_heat(self,
        St: float,
        K: float,
        T: float,
        r: Rate,
        sigma: float,
        call: bool,
        n_x: int = 400,
        n_t: int = 400,
        scheme: str = "implicit",
    ) -> float:
        """
        Price a European call/put by reducing Black-Scholes to the heat equation.
        """
        return float(
            self.heat_pricer.price(
                spot=St,
                strike=K,
                maturity=T,
                rate=r,
                volatility=sigma,
                is_call=call,
                n_x=n_x,
                n_t=n_t,
                scheme=scheme,
            )
        )

    def get_price_bs_geometric_asian(self,
        St: float,
        K: float,
        T: float,
        r: Rate,
        sigma: float,
        call: bool,
        engine: str = "analytical",
        n_paths: int | None = None,
        n_steps: int | None = None,
        seed: int | None = None,
    ) -> float:
        """
        Price a geometric Asian option (call or put) under Black-Scholes.
        """
        return float(
            self.black_scholes.price_bs_geometric_asian(
                spot=St,
                strike=K,
                maturity=T,
                rate=r,
                volatility=sigma,
                is_call=call,
                engine=engine,
                n_paths=n_paths,
                n_steps=n_steps,
                seed=seed,
            )
        )

    def get_greek_bs_european(self,
        St: float,
        K: float,
        T: float,
        r: Rate,
        sigma: float,
        call: bool,
        greek: str,
        engine: str = "analytical",
        greek_engine: str = "analytical",
        fd_scheme: str = "central",
        h: float | None = None,
        n_paths: int | None = None,
        seed: int | None = None,
    ) -> float:
        """
        Compute a Greek of a European call/put under Black-Scholes.
        """
        return float(
            self.greeks.greek(
                spot=St,
                strike=K,
                maturity=T,
                rate=r,
                volatility=sigma,
                is_call=call,
                greek=greek,
                engine=engine,
                greek_engine=greek_engine,
                fd_scheme=fd_scheme,
                h=h,
                n_paths=n_paths,
                seed=seed,
            )
        )


    def compute_static_arbitrage_quantity(self, surface: np.ndarray, strikes: np.ndarray | None = None,
                                         quantity: str = "vertical") -> np.ndarray:
        return self.arbitrage.compute(surface, strikes, quantity)


# Compatibility names refer directly to bound methods.
_DEFAULT_API = HesperidesAPI()
compute_static_arbitrage_quantity = _DEFAULT_API.compute_static_arbitrage_quantity
get_price_binomial_european = _DEFAULT_API.get_price_binomial_european
get_price_bs_european = _DEFAULT_API.get_price_bs_european
get_price_bs_european_dividend = _DEFAULT_API.get_price_bs_european_dividend
get_price_fx_option = _DEFAULT_API.get_price_fx_option
get_price_future_option = _DEFAULT_API.get_price_future_option
solve_heat_equation = _DEFAULT_API.solve_heat_equation
get_price_bs_european_heat = _DEFAULT_API.get_price_bs_european_heat
get_price_bs_geometric_asian = _DEFAULT_API.get_price_bs_geometric_asian
get_greek_bs_european = _DEFAULT_API.get_greek_bs_european
