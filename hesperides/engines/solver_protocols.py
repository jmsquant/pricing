"""Greek engines and finite-difference solver contracts."""
from __future__ import annotations
from typing import Protocol
import numpy as np
from hesperides.contracts.protocols import OptionContract
from hesperides.contracts.initial_condition import InitialCondition
from hesperides.market.curves import FlatDiscountCurve
from hesperides.models.black_scholes import BlackScholesModel
from hesperides.models.heat_equation import HeatEquationModel
from hesperides.greeks.black_scholes import PricingFunction

class AnalyticalGreekEngine(Protocol):

    def greek(
        self,
        spot: float,
        option: OptionContract,
        maturity: float,
        model: BlackScholesModel,
        discount_curve: FlatDiscountCurve,
        greek: str,
    ) -> float:
        ...

class FiniteDifferenceGreekEngine(Protocol):

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
        ...

class HeatEquationEngine(Protocol):

    def solve(
        self,
        initial_condition: InitialCondition,
        model: HeatEquationModel,
        scheme: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        ...
