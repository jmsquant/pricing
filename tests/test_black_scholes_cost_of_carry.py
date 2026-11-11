"""Tests for Assignment 5: dividends, FX, and futures under cost of carry."""

from __future__ import annotations

from functools import partial

from math import exp

import pytest

import hesperides.api as hapi


SPOT = 100.0
STRIKE = 100.0
MATURITY = 1.0
RATE = 0.05
VOLATILITY = 0.20
DIVIDEND_YIELD = 0.02


class TestBlackScholesCostOfCarry:
    @pytest.mark.parametrize("call", [True, False])
    def test_dividend_zero_yield_recovers_assignment_3_price(self, call: bool) -> None:
        dividend_price = hapi.get_price_bs_european_dividend(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            call,
            q=0.0,
        )
        plain_price = hapi.get_price_bs_european(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            call,
        )

        assert dividend_price == plain_price

    @pytest.mark.parametrize(
        ("call", "expected"),
        [
            (True, 9.227005508154036),
            (False, 6.330080627549918),
        ],
    )
    def test_dividend_analytical_closed_form_regression(self,
        call: bool,
        expected: float,
    ) -> None:
        price = hapi.get_price_bs_european_dividend(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            call,
            q=DIVIDEND_YIELD,
        )

        assert price == pytest.approx(expected, abs=1e-12)

    def test_dividend_put_call_parity(self) -> None:
        call = hapi.get_price_bs_european_dividend(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            True,
            q=DIVIDEND_YIELD,
        )
        put = hapi.get_price_bs_european_dividend(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            False,
            q=DIVIDEND_YIELD,
        )

        expected = SPOT * exp(-DIVIDEND_YIELD * MATURITY) - STRIKE * exp(
            -RATE * MATURITY
        )
        assert call - put == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize("call", [True, False])
    def test_dividend_reduction_formula_matches_shifted_spot(self, call: bool) -> None:
        dividend_price = hapi.get_price_bs_european_dividend(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            call,
            q=DIVIDEND_YIELD,
        )
        reduced_price = hapi.get_price_bs_european(
            SPOT * exp(-DIVIDEND_YIELD * MATURITY),
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            call,
        )

        assert dividend_price == pytest.approx(reduced_price, abs=1e-12)

    @pytest.mark.parametrize("call", [True, False])
    def test_fx_option_matches_stock_with_foreign_rate_as_yield(self, call: bool) -> None:
        fx_price = hapi.get_price_fx_option(
            1.15,
            1.12,
            0.75,
            r_d=0.035,
            r_f=0.015,
            sigma=0.22,
            call=call,
        )
        dividend_price = hapi.get_price_bs_european_dividend(
            1.15,
            1.12,
            0.75,
            r=0.035,
            sigma=0.22,
            call=call,
            q=0.015,
        )

        assert fx_price == pytest.approx(dividend_price, abs=1e-12)

    @pytest.mark.parametrize(
        ("call", "expected"),
        [
            (True, 0.10961159080129468),
            (False, 0.0634591101339721),
        ],
    )
    def test_fx_closed_form_regression_and_zero_foreign_rate_sanity(self,
        call: bool,
        expected: float,
    ) -> None:
        fx_price = hapi.get_price_fx_option(
            1.15,
            1.12,
            0.75,
            r_d=0.035,
            r_f=0.015,
            sigma=0.22,
            call=call,
        )
        zero_foreign_price = hapi.get_price_fx_option(
            SPOT,
            STRIKE,
            MATURITY,
            r_d=RATE,
            r_f=0.0,
            sigma=VOLATILITY,
            call=call,
        )
        plain_price = hapi.get_price_bs_european(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            call,
        )

        assert fx_price == pytest.approx(expected, abs=1e-12)
        assert zero_foreign_price == plain_price

    @pytest.mark.parametrize(
        ("call", "expected"),
        [
            (True, 8.568181273489337),
            (False, 12.373098971492198),
        ],
    )
    def test_future_option_black_76_closed_form_regression(self,
        call: bool,
        expected: float,
    ) -> None:
        price = hapi.get_price_future_option(
            96.0,
            100.0,
            1.25,
            r=0.04,
            sigma=0.25,
            call=call,
        )

        assert price == pytest.approx(expected, abs=1e-12)

    def test_future_option_put_call_parity_and_dividend_symmetry(self) -> None:
        call = hapi.get_price_future_option(
            96.0,
            100.0,
            1.25,
            r=0.04,
            sigma=0.25,
            call=True,
        )
        put = hapi.get_price_future_option(
            96.0,
            100.0,
            1.25,
            r=0.04,
            sigma=0.25,
            call=False,
        )
        dividend_call = hapi.get_price_bs_european_dividend(
            96.0,
            100.0,
            1.25,
            r=0.04,
            sigma=0.25,
            call=True,
            q=0.04,
        )

        assert call - put == pytest.approx(exp(-0.04 * 1.25) * (96.0 - 100.0), abs=1e-12)
        assert call == pytest.approx(dividend_call, abs=1e-12)

    @pytest.mark.parametrize("call", [True, False])
    def test_future_zero_rate_reduces_to_plain_european_price(self, call: bool) -> None:
        future_price = hapi.get_price_future_option(
            SPOT,
            STRIKE,
            MATURITY,
            r=0.0,
            sigma=VOLATILITY,
            call=call,
        )
        plain_price = hapi.get_price_bs_european(
            SPOT,
            STRIKE,
            MATURITY,
            r=0.0,
            sigma=VOLATILITY,
            call=call,
        )

        assert future_price == plain_price

    @pytest.mark.parametrize(
        ("api_call", "analytical"),
        [
            (
                partial(hapi.get_price_bs_european_dividend,
                    SPOT,
                    STRIKE,
                    MATURITY,
                    RATE,
                    VOLATILITY,
                    True,
                    q=DIVIDEND_YIELD,
                    engine="mc",
                    n_paths=100_000,
                    seed=1234,
                ),
                partial(hapi.get_price_bs_european_dividend,
                    SPOT,
                    STRIKE,
                    MATURITY,
                    RATE,
                    VOLATILITY,
                    True,
                    q=DIVIDEND_YIELD,
                ),
            ),
            (
                partial(hapi.get_price_future_option,
                    96.0,
                    100.0,
                    1.25,
                    r=0.04,
                    sigma=0.25,
                    call=True,
                    engine="mc",
                    n_paths=100_000,
                    seed=4321,
                ),
                partial(hapi.get_price_future_option,
                    96.0,
                    100.0,
                    1.25,
                    r=0.04,
                    sigma=0.25,
                    call=True,
                ),
            ),
        ],
    )
    def test_cost_of_carry_monte_carlo_matches_analytical(self,
        api_call,
        analytical,
    ) -> None:
        assert api_call() == pytest.approx(analytical(), abs=0.15)

    def test_negative_rates_and_yields_are_allowed(self) -> None:
        price = hapi.get_price_fx_option(
            1.15,
            1.12,
            0.75,
            r_d=-0.005,
            r_f=-0.015,
            sigma=0.22,
            call=True,
        )

        assert price > 0.0

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"St": 0.0}, "St"),
            ({"K": 0.0}, "K"),
            ({"T": -1.0}, "T"),
            ({"sigma": 0.0}, "sigma"),
            ({"sigma": -0.1}, "sigma"),
        ],
    )
    def test_dividend_api_rejects_invalid_market_parameters(self,
        kwargs: dict[str, float],
        message: str,
    ) -> None:
        params = {
            "St": SPOT,
            "K": STRIKE,
            "T": MATURITY,
            "r": RATE,
            "sigma": VOLATILITY,
            "call": True,
            "q": DIVIDEND_YIELD,
        }
        params.update(kwargs)
        with pytest.raises(ValueError, match=message):
            hapi.get_price_bs_european_dividend(**params)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"St": 0.0}, "St"),
            ({"K": 0.0}, "K"),
            ({"T": -1.0}, "T"),
            ({"sigma": 0.0}, "sigma"),
        ],
    )
    def test_fx_api_rejects_invalid_market_parameters(self,
        kwargs: dict[str, float],
        message: str,
    ) -> None:
        params = {
            "St": 1.15,
            "K": 1.12,
            "T": 0.75,
            "r_d": 0.035,
            "r_f": 0.015,
            "sigma": 0.22,
            "call": True,
        }
        params.update(kwargs)
        with pytest.raises(ValueError, match=message):
            hapi.get_price_fx_option(**params)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"F0": 0.0}, "F0"),
            ({"K": 0.0}, "K"),
            ({"T": -1.0}, "T"),
            ({"sigma": 0.0}, "sigma"),
        ],
    )
    def test_future_api_rejects_invalid_market_parameters(self,
        kwargs: dict[str, float],
        message: str,
    ) -> None:
        params = {
            "F0": 96.0,
            "K": 100.0,
            "T": 1.25,
            "r": 0.04,
            "sigma": 0.25,
            "call": True,
        }
        params.update(kwargs)
        with pytest.raises(ValueError, match=message):
            hapi.get_price_future_option(**params)

    @pytest.mark.parametrize(
        "factory",
        [
            partial(hapi.get_price_bs_european_dividend,
                SPOT,
                STRIKE,
                MATURITY,
                RATE,
                VOLATILITY,
                True,
                q=DIVIDEND_YIELD,
            ),
            partial(hapi.get_price_fx_option,
                1.15,
                1.12,
                0.75,
                r_d=0.035,
                r_f=0.015,
                sigma=0.22,
                call=True,
            ),
            partial(hapi.get_price_future_option,
                96.0,
                100.0,
                1.25,
                r=0.04,
                sigma=0.25,
                call=True,
            ),
        ],
    )
    def test_new_apis_reject_bad_engine_and_missing_mc_paths(self, factory) -> None:
        with pytest.raises(ValueError, match="engine"):
            factory(engine="tree")
        with pytest.raises(ValueError, match="n_paths"):
            factory(engine="mc")
        with pytest.raises(ValueError, match="n_paths"):
            factory(engine="mc", n_paths=0)
