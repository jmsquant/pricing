"""Tests for Assignment 4: Black-Scholes European option Greeks."""

from __future__ import annotations

from math import exp

import pytest

import hesperides.api as hapi


SPOT = 100.0
STRIKE = 100.0
MATURITY = 1.0
RATE = 0.05
VOLATILITY = 0.20


class TestBlackScholesGreeks:
    @pytest.mark.parametrize(
        ("call", "greek", "expected"),
        [
            (True, "delta", 0.6368306511756191),
            (True, "gamma", 0.018762017345846895),
            (True, "vega", 37.52403469169379),
            (True, "rho", 53.232481545376345),
            (False, "delta", -0.3631693488243809),
            (False, "gamma", 0.018762017345846895),
            (False, "vega", 37.52403469169379),
            (False, "rho", -41.89046090469506),
        ],
    )
    def test_analytical_greeks_match_textbook_references(self,
        call: bool,
        greek: str,
        expected: float,
    ) -> None:
        actual = hapi.get_greek_bs_european(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            call,
            greek,
            greek_engine="analytical",
        )
        assert actual == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize("call", [True, False])
    @pytest.mark.parametrize("greek", ["delta", "gamma", "vega", "rho"])
    def test_fd_over_analytical_matches_closed_form_with_default_bumps(self,
        call: bool,
        greek: str,
    ) -> None:
        analytical = hapi.get_greek_bs_european(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            call,
            greek,
        )
        finite_difference = hapi.get_greek_bs_european(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            call,
            greek,
            engine="analytical",
            greek_engine="fd",
            fd_scheme="central",
        )
        assert finite_difference == pytest.approx(analytical, abs=2e-5)

    def test_fd_over_monte_carlo_uses_crn_and_matches_delta_reference(self) -> None:
        analytical = hapi.get_greek_bs_european(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            True,
            "delta",
        )
        finite_difference_mc = hapi.get_greek_bs_european(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            True,
            "delta",
            engine="mc",
            greek_engine="fd",
            n_paths=100_000,
            seed=123,
        )
        assert finite_difference_mc == pytest.approx(analytical, abs=0.02)

    def test_fd_over_monte_carlo_requires_seed(self) -> None:
        with pytest.raises(ValueError, match="seed"):
            hapi.get_greek_bs_european(
                SPOT,
                STRIKE,
                MATURITY,
                RATE,
                VOLATILITY,
                True,
                "delta",
                engine="mc",
                greek_engine="fd",
                n_paths=10_000,
            )

    def test_fd_delta_converges_with_expected_order_before_roundoff(self) -> None:
        analytical = hapi.get_greek_bs_european(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            True,
            "delta",
        )

        central_coarse = hapi.get_greek_bs_european(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            True,
            "delta",
            greek_engine="fd",
            fd_scheme="central",
            h=1.0,
        )
        central_fine = hapi.get_greek_bs_european(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            True,
            "delta",
            greek_engine="fd",
            fd_scheme="central",
            h=0.5,
        )
        forward_coarse = hapi.get_greek_bs_european(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            True,
            "delta",
            greek_engine="fd",
            fd_scheme="forward",
            h=2.0,
        )
        forward_fine = hapi.get_greek_bs_european(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            True,
            "delta",
            greek_engine="fd",
            fd_scheme="forward",
            h=1.0,
        )

        assert abs(central_fine - analytical) < abs(central_coarse - analytical) / 3.0
        assert abs(forward_fine - analytical) < abs(forward_coarse - analytical) / 1.8

    @pytest.mark.parametrize("greek_engine", ["analytical", "fd"])
    def test_call_put_parity_per_greek(self, greek_engine: str) -> None:
        call_delta = hapi.get_greek_bs_european(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            True,
            "delta",
            greek_engine=greek_engine,
        )
        put_delta = hapi.get_greek_bs_european(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            False,
            "delta",
            greek_engine=greek_engine,
        )
        assert call_delta - put_delta == pytest.approx(1.0, abs=1e-10)

        for greek in ("gamma", "vega"):
            call_value = hapi.get_greek_bs_european(
                SPOT,
                STRIKE,
                MATURITY,
                RATE,
                VOLATILITY,
                True,
                greek,
                greek_engine=greek_engine,
            )
            put_value = hapi.get_greek_bs_european(
                SPOT,
                STRIKE,
                MATURITY,
                RATE,
                VOLATILITY,
                False,
                greek,
                greek_engine=greek_engine,
            )
            assert call_value == pytest.approx(put_value, abs=1e-8)

        call_rho = hapi.get_greek_bs_european(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            True,
            "rho",
            greek_engine=greek_engine,
        )
        put_rho = hapi.get_greek_bs_european(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            False,
            "rho",
            greek_engine=greek_engine,
        )
        assert call_rho - put_rho == pytest.approx(
            STRIKE * MATURITY * exp(-RATE * MATURITY),
            abs=1e-6,
        )

    @pytest.mark.parametrize("greek_engine", ["analytical", "fd"])
    def test_delta_bounds_and_non_negative_gamma_vega(self, greek_engine: str) -> None:
        call_delta = hapi.get_greek_bs_european(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            True,
            "delta",
            greek_engine=greek_engine,
        )
        put_delta = hapi.get_greek_bs_european(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            False,
            "delta",
            greek_engine=greek_engine,
        )
        assert 0.0 <= call_delta <= 1.0
        assert -1.0 <= put_delta <= 0.0

        gamma = hapi.get_greek_bs_european(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            True,
            "gamma",
            greek_engine=greek_engine,
        )
        vega = hapi.get_greek_bs_european(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            True,
            "vega",
            greek_engine=greek_engine,
        )
        assert gamma >= 0.0
        assert vega >= 0.0

    def test_rho_fd_bumps_model_rate_and_discount_curve_consistently(self) -> None:
        analytical = hapi.get_greek_bs_european(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            True,
            "rho",
        )
        finite_difference = hapi.get_greek_bs_european(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            True,
            "rho",
            greek_engine="fd",
        )
        assert finite_difference == pytest.approx(analytical, abs=1e-5)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"greek": "theta"}, "greek"),
            ({"engine": "tree"}, "engine"),
            ({"greek_engine": "pathwise"}, "greek_engine"),
            ({"St": 0.0}, "St"),
            ({"K": 0.0}, "K"),
            ({"T": 0.0}, "T"),
            ({"sigma": 0.0}, "sigma"),
        ],
    )
    def test_greek_facade_validates_declared_contract_branches(self,
        kwargs: dict[str, object],
        message: str,
    ) -> None:
        params = {
            "St": SPOT,
            "K": STRIKE,
            "T": MATURITY,
            "r": RATE,
            "sigma": VOLATILITY,
            "call": True,
            "greek": "delta",
        }
        params.update(kwargs)
        with pytest.raises(ValueError, match=message):
            hapi.get_greek_bs_european(**params)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"fd_scheme": "richardson"}, "fd_scheme"),
            ({"h": 0.0}, "h"),
            ({"h": -1.0}, "h"),
            ({"engine": "mc", "n_paths": None, "seed": 1}, "n_paths"),
            ({"engine": "mc", "n_paths": 0, "seed": 1}, "n_paths"),
        ],
    )
    def test_fd_mode_validates_fd_specific_contract_branches(self,
        kwargs: dict[str, object],
        message: str,
    ) -> None:
        params = {
            "St": SPOT,
            "K": STRIKE,
            "T": MATURITY,
            "r": RATE,
            "sigma": VOLATILITY,
            "call": True,
            "greek": "delta",
            "greek_engine": "fd",
        }
        params.update(kwargs)
        with pytest.raises(ValueError, match=message):
            hapi.get_greek_bs_european(**params)

    def test_analytical_greek_ignores_fd_and_mc_arguments_after_selector_validation(self) -> None:
        base = hapi.get_greek_bs_european(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            True,
            "delta",
        )
        ignored = hapi.get_greek_bs_european(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            True,
            "delta",
            engine="mc",
            greek_engine="analytical",
            fd_scheme="richardson",
            h=-1.0,
            n_paths=0,
            seed=None,
        )
        assert ignored == base
