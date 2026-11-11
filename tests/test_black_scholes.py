"""Tests for Assignment 3: Black-Scholes European and geometric Asian options."""

from __future__ import annotations

from math import exp, erf, log, sqrt

import pytest

import hesperides.api as hapi


SPOT = 100.0
STRIKE = 100.0
MATURITY = 1.0
RATE = 0.05
VOLATILITY = 0.20


class BlackScholesReferences:
    """Independent analytical reference calculations."""

    @classmethod
    def _normal_cdf(cls, x: float) -> float:
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    @classmethod
    def _geometric_asian_reference(cls, call: bool) -> float:
        mean = log(SPOT) + (RATE - 0.5 * VOLATILITY**2) * MATURITY / 2.0
        variance = VOLATILITY**2 * MATURITY / 3.0
        std = sqrt(variance)
        d1 = (mean - log(STRIKE) + variance) / std
        d2 = d1 - std
        expected_average = exp(mean + 0.5 * variance)
        discount = exp(-RATE * MATURITY)

        if call:
            return discount * (expected_average * cls._normal_cdf(d1) - STRIKE * cls._normal_cdf(d2))
        return discount * (STRIKE * cls._normal_cdf(-d2) - expected_average * cls._normal_cdf(-d1))


# Compatibility names refer directly to class methods.
_normal_cdf = BlackScholesReferences._normal_cdf
_geometric_asian_reference = BlackScholesReferences._geometric_asian_reference


class TestBlackScholes:
    def test_black_scholes_api_contract(self) -> None:
        assert hasattr(hapi, "get_price_bs_european")
        assert hasattr(hapi, "get_price_bs_geometric_asian")

        european = hapi.get_price_bs_european(
            SPOT, STRIKE, MATURITY, RATE, VOLATILITY, call=True
        )
        asian = hapi.get_price_bs_geometric_asian(
            SPOT, STRIKE, MATURITY, RATE, VOLATILITY, call=True
        )

        assert isinstance(european, float)
        assert isinstance(asian, float)

    def test_black_scholes_api_rejects_invalid_engine_and_missing_mc_inputs(self) -> None:
        with pytest.raises(ValueError):
            hapi.get_price_bs_european(
                SPOT, STRIKE, MATURITY, RATE, VOLATILITY, call=True, engine="tree"
            )
        with pytest.raises(ValueError):
            hapi.get_price_bs_geometric_asian(
                SPOT, STRIKE, MATURITY, RATE, VOLATILITY, call=True, engine="tree"
            )
        with pytest.raises(ValueError):
            hapi.get_price_bs_european(
                SPOT, STRIKE, MATURITY, RATE, VOLATILITY, call=True, engine="mc"
            )
        with pytest.raises(ValueError):
            hapi.get_price_bs_geometric_asian(
                SPOT,
                STRIKE,
                MATURITY,
                RATE,
                VOLATILITY,
                call=True,
                engine="mc",
                n_paths=1_000,
            )

    def test_black_scholes_monte_carlo_same_seed_is_reproducible(self) -> None:
        first = hapi.get_price_bs_european(
            SPOT, STRIKE, MATURITY, RATE, VOLATILITY, True, engine="mc", n_paths=10_000, seed=7
        )
        second = hapi.get_price_bs_european(
            SPOT, STRIKE, MATURITY, RATE, VOLATILITY, True, engine="mc", n_paths=10_000, seed=7
        )
        assert first == second

        first_asian = hapi.get_price_bs_geometric_asian(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            True,
            engine="mc",
            n_paths=5_000,
            n_steps=24,
            seed=11,
        )
        second_asian = hapi.get_price_bs_geometric_asian(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            True,
            engine="mc",
            n_paths=5_000,
            n_steps=24,
            seed=11,
        )
        assert first_asian == second_asian

    def test_european_analytical_known_values_and_put_call_parity(self) -> None:
        call = hapi.get_price_bs_european(
            SPOT, STRIKE, MATURITY, RATE, VOLATILITY, call=True
        )
        put = hapi.get_price_bs_european(
            SPOT, STRIKE, MATURITY, RATE, VOLATILITY, call=False
        )

        assert call == pytest.approx(10.450583572185565, abs=1e-12)
        assert put == pytest.approx(5.573526022256971, abs=1e-12)
        assert call - put == pytest.approx(
            SPOT - STRIKE * exp(-RATE * MATURITY),
            abs=1e-12,
        )

    def test_european_monte_carlo_matches_analytical_call_and_put(self) -> None:
        analytical_call = hapi.get_price_bs_european(
            SPOT, STRIKE, MATURITY, RATE, VOLATILITY, True
        )
        analytical_put = hapi.get_price_bs_european(
            SPOT, STRIKE, MATURITY, RATE, VOLATILITY, False
        )

        mc_call = hapi.get_price_bs_european(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            True,
            engine="mc",
            n_paths=100_000,
            seed=123,
        )
        mc_put = hapi.get_price_bs_european(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            False,
            engine="mc",
            n_paths=100_000,
            seed=456,
        )

        assert mc_call == pytest.approx(analytical_call, abs=0.15)
        assert mc_put == pytest.approx(analytical_put, abs=0.15)

    def test_european_monte_carlo_converges_with_more_paths(self) -> None:
        analytical_call = hapi.get_price_bs_european(
            SPOT, STRIKE, MATURITY, RATE, VOLATILITY, True
        )
        analytical_put = hapi.get_price_bs_european(
            SPOT, STRIKE, MATURITY, RATE, VOLATILITY, False
        )

        small_call = hapi.get_price_bs_european(
            SPOT, STRIKE, MATURITY, RATE, VOLATILITY, True, engine="mc", n_paths=2_000, seed=123
        )
        large_call = hapi.get_price_bs_european(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            True,
            engine="mc",
            n_paths=100_000,
            seed=123,
        )
        small_put = hapi.get_price_bs_european(
            SPOT, STRIKE, MATURITY, RATE, VOLATILITY, False, engine="mc", n_paths=2_000, seed=456
        )
        large_put = hapi.get_price_bs_european(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            False,
            engine="mc",
            n_paths=100_000,
            seed=456,
        )

        assert abs(large_call - analytical_call) < abs(small_call - analytical_call)
        assert abs(large_put - analytical_put) < abs(small_put - analytical_put)

    def test_geometric_asian_analytical_closed_form_and_parity(self) -> None:
        call = hapi.get_price_bs_geometric_asian(
            SPOT, STRIKE, MATURITY, RATE, VOLATILITY, call=True
        )
        put = hapi.get_price_bs_geometric_asian(
            SPOT, STRIKE, MATURITY, RATE, VOLATILITY, call=False
        )

        mean = log(SPOT) + (RATE - 0.5 * VOLATILITY**2) * MATURITY / 2.0
        variance = VOLATILITY**2 * MATURITY / 3.0
        expected_average = exp(mean + 0.5 * variance)

        assert call == pytest.approx(_geometric_asian_reference(call=True), abs=1e-12)
        assert put == pytest.approx(_geometric_asian_reference(call=False), abs=1e-12)
        assert call == pytest.approx(5.546818633789223, abs=1e-12)
        assert put == pytest.approx(3.4633319477385553, abs=1e-12)
        assert call - put == pytest.approx(
            exp(-RATE * MATURITY) * (expected_average - STRIKE),
            abs=1e-12,
        )

    def test_geometric_asian_monte_carlo_matches_analytical_call_and_put(self) -> None:
        analytical_call = hapi.get_price_bs_geometric_asian(
            SPOT, STRIKE, MATURITY, RATE, VOLATILITY, True
        )
        analytical_put = hapi.get_price_bs_geometric_asian(
            SPOT, STRIKE, MATURITY, RATE, VOLATILITY, False
        )

        mc_call = hapi.get_price_bs_geometric_asian(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            True,
            engine="mc",
            n_paths=60_000,
            n_steps=252,
            seed=789,
        )
        mc_put = hapi.get_price_bs_geometric_asian(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            False,
            engine="mc",
            n_paths=60_000,
            n_steps=252,
            seed=123,
        )

        assert mc_call == pytest.approx(analytical_call, abs=0.08)
        assert mc_put == pytest.approx(analytical_put, abs=0.08)

    def test_geometric_asian_monte_carlo_converges_with_more_paths(self) -> None:
        analytical_call = hapi.get_price_bs_geometric_asian(
            SPOT, STRIKE, MATURITY, RATE, VOLATILITY, True
        )
        small = hapi.get_price_bs_geometric_asian(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            True,
            engine="mc",
            n_paths=2_000,
            n_steps=252,
            seed=789,
        )
        large = hapi.get_price_bs_geometric_asian(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            True,
            engine="mc",
            n_paths=60_000,
            n_steps=252,
            seed=789,
        )

        assert abs(large - analytical_call) < abs(small - analytical_call)

    def test_geometric_asian_monte_carlo_converges_with_more_steps(self) -> None:
        analytical_call = hapi.get_price_bs_geometric_asian(
            SPOT, STRIKE, MATURITY, RATE, VOLATILITY, True
        )
        coarse = hapi.get_price_bs_geometric_asian(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            True,
            engine="mc",
            n_paths=60_000,
            n_steps=16,
            seed=789,
        )
        fine = hapi.get_price_bs_geometric_asian(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            True,
            engine="mc",
            n_paths=60_000,
            n_steps=256,
            seed=789,
        )

        assert abs(fine - analytical_call) < abs(coarse - analytical_call)

    @pytest.mark.parametrize(
        ("call", "expected"),
        [
            (True, 5.0),
            (False, 0.0),
        ],
    )
    def test_zero_maturity_returns_intrinsic_value(self, call: bool, expected: float) -> None:
        assert hapi.get_price_bs_european(105.0, 100.0, 0.0, RATE, VOLATILITY, call) == expected
        assert hapi.get_price_bs_geometric_asian(105.0, 100.0, 0.0, RATE, VOLATILITY, call) == expected
        assert (
            hapi.get_price_bs_european(
                105.0, 100.0, 0.0, RATE, VOLATILITY, call, engine="mc", n_paths=10, seed=1
            )
            == expected
        )
        assert (
            hapi.get_price_bs_geometric_asian(
                105.0,
                100.0,
                0.0,
                RATE,
                VOLATILITY,
                call,
                engine="mc",
                n_paths=10,
                n_steps=4,
                seed=1,
            )
            == expected
        )

    def test_zero_volatility_uses_deterministic_black_scholes_path(self) -> None:
        european_terminal = SPOT * exp(RATE * MATURITY)
        european_expected = exp(-RATE * MATURITY) * max(european_terminal - STRIKE, 0.0)
        asian_average = exp(log(SPOT) + RATE * MATURITY / 2.0)
        asian_expected = exp(-RATE * MATURITY) * max(asian_average - STRIKE, 0.0)

        assert hapi.get_price_bs_european(SPOT, STRIKE, MATURITY, RATE, 0.0, True) == pytest.approx(
            european_expected
        )
        assert hapi.get_price_bs_geometric_asian(
            SPOT, STRIKE, MATURITY, RATE, 0.0, True
        ) == pytest.approx(asian_expected)
        assert hapi.get_price_bs_european(
            SPOT, STRIKE, MATURITY, RATE, 0.0, True, engine="mc", n_paths=100, seed=1
        ) == pytest.approx(european_expected)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"St": 0.0}, "St"),
            ({"K": 0.0}, "K"),
            ({"T": -1.0}, "T"),
            ({"sigma": -0.1}, "sigma"),
        ],
    )
    def test_black_scholes_rejects_invalid_market_parameters(
        self,
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
        }
        params.update(kwargs)
        with pytest.raises(ValueError, match=message):
            hapi.get_price_bs_european(**params)
        with pytest.raises(ValueError, match=message):
            hapi.get_price_bs_geometric_asian(**params)

    @pytest.mark.parametrize("bad_n_paths", [0, -1, 1.5, True, None])
    def test_black_scholes_rejects_invalid_n_paths(self, bad_n_paths: object) -> None:
        with pytest.raises(ValueError, match="n_paths"):
            hapi.get_price_bs_european(
                SPOT,
                STRIKE,
                MATURITY,
                RATE,
                VOLATILITY,
                True,
                engine="mc",
                n_paths=bad_n_paths,
            )

    @pytest.mark.parametrize("bad_n_steps", [0, -1, 1.5, True, None])
    def test_geometric_asian_rejects_invalid_n_steps(self, bad_n_steps: object) -> None:
        with pytest.raises(ValueError, match="n_steps"):
            hapi.get_price_bs_geometric_asian(
                SPOT,
                STRIKE,
                MATURITY,
                RATE,
                VOLATILITY,
                True,
                engine="mc",
                n_paths=1_000,
                n_steps=bad_n_steps,
            )

    def test_analytical_engine_ignores_monte_carlo_arguments(self) -> None:
        base = hapi.get_price_bs_geometric_asian(
            SPOT, STRIKE, MATURITY, RATE, VOLATILITY, True
        )
        with_ignored_args = hapi.get_price_bs_geometric_asian(
            SPOT,
            STRIKE,
            MATURITY,
            RATE,
            VOLATILITY,
            True,
            engine="analytical",
            n_paths=0,
            n_steps=0,
            seed=123,
        )
        assert with_ignored_args == base
