"""Public tests for Assignment 2: Static arbitrage quantities on call surfaces."""

from __future__ import annotations

import numpy as np
import pytest

import hesperides.api as hapi


class TestStaticArbitrage:
    @pytest.fixture
    def call_surface(self) -> tuple[np.ndarray, np.ndarray]:
        """Call surface values on a strikes x maturities grid."""
        strikes = np.array([0.0, 60.0, 80.0, 90.0, 100.0, 110.0, 120.0, 140.0], dtype=float)
        surface_ext = np.array(
            [
                [100.0, 100.0, 100.0, 100.0, 100.0],
                [40.0, 40.00000023021229, 40.00040695032761, 40.026111811907235, 40.30569348391282],
                [20.0, 20.039914343421856, 20.309114475890524, 21.18592951321044, 23.082652301718596],
                [10.0, 10.712380896073668, 11.7724511004688, 13.589108116054796, 16.411067988848593],
                [0.0, 3.987761167674492, 5.6371977797016655, 7.965567455405804, 11.246291601828489],
                [0.0, 0.9539473918572199, 2.2112464335730806, 4.292010941409885, 7.467730366014841],
                [0.0, 0.14733226325696203, 0.7204125178557295, 2.1472988105781425, 4.830635378173774],
                [0.0, 0.0011741548650312517, 0.04835520601261578, 0.4500324519081067, 1.9050765273920778],
            ],
            dtype=float,
        )
        return surface_ext, strikes

    @pytest.mark.regression
    def test_surface_passes_vertical_spread_bounds(self, call_surface) -> None:
        surface, strikes = call_surface
        q = hapi.compute_static_arbitrage_quantity(surface, strikes, quantity="vertical")
        assert np.all(q >= -1e-10)
        assert np.all(q <= 1.0 + 1e-10)

    @pytest.mark.regression
    def test_vertical_spreads_match_expected_values(self, call_surface) -> None:
        surface, strikes = call_surface
        q = hapi.compute_static_arbitrage_quantity(surface, strikes, quantity="vertical")
        expected = np.array(
            [
                [1.0, 0.9999999961631285, 0.9999932174945398, 0.9995648031348794, 0.994905108601453],
                [1.0, 0.9980042943395215, 0.9845646237218542, 0.9420091149348397, 0.8611520591097112],
                [1.0, 0.9327533447348187, 0.8536663375421725, 0.7596821397155644, 0.6671584312870003],
                [1.0, 0.6724619728399176, 0.6135253320767134, 0.5623540660648991, 0.5164776387020105],
                [0.0, 0.30338137758172723, 0.3425951346128585, 0.3673556513995919, 0.3778561235813648],
                [0.0, 0.08066151286002579, 0.1490833915717351, 0.2144712130831742, 0.2637094987841067],
                [0.0, 0.00730790541959654, 0.03360286559215568, 0.0848633179335018, 0.14627794253908483],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(q, expected, atol=1e-12, rtol=0.0)

    @pytest.mark.regression
    def test_surface_passes_butterfly_non_negative(self, call_surface) -> None:
        surface, strikes = call_surface
        b = hapi.compute_static_arbitrage_quantity(surface, strikes, quantity="butterfly")
        assert np.all(b >= -1e-10)

    @pytest.mark.regression
    def test_surface_passes_calendar_non_negative(self, call_surface) -> None:
        surface, strikes = call_surface
        cs = hapi.compute_static_arbitrage_quantity(surface, strikes, quantity="calendar")
        assert np.all(cs >= -1e-10)
