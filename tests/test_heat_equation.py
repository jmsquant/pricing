"""Tests for Assignment 6: heat equation and Black-Scholes heat transform."""

from __future__ import annotations

from math import exp, pi
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.testing import assert_allclose

import hesperides.api as hapi


@dataclass(frozen=True)
class HeatInputs:
    domain_size: float = 1.0

    def sine(self, x_grid):
        return np.sin(pi * x_grid / self.domain_size)

    @staticmethod
    def constant(x):
        return np.full_like(x, 2.0)

    @staticmethod
    def invalid(x):
        return np.array([1.0, 2.0])

    @staticmethod
    def alternating_mode(x_grid: np.ndarray) -> np.ndarray:
        values = (-1.0) ** np.arange(x_grid.size)
        values[0] = 0.0
        values[-1] = 0.0
        return values


class HeatReferences:
    """Independent heat equation reference solutions."""

    @classmethod
    def _sine_initial(cls, domain_size: float):
        return HeatInputs(domain_size).sine

    @classmethod
    def _sine_solution(cls,
        x_grid: np.ndarray,
        kappa: float,
        domain_size: float,
        maturity: float,
    ) -> np.ndarray:
        decay = exp(-kappa * (pi / domain_size) ** 2 * maturity)
        return decay * np.sin(pi * x_grid / domain_size)


# Compatibility names refer directly to class methods.
_sine_initial = HeatReferences._sine_initial
_sine_solution = HeatReferences._sine_solution


class TestHeatEquation:
    @pytest.mark.parametrize(
        ("scheme", "n_t", "tolerance"),
        [
            ("explicit", 200, 1.0e-5),
            ("implicit", 40, 1.0e-4),
        ],
    )
    def test_heat_equation_sine_solution_matches_analytical(self,
        scheme: str,
        n_t: int,
        tolerance: float,
    ) -> None:
        kappa = 0.3
        domain_size = 2.0
        maturity = 0.1
        x_grid, values = hapi.solve_heat_equation(
            _sine_initial(domain_size),
            kappa=kappa,
            M=domain_size,
            T=maturity,
            n_x=80,
            n_t=n_t,
            scheme=scheme,
        )

        expected = _sine_solution(x_grid, kappa, domain_size, maturity)
        assert_allclose(values, expected, atol=tolerance, rtol=0.0)

    def test_explicit_heat_scheme_stability_threshold(self) -> None:
        kappa = 1.0
        domain_size = 1.0
        n_x = 40
        n_t = 30
        dx = domain_size / n_x

        alternating_mode = HeatInputs.alternating_mode

        stable_t = 0.4 * dx * dx * n_t / kappa
        _, stable = hapi.solve_heat_equation(
            alternating_mode,
            kappa=kappa,
            M=domain_size,
            T=stable_t,
            n_x=n_x,
            n_t=n_t,
            scheme="explicit",
        )

        unstable_t = 0.75 * dx * dx * n_t / kappa
        _, unstable = hapi.solve_heat_equation(
            alternating_mode,
            kappa=kappa,
            M=domain_size,
            T=unstable_t,
            n_x=n_x,
            n_t=n_t,
            scheme="explicit",
        )

        assert np.max(np.abs(stable)) <= 1.0
        assert np.max(np.abs(unstable)) > 1.0e6

    def test_heat_equation_converges_with_second_order_space_and_first_order_time(self) -> None:
        kappa = 0.25
        domain_size = 1.0
        maturity = 0.02
        errors = []

        for n_x in (40, 80, 160):
            n_t = n_x * n_x // 20
            x_grid, values = hapi.solve_heat_equation(
                _sine_initial(domain_size),
                kappa=kappa,
                M=domain_size,
                T=maturity,
                n_x=n_x,
                n_t=n_t,
                scheme="explicit",
            )
            expected = _sine_solution(x_grid, kappa, domain_size, maturity)
            errors.append(float(np.max(np.abs(values - expected))))

        assert errors[1] < 0.35 * errors[0]
        assert errors[2] < 0.35 * errors[1]

    def test_implicit_scheme_handles_large_time_steps_that_break_explicit(self) -> None:
        kappa = 0.3
        domain_size = 2.0
        maturity = 0.1
        x_grid, values = hapi.solve_heat_equation(
            _sine_initial(domain_size),
            kappa=kappa,
            M=domain_size,
            T=maturity,
            n_x=80,
            n_t=4,
            scheme="implicit",
        )

        expected = _sine_solution(x_grid, kappa, domain_size, maturity)
        assert_allclose(values, expected, atol=1.0e-3, rtol=0.0)

    @pytest.mark.parametrize("scheme", ["explicit", "implicit"])
    def test_neumann_boundaries_keep_constant_solution_constant(self, scheme: str) -> None:
        x_grid, values = hapi.solve_heat_equation(
            HeatInputs.constant,
            kappa=0.4,
            M=1.0,
            T=0.02,
            n_x=40,
            n_t=80,
            scheme=scheme,
            left_boundary=0.0,
            right_boundary=0.0,
            boundary_type="neumann",
        )

        assert_allclose(values, np.full_like(x_grid, 2.0), atol=1.0e-12, rtol=0.0)

    @pytest.mark.parametrize(
        ("spot", "strike", "maturity", "rate", "volatility"),
        [
            (100.0, 100.0, 1.0, 0.05, 0.20),
            (100.0, 90.0, 0.5, 0.03, 0.25),
            (100.0, 110.0, 1.5, 0.02, 0.30),
        ],
    )
    @pytest.mark.parametrize("call", [True, False])
    def test_black_scholes_heat_price_matches_closed_form(self,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        volatility: float,
        call: bool,
    ) -> None:
        heat_price = hapi.get_price_bs_european_heat(
            spot,
            strike,
            maturity,
            rate,
            volatility,
            call,
            n_x=320,
            n_t=320,
            scheme="implicit",
        )
        closed_form = hapi.get_price_bs_european(
            spot,
            strike,
            maturity,
            rate,
            volatility,
            call,
        )

        assert heat_price == pytest.approx(closed_form, abs=2.0e-2)

    @pytest.mark.parametrize("call", [True, False])
    def test_black_scholes_heat_explicit_price_matches_closed_form(self, call: bool) -> None:
        heat_price = hapi.get_price_bs_european_heat(
            100.0,
            100.0,
            1.0,
            0.05,
            0.20,
            call,
            n_x=300,
            n_t=400,
            scheme="explicit",
        )
        closed_form = hapi.get_price_bs_european(
            100.0,
            100.0,
            1.0,
            0.05,
            0.20,
            call,
        )

        assert heat_price == pytest.approx(closed_form, abs=3.0e-2)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"scheme": "crank"}, "scheme"),
            ({"boundary_type": "robin"}, "boundary_type"),
            ({"kappa": 0.0}, "kappa"),
            ({"M": 0.0}, "M"),
            ({"T": 0.0}, "T"),
            ({"n_x": 0}, "n_x"),
            ({"n_t": 0}, "n_t"),
        ],
    )
    def test_heat_solver_rejects_invalid_inputs(self, kwargs: dict[str, object], message: str) -> None:
        params = {
            "initial_condition": HeatInputs().sine,
            "kappa": 0.5,
            "M": 1.0,
            "T": 0.1,
            "n_x": 20,
            "n_t": 20,
        }
        params.update(kwargs)

        with pytest.raises(ValueError, match=message):
            hapi.solve_heat_equation(**params)

    def test_heat_solver_rejects_bad_initial_condition_shape(self) -> None:
        with pytest.raises(ValueError, match="initial_condition"):
            hapi.solve_heat_equation(
                HeatInputs.invalid,
                kappa=0.5,
                M=1.0,
                T=0.1,
                n_x=20,
                n_t=20,
            )

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"St": 0.0}, "St"),
            ({"K": 0.0}, "K"),
            ({"T": 0.0}, "T"),
            ({"sigma": 0.0}, "sigma"),
            ({"n_x": 0}, "n_x"),
            ({"n_t": 0}, "n_t"),
            ({"scheme": "tree"}, "scheme"),
        ],
    )
    def test_black_scholes_heat_rejects_invalid_inputs(self,
        kwargs: dict[str, object],
        message: str,
    ) -> None:
        params = {
            "St": 100.0,
            "K": 100.0,
            "T": 1.0,
            "r": 0.05,
            "sigma": 0.20,
            "call": True,
        }
        params.update(kwargs)

        with pytest.raises(ValueError, match=message):
            hapi.get_price_bs_european_heat(**params)
