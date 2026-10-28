"""p02_gradient のテスト."""

import math

import numpy as np
import pytest

from roadmap_examples.p02_gradient import check_gradient, derivative, forward_derivative, gradient


def test_derivative_of_square() -> None:
    # f(x) = x^2 の傾きは 2x. x = 3 なら 6
    assert derivative(lambda x: x**2, 3.0) == pytest.approx(6.0, abs=1e-8)


def test_central_difference_is_more_accurate_than_forward() -> None:
    f, x, want = math.exp, 1.0, math.exp(1.0)  # exp の微分は exp
    central = abs(derivative(f, x, h=1e-3) - want)
    forward = abs(forward_derivative(f, x, h=1e-3) - want)
    assert central < forward / 100


def test_too_small_h_loses_precision() -> None:
    # h を小さくしすぎると、引き算の桁落ちで誤差が増える
    f, want = math.exp, math.exp(1.0)
    assert abs(derivative(f, 1.0, h=1e-12) - want) > abs(derivative(f, 1.0, h=1e-5) - want)


def test_gradient_of_quadratic() -> None:
    # f(x, y) = x^2 + 3xy の勾配は (2x + 3y, 3x)
    def f(v: np.ndarray) -> float:
        return float(v[0] ** 2 + 3 * v[0] * v[1])

    assert np.allclose(gradient(f, [1.0, 2.0]), [8.0, 3.0])


def test_check_gradient_finds_wrong_formula() -> None:
    def f(v: np.ndarray) -> float:
        return float(np.sum(v**2))

    assert check_gradient(f, lambda v: 2 * v, [1.0, -2.0, 0.5]) < 1e-8
    with pytest.raises(ValueError, match="勾配が合いません"):
        check_gradient(f, lambda v: v, [1.0, -2.0, 0.5])  # 2 を掛け忘れた
