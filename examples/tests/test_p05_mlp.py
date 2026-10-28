"""p05_mlp.py のテスト（05-1 ニューラルネットワークと誤差逆伝播）."""

import numpy as np
import pytest

from roadmap_examples.p05_mlp import XOR_X, XOR_Y, init_params, loss_and_grads, numerical_grad, predict, train_xor


def test_backprop_matches_numerical_gradient() -> None:
    rng = np.random.default_rng(1)
    params = init_params(3, 5, 4, seed=1)
    x, y = rng.normal(size=(6, 3)), np.array([0, 1, 2, 3, 1, 0])
    _, grads = loss_and_grads(params, x, y)
    for name, p in params.items():
        expected = numerical_grad(lambda: loss_and_grads(params, x, y)[0], p)
        assert grads[name] == pytest.approx(expected, abs=1e-7), name


def test_learns_xor() -> None:
    params, losses = train_xor()
    assert losses[-1] < losses[0] / 10
    assert predict(params, XOR_X).tolist() == XOR_Y.tolist()


def test_zero_init_cannot_break_symmetry() -> None:
    """重みを全部同じ値（0）で始めると、隠れ層の素子が同じ勾配を受け、違いが生まれない."""
    params = {k: np.zeros_like(v) for k, v in init_params(2, 4, 2).items()}
    _, grads = loss_and_grads(params, XOR_X, XOR_Y)
    assert np.allclose(grads["w1"], 0.0)
