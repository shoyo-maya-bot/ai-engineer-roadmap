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


def test_zero_init_freezes_the_weights() -> None:
    """全部 0 で始めると、tanh(0) = 0 で隠れ層の出力が 0 になり、動くのは出力のバイアス b2 だけ."""
    params = {k: np.zeros_like(v) for k, v in init_params(2, 4, 2).items()}
    _, grads = loss_and_grads(params, XOR_X, XOR_Y)
    for name in ("w1", "b1", "w2"):
        assert np.allclose(grads[name], 0.0), name


def test_same_nonzero_init_keeps_hidden_units_identical() -> None:
    """0 以外の同じ値で始めると勾配は 0 にならないが、隠れ層の素子はいつまでも同じ動きをする（素子 1 個と同じ）."""
    params = {k: np.full_like(v, 0.5) for k, v in init_params(2, 4, 2).items()}
    for _ in range(200):
        _, grads = loss_and_grads(params, XOR_X, XOR_Y)
        for name in params:
            params[name] -= 0.5 * grads[name]
    assert not np.allclose(params["w1"], 0.5)  # 重みは動いた
    assert np.allclose(params["w1"], params["w1"][:, :1])  # 4 個の素子の重みは、どれも同じまま
