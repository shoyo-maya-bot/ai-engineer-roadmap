"""05-1 ニューラルネットワークと誤差逆伝播: 2 層のネットワークを NumPy で書き、勾配を手で計算する.

入力 → 全結合 → tanh → 全結合 → softmax（分類ごとの確率）. 損失は交差エントロピー.
勾配は、出力の側から入力の側へ、連鎖律で順に計算する（誤差逆伝播）. 正しさは数値微分と比べて確かめる.
PyTorch は、この「逆向きの計算」を自動でする（05-2）.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Params = dict[str, np.ndarray]

# XOR: 2 つの入力が違うときだけ 1. 直線 1 本では分けられない
XOR_X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR_Y = np.array([0, 1, 1, 0])


def init_params(n_in: int, n_hidden: int, n_out: int, seed: int = 0) -> Params:
    """重みは小さな乱数、バイアスは 0 で始める（全部 0 だと、隠れ層の素子がみな同じ動きになる）."""
    rng = np.random.default_rng(seed)
    return {
        "w1": rng.normal(0.0, 1.0 / np.sqrt(n_in), (n_in, n_hidden)),
        "b1": np.zeros(n_hidden),
        "w2": rng.normal(0.0, 1.0 / np.sqrt(n_hidden), (n_hidden, n_out)),
        "b2": np.zeros(n_out),
    }


def softmax(z: np.ndarray) -> np.ndarray:
    """行ごとに確率（足して 1）にする. 最大値を引いてから exp を取り、大きな数で溢れないようにする."""
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def forward(params: Params, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """順伝播. 逆伝播で使う途中の値（cache）も返す."""
    h = np.tanh(x @ params["w1"] + params["b1"])
    probs = softmax(h @ params["w2"] + params["b2"])
    return probs, {"x": x, "h": h, "probs": probs}


def loss_and_grads(params: Params, x: np.ndarray, y: np.ndarray) -> tuple[float, Params]:
    """交差エントロピー（正解の確率の -log の平均）と、各パラメーターの勾配."""
    probs, cache = forward(params, x)
    n = len(y)
    loss = float(-np.log(probs[np.arange(n), y] + 1e-12).mean())
    # softmax と交差エントロピーを合わせた勾配は「確率 - 正解の one-hot」になる
    dz2 = probs.copy()
    dz2[np.arange(n), y] -= 1.0
    dz2 /= n
    grads = {"w2": cache["h"].T @ dz2, "b2": dz2.sum(axis=0)}
    dh = dz2 @ params["w2"].T
    dz1 = dh * (1.0 - cache["h"] ** 2)  # tanh の微分は 1 - tanh^2
    grads["w1"] = cache["x"].T @ dz1
    grads["b1"] = dz1.sum(axis=0)
    return loss, grads


def numerical_grad(f: Callable[[], float], param: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """中心差分（(f(p+ε) - f(p-ε)) / 2ε）で、param の各要素の勾配を近似する. 遅いので確かめるときだけ使う."""
    grad = np.zeros_like(param)
    for i in np.ndindex(param.shape):
        old = param[i]
        param[i] = old + eps
        plus = f()
        param[i] = old - eps
        minus = f()
        param[i] = old
        grad[i] = (plus - minus) / (2 * eps)
    return grad


def predict(params: Params, x: np.ndarray) -> np.ndarray:
    return forward(params, x)[0].argmax(axis=1)


def train_xor(steps: int = 2000, lr: float = 0.5, n_hidden: int = 4, seed: int = 0) -> tuple[Params, list[float]]:
    """XOR を勾配降下で学習する. 損失の推移も返す."""
    params = init_params(2, n_hidden, 2, seed)
    losses = []
    for _ in range(steps):
        loss, grads = loss_and_grads(params, XOR_X, XOR_Y)
        for name in params:
            params[name] -= lr * grads[name]
        losses.append(loss)
    return params, losses
