"""02-2 微分と勾配: 傾き（微分）と勾配を、数値で近似して確かめる.

式で求めた微分（解析的な微分）が合っているかを、数値の微分と比べて確かめる「勾配の確認」は、
02-4 の線形回帰や 05 の誤差逆伝播でも使う.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def derivative(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """f の x での傾きを、中心差分 (f(x+h) - f(x-h)) / 2h で近似する."""
    return (f(x + h) - f(x - h)) / (2 * h)


def forward_derivative(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """前進差分 (f(x+h) - f(x)) / h. 中心差分より誤差が大きい（比べるために置く）."""
    return (f(x + h) - f(x)) / h


def gradient(f: Callable[[NDArray[np.float64]], float], x: ArrayLike, h: float = 1e-5) -> NDArray[np.float64]:
    """多変数の関数 f の勾配（各変数での傾きを並べたベクトル）を、1 変数ずつ中心差分で近似する."""
    x = np.asarray(x, dtype=np.float64)
    grad = np.zeros_like(x)
    for i in range(x.size):
        step = np.zeros_like(x)
        step.flat[i] = h
        grad.flat[i] = (f(x + step) - f(x - step)) / (2 * h)
    return grad


def check_gradient(
    f: Callable[[NDArray[np.float64]], float],
    grad_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x: ArrayLike,
    tol: float = 1e-6,
) -> float:
    """式で求めた勾配 grad_f と数値の勾配の、相対的な差. tol を超えたら ValueError（式の誤りの疑い）."""
    x = np.asarray(x, dtype=np.float64)
    analytic, numeric = grad_f(x), gradient(f, x)
    diff = float(np.linalg.norm(analytic - numeric) / max(np.linalg.norm(analytic) + np.linalg.norm(numeric), 1e-12))
    if diff > tol:
        raise ValueError(f"勾配が合いません（相対的な差 {diff:.2e}）")
    return diff
