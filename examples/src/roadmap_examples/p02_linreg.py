"""02-4 損失関数と最適化（02 の成果物の部品）: 線形回帰を勾配降下で解き、閉じた式（最小二乗）の答えと照合する.

題材（架空）: 問い合わせ文の文字数と添付ファイルの数から、対応にかかる分数を予測する.
成果物のノートブックは `examples/notebooks/p02_linreg.py`（このファイルの関数を順に使う）.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]

# 架空のデータを作るときの「本当の」係数（勾配降下でここに近づくかを確かめる）
TRUE_W = np.array([0.3, 8.0])  # 1 文字あたり 0.3 分、添付 1 つあたり 8 分
TRUE_B = 5.0


def make_data(n: int = 200, noise: float = 4.0, seed: int = 0) -> tuple[Array, Array]:
    """X（n 行 × 2 列: 文字数・添付の数）と y（対応の分数）. seed が同じなら毎回同じデータ."""
    rng = np.random.default_rng(seed)
    length = rng.uniform(10, 200, size=n)
    attachments = rng.integers(0, 4, size=n).astype(np.float64)
    X = np.column_stack([length, attachments])
    y = X @ TRUE_W + TRUE_B + rng.normal(0, noise, size=n)
    return X, y


def predict(X: ArrayLike, w: ArrayLike, b: float) -> Array:
    return np.asarray(X, dtype=np.float64) @ np.asarray(w, dtype=np.float64) + b


def mse(y: ArrayLike, y_pred: ArrayLike) -> float:
    """平均二乗誤差: 外れの 2 乗の平均. 大きな外れほど重く数える."""
    return float(np.mean((np.asarray(y) - np.asarray(y_pred)) ** 2))


def mae(y: ArrayLike, y_pred: ArrayLike) -> float:
    """平均絶対誤差: 外れの大きさの平均. 外れ値に引っ張られにくい."""
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(y_pred))))


def log_loss(y: ArrayLike, p: ArrayLike, eps: float = 1e-15) -> float:
    """2 値分類の交差エントロピー: 正解の側に付けた確率の、対数のマイナスの平均. 自信を持って外すほど大きい."""
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)  # log(0) を避ける
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def mse_gradients(X: ArrayLike, y: ArrayLike, w: ArrayLike, b: float) -> tuple[Array, float]:
    """MSE を w と b で微分した勾配. 誤差 e = Xw + b - y とすると、∂/∂w = 2/n Xᵀe、∂/∂b = 2/n Σe."""
    X = np.asarray(X, dtype=np.float64)
    e = predict(X, w, b) - np.asarray(y, dtype=np.float64)
    n = len(e)
    return 2 / n * X.T @ e, float(2 / n * e.sum())


def standardize(X: ArrayLike) -> tuple[Array, Array, Array]:
    """列ごとに、平均 0・標準偏差 1 にそろえる. 戻り値は（そろえた X, 平均, 標準偏差）."""
    X = np.asarray(X, dtype=np.float64)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std = np.where(std == 0, 1.0, std)  # 値が 1 種類しかない列は割らない
    return (X - mean) / std, mean, std


@dataclass
class Fit:
    w: Array
    b: float
    history: list[float] = field(default_factory=list)  # ステップごとの MSE


def gradient_descent(X: ArrayLike, y: ArrayLike, lr: float, steps: int) -> Fit:
    """w と b を 0 から始め、勾配の逆向きに lr（学習率）ずつ進む、を steps 回. 損失が発散したら止める."""
    X = np.asarray(X, dtype=np.float64)
    w, b = np.zeros(X.shape[1]), 0.0
    history: list[float] = []
    for _ in range(steps):
        loss = mse(y, predict(X, w, b))
        history.append(loss)
        if not np.isfinite(loss) or loss > 1e12:
            break  # 学習率が大きすぎて発散している
        grad_w, grad_b = mse_gradients(X, y, w, b)
        w, b = w - lr * grad_w, b - lr * grad_b
    return Fit(w, b, history)


def to_original_scale(fit: Fit, mean: Array, std: Array) -> tuple[Array, float]:
    """標準化した X で学習した係数を、元の単位（1 文字あたり何分）に戻す."""
    w = fit.w / std
    return w, float(fit.b - mean @ w)


def least_squares(X: ArrayLike, y: ArrayLike) -> tuple[Array, float]:
    """閉じた式の答え（最小二乗）. X に 1 の列を足して np.linalg.lstsq で解く."""
    X = np.asarray(X, dtype=np.float64)
    A = np.column_stack([X, np.ones(len(X))])
    coef, *_ = np.linalg.lstsq(A, np.asarray(y, dtype=np.float64), rcond=None)
    return coef[:-1], float(coef[-1])
