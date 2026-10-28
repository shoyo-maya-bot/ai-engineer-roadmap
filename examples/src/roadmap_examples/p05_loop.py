"""05-2 学習ループ: データを小分け（ミニバッチ）にして、損失 → 勾配 → 更新を繰り返し、検証の損失で止める.

PyTorch の学習ループ（DataLoader・optimizer.zero_grad()・loss.backward()・optimizer.step()・model.eval()）と
同じ組み立てを、NumPy で書いたもの. 勾配の計算は 05-1 の `loss_and_grads`（PyTorch では loss.backward() がする）.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np
from sklearn.datasets import make_moons

from roadmap_examples.p05_mlp import Params, init_params, loss_and_grads, predict

LossAndGrads = Callable[[Params, np.ndarray, np.ndarray], tuple[float, Params]]


class SGD:
    """確率的勾配降下法: パラメーター -= 学習率 × 勾配."""

    def __init__(self, params: Params, lr: float = 0.1) -> None:
        self.params, self.lr = params, lr

    def step(self, grads: Params) -> None:
        for name, p in self.params.items():
            p -= self.lr * grads[name]


class Adam:
    """Adam: 勾配の移動平均（向き）と、勾配の 2 乗の移動平均（大きさ）で、パラメーターごとに歩幅を変える."""

    def __init__(self, params: Params, lr: float = 0.01, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        self.params, self.lr, self.betas, self.eps = params, lr, betas, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: Params) -> None:
        self.t += 1
        b1, b2 = self.betas
        for name, p in self.params.items():
            self.m[name] = b1 * self.m[name] + (1 - b1) * grads[name]
            self.v[name] = b2 * self.v[name] + (1 - b2) * grads[name] ** 2
            m_hat = self.m[name] / (1 - b1**self.t)  # 始めのうち 0 に寄る分を直す
            v_hat = self.v[name] / (1 - b2**self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def batches(n: int, batch_size: int, rng: np.random.Generator) -> Iterator[np.ndarray]:
    """0〜n-1 を混ぜて、batch_size ずつの番号の配列にする（PyTorch の DataLoader(shuffle=True) にあたる）."""
    order = rng.permutation(n)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


@dataclass
class History:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_params: Params = field(default_factory=dict)


def fit(
    params: Params,
    optimizer: SGD | Adam,
    train: tuple[np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray],
    epochs: int = 200,
    batch_size: int = 32,
    patience: int = 20,
    loss_fn: LossAndGrads = loss_and_grads,
    seed: int = 0,
) -> History:
    """1 エポック = 学習データを 1 周. 検証の損失が patience エポック続けて下がらなければ止める（早期終了）."""
    rng = np.random.default_rng(seed)
    x, y = train
    history = History()
    best = float("inf")
    for epoch in range(epochs):
        total = 0.0
        for idx in batches(len(x), batch_size, rng):
            loss, grads = loss_fn(params, x[idx], y[idx])  # PyTorch: loss = ...; loss.backward()
            optimizer.step(grads)  # PyTorch: optimizer.step()
            total += loss * len(idx)
        history.train_losses.append(total / len(x))
        val_loss, _ = loss_fn(params, *val)  # 検証は更新しない（PyTorch: model.eval() と torch.no_grad()）
        history.val_losses.append(val_loss)
        if val_loss < best:
            best, history.best_epoch, history.best_params = val_loss, epoch, copy.deepcopy(params)
        elif epoch - history.best_epoch >= patience:
            break
    return history


def moons(n: int = 300, noise: float = 0.25, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """2 つの三日月の形に並んだ点（直線では分けられない. scikit-learn の make_moons）."""
    x, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    return x, y


def train_moons(optimizer: str = "adam", epochs: int = 300, seed: int = 0) -> tuple[History, float]:
    """三日月のデータを 学習 70% / 検証 30% に分けて学習し、検証の正解率（いちばん良かった時点）も返す."""
    x, y = moons(seed=seed)
    split = int(len(x) * 0.7)
    params = init_params(2, 16, 2, seed)
    opt = Adam(params, lr=0.01) if optimizer == "adam" else SGD(params, lr=0.1)
    history = fit(params, opt, (x[:split], y[:split]), (x[split:], y[split:]), epochs=epochs, seed=seed)
    accuracy = float((predict(history.best_params, x[split:]) == y[split:]).mean())
    return history, accuracy
