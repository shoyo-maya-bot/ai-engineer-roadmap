"""05-3 文章の扱い: RNN（前から 1 つずつ読む）と、アテンション（全部の位置を一度に見比べる）.

- `rnn_states`: 隠れ状態に、それまでの内容を詰め込みながら 1 文字ずつ進む
- `first_token_influence`: 最初の入力を少し変えたとき、最後の隠れ状態がどれだけ変わるか（長いと伝わりにくい）
- `attention`: 各位置が、ほかの全部の位置との近さ（内積）で重みを決め、値を混ぜる（scaled dot-product attention）
"""

from __future__ import annotations

import numpy as np

from roadmap_examples.p05_mlp import softmax


def rnn_states(xs: np.ndarray, w_x: np.ndarray, w_h: np.ndarray) -> np.ndarray:
    """h_t = tanh(x_t W_x + h_{t-1} W_h). xs は（長さ, 入力の次元）. 各時点の隠れ状態（長さ, 隠れの次元）を返す."""
    h = np.zeros(w_h.shape[0])
    states = []
    for x in xs:  # 前の結果を待たないと次に進めない（並べて計算できない）
        h = np.tanh(x @ w_x + h @ w_h)
        states.append(h)
    return np.array(states)


def first_token_influence(length: int, scale: float = 0.5, dim: int = 8, seed: int = 0, eps: float = 1e-4) -> float:
    """最初の入力だけを eps ずらしたとき、最後の隠れ状態が動く大きさ ÷ eps（大きいほど、最初の文字が届いている）."""
    rng = np.random.default_rng(seed)
    w_x = rng.normal(0.0, scale / np.sqrt(dim), (dim, dim))
    w_h = rng.normal(0.0, scale / np.sqrt(dim), (dim, dim))
    xs = rng.normal(0.0, 1.0, (length, dim))
    moved = xs.copy()
    moved[0] += eps
    return float(np.linalg.norm(rnn_states(moved, w_x, w_h)[-1] - rnn_states(xs, w_x, w_h)[-1]) / eps)


def causal_mask(length: int) -> np.ndarray:
    """位置 i が見てよいのは 0〜i だけ（未来の文字を見ない）. 見てよいところが True の（長さ, 長さ）の表."""
    return np.tril(np.ones((length, length), dtype=bool))


def attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """softmax(Q K^T / √d) V. 出力と、重み（行ごとに足して 1）を返す. mask が False の位置の重みは 0 にする."""
    scores = q @ k.swapaxes(-1, -2) / np.sqrt(q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)  # exp(-inf) = 0 なので、重みが 0 になる
    weights = softmax(scores)
    return weights @ v, weights
