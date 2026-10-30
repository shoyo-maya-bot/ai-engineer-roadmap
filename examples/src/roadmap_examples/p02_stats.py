"""02-3 確率・分布・検定の考え方: 「その差は偶然でも出るか」を、シミュレーションで確かめる.

題材は 07 の評価セット（30 件）. 2 つの分類の正解率の差が、たまたま出た差なのかを調べる.
式（t 分布など）の代わりに、乱数で何千回もやり直して確かめる（ブートストラップ・並べ替え検定）.
乱数は seed を決めた Generator を使い、毎回同じ結果にする.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def simulate_accuracy(p: float, n: int, trials: int, seed: int = 0) -> np.ndarray:
    """本当の正解率が p の分類を、n 件の評価セットで trials 回測ったときの、正解率の一覧（ばらつきを見る）."""
    rng = np.random.default_rng(seed)
    return rng.binomial(n, p, size=trials) / n


def bootstrap_ci(scores: ArrayLike, resamples: int = 10_000, level: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """平均の信頼区間（ブートストラップ）. 件を重複ありで選び直して平均を取る、を繰り返し、その分布の両端を返す."""
    scores = np.asarray(scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(scores), size=(resamples, len(scores)))
    means = scores[idx].mean(axis=1)
    tail = (1 - level) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return float(low), float(high)


def paired_permutation_test(a: ArrayLike, b: ArrayLike, resamples: int = 10_000, seed: int = 0) -> float:
    """同じ件を 2 つの方法で採点した点 a・b の、平均の差の p 値（両側）.

    「a と b に差は無い」なら、件ごとに a と b を入れ替えても差の分布は変わらないはず.
    入れ替え（差の符号の反転）をランダムに繰り返し、実際の差以上に偏る割合を数える.
    """
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    observed = abs(diff.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(resamples, len(diff)))
    null = np.abs((signs * diff).mean(axis=1))
    # 実際の並びも 1 回分として数える（p 値が 0 にならないように）
    return float((np.sum(null >= observed - 1e-12) + 1) / (resamples + 1))
