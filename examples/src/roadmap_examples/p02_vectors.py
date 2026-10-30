"""02-1 ベクトル・行列とコサイン類似度: 内積・ノルム・類似度を、for 文と NumPy の両方で計算する.

for 文の版は式をそのまま書いたもの. NumPy の版は同じ計算を配列の演算でまとめて行う（速く、08 の検索でも使う形）.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """内積: 同じ位置どうしを掛けて足す. a·b = Σ a_i b_i."""
    if len(a) != len(b):
        raise ValueError(f"ベクトルの長さが違います（{len(a)} と {len(b)}）")
    return sum(x * y for x, y in zip(a, b, strict=True))


def norm(a: Sequence[float]) -> float:
    """ノルム（長さ）: 自分との内積の平方根. ||a|| = √(a·a)."""
    return math.sqrt(dot(a, a))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """コサイン類似度: 内積を、2 つの長さで割る. 向きだけを比べる（-1〜1）. 長さ 0 なら 0 とする."""
    denom = norm(a) * norm(b)
    return dot(a, b) / denom if denom else 0.0


def normalize_rows(m: ArrayLike) -> NDArray[np.float64]:
    """行列の各行を長さ 1 にする. 長さ 0 の行は 0 のまま."""
    m = np.asarray(m, dtype=np.float64)
    lengths = np.linalg.norm(m, axis=1, keepdims=True)
    return np.divide(m, lengths, out=np.zeros_like(m), where=lengths > 0)


def cosine_matrix(queries: ArrayLike, docs: ArrayLike) -> NDArray[np.float64]:
    """queries の各行と docs の各行のコサイン類似度を、行列の掛け算 1 回でまとめて求める（形は 質問数 × 文書数）."""
    return normalize_rows(queries) @ normalize_rows(docs).T


def top_k(query: ArrayLike, docs: ArrayLike, k: int) -> list[tuple[int, float]]:
    """query に近い文書の（行番号, 類似度）を、近い順に k 件."""
    sims = cosine_matrix(np.atleast_2d(query), docs)[0]
    order = np.argsort(-sims, kind="stable")[:k]  # 同じ値は行番号の小さい順（毎回同じ結果にする）
    return [(int(i), float(sims[i])) for i in order]
