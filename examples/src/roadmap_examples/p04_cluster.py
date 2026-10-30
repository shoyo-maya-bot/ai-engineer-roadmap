"""04-3 クラスタリング: 正解の分類が無いデータを、近いものどうしでまとめる（k-means）.

- 会員の利用の様子（月の利用時間と月の問い合わせ数. 架空）を、単位をそろえてから k 個にまとめる.
  k はシルエット係数（まとまりの良さ. -1〜1）を比べて選ぶ
- 問い合わせ文も、文字の組の TF-IDF にすればまとめられる. ただし、まとまりが人の決めた分類と一致するとは限らない
  （一致の度合いは調整ランド指数で測る. 0 は偶然と同じ、1 は完全に一致）
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# （月の利用時間（分）, 月の問い合わせ数）. 架空の会員 18 人. 列ごとに単位と大きさが違う
MEMBERS: list[tuple[float, float]] = [
    (900, 0), (1300, 1), (1000, 0), (1200, 1), (950, 0), (1250, 0),  # よく使い、ほとんど問い合わせない
    (920, 6), (1280, 7), (1050, 5), (1180, 8), (980, 6), (1220, 7),  # よく使い、問い合わせも多い
    (40, 0), (120, 1), (60, 0), (150, 0), (30, 1), (90, 0),  # ほとんど使っていない
]  # fmt: skip


def cluster_members(k: int, scale: bool = True, members: Sequence[tuple[float, float]] = MEMBERS) -> np.ndarray:
    """会員を k 個にまとめ、まとまりの番号（0〜k-1）を返す. scale=True なら列ごとに平均 0・標準偏差 1 にそろえる."""
    x = np.array(members, dtype=float)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    model = make_pipeline(StandardScaler(), kmeans) if scale else kmeans
    return model.fit_predict(x)


def silhouette_by_k(ks: Sequence[int], members: Sequence[tuple[float, float]] = MEMBERS) -> dict[int, float]:
    """k ごとのシルエット係数（単位をそろえた後の距離で測る）. 大きいほど、まとまりがはっきりしている."""
    x = StandardScaler().fit_transform(np.array(members, dtype=float))
    return {k: float(silhouette_score(x, cluster_members(k, members=members))) for k in ks}


def cluster_texts(texts: Sequence[str], k: int) -> tuple[np.ndarray, list[list[str]]]:
    """問い合わせ文を k 個にまとめる. まとまりの番号と、まとまりごとに重みの大きい文字の組（上位 5 つ）を返す."""
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 2), min_df=2)
    x = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(x)
    names = vectorizer.get_feature_names_out()
    terms = [[str(names[i]) for i in center.argsort()[::-1][:5]] for center in kmeans.cluster_centers_]
    return kmeans.labels_, terms


def agreement(labels: Sequence[int], truth: Sequence[str]) -> float:
    """まとまりと人の決めた分類の一致の度合い（調整ランド指数. まとまりの番号の付け方には左右されない）."""
    return float(adjusted_rand_score(truth, labels))
