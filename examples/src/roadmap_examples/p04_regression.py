"""04-1 回帰: 問い合わせの対応にかかる時間（分）を、文字数と添付の数から予測する.

題材は架空の数字. 分類（どの窓口か）と違い、回帰は「数」を予測する. 指標も違う（正解率ではなく、誤差の大きさ）.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# （文字数, 添付の数, 対応にかかった時間（分））. 架空のデータ
HANDLING: list[tuple[int, int, float]] = [
    (40, 0, 9.0),
    (80, 0, 13.5),
    (120, 0, 16.0),
    (60, 1, 19.0),
    (100, 1, 22.5),
    (150, 1, 27.0),
    (50, 2, 26.0),
    (90, 2, 30.5),
    (200, 0, 25.5),
    (130, 2, 33.0),
    (70, 0, 12.0),
    (180, 1, 29.5),
]
TEST: list[tuple[int, int, float]] = [(110, 0, 15.5), (75, 1, 20.0), (160, 2, 37.0), (45, 1, 17.5)]


def to_xy(rows: list[tuple[int, int, float]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[chars, files] for chars, files, _ in rows], dtype=float)
    y = np.array([minutes for _, _, minutes in rows], dtype=float)
    return x, y


@dataclass(frozen=True)
class RegressionResult:
    per_char: float  # 1 文字あたり何分増えるか（係数）
    per_file: float  # 添付 1 つあたり何分増えるか（係数）
    intercept: float  # 文字数も添付も 0 のときの分（切片）
    mae: float  # 予測と実際の差の絶対値の平均（分）
    baseline_mae: float  # 学習データの平均をいつも答えたときの MAE（比べる基準）


def fit_and_score(
    train: list[tuple[int, int, float]] = HANDLING, test: list[tuple[int, int, float]] = TEST
) -> tuple[LinearRegression, RegressionResult]:
    """学習データで直線（平面）を当てはめ、学習に使っていないデータで誤差を測る."""
    x_train, y_train = to_xy(train)
    x_test, y_test = to_xy(test)
    model = LinearRegression().fit(x_train, y_train)
    baseline = np.full_like(y_test, y_train.mean())
    return model, RegressionResult(
        per_char=float(model.coef_[0]),
        per_file=float(model.coef_[1]),
        intercept=float(model.intercept_),
        mae=float(mean_absolute_error(y_test, model.predict(x_test))),
        baseline_mae=float(mean_absolute_error(y_test, baseline)),
    )


def least_squares(rows: list[tuple[int, int, float]] = HANDLING) -> np.ndarray:
    """同じ問題を NumPy の最小二乗法で解く（[係数 1, 係数 2, 切片]）. 02 の勾配降下の答えとも同じになる."""
    x, y = to_xy(rows)
    x1 = np.hstack([x, np.ones((len(x), 1))])
    solution, *_ = np.linalg.lstsq(x1, y, rcond=None)
    return solution
