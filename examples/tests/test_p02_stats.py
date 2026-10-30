"""p02_stats のテスト."""

import numpy as np
import pytest
from scipy import stats

from roadmap_examples.p02_stats import bootstrap_ci, paired_permutation_test, simulate_accuracy


def test_simulated_accuracy_varies_around_p() -> None:
    acc = simulate_accuracy(p=0.7, n=30, trials=10_000, seed=0)
    assert acc.mean() == pytest.approx(0.7, abs=0.01)
    # 30 件だと、本当は 70% の分類でも、60% 以下や 80% 以上が珍しくない
    assert np.mean((acc <= 0.6) | (acc >= 0.8)) > 0.2


def test_simulation_is_reproducible_with_seed() -> None:
    assert np.array_equal(simulate_accuracy(0.7, 30, 100, seed=1), simulate_accuracy(0.7, 30, 100, seed=1))


def test_bootstrap_ci_contains_mean_and_narrows_with_more_cases() -> None:
    small = np.array([1] * 20 + [0] * 10)  # 30 件で 66.7%
    large = np.array([1] * 200 + [0] * 100)  # 300 件で 66.7%
    low_s, high_s = bootstrap_ci(small, seed=0)
    low_l, high_l = bootstrap_ci(large, seed=0)
    assert low_s < small.mean() < high_s
    assert (high_l - low_l) < (high_s - low_s) / 2  # 件数が 10 倍なら、幅はおよそ 1/√10


def test_paired_test_small_difference_is_not_significant() -> None:
    # 30 件中、a は 20 件・b は 24 件が正解. 違うのは 4 件だけ
    a = np.array([1] * 20 + [0] * 10)
    b = np.array([1] * 24 + [0] * 6)
    assert paired_permutation_test(a, b, seed=0) > 0.05


def test_paired_test_matches_scipy() -> None:
    a = np.array([1] * 20 + [0] * 10)
    b = np.array([1] * 24 + [0] * 6)
    want = stats.permutation_test(
        (a, b),
        lambda x, y, axis: np.mean(x - y, axis=axis),
        permutation_type="samples",  # 対応のある 2 群（件ごとに a と b を入れ替える）
        vectorized=True,
        n_resamples=10_000,
        random_state=0,
    ).pvalue
    assert paired_permutation_test(a, b, seed=0) == pytest.approx(want, abs=0.02)
    assert want == pytest.approx(0.125, abs=0.02)  # 違う 4 件がすべて b の勝ち: 2 × (1/2)^4 = 0.125


def test_paired_test_large_difference_is_significant() -> None:
    a = np.array([1] * 10 + [0] * 20)
    b = np.array([1] * 25 + [0] * 5)
    assert paired_permutation_test(a, b, seed=0) < 0.01


def test_paired_test_identical_scores() -> None:
    a = np.array([1, 0, 1, 1])
    assert paired_permutation_test(a, a, resamples=100) == 1.0
