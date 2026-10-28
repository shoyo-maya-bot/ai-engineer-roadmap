"""p04_regression.py のテスト（04-1 回帰）."""

import pytest

from roadmap_examples.p04_regression import fit_and_score, least_squares


def test_regression_learns_the_trend_and_beats_the_mean() -> None:
    _, result = fit_and_score()
    assert result.per_char == pytest.approx(0.094, abs=0.001)  # 100 文字で約 9 分
    assert result.per_file == pytest.approx(7.8, abs=0.1)  # 添付 1 つで約 8 分
    assert result.mae < 1.0 < result.baseline_mae


def test_same_answer_as_least_squares() -> None:
    model, _ = fit_and_score()
    assert least_squares() == pytest.approx([*model.coef_, model.intercept_])
