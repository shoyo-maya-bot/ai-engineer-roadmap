"""p02_linreg と、成果物のノートブック（notebooks/p02_linreg.py）のテスト."""

import runpy
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import log_loss as sk_log_loss
from sklearn.metrics import mean_absolute_error, mean_squared_error

from roadmap_examples.p02_gradient import gradient
from roadmap_examples.p02_linreg import (
    TRUE_B,
    TRUE_W,
    gradient_descent,
    least_squares,
    log_loss,
    mae,
    make_data,
    mse,
    mse_gradients,
    predict,
    standardize,
    to_original_scale,
)

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "p02_linreg.py"


def test_losses_match_scikit_learn() -> None:
    y, y_pred = np.array([3.0, 5.0, 10.0]), np.array([2.0, 5.0, 14.0])
    assert mse(y, y_pred) == pytest.approx(mean_squared_error(y, y_pred))
    assert mae(y, y_pred) == pytest.approx(mean_absolute_error(y, y_pred))
    labels, probs = np.array([1, 0, 1]), np.array([0.9, 0.2, 0.6])
    assert log_loss(labels, probs) == pytest.approx(sk_log_loss(labels, probs))


def test_confident_mistake_costs_more() -> None:
    assert log_loss([1], [0.01]) > log_loss([1], [0.4]) > log_loss([1], [0.9])


def test_mse_gradients_match_numerical_gradient() -> None:
    X, y = make_data(n=20, seed=1)
    w, b = np.array([0.1, 2.0]), 1.0
    grad_w, grad_b = mse_gradients(X, y, w, b)
    numeric = gradient(lambda p: mse(y, predict(X, p[:2], p[2])), np.append(w, b))
    assert np.allclose(np.append(grad_w, grad_b), numeric, rtol=1e-5)


def test_gradient_descent_matches_least_squares_and_scikit_learn() -> None:
    X, y = make_data()
    Xs, mean, std = standardize(X)
    w_gd, b_gd = to_original_scale(gradient_descent(Xs, y, lr=0.1, steps=500), mean, std)
    w_ls, b_ls = least_squares(X, y)
    model = LinearRegression().fit(X, y)
    assert np.allclose(w_gd, w_ls, atol=1e-3) and b_gd == pytest.approx(b_ls, abs=1e-2)
    assert np.allclose(w_ls, model.coef_) and b_ls == pytest.approx(model.intercept_)
    assert np.allclose(w_ls, TRUE_W, atol=0.6) and b_ls == pytest.approx(TRUE_B, abs=2)  # 雑音があるので近いだけ


def test_loss_decreases_with_good_learning_rate() -> None:
    X, y = make_data()
    history = gradient_descent(standardize(X)[0], y, lr=0.1, steps=100).history
    assert all(a >= b - 1e-9 for a, b in zip(history, history[1:], strict=False))  # 小数の丸めの分だけ許す


def test_too_large_learning_rate_diverges_and_stops() -> None:
    X, y = make_data()
    history = gradient_descent(standardize(X)[0], y, lr=1.5, steps=500).history
    assert len(history) < 500 and history[-1] > history[0]


def test_unscaled_features_need_tiny_learning_rate() -> None:
    X, y = make_data()
    assert gradient_descent(X, y, lr=1e-4, steps=500).history[-1] > 1e6  # 列をそろえないと、この学習率でも発散
    slow = gradient_descent(X, y, lr=1e-5, steps=500).history[-1]
    fast = gradient_descent(standardize(X)[0], y, lr=0.1, steps=500).history[-1]
    assert slow > 5 * fast  # 発散しない学習率では、500 回で止まり切らない


def test_standardize_constant_column() -> None:
    Xs, _, std = standardize([[1.0, 5.0], [3.0, 5.0]])
    assert np.allclose(Xs[:, 0], [-1, 1]) and np.allclose(Xs[:, 1], 0) and std[1] == 1.0


def test_notebook_runs(capsys) -> None:
    runpy.run_path(str(NOTEBOOK), run_name="__main__")
    assert "3 つの結果が一致しました" in capsys.readouterr().out
