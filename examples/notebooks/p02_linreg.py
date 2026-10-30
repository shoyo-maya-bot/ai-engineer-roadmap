# %% [markdown]
# # 02 の成果物: 線形回帰を勾配降下で解く
#
# 架空の問い合わせについて、文字数と添付ファイルの数から、対応にかかる分数を予測します。
# 勾配降下で係数を求め、閉じた式（最小二乗）と scikit-learn の結果と照合します。
#
# - `# %%` で区切ったセルの形式です。VS Code では、セルごとに実行できます
# - まとめて実行する場合: `uv run python notebooks/p02_linreg.py`（examples/ で）

# %%
import numpy as np
from sklearn.linear_model import LinearRegression

from roadmap_examples.p02_gradient import check_gradient
from roadmap_examples.p02_linreg import (
    TRUE_B,
    TRUE_W,
    gradient_descent,
    least_squares,
    make_data,
    mse,
    mse_gradients,
    predict,
    standardize,
    to_original_scale,
)

np.set_printoptions(precision=3, suppress=True)

# %% [markdown]
# ## 1. データを作る

# %%
X, y = make_data(n=200, seed=0)
print("X の形:", X.shape, " y の形:", y.shape)
print("先頭 3 行:", X[:3], y[:3])

# %% [markdown]
# ## 2. 勾配の式を、数値の微分で確かめる

# %%
w0, b0 = np.array([0.1, 1.0]), 2.0


def loss_of(params: np.ndarray) -> float:
    return mse(y, predict(X, params[:2], params[2]))


def grad_of(params: np.ndarray) -> np.ndarray:
    grad_w, grad_b = mse_gradients(X, y, params[:2], params[2])
    return np.append(grad_w, grad_b)


print("相対的な差:", check_gradient(loss_of, grad_of, np.append(w0, b0)))

# %% [markdown]
# ## 3. 勾配降下で解く（列をそろえてから）

# %%
Xs, mean, std = standardize(X)
fit = gradient_descent(Xs, y, lr=0.1, steps=500)
w_gd, b_gd = to_original_scale(fit, mean, std)
print("損失（最初・最後）:", round(fit.history[0], 2), round(fit.history[-1], 2))
print("勾配降下:", w_gd, round(b_gd, 3))

# %% [markdown]
# ## 4. 閉じた式・scikit-learn と照合する

# %%
w_ls, b_ls = least_squares(X, y)
model = LinearRegression().fit(X, y)
print("最小二乗:", w_ls, round(b_ls, 3))
print("scikit-learn:", model.coef_, round(float(model.intercept_), 3))
print("本当の値:", TRUE_W, TRUE_B)
assert np.allclose(w_gd, w_ls, atol=1e-3) and abs(b_gd - b_ls) < 1e-2
assert np.allclose(w_ls, model.coef_) and np.isclose(b_ls, model.intercept_)
print("3 つの結果が一致しました")

# %% [markdown]
# ## 5. 学習率を変えてみる
#
# 大きすぎると損失が増え続け（発散）、小さすぎると 500 回では止まり切りません。

# %%
for lr in (0.001, 0.1, 1.5):
    h = gradient_descent(Xs, y, lr=lr, steps=500).history
    print(f"学習率 {lr}: {len(h)} 回で、損失 {h[0]:.1f} → {h[-1]:.3g}")
