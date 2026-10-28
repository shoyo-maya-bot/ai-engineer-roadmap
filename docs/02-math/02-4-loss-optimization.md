# 02-4 損失関数と最適化

!!! abstract "到達目標"
    - 損失関数（MSE・MAE・交差エントロピー）が「何を悪いとみなすか」の違いを説明し、課題に合うものを選べる
    - 線形回帰を勾配降下で解き、閉じた式（最小二乗）と scikit-learn の結果と照合できる
    - 学習率と、特徴量の大きさをそろえること（標準化）が、学習の進み方にどう効くかを説明できる

## 何か

**損失関数** は、予測がどれだけ外れているかを 1 つの数にする関数です。学習は、損失を小さくするパラメーターを探すことです（**最適化**）。

| 損失 | 式（y は正解、ŷ は予測） | 何を悪いとみなすか | 使う場面 |
|---|---|---|---|
| MSE（平均二乗誤差） | 平均 (y − ŷ)² | 大きな外れをとても重く数える | 数を予測する（回帰） |
| MAE（平均絶対誤差） | 平均 \|y − ŷ\| | 外れの大きさをそのまま数える。外れ値に強い | 回帰で外れ値が多いとき |
| 交差エントロピー | 平均 −log（正解の側に付けた確率） | 自信を持って外すほど、とても重く数える | 分類（04・05） |

**勾配降下** は、02-2 の勾配を使って、損失が減る向きにパラメーターを少しずつ動かす方法です。
1 回に動かす幅を決めるのが **学習率** です。

この章の題材は **線形回帰** です。架空の問い合わせについて、文字数と添付ファイルの数から、対応にかかる分数を予測します。

```text
予測（分） = w₁ × 文字数 + w₂ × 添付の数 + b
```

w₁・w₂（重み）と b（切片）を、損失（MSE）が最も小さくなるように決めます。

## なぜ必要か

04 の分類モデルも、05 のニューラルネットも、LLM の学習も、「損失を決めて、勾配で小さくする」という同じ形です。
ライブラリを使えば `fit` の 1 行で済みますが、中身を知らないと、うまくいかないときに手が打てません。

- 損失が減らない・増え続ける（学習率が大きすぎる）
- いつまでも学習が終わらない（特徴量の大きさがそろっていない）
- 指標では良いのに、業務では困る予測をする（損失が、業務の「悪さ」と合っていない）

## どうやるか

### データと損失

`make_data` は、決まった係数（1 文字あたり 0.3 分・添付 1 つあたり 8 分・切片 5 分）に雑音を足して、架空のデータを作ります。
「本当の答え」が分かっているので、学習の結果が正しいかを確かめられます。

<!-- src: examples/src/roadmap_examples/p02_linreg.py -->
```python title="examples/src/roadmap_examples/p02_linreg.py"
def mse(y: ArrayLike, y_pred: ArrayLike) -> float:
    """平均二乗誤差: 外れの 2 乗の平均. 大きな外れほど重く数える."""
    return float(np.mean((np.asarray(y) - np.asarray(y_pred)) ** 2))


def mae(y: ArrayLike, y_pred: ArrayLike) -> float:
    """平均絶対誤差: 外れの大きさの平均. 外れ値に引っ張られにくい."""
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(y_pred))))


def log_loss(y: ArrayLike, p: ArrayLike, eps: float = 1e-15) -> float:
    """2 値分類の交差エントロピー: 正解の側に付けた確率の、対数のマイナスの平均. 自信を持って外すほど大きい."""
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)  # log(0) を避ける
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
```

3 つとも、scikit-learn の同じ名前の関数と同じ値になることを、テストで確かめています（`examples/tests/test_p02_linreg.py`）。

外れ値が 1 件あると、MSE と MAE は大きく違います。

```pycon
>>> from roadmap_examples.p02_linreg import log_loss, mae, mse
>>> mse([10, 12, 11, 100], [11, 11, 11, 11]), mae([10, 12, 11, 100], [11, 11, 11, 11])
(1980.75, 22.75)
>>> log_loss([1], [0.9]), log_loss([1], [0.4]), log_loss([1], [0.01])
(0.10536051565782628, 0.916290731874155, 4.605170185988091)
```

- MSE は、100 の 1 件（外れ 89）の 2 乗が、ほぼすべてを占めます。MSE で学習すると、その 1 件に合わせようとして、ほかの件の予測がずれます
- 交差エントロピーは、正解に 1% の確率しか付けなかった（自信を持って外した）ときに、とても大きくなります

### 勾配の式を書き、確かめる

誤差を e = Xw + b − y とすると、MSE の勾配は、w について 2/n Xᵀe、b について 2/n Σe です。

<!-- src: examples/src/roadmap_examples/p02_linreg.py -->
```python title="examples/src/roadmap_examples/p02_linreg.py"
def mse_gradients(X: ArrayLike, y: ArrayLike, w: ArrayLike, b: float) -> tuple[Array, float]:
    """MSE を w と b で微分した勾配. 誤差 e = Xw + b - y とすると、∂/∂w = 2/n Xᵀe、∂/∂b = 2/n Σe."""
    X = np.asarray(X, dtype=np.float64)
    e = predict(X, w, b) - np.asarray(y, dtype=np.float64)
    n = len(e)
    return 2 / n * X.T @ e, float(2 / n * e.sum())
```

使う前に、02-2 の数値の勾配と比べます。

<!-- src: examples/tests/test_p02_linreg.py -->
```python title="examples/tests/test_p02_linreg.py"
def test_mse_gradients_match_numerical_gradient() -> None:
    X, y = make_data(n=20, seed=1)
    w, b = np.array([0.1, 2.0]), 1.0
    grad_w, grad_b = mse_gradients(X, y, w, b)
    numeric = gradient(lambda p: mse(y, predict(X, p[:2], p[2])), np.append(w, b))
    assert np.allclose(np.append(grad_w, grad_b), numeric, rtol=1e-5)
```

### 勾配降下

w と b を 0 から始め、勾配の逆向きに学習率ずつ進みます。損失の推移（`history`）も残します。

<!-- src: examples/src/roadmap_examples/p02_linreg.py -->
```python title="examples/src/roadmap_examples/p02_linreg.py"
def gradient_descent(X: ArrayLike, y: ArrayLike, lr: float, steps: int) -> Fit:
    """w と b を 0 から始め、勾配の逆向きに lr（学習率）ずつ進む、を steps 回. 損失が発散したら止める."""
    X = np.asarray(X, dtype=np.float64)
    w, b = np.zeros(X.shape[1]), 0.0
    history: list[float] = []
    for _ in range(steps):
        loss = mse(y, predict(X, w, b))
        history.append(loss)
        if not np.isfinite(loss) or loss > 1e12:
            break  # 学習率が大きすぎて発散している
        grad_w, grad_b = mse_gradients(X, y, w, b)
        w, b = w - lr * grad_w, b - lr * grad_b
    return Fit(w, b, history)
```

### 学習率と標準化

文字数（10〜200）と添付の数（0〜3）は、大きさが 50 倍以上違います。このまま学習すると、文字数の勾配だけが大きくなり、
文字数に合わせて学習率を小さくすると、添付の数の重みがなかなか動きません。

そこで、列ごとに平均 0・標準偏差 1 にそろえます（**標準化**）。

<!-- src: examples/src/roadmap_examples/p02_linreg.py -->
```python title="examples/src/roadmap_examples/p02_linreg.py"
def standardize(X: ArrayLike) -> tuple[Array, Array, Array]:
    """列ごとに、平均 0・標準偏差 1 にそろえる. 戻り値は（そろえた X, 平均, 標準偏差）."""
    X = np.asarray(X, dtype=np.float64)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std = np.where(std == 0, 1.0, std)  # 値が 1 種類しかない列は割らない
    return (X - mean) / std, mean, std
```

200 件のデータで、500 回まで進めた結果です（成果物のノートブックとテストで確かめた値）。

| 特徴量 | 学習率 | 結果（MSE） |
|---|---|---|
| そろえない | 0.0001 | 発散（14 回で 10¹² を超えて止まる） |
| そろえない | 0.00001 | 500 回で 3011 → 132（まだ下がり切らない） |
| そろえる | 0.001 | 500 回で 3011 → 413（小さすぎて遅い） |
| そろえる | 0.1 | 500 回で 3011 → 15.7（下がり切る） |
| そろえる | 1.5 | 発散（14 回で止まる） |

!!! failure "悪い例"
    特徴量をそろえずに学習し、発散したら学習率を下げ、遅ければ回数を増やす、を繰り返す。

!!! success "良い例"
    先に標準化する。学習率は 0.001・0.01・0.1 のように 10 倍ずつ変えて試し、損失の推移（`history`）を見て決める。

標準化した X で学習した係数は、「1 標準偏差あたり何分」の単位です。元の単位（1 文字あたり何分）に戻すには、`to_original_scale` を使います。

標準化に使う平均と標準偏差は、学習用のデータだけから求めます。評価用のデータの値を混ぜると、評価が甘くなります（03-4 で扱います）。

### 閉じた式・scikit-learn と照合する

線形回帰は、勾配降下を使わなくても、式で答えが求まります（最小二乗法）。NumPy の `np.linalg.lstsq` で解けます
（[公式ドキュメント](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)）。

<!-- src: examples/src/roadmap_examples/p02_linreg.py -->
```python title="examples/src/roadmap_examples/p02_linreg.py"
def least_squares(X: ArrayLike, y: ArrayLike) -> tuple[Array, float]:
    """閉じた式の答え（最小二乗）. X に 1 の列を足して np.linalg.lstsq で解く."""
    X = np.asarray(X, dtype=np.float64)
    A = np.column_stack([X, np.ones(len(X))])
    coef, *_ = np.linalg.lstsq(A, np.asarray(y, dtype=np.float64), rcond=None)
    return coef[:-1], float(coef[-1])
```

1 の列を足すのは、切片 b を「いつも 1 の特徴量の重み」として一緒に解くためです。

3 つの方法（勾配降下・最小二乗・scikit-learn の `LinearRegression`）の結果が一致することを、テストで確かめています
（[scikit-learn: LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)）。

<!-- src: examples/tests/test_p02_linreg.py -->
```python title="examples/tests/test_p02_linreg.py"
def test_gradient_descent_matches_least_squares_and_scikit_learn() -> None:
    X, y = make_data()
    Xs, mean, std = standardize(X)
    w_gd, b_gd = to_original_scale(gradient_descent(Xs, y, lr=0.1, steps=500), mean, std)
    w_ls, b_ls = least_squares(X, y)
    model = LinearRegression().fit(X, y)
    assert np.allclose(w_gd, w_ls, atol=1e-3) and b_gd == pytest.approx(b_ls, abs=1e-2)
    assert np.allclose(w_ls, model.coef_) and b_ls == pytest.approx(model.intercept_)
    assert np.allclose(w_ls, TRUE_W, atol=0.6) and b_ls == pytest.approx(TRUE_B, abs=2)  # 雑音があるので近いだけ
```

結果は、勾配降下・最小二乗・scikit-learn のどれも、1 文字あたり 0.297 分・添付 1 つあたり 7.543 分・切片 6.007 分でした。
本当の値（0.3・8・5）とは少し違います。雑音の入った 200 件から推定しているからです。

線形回帰のように式で解ける問題でも勾配降下を学ぶのは、式で解けない大きなモデル（05 以降）では、勾配降下しか手が無いからです。

### 02 の成果物: ノートブック

成果物は `examples/notebooks/p02_linreg.py` です。ここまでの手順（データ・勾配の確認・勾配降下・照合・学習率の比較）を、セルに分けて順に実行します。

<!-- src: examples/notebooks/p02_linreg.py -->
```python title="examples/notebooks/p02_linreg.py"
# %%
w_ls, b_ls = least_squares(X, y)
model = LinearRegression().fit(X, y)
print("最小二乗:", w_ls, round(b_ls, 3))
print("scikit-learn:", model.coef_, round(float(model.intercept_), 3))
print("本当の値:", TRUE_W, TRUE_B)
assert np.allclose(w_gd, w_ls, atol=1e-3) and abs(b_gd - b_ls) < 1e-2
assert np.allclose(w_ls, model.coef_) and np.isclose(b_ls, model.intercept_)
print("3 つの結果が一致しました")
```

- `# %%` で区切った、セルの形式の Python のファイルです。VS Code では、セルごとに実行して結果を見られます
  （[VS Code: Python Interactive](https://code.visualstudio.com/docs/python/jupyter-support-py)）
- まとめて実行するときは、`examples/` で `uv run python notebooks/p02_linreg.py` です
- ふつうの `.py` なので、差分が読みやすく、テスト（`test_notebook_runs`）で最後まで動くことを確かめています

## 演習

1. 成果物のノートブックを実行し、勾配降下・最小二乗・scikit-learn の係数と、学習率ごとの損失を記録してください
2. ミニバッチの確率的勾配降下（SGD）を書いてください。毎回 32 件をランダムに選んで勾配を求め、全件を 1 巡するのを 50 回繰り返します。
   最小二乗の結果と比べ、損失と係数がどれくらい近いかをテストで確かめます
3. データの 5 件だけ、対応時間を 200 分多くして（入力の誤りの真似）、最小二乗の係数がどう変わるかを確かめてください。MSE の性質から理由を説明します

- できたことの確かめ方: 1 は記録した表。2 はテストが通る。3 は係数の変化と 1〜2 文の説明
- 目安の時間: 4 時間

解答: [02-4 の解答](answers/02-4.md)

## もし〇〇がなかったら？

**損失関数を意識せずにライブラリの既定のまま学習していたら**、入力の誤りによる外れ値に、モデル全体が引っ張られます。
「たまにある大きな外れ」と「いつも少しずれる」のどちらを避けたいかは業務で決まるので、損失もそれに合わせて選びます。

**標準化をしていなかったら**、学習率をいくら調整しても、ある特徴量の重みだけが動かない、ということが起きます。
04 のモデル（ロジスティック回帰など）も勾配で学習するので、同じ理由で、特徴量をそろえてから学習します。

## 理解度チェック

1. 外れ値が多いデータで、MSE より MAE が向いているのはなぜですか。

    ??? note "答え"
        MSE は外れを 2 乗するので、大きな外れ値が損失のほとんどを占め、モデルがその件に合わせようとするからです。
        MAE は外れの大きさをそのまま数えるので、外れ値に引っ張られにくくなります。

2. 学習率が大きすぎると、何が起きますか。小さすぎるとどうですか。

    ??? note "答え"
        大きすぎると、谷を飛び越えて損失が増え続けます（発散）。小さすぎると、損失は減りますが、とても遅くなります。
        損失の推移を見て、10 倍ずつ変えて試します。

3. 特徴量を標準化すると、学習が速く進むのはなぜですか。

    ??? note "答え"
        特徴量の大きさがそろうと、どの重みの勾配も同じくらいの大きさになり、1 つの学習率でどの重みも同じように動かせるからです。

4. 線形回帰は式で解けるのに、勾配降下を学ぶのはなぜですか。

    ??? note "答え"
        05 以降のニューラルネットのように、式で答えを求められない大きなモデルでは、勾配降下（とその改良版）で学習するしかないからです。
        線形回帰は、式の答えと照合できるので、勾配降下の練習に向いています。
