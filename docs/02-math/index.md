# 02 AI のための数学

!!! abstract "到達目標"
    - ベクトルの類似度と勾配降下を、NumPy で自分で計算して説明できる

## 前提

- 前提の部: [01 Python と基礎](../01-python/index.md)
- 式の証明より「コードで確かめる」を優先します。演習のコードは `examples/` にあり、NumPy と scikit-learn を使います。

## 章の一覧

どの章も「何か → なぜ必要か → どうやるか → 演習 → もし〇〇がなかったら？ → 理解度チェック」の順に並びます。演習の解答は別のページです。

| 章 | 内容 | 状態 |
|---|---|---|
| [02-1 ベクトル・行列とコサイン類似度](02-1-vectors-cosine.md) | 内積・ノルム・類似度 | 公開 |
| [02-2 微分と勾配](02-2-derivative-gradient.md) | 傾きを数値で確かめる | 公開 |
| [02-3 確率・分布・検定の考え方](02-3-probability-testing.md) | ばらつきと偶然の見分け方 | 公開 |
| [02-4 損失関数と最適化](02-4-loss-optimization.md) | 損失を小さくする仕組み | 公開 |

## 成果物

線形回帰を勾配降下で解くノートブック（ライブラリの結果と照合する。[02-4](02-4-loss-optimization.md)）

- ノートブック: `examples/notebooks/p02_linreg.py`（`# %%` で区切ったセルの形式。部品は `examples/src/roadmap_examples/p02_linreg.py`）
- 実行: `uv run python notebooks/p02_linreg.py`（`examples/` で）

類似度は 08 の検索、勾配と損失は 04・05 の学習の理解に使います。

## 目安時間

16 時間（週 8 時間で約 2 週）
