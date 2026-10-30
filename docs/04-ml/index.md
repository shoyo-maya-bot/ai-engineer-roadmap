# 04 機械学習

!!! abstract "到達目標"
    - scikit-learn で分類モデルを作り、「どの指標で、どれだけ良いか」を根拠つきで言える

## 前提

- 前提の部: [02 AI のための数学](../02-math/index.md)・[03 データエンジニアリング](../03-data/index.md)
- 演習では、03 の成果物（`examples/data/p03/inquiries_clean.csv`、架空の問い合わせ 130 件）の本文と分類で学習し、07 と同じ評価セット（`examples/data/p07/inquiries.jsonl`、30 件）で測ります。整えた表はリポジトリにあるので、03 を後回しにしても進められます。
- ライブラリは scikit-learn を使います（`examples/` の `uv sync --locked` で入ります）。GPU も LLM も使いません。

## 章の一覧

どの章も「何か → なぜ必要か → どうやるか → 演習 → もし〇〇がなかったら？ → 理解度チェック」の順に並びます。演習の解答は別のページです。

| 章 | 内容 | 状態 |
|---|---|---|
| [04-1 回帰と分類](04-1-regression-classification.md) | 予測の種類と基本のモデル | 公開 |
| [04-2 決定木・ランダムフォレスト](04-2-trees-forests.md) | 木のモデルと特徴量の重要度 | 公開 |
| [04-3 クラスタリング](04-3-clustering.md) | 正解の無いデータをまとめる | 公開 |
| [04-4 指標・交差検証・過学習](04-4-metrics-validation.md) | 正しく測る・測りすぎない | 公開 |

## 成果物

問い合わせの自動分類モデルと評価レポート（[04-4](04-4-metrics-validation.md) の「04 の成果物: 評価レポート」）

- モデル: 文字の組の TF-IDF + ロジスティック回帰。03 の成果物の本文と分類だけで学習（`examples/src/roadmap_examples/p04_classify.py`）
- 実行: `uv run python -m roadmap_examples.p04_classify --out results-p04.jsonl --report report-p04.md`
- 評価セットでの正解率は 80.0%（30 件中 24 件。07 の仮のルールベース分類は 66.7%）。学習データの交差検証では 73.8% です

07 で、同じ評価セットを使って LLM での分類と比べる基準にします。07 の仮のルールベース分類からの差し替え方は、
[04-4](04-4-metrics-validation.md) の「07 の基準を差し替える」にあります。

## 目安時間

20 時間（週 8 時間で約 2.5 週）
