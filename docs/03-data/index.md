# 03 データエンジニアリング

!!! abstract "到達目標"
    - 汚れた業務データを、学習に使える表にする処理を、再実行できる形で書ける

## 前提

- 前提の部: [01 Python と基礎](../01-python/index.md)
- 02 は後回しにしても進められます。
- 題材は架空の問い合わせデータです（`examples/data/p03/`）。演習のコードは `examples/` にあり、pandas と scikit-learn を使います。

## 章の一覧

どの章も「何か → なぜ必要か → どうやるか → 演習 → もし〇〇がなかったら？ → 理解度チェック」の順に並びます。演習の解答は別のページです。

| 章 | 内容 | 状態 |
|---|---|---|
| [03-1 NumPy と pandas](03-1-numpy-pandas.md) | 配列と表の基本操作 | 公開 |
| [03-2 SQL での集計と結合](03-2-sql.md) | GROUP BY・JOIN と pandas との対応 | 公開 |
| [03-3 欠損・外れ値・表記ゆれの整理](03-3-cleaning.md) | 汚れの見つけ方と直し方 | 公開 |
| [03-4 特徴量と前処理パイプライン](03-4-features-pipeline.md) | 同じ処理を何度でも再現する | 公開 |

## 成果物

架空の問い合わせデータを整えるパイプライン（[03-3](03-3-cleaning.md)・[03-4](03-4-features-pipeline.md)）

- 入力: `examples/data/p03/inquiries_raw.csv`（汚れた問い合わせ 147 件）と `examples/data/p03/customers.csv`（顧客 40 人）
- 出力: `examples/data/p03/inquiries_clean.csv`（130 件。列は `id`・`received_at`・`customer_id`・`plan`・`channel`・`input`・`category`・`response_minutes`）
- 分類（`category`）は 07 の評価セット（`examples/data/p07/inquiries.jsonl`）と同じ 5 種類（billing・account・bug・contract・other）で、各 26 件
- 実行: `uv run python -m roadmap_examples.p03_pipeline --out inquiries_clean.csv`（`--cases` で 07 と同じ形式の JSONL も書く）
- 前処理: `examples/src/roadmap_examples/p03_features.py`（07 の評価セットで測るモデルは、本文だけを使う `build_preprocessor(text_only=True)`）

整えたデータは 04 の分類モデルの学習に使います。

## 目安時間

16 時間（週 8 時間で約 2 週）
