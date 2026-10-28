# 01 Python と基礎

!!! abstract "到達目標"
    - 他の言語で書ける処理を Python で書き、テストと lint を通して Git で出せる

## 前提

- 最初の部です。何か 1 つの言語で業務のコードを書いた経験があれば始められます。
- Java などとの違いから覚える形で進めます。演習のコードは `examples/` にあり、テストはネットワークを使いません。

## 章の一覧

どの章も「何か → なぜ必要か → どうやるか → 演習 → もし〇〇がなかったら？ → 理解度チェック」の順に並びます。演習の解答は別のページです。

| 章 | 内容 | 状態 |
|---|---|---|
| [01-1 他の言語との違いで覚える Python](01-1-python-basics.md) | 型ヒント・仮想環境・uv | 公開 |
| [01-2 Linux の基本操作と HTTP API の呼び方](01-2-linux-http.md) | シェル・ファイル操作・HTTP の要求と応答 | 公開 |
| [01-3 データ構造と計算量](01-3-data-structures.md) | list・dict・set の使い分けと速さ | 公開 |
| [01-4 クラス設計・pytest・ruff](01-4-class-pytest-ruff.md) | クラスと依存の差し替え・テスト・静的解析 | 公開 |

## 成果物

公開 API からデータを取って CSV にする CLI（テスト付き。[01-2](01-2-linux-http.md)・[01-4](01-4-class-pytest-ruff.md)）

- コード: `examples/src/roadmap_examples/p01_api_to_csv.py`（テストは `examples/tests/test_p01_api_to_csv.py`）
- 実行: `uv run api-to-csv --url <URL> --fields id,name -o out.csv`（`examples/` で。手元のサーバーで試す方法は 01-2）

作り方（依存の差し替え・テスト・lint）は、以降のすべての部の演習で使います。

## 目安時間

16 時間（週 8 時間で約 2 週）
