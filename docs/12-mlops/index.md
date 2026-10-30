# 12 MLOps とデプロイ

!!! abstract "到達目標"
    - AI アプリをコンテナで配備し、精度・費用・応答時間を監視できる

## 前提

- 前提の部: [10 AI エージェント](../10-agents/index.md)・[11 AI の安全と評価](../11-safety/index.md)
- 配備は、自分の PC やサーバーで Docker と compose を使う方法（費用なし）を基本にします。クラウドに置く手順は任意（有料）です。
- API は標準ライブラリだけで書いています（`examples/`）。FastAPI での書き方は、本文で例として比べます。

## 章の一覧

どの章も「何か → なぜ必要か → どうやるか → 演習 → もし〇〇がなかったら？ → 理解度チェック」の順に並びます。演習の解答は別のページです。

| 章 | 内容 | 状態 |
|---|---|---|
| [12-1 FastAPI で API にする](12-1-fastapi.md) | エージェントを呼び出せる形にする | 公開 |
| [12-2 Docker と CI/CD](12-2-docker-cicd.md) | 同じものを何度でも作って配る | 公開 |
| [12-3 クラウドへの配備](12-3-cloud.md) | 必要な分だけのクラウドの使い方 | 公開 |
| [12-4 監視とモデル・プロンプトの版管理](12-4-monitoring-versioning.md) | トークンと費用・応答時間・追跡 | 公開 |

## 成果物

10 のエージェントを API にして配備する手順と、監視の画面

- API: `uv run python -m roadmap_examples.p12_api --model <モデル名>`（11 のガードレールつき. 承認は `202` で返し、鍵つきの `/approvals` で受ける. [12-1](12-1-fastapi.md)）
- 配備: `examples/Dockerfile` と `examples/compose.yaml`（Ollama と一緒に動かす. [12-2](12-2-docker-cicd.md)）と、配備の手順書の形（[12-3](12-3-cloud.md)）
- 監視: `uv run python -m roadmap_examples.p12_monitor <記録> --html monitor.html`（応答時間・トークンと費用・止めた割合・版ごとの内訳. [12-4](12-4-monitoring-versioning.md)）

13 の実践プロジェクトの配備と監視の型にします。

## 目安時間

14 時間（週 8 時間で約 1.8 週）
