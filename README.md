# ai-engineer-roadmap

業務開発の経験がある人が、約 7 か月（週 8 時間、合計約 220 時間）で、**RAG とエージェントを使う AI システムを作り、本番で動かせるようになる** ための教材です（Python）。
MkDocs でサイトにして GitHub Pages で公開します。

**サイト**: https://shoyo-maya-bot.github.io/ai-engineer-roadmap/

- **対象**: 何か 1 つの言語（Java など）で業務開発の経験がある人。Python の経験は問いません
- **費用**: 演習は無料の範囲で完結します。LLM はローカルのモデル（Ollama など）が基本で、従量課金の API を使う演習は「任意」です
- **形式**: どの章も同じ型（何か → なぜ必要か → どうやるか → 演習 → もし〇〇がなかったら？ → 理解度チェック）。章の構成とコードは CI で検査します
- いまは各部の到達目標と章の一覧を公開しています。章の本文は順に追加します

## 目次

| 段階 | 部 | 内容 | 目安 |
|---|---|---|---|
| 基礎 | 01 Python と基礎 | Python・Linux と HTTP・データ構造・テストと lint | 16 時間 |
| 基礎 | 02 AI のための数学 | ベクトルと類似度・微分と勾配・確率・最適化 | 16 時間 |
| 基礎 | 03 データエンジニアリング | NumPy と pandas・SQL・データの整理・前処理 | 16 時間 |
| 機械学習・深層学習 | 04 機械学習 | 回帰と分類・決定木・クラスタリング・指標と検証 | 20 時間 |
| 機械学習・深層学習 | 05 ディープラーニング | 誤差逆伝播・PyTorch・アテンション・小さなトランスフォーマー | 24 時間 |
| LLM 応用 | 06 LLM の基礎 | トークン・コンテキスト・学習の種類・モデルの選び方 | 10 時間 |
| LLM 応用 | 07 プロンプトと API | プロンプト設計・構造化出力・ツール使用・費用・評価セット | 12 時間 |
| LLM 応用 | 08 RAG とベクトルデータベース | 取り込み・チャンク分割・検索・RAG の評価 | 20 時間 |
| LLM 応用 | 09 ファインチューニングと最適化（発展） | RAG か学習か・LoRA・量子化 | 14 時間 |
| LLM 応用 | 10 AI エージェント | ループとワークフロー・状態・MCP・人の承認と上限 | 18 時間 |
| 安全・運用・実践 | 11 AI の安全と評価 | 幻覚・プロンプトインジェクション・ガードレール・評価の自動化 | 12 時間 |
| 安全・運用・実践 | 12 MLOps とデプロイ | API 化・コンテナと CI/CD・配備・監視と版管理 | 14 時間 |
| 安全・運用・実践 | 13 実践プロジェクト | 要件から配備までを 1 人で通す | 24 時間 |

LLM アプリだけが目的なら、01・03・06〜08・10〜13（約 140 時間）の近道で進められます。

## 構成

| 場所 | 中身 |
|---|---|
| `docs/` | 教材の本文（部ごとのフォルダ。章と解答） |
| `examples/` | 演習のコード（Python 3.12・uv。本文が引用する実物とテスト） |
| `scripts/` | 教材の検査スクリプト（CI で実行） |
| `tests/` | 検査スクリプトのテスト |
| `mkdocs.yml` | サイトの設定と、本文に書く版（`extra.versions`） |

## 書く・直す

執筆のルールと章の型は [CONTRIBUTING.md](CONTRIBUTING.md) にあります。
AI（Copilot など）に書かせるときの指示は [AGENTS.md](AGENTS.md) です。

```bash
pip install --require-hashes -r constraints.txt   # 版とハッシュは constraints.txt で固定（.github/workflows/deps-lock.yml が自動で作る）
mkdocs serve                              # http://127.0.0.1:8000
mkdocs build --strict                     # CI と同じ（サイト内のリンク切れで失敗）
python scripts/check_chapters.py          # 章の型（到達目標・必須の見出し・解答）・nav・章の一覧と、禁止語
python scripts/check_code_quotes.py       # 本文に引用したコードが examples/ の実物と一致するか
python scripts/check_python_examples.py   # 説明用の Python のコード例の構文
python scripts/check_versions.py          # 本文の版（mkdocs.yml の extra.versions）が実物と一致するか
python scripts/check_docs_match.py        # README などに書いたコマンド・ファイル・版が実物とずれていないか
python scripts/check_links.py site        # 外部リンクの切れ（CI では週 1 回だけ）
python -m pytest                          # 検査スクリプトのテスト

cd examples
uv sync --locked                          # 演習のコード（Python 3.12・uv 0.8.17）
uv run pytest                             # 演習のテスト（LLM もネットワークも使わない）
```

本文に書く版（Python・uv・pytest・ruff）は `mkdocs.yml` の `extra.versions` の 1 か所にまとめ、本文では `{{ versions.python }}` のように参照します（そのページの先頭に `render_macros: true`）。

main に push すると GitHub Pages に公開されます（初回だけ、リポジトリの Settings → Pages → Source を **GitHub Actions** にしてください）。
検査（章の型・引用・コード例・版・説明のずれ・テスト・lint・セキュリティ）が通らないと公開しません。
週 1 回の定期実行で、外部リンクを確かめ、切れていれば Issue「外部リンクが切れています（週 1 回の検査）」が起票されます（push・PR では確かめません）。

## 変更点

| 時期 | 変更 |
|---|---|
| 2026-10 | 土台を作成: サイト（はじめに・学び方・13 部の到達目標と章の一覧）、章の型（CONTRIBUTING.md）、演習のコードの土台（`examples/`）、検査スクリプトとテスト、CI（検査・Windows・セキュリティ・CodeQL・依存の固定・Pages 公開・週 1 回の外部リンク検査） |

## セキュリティ・ライセンス

- 脆弱性を見つけたら、公開の Issue ではなく [SECURITY.md](SECURITY.md) の方法で知らせてください
- 秘密情報（API キーなど）は `.env` にだけ書きます（`.env.example` を写して使う。`.env` はコミットしない）
- ライセンスは [MIT](LICENSE) です
