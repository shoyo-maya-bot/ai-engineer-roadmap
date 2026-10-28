# examples（演習のコード）

教材の本文が引用する、実行できるコードの置き場所です。本文のコードはここからの引用で、
`scripts/check_code_quotes.py`（CI）が一致を確かめます。

```bash
cd examples
uv sync --locked      # 固定した版で入れる（.venv を作る）
uv run pytest         # テスト（LLM もネットワークも使わない）
uv run ruff check . && uv run ruff format --check .
uv run api-to-csv --help
```

| ファイル | 部 | 内容 |
|---|---|---|
| `src/roadmap_examples/p01_api_to_csv.py` | 01 | 公開 API から JSON を取り、CSV にする CLI の雛形 |
| `src/roadmap_examples/llm.py` | 06 以降 | LLM を呼ぶ部分のインターフェース（`LLMClient`）と、テスト用の偽の実装（`FakeLLM`）・呼び出し回数の上限（`LimitedLLM`） |
| `src/roadmap_examples/p06_*.py` | 06 | 小さな BPE とコサイン類似度・temperature とコンテキストに収める処理・チャットのテンプレート・成果物（モデルの比べる表） |
| `data/p06/questions.jsonl` | 06 | 06 の成果物で比べる質問（8 問。評価セットの形式） |
| `src/roadmap_examples/ollama_client.py` | 07 以降 | ローカルの LLM（Ollama）を呼ぶ本物の実装（会話・構造化出力・ツール使用・エンベディング） |
| `src/roadmap_examples/evalset.py` | 07 以降 | 評価セット（JSONL）の読み書き・採点・前回の結果との比較 |
| `src/roadmap_examples/p07_*.py` | 07 | プロンプトの組み立て・構造化出力の検証・ツール使用・費用の上限とキャッシュ・成果物（分類の比較） |
| `data/p07/inquiries.jsonl` | 07 | 07 の評価セット（架空の問い合わせ 30 件） |
| `src/roadmap_examples/p08_*.py` | 08 | 文書の取り込み（PDF・Excel）・チャンク分割・検索（ベクトル・BM25・RRF・並べ替え）・出典つきの Q&A・検索と生成を分けた評価 |
| `data/p08/` | 08 | 08 の題材（架空の設計書）と評価セット（14 問） |
| `src/roadmap_examples/p10_*.py` | 10 | ワークフロー・承認と上限つきのエージェント（成果物）・状態の保存と会話の削り方・MCP の最小の実装（stdio）・エージェントの評価 |
| `data/p10/agent_cases.jsonl` | 10 | 10 のエージェントの評価セット（8 件. 起票を提案すべきか・呼ぶべきツール） |
| `src/roadmap_examples/p09_*.py` | 09 | NumPy だけの LoRA（成果物. r は検証用で選び、07 の評価セットで前後を比べる）・SFT の学習の例の書き出し・量子化とメモリの見積もり |
| `data/p09/` | 09 | 09 の学習の例（汎用 25 件・業務 40 件・検証用 10 件. 07 の評価セットを見ずに書いた. 完全一致は無い. 似た文は `p09_sft.nearest_pairs` で確かめた） |
| `src/roadmap_examples/p11_*.py` | 11 | ガードレール（入力・資料・書き換えの前・出力）・攻撃例を流す道具（成果物）・正解率の区間と符号検定・数字の裏付け |
| `data/p11/attacks.jsonl` | 11 | 10 のエージェントへの攻撃例（20 件. 攻撃に従う台本と、漏れてはいけない文字列つき） |
| `data/p11/holdout.jsonl` | 11 | 守りを見ずに書かれた攻撃例（7 件. 未知の言い換えを測る. 守りをこれに合わせて直さない） |
| `src/roadmap_examples/p12_*.py` | 12 | エージェントの API（標準ライブラリの WSGI. 承認待ち・記録・版）・監視（集計と HTML の画面） |
| `data/p12/requests-sample.jsonl` | 12 | 監視を試すための、API の記録のサンプル（架空の 40 件） |
| `Dockerfile`・`compose.yaml` | 12 | API のイメージと、Ollama と一緒に動かす compose |
| `src/roadmap_examples/p13_project.py` | 13 | 実践プロジェクトの提出物のひな形・費用の見積もり・提出前の確かめ |

- ファイル名は `p<部の番号>_<内容>.py`。テストは `tests/test_<同じ名前>.py`
- LLM やネットワークを使う部分はインターフェースにして、テストでは偽の実装に差し替える。
  `tests/conftest.py` がテスト中のネットワーク接続を止めるので、うっかり本物を呼ぶとテストが失敗する
