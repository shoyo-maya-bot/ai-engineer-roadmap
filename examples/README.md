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

- ファイル名は `p<部の番号>_<内容>.py`。テストは `tests/test_<同じ名前>.py`
- LLM やネットワークを使う部分はインターフェースにして、テストでは偽の実装に差し替える。
  `tests/conftest.py` がテスト中のネットワーク接続を止めるので、うっかり本物を呼ぶとテストが失敗する
