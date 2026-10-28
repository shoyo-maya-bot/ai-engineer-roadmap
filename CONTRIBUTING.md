# 教材の書き方（執筆ルール）

この教材は、どの章も同じ型で書き、章の構成・コード・版を CI で検査します。
AI（Copilot など）に書かせるときは、あわせて [AGENTS.md](AGENTS.md) を読ませてください。

## ファイルの置き場所

| もの | 場所 | 例 |
|---|---|---|
| 部 | `docs/<NN>-<英小文字の名前>/index.md` | `docs/01-python/index.md` |
| 章 | `docs/<NN>-<名前>/<NN>-<N>-<英小文字と数字とハイフン>.md` | `docs/01-python/01-4-class-pytest-ruff.md` |
| 解答 | `docs/<NN>-<名前>/answers/<NN>-<N>.md` | `docs/01-python/answers/01-4.md` |
| 演習のコード | `examples/src/roadmap_examples/p<NN>_<内容>.py` | `examples/src/roadmap_examples/p01_api_to_csv.py` |
| 演習のテスト | `examples/tests/test_p<NN>_<内容>.py` | `examples/tests/test_p01_api_to_csv.py` |

- 章を足したら、`mkdocs.yml` の `nav`（章と解答の両方）と、部の `docs/<NN>-<名前>/index.md` の「章の一覧」（「準備中」をリンクに変える）も直す
- 空の章ファイルは作らない（「準備中」は部の `docs/<NN>-<名前>/index.md` の表で示す）

## 1 章の構成（この順で、見出しもこの名前で）

```markdown
# 01-4 クラス設計・pytest・ruff

!!! abstract "到達目標"
    - 依存を差し替えられるクラスを書き、偽の実装でテストできる
    - （2〜4 個。「〜を説明できる」「〜を書ける」「〜を判断できる」の形）

## 何か
（その仕組み・考え方を 1〜3 段落で。図があれば mermaid で）

## なぜ必要か
（無いと業務で何に困るか。ほかの言語の経験と比べてよい）

## どうやるか
（手順とコード。「悪い例」と「良い例」を並べる。!!! failure "悪い例" / !!! success "良い例"）

## 演習
（課題。手順・できたことの確かめ方（テストが通る、など）・目安の時間。
  最後に「解答: [01-4 の解答](answers/01-4.md)」）

## もし〇〇がなかったら？
（その仕組みが無いと何が起きるか。1〜2 段落）

## 理解度チェック
（3〜5 問。答えは ??? note "答え"（折りたたみ）に書く）
```

- `scripts/check_chapters.py`（CI）が、到達目標・見出しの順番・解答ページの有無とリンク・`nav` と「章の一覧」への登録を確かめる
- 解答ページは「# 01-4 解答」で始め、解答の一例と、なぜそうするかを書く。別の書き方でもよい点（テストが通ればよい、など）も書く
- 文体: です・ます。1 文 60 字程度まで。専門用語は初出で説明する。Java などほかの言語での書き方と比べると伝わりやすい
- 1 章は 150〜400 行を目安。長くなる場合は分割を相談する
- 部の `docs/<NN>-<名前>/index.md` は「到達目標・前提・章の一覧・成果物・目安時間」の見出しをそろえる（13 は「課題」「提出物」）

## 費用の方針（従量課金の API）

- 演習は **無料の範囲で完結** させる。LLM はローカルのモデル（Ollama など）を基本にする
- 従量課金の API を使う演習は、見出しか admonition で **「任意」** と明記し、ローカルのモデルで同じことをする道も書く:

    ```markdown
    !!! warning "任意: 従量課金の API を使います"
        ローカルのモデルだけで進める場合は、この手順を飛ばしてください（成果物は作れます）。
        試す場合は、先に費用の上限と止め方を決めてください（学び方の「従量課金の API は任意」）。
    ```

- 費用がかかる手順には、目安（1 回あたりのトークン数など）と、上限の設定・キーの無効化の手順を同じ章に書く
- 演習のコードのテストと CI は LLM を呼ばない。LLM を使う部分は `LLMClient` を受け取る形にし、テストでは `FakeLLM`（決まった応答）を使う。
  `examples/tests/conftest.py` がテスト中のネットワーク接続を止める
- キーは `.env` だけに書く（`.env.example` を写す）。本文の例でも本物らしいキーを書かない（`<API キー>` のように伏せる）

## コードの書き方

### 実行できるコードは examples/ に置き、本文から引用する

実行して確かめるコードは `examples/` に実物を置き、テストを付けます。本文はそこから引用し、直前に引用元をコメントで書きます。
`scripts/check_code_quotes.py`（CI）が、引用と実物が一致するかを確かめます（実物を直したのに本文が古いまま、を防ぐ）。

````markdown
<!-- src: examples/src/roadmap_examples/p01_api_to_csv.py#L52-L57 -->
```python title="examples/src/roadmap_examples/p01_api_to_csv.py" linenums="52"
（引用したコード）
```
````

- パスはリポジトリ直下から書く（`examples/` の下だけ）
- 本文や `linenums` で行番号を示すときは `#L52-L57` で範囲も書く。範囲を書くと、その行と一致するか（行番号のずれ）まで確かめる。
  離れた行の抜粋は `#L10-L12,L20-L25` のように並べる
- 行番号を示さない引用は範囲を省く（`<!-- src: examples/src/roadmap_examples/llm.py -->`）。ファイルのどこかに、そのまま連続してあればよい
- 字下げと空行は比べない。途中を飛ばすときは `# ...`・`# ...（説明）`・`# （中略）` だけの行を置く
- title が `examples/` で始まるブロックは、src コメントが無いと失敗する。説明用に書き換えた例（「例:」「悪い例」「修正案」）には付けない

### 説明用のコード例

src コメントの無い ```` ```python ```` のブロックは、`scripts/check_python_examples.py`（CI）が Python の構文として読めるかを確かめます
（版は `extra.versions.python`）。型・import・名前の有無は見ません。

- 断片でよい。中身を飛ばすときは `...`（Python の式として正しい）か `pass` と `# ...` を書く
- 対話の例（`>>>`）は ```` ```pycon ```` で、シェルのコマンドは ```` ```bash ```` で書く（対象外）
- わざと構文が誤ったコードを見せるときだけ、直前に `<!-- python-syntax: skip 理由 -->` を書く（理由は必須）
- 対象と skip の一覧は `python scripts/check_python_examples.py --list`

### サンプルのデータ

- 架空のものだけを使う（会員・問い合わせ・設計書・課題など）。実在の会社名・顧客名・個人名・メールアドレス・社内の URL・チャットの ID を書かない
- メールアドレスは `example.com`、パスは `<作業フォルダ>` のように伏せる。`banned-terms.txt` の語や形は `scripts/check_chapters.py` が検出する

## 根拠の示し方

本文の説明（「この指標は不均衡なデータで誤解を招く」「この関数は非推奨」など）の正しさは、CI では確かめられません。

- 挙動や仕様の説明には、公式ドキュメント（Python・ライブラリ・ツールの公式サイト）へのリンクを添える
- 動きで示せるものは、`examples/` のコードとテストで示し、本文から引用する（「テストで確かめた」と書ける形にする）
- 数字（精度・費用・速さ）は、測った条件（モデル・データの件数・手元の環境）と一緒に書く。費用の単価のように変わるものは、公式の料金ページへのリンクにして数字を固定しない
- 根拠を添えられない主張は書かないか、PR で根拠を示してレビューを受ける（PR のテンプレートのチェック項目）
- 外部リンクは週 1 回 `scripts/check_links.py` が確かめ、切れていると Issue「外部リンクが切れています（週 1 回の検査）」が起票される

## 版の書き方

本文（コードブロックの外）に版を直接書かず、`mkdocs.yml` の `extra.versions` を参照します。値は演習のコードで実際に使う版です。

```markdown
---
render_macros: true
---

演習は Python {{ versions.python }} と uv {{ versions.uv }} で確かめています。
```

- `{{ versions.* }}` を使うページは、先頭に `render_macros: true` を書く（無いと `{{ }}` がそのまま表示される）
- `scripts/check_versions.py`（CI）が、`extra.versions` と実物（`examples/pyproject.toml`・`examples/.python-version`・`examples/uv.lock`・
  `requirements.txt`・`constraints.txt`・`.github/workflows`・`examples/Dockerfile` の uv のイメージ）を照合し、本文に直接書いた版（「Python 3.12」「pytest 9.1」など）を検出する
- 版を上げるときは、実物と `extra.versions` を同じ PR で直す
- README・CONTRIBUTING などの説明に書いた版は `scripts/check_docs_match.py` が `extra.versions` と照合する

## 依存の固定

- サイトの生成と検査に使うものは `requirements.txt`（範囲）に書く。実際に入れる版とハッシュは `constraints.txt` に固定する。
  `constraints.txt` は `.github/workflows/deps-lock.yml` が自動で作る（`requirements.txt` を変えて main 以外のブランチに push すると、同じブランチにコミットされる）。手で書き換えない
- 演習のコードの依存は `examples/pyproject.toml` に書き、`uv lock` で `examples/uv.lock` を更新する（deps-lock でも更新される）
- 依存の更新は Dependabot が月 1 回 PR で提案する

## 確認

```bash
pip install --require-hashes -r constraints.txt   # サイトの生成と検査（版とハッシュは constraints.txt で固定）
mkdocs serve                                       # http://127.0.0.1:8000 でプレビュー
mkdocs build --strict                              # リンク切れ・設定の誤りで失敗する（CI と同じ）
python scripts/check_chapters.py                   # 章の型・解答・nav・章の一覧と、禁止語（banned-terms.txt）
python scripts/check_code_quotes.py                # 本文に引用したコードが examples/ の実物と一致するか
python scripts/check_python_examples.py            # 説明用の Python のコード例が構文として読めるか
python scripts/check_versions.py                   # 本文の版（extra.versions）が実物と一致するか
python scripts/check_docs_match.py                 # README・CONTRIBUTING などの説明が実物とずれていないか
python scripts/check_links.py site                 # 外部リンク（mkdocs build の後。CI では週 1 回だけ）
python -m pytest                                   # 検査スクリプトのテスト（tests/）

cd examples
uv sync --locked                                   # 演習のコード（Python 3.12・uv 0.8.17）
uv run pytest                                      # 演習のテスト（LLM もネットワークも使わない）
uv run ruff check .. && uv run ruff format --check ..   # リポジトリ全体の静的解析と書式（ruff は examples の版を使う）
```

Windows（PowerShell）でも同じコマンドで動きます（CI の `windows` ジョブで確認）。結果をファイルに保存したりパイプに渡したりするときは、
先に `$env:PYTHONUTF8 = '1'` を実行すると、日本語の出力で止まりません。
