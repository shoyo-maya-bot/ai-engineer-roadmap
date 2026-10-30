# 07-5 小さな評価セットを最初に作る

!!! abstract "到達目標"
    - 正解つきの小さな評価セットを、JSONL の決まった形式で作れる
    - 1 件ごとの点と平均で採点し、前回の結果と比べて後退した件を見つけられる
    - LLM と比べる基準（07 では仮のルールベース分類）を、同じ評価セットで、精度と費用の両方で比べられる

## 何か

評価セットは、「この入力には、この答えが正しい」という組を集めたものです。
プロンプト・モデル・設定を変えるたびに同じ評価セットを流し、点数で良し悪しを判断します。

この教材では、評価セットの形式と採点の仕方を、ここで 1 つに決めます。08（RAG）・09（ファインチューニング）・11（安全と評価）でも同じ形式を使います。

| 決めごと | 内容 |
|---|---|
| ファイル | JSONL（1 行に 1 件の JSON） |
| 1 件の形 | `id`（一意な名前）・`input`（入力）・`expected`（正解）・`tags`（任意の分類） |
| 採点 | 1 件ごとに 0.0〜1.0 の点を付け、平均する。点の付け方は課題ごとに決める |
| 結果 | 1 行目にまとめ、2 行目から 1 件ずつの JSONL で残す。前回の結果と比べられる |

07 の評価セットは `examples/data/p07/inquiries.jsonl`（架空の問い合わせ 30 件）です。

```json
{"id": "c005", "input": "解約したはずの月の分まで引き落とされています。", "expected": {"category": "billing"}, "tags": ["紛らわしい"]}
```

`expected` の中身は課題ごとに決めます。分類なら `category`、08 の RAG なら「答えの出典になる文書」などを入れます。

## なぜ必要か

LLM の出力は、毎回少しずつ違います。2〜3 件を手で試して「良くなった気がする」では、判断を誤ります。

- 例示を足したら、紛らわしい件は良くなったが、簡単な件が 2 件悪くなった（気づかない）
- 安いモデルに替えたら、費用は半分になったが、正解率も 20 ポイント下がった（測っていない）
- 指示の混じった問い合わせで、分類が乗っ取られていた（試していない）

だから評価セットは「最初に」作ります。プロンプトを書く前に作れば、最初の版から点数で比べられます。
Java の開発で、実装の前にテストを書くのと同じ考え方です。

30 件のように小さくてかまいません。小さければ、ローカルのモデルでも数分で流せて、何度も比べられます。
誤りを見つけるたびに件数を増やし、育てていきます。

## どうやるか

### 評価セットを作る

件を集めるときは、次の 3 種類を混ぜます。`tags` で分けておくと、種類ごとの点数を出せます。

| タグ | 中身 | 07 の件数 |
|---|---|---|
| 基本 | 迷わずに分けられる件。各分類から同じくらい | 19 |
| 紛らわしい | 2 つの分類にまたがる件（「解約したのに引き落とされた」は billing か contract か） | 9 |
| 指示混入 | 問い合わせ文に、分類を変えさせる指示が書かれた件 | 2 |

注意点:

- 正解は、作った人とは別の人が見直す。正解そのものが割れる件は、判断の基準を決めてから入れる
- 07-1 の例示に使った文を、評価セットに入れない（答えを見せた試験になる）
- 本物の問い合わせを使うときは、名前・連絡先などの個人の情報を消してから入れる（この教材の評価セットはすべて架空です）

読み込みでは、形式の誤りを行番号つきで止めます。`id` の重複も誤りにします（結果を比べるときの目印なので）。

<!-- src: examples/src/roadmap_examples/evalset.py -->
```python title="examples/src/roadmap_examples/evalset.py"
def load_cases(path: Path) -> list[EvalCase]:
    """JSONL の評価セットを読む. 空行は飛ばす. 形の誤りと id の重複は行番号つきで止める."""
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        where = f"{path.name}:{lineno}"
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise EvalSetError(f"{where}: JSON として読めません（{e.msg}）") from e
        case = parse_case(obj, where)
        if case.id in seen:
            raise EvalSetError(f"{where}: id「{case.id}」が重複しています")
        seen.add(case.id)
        cases.append(case)
```

### 採点する

1 件の点の付け方（scorer）を関数で渡します。分類は「正解と同じなら 1 点」です。

<!-- src: examples/src/roadmap_examples/evalset.py -->
```python title="examples/src/roadmap_examples/evalset.py"
def exact_match(key: str) -> Scorer:
    """expected[key] と出力（dict なら出力[key]）が同じなら 1 点."""

    def score(case: EvalCase, output: Any) -> float:
        got = output.get(key) if isinstance(output, dict) else output
        return 1.0 if got == case.expected.get(key) else 0.0

    return score
```

点を 0.0〜1.0 にそろえておくと、部分点のある課題（08 の「出典を 3 つ中 2 つ見つけた」など）も、同じ平均で扱えます。

`run_eval` は全件を順に予測して採点します。1 件が失敗（例外）しても 0 点として記録し、残りを続けます。
ただし、費用や回数の上限に達したときは、続けても無駄なので、`stop_on` に書いた例外で止めます。

### 結果を残し、前回と比べる

結果は JSONL で残します。平均点だけでなく、1 件ずつの出力も残すと、何が変わったかを後から調べられます。
前回の結果と比べて、点が下がった件（後退）を先に並べます。

<!-- src: examples/src/roadmap_examples/evalset.py -->
```python title="examples/src/roadmap_examples/evalset.py"
def compare(before: Report, after: Report) -> list[tuple[str, float, float]]:
    """両方にある件のうち、点が変わったもの（id, 前, 後）. 点が下がった件（後退）を先に並べる."""
    old = {r.id: r.score for r in before.results}
    changed = [(r.id, old[r.id], r.score) for r in after.results if r.id in old and old[r.id] != r.score]
    return sorted(changed, key=lambda c: (c[2] - c[1], c[0]))
```

平均が上がっていても、後退した件があれば中身を読みます。平均だけを見ると、大事な件の悪化を見落とします。

### 07 の成果物: 比べる基準と LLM を比べる

07 の成果物は、04 の分類を LLM でやり直し、同じ評価セットで精度と費用を比べることです。

!!! note "比べる基準について"
    04 を進めた人は、04 の分類モデルを基準にします。差し替え方（04 の結果のファイルを `--compare` に渡す、
    または `baseline_predict` の代わりに 04 のモデルの予測を渡す）は [04-4](../04-ml/04-4-metrics-validation.md) の「07 の基準を差し替える」にあります
    （評価セットと採点はそのまま使えます）。近道で 04 を飛ばした人は、キーワードで振り分ける **仮のルールベース分類** を基準にします。

<!-- src: examples/src/roadmap_examples/p07_classify.py -->
```python title="examples/src/roadmap_examples/p07_classify.py"
def rule_based(text: str) -> str:
    for category, words in RULES:
        if any(w in text for w in words):
            return category
    return "other"


def baseline_predict(case: EvalCase) -> Prediction:
    return Prediction({"category": rule_based(case.input)})


def llm_predict(llm: LLMClient, spec: PromptSpec = DEFAULT_SPEC, max_attempts: int = 2) -> Predict:
    """LLM で分類する予測の関数を作る（プロンプトは 07-1、検証と頼み直しは 07-2）."""

    def predict(case: EvalCase) -> Prediction:
        result = ask_json(llm, build_messages(spec, case.input), CLASSIFY_SCHEMA, max_attempts)
        return Prediction(result.value, result.input_tokens, result.output_tokens)

    return predict
```

基準と LLM を、同じコマンドで流します（`examples/` で実行）。LLM の側は、呼び出しの回数とトークン数に上限を設けて動きます。

```bash
uv run python -m roadmap_examples.p07_classify --mode baseline --out results-baseline.jsonl
uv run python -m roadmap_examples.p07_classify --mode ollama --model <モデル名> \
    --out results-llm.jsonl --compare results-baseline.jsonl
```

仮のルールベース分類の結果は、テストで確かめています（30 件中 20 件が正解。紛らわしい件に弱い）。

<!-- src: examples/tests/test_p07_classify.py -->
```python title="examples/tests/test_p07_classify.py"
def test_main_baseline_writes_results(tmp_path: Path) -> None:
    out = tmp_path / "baseline.jsonl"
    stdout = io.StringIO()
    assert main(["--mode", "baseline", "--out", str(out)], stdout=stdout) == 0
    assert "正解率: 66.7%（20 / 30 件）" in stdout.getvalue()
    assert load_results(out).meta == {"mode": "baseline"}
```

LLM の結果は、モデルと手元の環境で変わります。次の表を自分で埋めて、成果物にします。
04 を進めた人は、基準の列を「04 のモデル」（評価セットで 80.0%。04-4）にして埋めます。

| 項目 | 仮のルールベース分類 | LLM（モデル名） |
|---|---|---|
| 正解率（全体） | 66.7% | （測った値） |
| 正解率（紛らわしい） | （表示された値） | （測った値） |
| 正解率（指示混入） | （表示された値） | （測った値） |
| 入力・出力のトークン数 | 0 | （測った値） |
| 費用の見積もり | 0 | ローカルは 0。クラウドの単価なら（`--price-in`・`--price-out`） |
| 時間（30 件） | （表示された値） | （測った値） |

表には、測った条件（モデル名・件数・`temperature`・手元の PC）も書き添えます。
「LLM は正解率が高いが、1 件ごとに時間とトークンがかかる」のように、精度と費用の両方で結論を書きます。

## 演習

1. 評価セットを写し、自分で考えた問い合わせを 10 件足してください（紛らわしい件を半分以上）。
   `load_cases` で読めることをテストで確かめます
2. 仮のルールベース分類と LLM を、40 件の評価セットで比べ、上の表を埋めてください。
   LLM はローカルのモデルで動かします（任意: クラウドの API。07-4 の上限を設定してから）
3. 07-1 のプロンプトの部品を 1 つ変え、`--compare` で前の結果と比べて、後退した件を読んでください

- できたことの確かめ方: `uv run pytest` が通る。2 は埋めた表、3 は後退した件とその理由の考察
- 目安の時間: 3 時間

解答: [07-5 の解答](answers/07-5.md)

## もし〇〇がなかったら？

**評価セットがなかったら**、プロンプトやモデルを変えるたびに、数件を手で試して判断することになります。
試した件が良くなれば、ほかの件の悪化に気づかないまま採用します。後で「先月より分類がおかしい」と言われても、
いつ、どの変更で悪くなったのかを調べる手がかりがありません。

**形式が部ごとにばらばらだったら**、08 や 11 で評価のコードを毎回作り直すことになり、
部をまたいで「同じ問いへの答えが良くなったか」を比べられません。

## 理解度チェック

1. 評価セットを、プロンプトを書く「前」に作るのはなぜですか。

    ??? note "答え"
        最初の版から点数で比べられるようにするためです。後から作ると、それまでの変更が良かったのか悪かったのかが分かりません。
        また、プロンプトに合わせて評価セットを選んでしまう（都合の良い件だけ入れる）ことも防げます。

2. 1 件の点を 0.0〜1.0 にそろえるのはなぜですか。

    ??? note "答え"
        正解か不正解かの課題（分類）と、部分点のある課題（08 の出典の数など）を、同じ「平均」で扱えるようにするためです。
        評価のコード（読み込み・結果の保存・比較）を、部をまたいで使い回せます。

3. 平均点が上がったのに、変更を採用する前に結果を読むのはなぜですか。

    ??? note "答え"
        平均が上がっても、点が下がった件（後退）があるかもしれないからです。
        後退した件が大事な件（指示混入など）なら、平均が上がっても採用しないこともあります。

4. 07 で LLM と比べる基準は何ですか。04 を飛ばした場合は、どうしますか。

    ??? note "答え"
        04 の分類モデルです。04 の結果のファイルを `--compare` に渡すか、`baseline_predict` の代わりに 04 のモデルの予測を渡します（04-4）。
        04 を飛ばした場合は、`--mode baseline` の仮のルールベース分類を基準にします。どちらでも、評価セットと採点はそのまま使います。
