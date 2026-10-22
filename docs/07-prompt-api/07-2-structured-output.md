# 07-2 構造化出力とスキーマ検証

!!! abstract "到達目標"
    - LLM の出力の形を JSON スキーマで決め、モデルに渡せる
    - 受け取った JSON をスキーマで確かめ、合わなければ誤りを伝えて頼み直すコードを書ける
    - 「形を指定する」と「形を確かめる」の両方が要る理由を説明できる

## 何か

構造化出力（structured output）は、LLM に決まった形のデータ（多くは JSON）で答えさせることです。
形は JSON スキーマで書きます。JSON スキーマは「どのキーがあり、値はどの型か」を JSON で書く決まりです。

07 の分類では、出力を次の形に決めます。

<!-- src: examples/src/roadmap_examples/p07_structured.py -->
```python title="examples/src/roadmap_examples/p07_structured.py"
CLASSIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "reason": {"type": "string", "maxLength": 80},
    },
    "required": ["category", "reason"],
    "additionalProperties": False,
}
```

- `enum`: 選べる値の一覧。分類の 5 つのキーのどれか
- `required`: 必ずあるキー
- `additionalProperties: false`: 決めていないキーを許さない

Java の経験があれば、JSON を受け取るクラス（DTO）と、Bean Validation の注釈を思い浮かべてください。
JSON スキーマは、それを言語に依らない JSON で書いたものです。

## なぜ必要か

LLM の出力を業務のコードに渡すには、形が毎回そろっている必要があります。
「billing」と返る回と「請求」と返る回があると、後ろの処理（担当の振り分け・集計）が動きません。

07-1 のプロンプトでも、ほとんどの回は JSON で返ります。しかし「ほとんど」では困ります。
1 日に 1 万件を処理すれば、1% の崩れでも 100 件です。

そこで、次の 2 つを組み合わせます。

| 方法 | どこで | 役目 |
|---|---|---|
| 形を指定する | モデルに渡す（Ollama の `format`） | 崩れにくくする |
| 形を確かめる | 受け取った後のコード | 崩れたら気づき、頼み直すか止める |

指定だけでは足りません。Ollama の公式ドキュメントの例も、受け取った後に検証しています
（[公式ドキュメント: 構造化出力](https://docs.ollama.com/capabilities/structured-outputs)）。
外部から受け取るデータは、相手が約束していても確かめる。API の入力と同じ考え方です。

## どうやるか

### 形をモデルに指定する

Ollama のチャットの API は、`format` に JSON スキーマを渡せます
（[公式ドキュメント: チャットの API](https://docs.ollama.com/api/chat) の `format`）。
演習のコードの `OllamaChat` は、`format` をそのまま送ります。

```python
from roadmap_examples.ollama_client import OllamaChat
from roadmap_examples.p07_structured import CLASSIFY_SCHEMA

llm = OllamaChat("<モデル名>", format=CLASSIFY_SCHEMA, options={"temperature": 0})
```

公式ドキュメントは、スキーマをプロンプトの中にも文字列で書くとよいとしています。
07-1 の system に出力の形を書いたのは、このためです。

### 受け取った JSON を確かめる

確かめる側は、スキーマのうち、よく使う一部だけを読む小さな関数 `validate` で書きました。
返り値は「合わない点の一覧」で、空なら合っています。

<!-- src: examples/src/roadmap_examples/p07_structured.py -->
```python title="examples/src/roadmap_examples/p07_structured.py"
def validate(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """スキーマに合わない点の一覧（空なら合っている）. 場所は $.category のように書く."""
    errors: list[str] = []
    kind = schema.get("type")
    if kind and not _is_type(value, kind):
        return [f"{path}: 型が {kind} ではありません"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} は選べる値（{', '.join(map(str, schema['enum']))}）にありません")
    # ...（文字数・数の範囲）
    if isinstance(value, dict):
        props = schema.get("properties", {})
        errors += [f"{path}: 「{key}」がありません" for key in schema.get("required", []) if key not in value]
        if schema.get("additionalProperties") is False:
            errors += [f"{path}: 「{key}」は決めていない項目です" for key in value if key not in props]
        for key, sub in props.items():
            if key in value:
                errors += validate(value[key], sub, f"{path}.{key}")
```

ここでは依存を増やさないために自分で書きました。業務では、JSON スキーマの検証ライブラリ（jsonschema）や、
型から検証とスキーマの両方を作るライブラリ（Pydantic）を使うのがふつうです。
どれを使っても「受け取ったら確かめる」場所は同じです。

!!! failure "悪い例"
    ```python
    value = json.loads(result.text)
    route_to(value["category"])
    ```

    JSON として読めない・キーが無い・`category` が一覧に無い、のどれでも、ここで例外になるか、
    知らない部署名で振り分けが進みます。どの件で何が起きたのかも残りません。

### 合わなければ、誤りを伝えて頼み直す

形が崩れたら、モデルの答えと「どこが合わないか」を会話に足して、もう一度頼みます。
回数には上限を設けます。上限まで直らなければ、例外にして呼んだ側に任せます。

<!-- src: examples/src/roadmap_examples/p07_structured.py -->
```python title="examples/src/roadmap_examples/p07_structured.py"
def ask_json(
    llm: LLMClient, messages: Sequence[Message], schema: dict[str, Any], max_attempts: int = 2
) -> StructuredResult:
    """JSON で答えさせ、スキーマで確かめる. 合わなければ誤りを伝えて頼み直す（合計 max_attempts 回まで）."""
    history = list(messages)
    tokens_in = tokens_out = 0
    text = ""
    for attempt in range(1, max_attempts + 1):
        result = llm.complete(history)
        tokens_in += result.input_tokens
        tokens_out += result.output_tokens
        text = result.text
        try:
            value = parse_json(text)
            problems = validate(value, schema)
        except json.JSONDecodeError as e:
            problems = [f"JSON として読めません（{e.msg}）"]
        if not problems:
            return StructuredResult(value, attempt, tokens_in, tokens_out)
        history += [
            Message("assistant", text),
            Message(
                "user", "出力が決めた形に合いません。直して JSON だけを出力してください:\n- " + "\n- ".join(problems)
            ),
        ]
    raise StructuredOutputError(f"{max_attempts} 回試しても、形に合う JSON が返りませんでした", text)
```

- 頼み直しも費用がかかります。トークン数は、頼み直した分も足して返します（07-4 で使う）
- 上限の回数は小さく（2〜3 回）します。頼み直すたびに、費用と待ち時間が増えるからです。
  何回目で直ったかは `attempts` に残るので、評価セットで回数の決め方を確かめられます
- 最後の出力（`last_text`）を例外に残します。後で原因を調べられます

頼み直しの流れは、`FakeLLM` に「崩れた応答 → 正しい応答」を順に返させて確かめています。

<!-- src: examples/tests/test_p07_structured.py -->
```python title="examples/tests/test_p07_structured.py"
def test_ask_json_retries_with_the_problems() -> None:
    fake = FakeLLM.of(["カテゴリは billing です", json.dumps(GOOD, ensure_ascii=False)])
    result = ask_json(fake, [Message("user", "分類して")], CLASSIFY_SCHEMA)
    assert result.attempts == 2
    retry = fake.calls[1]
    assert [m.role for m in retry] == ["user", "assistant", "user"]
    assert "JSON として読めません" in retry[-1].content
```

### 形が合っても、中身が正しいとは限らない

スキーマで確かめられるのは形だけです。`{"category": "billing"}` が形に合っていても、正解が bug なら誤りです。
中身の正しさは、正解つきの問い合わせで測ります。それが 07-5 の評価セットです。

## 演習

1. `CLASSIFY_SCHEMA` を写して、急ぎかどうかを表す `urgency`（`"high"` か `"normal"`）を必須のキーに足したスキーマを作ってください
2. そのスキーマで、次の 3 つを確かめるテストを書いてください
    - `{"category": "bug", "reason": "x", "urgency": "high"}` は合う
    - `urgency` が無いと「「urgency」がありません」が返る
    - `FakeLLM` が 1 回目に `urgency` の無い JSON、2 回目に正しい JSON を返すと、`ask_json` は 2 回目の値を返す
3. 任意: Ollama で、`format` を渡す場合と渡さない場合に、評価セットの 30 件で形が崩れた回数を数えてください

- できたことの確かめ方: `uv run pytest` が通る。3 は、崩れた回数と、使ったモデル名
- 目安の時間: 1.5 時間

解答: [07-2 の解答](answers/07-2.md)

## もし〇〇がなかったら？

**検証がなかったら**、崩れた出力はそのまま後ろの処理に流れます。
知らない分類名で振り分けが止まる、集計の表に「請求関係」という列が増える、のような事故が、
しばらく経ってから見つかります。どの入力で崩れたのかも分かりません。

**頼み直しの上限がなかったら**、直らないモデルに何度も頼み続け、時間と費用が際限なく増えます。
上限で止めて例外にすれば、その件だけを人が見る、という扱いにできます。

## 理解度チェック

1. Ollama の `format` で形を指定しているのに、受け取った後でも検証するのはなぜですか。

    ??? note "答え"
        指定は崩れにくくするための仕組みで、必ず形が合うとは書かれていないからです。
        また、モデルや API を差し替えたときに、指定の効き方が変わることもあります。外から受け取るデータは確かめます。

2. `additionalProperties: false` にすると、何を防げますか。

    ??? note "答え"
        決めていないキー（`score` や `memo` など）が混じった出力を、誤りとして見つけられます。
        モデルが勝手に項目を足しても、気づかずに通すことがなくなります。

3. 頼み直しのメッセージに、モデルの前の答え（assistant）を入れるのはなぜですか。

    ??? note "答え"
        モデルが「自分が何を出力し、どこが誤りだったか」を見て直せるようにするためです。
        前の答えが無いと、誤りの説明だけが届き、何を直すのかが伝わりにくくなります。

4. スキーマに合った出力が、誤りであることはありますか。

    ??? note "答え"
        あります。スキーマは形だけを確かめます。`category` が一覧の値でも、正解と違えば誤りです。
        中身の正しさは、正解つきの評価セット（07-5）で測ります。
