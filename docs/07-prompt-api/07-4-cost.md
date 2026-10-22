# 07-4 費用の最適化（キャッシュ・モデルの使い分け・上限）

!!! abstract "到達目標"
    - トークン数と単価から、LLM を使う処理の費用を見積もれる
    - キャッシュ・モデルの使い分け・上限の 3 つを、コードで組み合わせられる
    - 従量課金の API を試す前に、費用の上限と止め方を決められる

## 何か

LLM の費用は、ほとんどがトークン数で決まります（トークンは 06 で扱いました）。

```text
費用 = 入力のトークン数 × 入力の単価 + 出力のトークン数 × 出力の単価
```

クラウドの API は、使った分だけお金がかかります（従量課金）。単価はモデルごとに違い、入力と出力でも違います。
ローカルのモデル（Ollama）は API の料金がかかりませんが、時間はかかります。電気代や、PC を占有することも費用です。

費用を下げる方法は、大きく 3 つです。

| 方法 | 考え方 | この章のコード |
|---|---|---|
| キャッシュ | 同じ入力に、もう一度お金を払わない | `CachedLLM` |
| モデルの使い分け | 安いモデルで足りる件は、安いモデルで済ませる | `cascade` |
| 上限 | 想定を超えたら止める | `LimitedLLM`（06）・`BudgetedLLM` |

## なぜ必要か

LLM を使う処理は、作り方しだいで費用が何倍にもなります。

- 07-1 の例示を 3 件から 10 件に増やすと、1 件ごとの入力のトークン数が増える。1 万件を処理すれば、その差も 1 万倍になる
- 07-2 の頼み直しや、07-3 のツールの往復は、1 件で何回もモデルを呼ぶ
- 10 のエージェントは、止まらないと、気づかないうちに呼び出しを続ける

業務では「1 件あたりいくらか」「月にいくらまでか」を、作る前に見積もり、動かした後に測ります。
Java のバッチで、1 件あたりの処理時間と件数から全体の時間を見積もるのと同じです。

## どうやるか

### 見積もる

トークン数から費用を計算する関数です。単価は変わるので、コードに固定しません。
使う API の公式の料金ページで確かめた値を、実行のたびに渡します。

<!-- src: examples/src/roadmap_examples/p07_cost.py -->
```python title="examples/src/roadmap_examples/p07_cost.py"
@dataclass(frozen=True)
class Price:
    """100 万トークンあたりの単価（入力・出力）. 通貨は使う API に合わせる."""

    input_per_million: float
    output_per_million: float


FREE = Price(0.0, 0.0)  # ローカルのモデル（電気代などは別）


def cost(input_tokens: int, output_tokens: int, price: Price) -> float:
    return (input_tokens * price.input_per_million + output_tokens * price.output_per_million) / 1_000_000
```

トークン数は、モデルの応答に書かれた数を使います。Ollama は、入力のトークン数を `prompt_eval_count`、
出力のトークン数を `eval_count` で返します（[公式ドキュメント: チャットの API](https://docs.ollama.com/api/chat)）。
`OllamaChat` は、この 2 つを `Completion` の `input_tokens`・`output_tokens` に入れています。

見積もりの手順は次のとおりです。

1. ローカルのモデルで、評価セット（07-5）の数十件を流し、1 件あたりの入力と出力のトークン数を測る
2. 使う予定の API の単価を、公式の料金ページで確かめる
3. 「1 件あたりのトークン数 × 単価 × 月の件数」で、月の費用を見積もる

トークンの数え方はモデルごとに違います（06）。ローカルのモデルで測った数は目安です。
クラウドの API に替えたら、その API が返すトークン数で測り直します。

### キャッシュする

まったく同じメッセージには、前の応答を返します。呼ばないので、費用も時間もかかりません。

<!-- src: examples/src/roadmap_examples/p07_cost.py -->
```python title="examples/src/roadmap_examples/p07_cost.py"
    def complete(self, messages: Sequence[Message]) -> Completion:
        key = cache_key(messages)
        if key in self.store:
            self.hits += 1
            return Completion(self.store[key], 0, 0)
        self.misses += 1
        result = self.inner.complete(messages)
        self.store[key] = result.text
        return result
```

効くのは、同じ入力が何度も来る場面です（同じ文書の要約・開発中に同じ評価セットを何度も流す、など）。
問い合わせのように毎回ちがう文章には、ほとんど効きません。

注意点:

- 鍵はメッセージの中身だけから作っています。モデルや `temperature` を替えたら、キャッシュを空にします（解答の例を参照）
- 1 文字でも違えば別の鍵です。日時のような毎回変わる値をプロンプトに入れると、キャッシュは効きません

API の側にも、先頭が同じ入力を安く扱う仕組み（プロンプトキャッシュ）を持つものがあります。
Ollama も、キャッシュから読んだ入力のトークン数を `prompt_eval_cached_count` で返します（[公式ドキュメント: チャットの API](https://docs.ollama.com/api/chat)）。
クラウドの API の条件と料金は、使う API の公式ドキュメントで確かめてください。
どの場合も、変わらない部分（役割・手順・例示）を前に、変わる部分（問い合わせ文）を後ろに置くと効きやすくなります。
07-1 の `build_messages` は、この順に並べています。

### モデルを使い分ける

すべての件を、いちばん良い（高い）モデルで処理する必要はありません。
安いモデルで先に試し、決めた条件のときだけ高いモデルに回します（カスケードと呼びます）。

<!-- src: examples/src/roadmap_examples/p07_cost.py -->
```python title="examples/src/roadmap_examples/p07_cost.py"
    def predict(text: str) -> Prediction:
        try:
            first = cheap(text)
        except Exception:  # noqa: BLE001 — 安いモデルの失敗（形の崩れなど）は、高いモデルに回す理由になる
            return strong(text)
        if not escalate(first):
            return first
        second = strong(text)
        return Prediction(
            second.output, first.input_tokens + second.input_tokens, first.output_tokens + second.output_tokens
        )
```

高いモデルに回す条件（`escalate`）の例:

- 安いモデルの出力が、スキーマに合わなかった（07-2 で頼み直しても直らない）
- 安いモデルが other（その他）を選んだ。迷った件は other になりやすい
- 07-5 の評価セットで、安いモデルが苦手と分かった種類（タグ）の件

モデルに「自信の度合い」を数字で答えさせる方法もありますが、その数字が当てになるかは、評価セットで確かめてから使います。
使い分けの効果は、正解率と費用の両方で比べます（07-5）。費用は、モデルごとに単価が違うので、
モデルごとに `BudgetedLLM` を挟み、それぞれの `spent` を足して数えます。

### 上限で止める

06 の `LimitedLLM` は、呼び出しの回数とトークン数で止めます。`BudgetedLLM` は、見積もった費用で止めます。

<!-- src: examples/src/roadmap_examples/p07_cost.py -->
```python title="examples/src/roadmap_examples/p07_cost.py"
    def complete(self, messages: Sequence[Message]) -> Completion:
        if self.spent >= self.max_cost:
            raise BudgetExceeded(f"費用の見積もりが上限（{self.max_cost}）に達しました（{self.spent:.4f}）")
        result = self.inner.complete(messages)
        self.spent += cost(result.input_tokens, result.output_tokens, self.price)
        return result
```

1 回の費用は、呼んだ後でないと分かりません。そのため、合計は上限を 1 回分だけ超えることがあります。
テストでも、この動きを確かめています（`examples/tests/test_p07_cost.py`）。

<!-- src: examples/tests/test_p07_cost.py -->
```python title="examples/tests/test_p07_cost.py"
def test_budgeted_llm_stops_when_budget_is_used() -> None:
    fake = FakeLLM.of(["x" * 400] * 5)  # 1 回の出力はおよそ 100 トークン → 0.04
    llm = BudgetedLLM(fake, PRICE, max_cost=0.05)
    llm.complete([Message("user", "1")])
    llm.complete([Message("user", "2")])  # 上限に達していないので呼ぶ（合計は上限を 1 回分超える）
    with pytest.raises(BudgetExceeded, match="上限"):
        llm.complete([Message("user", "3")])
    assert len(fake.calls) == 2
```

部品はどれも `LLMClient` を受け取って `LLMClient` になるので、重ねて使えます。

```python
from roadmap_examples.llm import LimitedLLM
from roadmap_examples.ollama_client import OllamaChat
from roadmap_examples.p07_cost import FREE, BudgetedLLM, CachedLLM

price = FREE  # クラウドの API なら、公式の料金ページで確かめた単価を Price(入力, 出力) で入れる
limited = LimitedLLM(OllamaChat("<モデル名>"), max_calls=100, max_tokens=200_000)
llm = CachedLLM(BudgetedLLM(limited, price, max_cost=1.0))
```

重ねる順番で動きが変わります。この例では、キャッシュが外側にあるので、キャッシュから返した分は上限に数えません。
どの順番が良いかは、演習で考えます。

### 任意（有料）: クラウドの API を試す

!!! warning "任意: 従量課金の API を使います"
    ローカルのモデルだけで進める場合は、この節を飛ばしてください（07 の成果物は作れます）。
    試す場合は、先に費用の上限と止め方を決めてください（[学び方](../guide.md) の「従量課金の API は任意」）。

クラウドの API を試すときは、呼び出す前に、次の 4 つを決めて実行します。

1. **上限の金額を決める**: 例えば「この演習で使うのは月に 500 円まで」。
   評価セット 30 件を 1 回流すトークン数を、ローカルのモデルで測っておくと、何回流せるかが分かる
2. **API の管理画面で、利用の上限と通知（予算アラート）を設定する**: 上限の 50%・80% で通知が届くようにする。
   設定の場所と、上限で止まるのか通知だけなのかは、API ごとに違う。使う API の公式ドキュメントで確かめる
3. **キーは `.env` にだけ書く**: `.env.example` を写して使い、コミットしない。本文やコードに、キーの値を書かない
4. **止め方を先に確かめる**: 管理画面でキーを無効にする・削除する手順を、使い始める前に一度見ておく。使い終わったらキーを消す

コードの側でも、管理画面の上限とは別に止めます。管理画面の通知は、届くまでに時間がかかることがあるからです。

- `BudgetedLLM` の `max_cost` に、1 回の実行で使ってよい金額を入れる（単価は公式の料金ページで確かめる）
- `LimitedLLM` で、呼び出しの回数とトークン数にも上限を設ける
- 07 の成果物のスクリプトは、`--price-in`・`--price-out` で単価を渡すと、費用の見積もりを表示する

クラウドの API で動かすには、その API の公式の SDK を使って、`complete(messages)` を持つクラス（`LLMClient`）を書きます。
`OllamaChat` と差し替えるだけで、プロンプト・検証・上限・評価のコードはそのまま使えます。

## 演習

1. `CachedLLM`・`BudgetedLLM`・`LimitedLLM` を重ねる順番を 2 通り考え、それぞれで次の動きになるかを、テストで確かめてください
    - 同じメッセージで 2 回呼んだとき、2 回目は内側の LLM を呼ばない
    - 2 回目（キャッシュから返した分）は、`BudgetedLLM.spent` を増やさない
2. `CachedLLM` の鍵に、モデル名を含める方法を考えてください（モデルを替えたときに、古い答えを返さないため）
3. 任意: 07-5 の評価セットをローカルのモデルで流し、1 件あたりの入力と出力のトークン数の平均を出して、
   「月 1 万件を処理したら、何トークンか」を見積もってください

- できたことの確かめ方: `uv run pytest` が通る。3 は、見積もりの式と、測った条件（モデル名・件数）
- 目安の時間: 2 時間

解答: [07-4 の解答](answers/07-4.md)

## もし〇〇がなかったら？

**上限がなかったら**、ループの誤りや想定外の入力で呼び出しが続いたとき、気づくのは請求が届いてからです。
管理画面の通知だけでは、通知が届くまでの間も呼び出しは続きます。コードの上限は、その間も止めてくれます。

**見積もりがなかったら**、「動いたけれど、月の費用が予算の 10 倍だった」ことが、作り終えた後に分かります。
プロンプトの書き方やモデルの選び方を、作り直すことになります。

## 理解度チェック

1. 例示を 3 件から 10 件に増やすと、費用はどう変わりますか。

    ??? note "答え"
        1 件ごとの入力のトークン数が増えるので、処理する件数に比例して費用が増えます。
        正解率がどれだけ上がったかと合わせて、増やす価値があるかを評価セットで判断します。

2. 問い合わせの分類に、`CachedLLM` があまり効かないのはなぜですか。

    ??? note "答え"
        問い合わせ文は毎回ちがうので、まったく同じメッセージが来ることが少ないからです。
        同じ入力が繰り返し来る場面（開発中の評価の繰り返しなど）では効きます。

3. `BudgetedLLM` の費用の合計が、上限を超えることがあるのはなぜですか。

    ??? note "答え"
        1 回の費用は、呼んだ後で返るトークン数を見ないと分からないからです。呼ぶ前に上限に達していなければ呼ぶので、
        その 1 回分だけ超えることがあります。超えて困る場合は、上限を 1 回の最大の費用だけ低くしておきます。

4. クラウドの API の管理画面に上限と通知を設定しても、コードに上限を書くのはなぜですか。

    ??? note "答え"
        管理画面の上限や通知は、設定の内容が API ごとに違い、通知が届くまでに時間がかかることもあるからです。
        コードの上限なら、その実行の中ですぐに止まります。2 つを重ねて使います。
