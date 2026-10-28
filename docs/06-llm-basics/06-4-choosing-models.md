# 06-4 モデルの選び方（ローカルとクラウド、精度と費用）

!!! abstract "到達目標"
    - ローカルのモデルとクラウドの API の違いを、精度・費用・速さ・データの置き場所で比べられる
    - モデルの大きさと量子化から、必要なメモリを見積もれる
    - 同じ質問の答えとトークン数をモデルごとに測る表を作り、それをもとに使うモデルを決められる

## 何か

LLM を使うとき、最初に決めるのは「どのモデルを、どこで動かすか」です。

| 置き場所 | 例 | 費用 | データ | 精度・速さ |
|---|---|---|---|---|
| ローカル | 手元の PC で Ollama を動かす | API の料金はかからない（PC と電気代） | 手元から出ない | PC のメモリに入る大きさまで。大きいモデルほど遅い |
| クラウドの API | 事業者が動かすモデルを HTTP で呼ぶ | トークン数に応じた従量課金（07-4） | 事業者に送る。利用規約とデータの扱いを確かめる | 大きく精度の高いモデルを使える。速さは回線と混み具合しだい |

モデルの大きさは、**パラメータ**（学習で決まる数。重みとも呼びます）の数で表します。
`8B` は 80 億（B は billion、10 億）です。大きいほど知識が多く、複雑な指示に強い傾向がありますが、メモリを多く使い、遅くなります。

**量子化** は、パラメータ 1 つを少ないビット数で表して、モデルを小さくする方法です。16 ビットを約 4 ビットにすれば、
重みのメモリはおよそ 4 分の 1 になります。その代わり、精度が少し下がることがあります。

## なぜ必要か

「いちばん大きくて新しいモデル」がいつも正解とは限りません。

- 問い合わせの分類のような決まった仕事なら、小さなモデルで足りることが多い。足りるかどうかは、評価セットで測って決める（07-5）
- メモリに入らないモデルは、CPU とメモリの間でやり取りが増え、とても遅くなる。演習が進まなくなる
- 社外に出せないデータを扱うなら、クラウドの API が使えないこともある
- クラウドの API は、月の件数が増えると費用も比例して増える（07-4）

この教材の演習は、ローカルのモデルで進めます。06 の最後に、手元で動く 2〜3 個のモデルを同じ質問で比べ、07 以降で使うモデルを決めます。

## どうやるか

### メモリを見積もる

重みだけのメモリは、「パラメータの数 × 1 つあたりのビット数 ÷ 8」で見積もれます。

<!-- src: examples/src/roadmap_examples/p06_compare.py -->
```python title="examples/src/roadmap_examples/p06_compare.py"
def weights_gb(params_billion: float, bits: float) -> float:
    """重み（パラメータ）だけのメモリの目安（GB）. 実際は、これにコンテキストの分（06-2）と実行の分が足される."""
    return params_billion * bits / 8
```

例えば 8B のモデルを 16 ビットで持つと約 16 GB、約 4.9 ビットの量子化（`Q4_K_M`）なら約 4.9 GB です。
量子化の名前ごとの 1 つあたりのビット数は、llama.cpp の量子化ツールの説明に表があります
（[公式ドキュメント: llama-quantize](https://github.com/ggml-org/llama.cpp/tree/master/tools/quantize)。
Ollama は、モデルの取り込みでこのツールを案内しています: [公式ドキュメント: モデルの取り込み](https://docs.ollama.com/import)）。

実際には、重みのほかに、コンテキストの分（長いほど多い）と実行に使う分が要ります。見積もりには余裕を持たせ、
動かした後に `ollama ps` で、GPU とメモリのどちらに載ったかを確かめます（[公式ドキュメント: FAQ](https://docs.ollama.com/faq)）。

| `ollama ps` の Processor | 意味 |
|---|---|
| `100% GPU` | 全部が GPU に載っている（速い） |
| `100% CPU` | 全部がふつうのメモリに載っている（GPU が無い PC ではこれ） |
| `48%/52% CPU/GPU` | 両方にまたがっている（GPU に入りきらなかった） |

### モデルの詳細を調べる

Ollama のモデルの詳細（`/api/show`）には、大きさ・量子化・扱える最大のコンテキスト長・できること（ツール使用など）が書かれています
（[公式ドキュメント: モデルの詳細](https://docs.ollama.com/api-reference/show-model-details)）。

<!-- src: examples/src/roadmap_examples/p06_compare.py -->
```python title="examples/src/roadmap_examples/p06_compare.py"
def model_details(model: str, base_url: str, transport: Transport, timeout: float = 30.0) -> dict[str, str]:
    """Ollama の /api/show から、大きさ・量子化・コンテキスト長・できること（capabilities）を取り出す."""
    data: dict[str, Any] = transport.post_json(f"{base_url.rstrip('/')}/api/show", {"model": model}, timeout)
    details = data.get("details", {})
    info = data.get("model_info", {})
    context = next((str(v) for k, v in info.items() if k.endswith(".context_length")), "")
    return {
        "parameter_size": str(details.get("parameter_size", "")),
        "quantization": str(details.get("quantization_level", "")),
        "context_length": context,
        "capabilities": ", ".join(data.get("capabilities", [])),
    }
```

コンテキスト長のキーは `<モデルの種類>.context_length` の形なので、末尾で探しています。
ここで分かるのは、モデルが扱える最大の長さです。実際に使う長さは Ollama の設定で決まります（06-2）。

07-3 のツール使用と 10 のエージェントでは、`capabilities` に `tools` があるモデルが要ります。

### 上限を決めてから呼ぶ

比べるときも、呼び出しの回数とトークン数に上限を設けます。`LimitedLLM` は、上限に達したら LLM を呼ばずに止めます。

<!-- src: examples/src/roadmap_examples/llm.py -->
```python title="examples/src/roadmap_examples/llm.py"
    def complete(self, messages: Sequence[Message]) -> Completion:
        if self.calls >= self.max_calls:
            raise LimitExceeded(f"呼び出しの回数が上限（{self.max_calls} 回）に達しました")
        if self.tokens >= self.max_tokens:
            raise LimitExceeded(f"トークン数が上限（{self.max_tokens}）に達しました")
        result = self.inner.complete(messages)
        self.calls += 1
        self.tokens += result.input_tokens + result.output_tokens
        return result
```

ローカルのモデルなら料金はかかりませんが、誤ったループで PC を長く占有するのを防げます。
クラウドの API に替えたときは、この上限がそのまま費用の上限になります（07-4 で、費用の見積もりの上限 `BudgetedLLM` も足します）。

### 同じ質問で比べる（06 の成果物）

06 の成果物は、同じ質問をモデルごとに聞き、答えとトークン数を並べる表です。
質問は `examples/data/p06/questions.jsonl`（8 問）で、07-5 の評価セットと同じ形式です。日本語と英語の同じ質問の組、計算、要約、
分類、モデルが知るはずのないこと（架空の会社の売上）を入れています。

<!-- src: examples/src/roadmap_examples/p06_compare.py -->
```python title="examples/src/roadmap_examples/p06_compare.py"
def run_compare(
    models: Sequence[str],
    cases: Sequence[EvalCase],
    make_llm: Callable[[str], LLMClient],
    max_tokens: int,
    clock: Callable[[], float] = time.perf_counter,
) -> list[Row]:
    """モデルごとに、全部の質問を 1 回ずつ聞く. 呼び出しの回数（質問の数）とトークン数に、モデルごとの上限を設ける."""
    rows = []
    for model in models:
        llm = LimitedLLM(make_llm(model), max_calls=len(cases), max_tokens=max_tokens)
        for case in cases:
            started = clock()
            result = llm.complete([Message("user", case.input)])
            seconds = clock() - started
            text = result.text.strip()
            rows.append(
                Row(model, case.id, text, result.input_tokens, result.output_tokens, seconds, keyword_score(case, text))
            )
    return rows
```

- `make_llm` は「モデル名から `LLMClient` を作る関数」です。本物では `OllamaChat`、テストでは `FakeLLM` を返します
- `clock`（時計）も差し替えられます。テストでは決まった秒数が進む時計を渡し、毎回同じ表になることを確かめています
- 「語」の列は、答えに入るべき語（`keywords`）の割合です。粗い目安で、答えの正しさは表を読んで自分でも確かめます

手元で、2 つ以上のモデルを比べます（`examples/` で実行）。

```bash
ollama pull <モデル名 1>
ollama pull <モデル名 2>
uv run python -m roadmap_examples.p06_compare --model <モデル名 1> --model <モデル名 2> --out compare.md
```

`--num-predict`（1 回の答えの出力トークン数の上限）と `--max-tokens`（モデルごとのトークン数の上限）で、長く動き続けないようにしています。
書き出される表は、次の形です（数字はモデルと PC で変わるので、ここでは形だけを示します）。

| 質問 | モデル | 答え（先頭） | 入力トークン | 出力トークン | 秒 | 語 |
|---|---|---|---:|---:|---:|---:|
| q01 | （モデル名 1） | （答え） | （値） | （値） | （値） | （割合） |
| q01 | （モデル名 2） | （答え） | （値） | （値） | （値） | （割合） |

| モデル | 大きさ | 量子化 | コンテキスト長 | 入力の平均 | 出力の平均 | 秒の合計 | 語の平均 |
|---|---|---|---:|---:|---:|---:|---:|
| （モデル名 1） | （/api/show の値） | （値） | （値） | （値） | （値） | （値） | （割合） |

表から読み取ることの例:

- 同じモデルで、q01（日本語）と q02（英語）の入力トークンがどれだけ違うか（06-1）
- 小さいモデルと大きいモデルで、語の割合と秒の合計がどれだけ違うか。足りる精度で、いちばん速い（安い）ものはどれか
- q07（知らないこと）に「分からない」と答えたか、もっともらしい数字を作ったか（「語」は空欄。答えを読んで判断する）

初回は、モデルをメモリに読み込む時間が q01 の秒数に入ります。秒数を比べるときは、2 回流して 2 回目を使うと公平です。

!!! failure "悪い例"
    「いちばん大きいモデルを使えば間違いない」と、比べずに決める。

    手元のメモリに入らず遅くなったり、小さいモデルで足りる仕事に何倍もの時間（クラウドなら費用）をかけたりします。

!!! success "良い例"
    メモリに入る大きさの候補を 2〜3 個選び、同じ質問で表を作ってから決める。決めた理由（表の数字と条件）を残す。

### 任意（有料）: クラウドの API と比べる

!!! warning "任意: 従量課金の API を使います"
    ローカルのモデルだけで進める場合は、この節を飛ばしてください（06 の成果物は作れます）。
    試す場合は、先に費用の上限と止め方を決めてください（[学び方](../guide.md) の「従量課金の API は任意」）。

クラウドの API のモデルも、同じ表に並べられます。その API の公式の SDK で `complete(messages)` を持つクラス（`LLMClient`）を書き、
`run_compare` の `make_llm` に渡します。呼ぶ前に、次を決めます。

1. **上限の金額**: 8 問を 1 回流すトークン数は、ローカルのモデルで作った表の「入力の平均」「出力の平均」から見積もれる。
   単価は、使う API の公式の料金ページで確かめる（07-4 の `Price` と `cost`）
2. **管理画面の上限と通知**: 利用の上限と予算の通知を設定する。上限で止まるのか通知だけなのかは、API ごとに違う
3. **コードの上限**: `--max-tokens` と `--num-predict` に当たる上限を、クラウドの呼び出しにも設ける（`LimitedLLM` で包む）
4. **止め方**: 管理画面でキーを無効にする・削除する手順を先に確かめ、使い終わったらキーを消す。キーは `.env` にだけ書く

## 演習

1. 手元の PC のメモリ（GPU があれば GPU のメモリも）を調べ、`weights_gb` を使って、`Q4_K_M` で動かせそうなモデルの大きさの上限を見積もってください。
   重みの 1.5 倍を必要なメモリとみなすなど、余裕の持たせ方も決めて書きます
2. `to_markdown` の「モデルごと」の表に、「出力の速さ（出力トークンの合計 ÷ 秒の合計）」の列を足し、テストで確かめてください
3. 任意（成果物）: 手元の Ollama で、2 つ以上のモデルを `p06_compare` で比べ、表を作ってください。
   表をもとに、07 以降で使うモデルを 1 つ選び、選んだ理由を 3 行で書きます（ツール使用が要るので `capabilities` も見る）

- できたことの確かめ方: `uv run pytest` が通る。1 は見積もりの式と結果。3 は表（モデル名・PC の条件を添える）と選んだ理由
- 目安の時間: 4 時間

解答: [06-4 の解答](answers/06-4.md)

## もし〇〇がなかったら？

**比べる表がなかったら**、モデルは評判や大きさで選ぶことになります。07 以降で「遅くて演習が進まない」
「分類の精度が足りない」と分かってから選び直すと、それまでのプロンプトや評価をやり直すことになります。

**メモリの見積もりがなかったら**、取ってきたモデルが動かない、または極端に遅い理由が分かりません。
大きいモデルを何度も取り直して、ディスクと時間を使います。

## 理解度チェック

1. 8B のモデルを、約 4.9 ビットの量子化で動かすとき、重みだけで何 GB ほど要りますか。実際にはそれより多く要るのはなぜですか。

    ??? note "答え"
        8 × 4.9 ÷ 8 で、約 4.9 GB です。実際には、コンテキストの分（長いほど多い）と、実行に使う分が足されるからです。
        動かした後に `ollama ps` で確かめます。

2. ローカルのモデルとクラウドの API を選ぶとき、精度のほかに比べることを 3 つ挙げてください。

    ??? note "答え"
        費用（従量課金か、PC と電気代か）、データの置き場所（社外に送ってよいか）、速さ（PC の性能か、回線と混み具合か）です。
        ほかに、メモリに入る大きさかどうか、事業者の利用規約も確かめます。

3. 06 の成果物の表で、「語」の列だけを見てモデルを決めてはいけないのはなぜですか。

    ??? note "答え"
        語が入っているかは粗い目安で、「1680 円ではありません」のような誤りも正しいと数えるからです。
        q07 のように語を決めていない質問もあります。答えそのものを読み、秒数やトークン数と合わせて判断します。

4. 比べるときにも `LimitedLLM` で上限を設けるのはなぜですか。

    ??? note "答え"
        誤ったループや想定より長い答えで、呼び出しやトークンが増え続けるのを止めるためです。
        ローカルなら PC の占有を、クラウドの API なら費用を、その実行の中で止められます。
