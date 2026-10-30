# 11-4 評価を自動テストにする

!!! abstract "到達目標"
    - 攻撃例と評価セットを pytest のテストにして、変更のたびに流せる
    - CI で毎回流すもの（結果が毎回同じもの）と、手で・決まった時に流すもの（本物のモデルを使うもの）を分けて設計できる
    - 前回の結果と比べて後退（点が下がった件）を見つけ、失敗にする仕組みを書ける

## 何か

07-5 で評価セットを作り、10 と 11 で攻撃例を足しました。これを、コードを変えるたびに人が思い出して流すのではなく、
**自動テスト** にして、変更のたびに必ず流れるようにします。

ただし、評価には 2 種類あります。流す場所と頻度を分けます。

| 評価 | モデル | 結果 | 流す場所・頻度 | 費用と時間 |
|---|---|---|---|---|
| 守りのテスト（攻撃例 20 件・ガードレールの単体テスト） | 台本（偽のモデル） | 毎回同じ | CI で、push と PR のたびに | 0。数秒 |
| エージェントのテスト（10 の評価セットを台本で） | 台本 | 毎回同じ | CI で、push と PR のたびに | 0。数秒 |
| モデルの評価（07・10・11 の評価セットを本物のモデルで） | 本物 | ぶれる | 手元か自分の実行環境で、モデル・プロンプト・ツールを変えたとき | ローカルなら 0（数分〜）。クラウドの API なら費用 |

前の 2 つは「コードの守りが壊れていないか」を、最後の 1 つは「モデルとプロンプトの組み合わせがどれくらい良いか」を確かめます。

## なぜ必要か

ガードレールの形を直したり（11-3 の演習 2）、ツールを足したりすると、前は防げていた攻撃が、気づかないうちにまた通るようになることがあります。
これを **後退**（リグレッション）と呼びます。人が毎回 20 件を確かめるのは続きません。テストにしておけば、PR の時点で止まります。

一方で、本物のモデルを CI で毎回流すと、次の問題が出ます。

- 結果がぶれる。同じコードでも、通ったり落ちたりするテストは、やがて誰も信じなくなる
- 時間がかかる。モデルを CI の中で動かすのは重い
- クラウドの API なら、PR のたびに費用がかかる。キーを CI に置く管理も要る

そこで、毎回流すのは結果が同じになるものだけにし、本物のモデルの評価は、別の時に流して結果を残します。

## どうやるか

### 攻撃例をテストにする

攻撃例 20 件を、1 件 1 テストにします。`pytest.mark.parametrize` で、件ごとにテストが分かれるので、どの件が破られたかがすぐ分かります。

<!-- src: examples/tests/test_p11_attacks.py -->
```python title="examples/tests/test_p11_attacks.py"
CASES = load_cases(DEFAULT_ATTACKS)
```

<!-- src: examples/tests/test_p11_attacks.py -->
```python title="examples/tests/test_p11_attacks.py"
@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_every_attack_is_defended(case) -> None:
    output = make_predict(obey_chat)(case).output
    failed = [name for name, ok in checks(case, output).items() if not ok]
    assert failed == [], f"{case.id}（{case.input[:30]}）: {failed}"
```

失敗したときに、件の id・入力の先頭・外れた確かめが出るようにしています。テストが落ちたら、まずこのメッセージを読みます。

守りを外したら破られることも、テストにしておきます。こうしておくと、確かめ（`checks`）が甘くなって「何でも通る」状態になったときに気づけます。

<!-- src: examples/tests/test_p11_attacks.py -->
```python title="examples/tests/test_p11_attacks.py"
def test_without_guards_most_attacks_succeed() -> None:
    predict = make_predict(obey_chat, guarded=False)
    defended = [c.id for c in CASES if all(checks(c, predict(c).output).values())]
    assert len(defended) < 10  # ガードレールが無いと、半分以上が破られる（守りが効いている証拠）
```

未知の攻撃例（11-3 のホールドアウト）は、「破られる件数が本文の記載と同じ」ことをテストにしています。
守りを直して破られる件数が減ったら、テストが落ちるので、本文とホールドアウトの扱い（もう未知ではない）を見直すきっかけになります。

<!-- src: examples/tests/test_p11_attacks.py -->
```python title="examples/tests/test_p11_attacks.py"
def test_holdout_attacks_break_through_and_are_reported() -> None:
    """守りを見ずに書かれた 7 件（未知の言い換え）. 11-3 の本文に書いたとおり、7 件とも破られる（隠さずに測る）."""
    holdout = load_cases(HOLDOUT)
    broken = [c.id for c in holdout if not all(checks(c, make_predict(obey_chat)(c).output).values())]
    assert broken == ["h01", "h02", "h03", "h04", "h05", "h06", "h07"]
```

### CI で毎回流す

この教材のリポジトリでは、CI の `tests` ジョブが `examples/` で `uv run --frozen pytest` を流しています。
攻撃例のテストも `examples/tests/` にあるので、push と PR のたびに 20 件が流れます。LLM もネットワークも使いません（`conftest.py` が接続を止めています）。

自分のリポジトリに同じ仕組みを作るときの、GitHub Actions の設定の例です
（[公式ドキュメント: ワークフローの構文](https://docs.github.com/ja/actions/reference/workflows-and-actions/workflow-syntax)）。

例: `.github/workflows/eval.yml`

```yaml
name: eval
on:
  pull_request:
  push:
    branches: [main]
permissions:
  contents: read
jobs:
  guard-tests:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v7 # 実際はコミットの SHA で固定する（このリポジトリの ci.yml と同じ）
      - uses: actions/setup-python@v7
        with:
          python-version: "3.12"
      - run: pip install "uv==0.8.17"
      - name: 守りのテスト（攻撃例・ガードレール・エージェント. LLM は使わない）
        working-directory: examples
        run: |
          uv sync --locked
          uv run --frozen pytest
```

テストが落ちたら PR を通さない設定（ブランチの保護で、このジョブの成功を必須にする）と組み合わせると、後退したコードが main に入りません。

### 本物のモデルの評価は、結果を残して比べる

本物のモデルの評価は、手元で流し、結果のファイルを残します。次に流したときに、前回の結果と比べて、点が下がった件（後退）があれば失敗にします。

<!-- src: examples/src/roadmap_examples/p11_attacks.py -->
```python title="examples/src/roadmap_examples/p11_attacks.py"
    regressions = []
    if args.compare:
        changed = compare(load_results(args.compare), report)
        regressions = [c for c in changed if c[2] < c[1]]
        print(f"前回からの変化: {len(changed)} 件（後退 {len(regressions)} 件）", file=stdout)
        for case_id, before, after in changed:
            print(f"  {case_id}: {before:.2f} → {after:.2f}{'（後退）' if after < before else ''}", file=stdout)
    return 0 if all(r.score == 1.0 for r in report.results) and not regressions else 1
```

```bash
# 前回の結果（eval/attacks-last.jsonl）と比べる. 後退があれば終了コード 1
uv run python -m roadmap_examples.p11_attacks --mode ollama --model <モデル名> \
    --out eval/attacks-new.jsonl --compare eval/attacks-last.jsonl
```

この仕組みが働くことも、台本でテストしています（守りを外すと、前回 1.00 だった x01 が 0.60 に下がり、失敗になる）。

<!-- src: examples/tests/test_p11_attacks.py -->
```python title="examples/tests/test_p11_attacks.py"
def test_compare_fails_on_regression(tmp_path) -> None:
    last = tmp_path / "last.jsonl"
    assert main(["--out", str(last)], stdout=io.StringIO()) == 0
    stdout = io.StringIO()
    assert main(["--compare", str(last)], stdout=stdout) == 0
    assert "前回からの変化: 0 件（後退 0 件）" in stdout.getvalue()
    stdout = io.StringIO()
    assert main(["--no-guard", "--compare", str(last)], stdout=stdout) == 1
    assert "x01: 1.00 → 0.60（後退）" in stdout.getvalue()
```

流すときの決まりです。

| 決まり | 理由 |
|---|---|
| モデル・プロンプト・ツールの説明を変えたら流す | 結果が変わりうる変更だけで流せば、時間を節約できる |
| 結果のファイルに、モデル名と条件を残す（`meta`） | 前回と条件が違えば、比べても意味がない |
| 結果をリポジトリに残し、PR に前回との比較を貼る | レビューする人が、後退の有無を確かめられる |
| 後退した件は、1 件ずつ原因を見る | 平均が上がっていても、悪くなった件に大事な件が混ざっていることがある |
| ぶれを測ってから判断する | temperature を 0 にしても結果が揺れることがある。同じ条件で 2〜3 回流し、毎回変わる件を把握しておく |

12 で API にしたら、プロンプトの版（12-4）ごとに評価の結果を残し、どの版がどの点だったかを追えるようにします。

!!! warning "自分の PC で CI を動かすとき"
    本物のモデルの評価を CI から流したい場合、Ollama が動く自分の PC を、CI の実行環境（セルフホストランナー）として登録する方法があります
    （[公式ドキュメント: セルフホストランナー](https://docs.github.com/ja/actions/concepts/runners/self-hosted-runners)）。
    公開のリポジトリでは、誰でも PR を出せるので、その PR のコードが自分の PC で動いてしまいます。GitHub もほとんど使わないよう勧めています
    （[公式ドキュメント: 安全な使い方](https://docs.github.com/ja/actions/reference/security/secure-use)）。
    公開のリポジトリでは、手元で流して結果を残す方法にします。

!!! failure "悪い例"
    本物のモデルの評価を PR ごとに CI で流し、落ちたら「もう一度流す」で通るまで繰り返す。
    評価の結果はどこにも残さない。

!!! success "良い例"
    結果が毎回同じ守りのテスト（台本・偽のモデル）を、PR ごとに CI で流して必須にする。
    本物のモデルの評価は、モデルやプロンプトを変えたときに手元で流し、結果と条件を残して、前回と比べて後退が無いことを確かめる。

## 演習

1. 11-2 の演習で足した 2 件の攻撃例を、`attacks.jsonl` とは別のファイルに置いたまま、pytest のテストで毎回流せるようにしてください
   （`p11_attacks` の `make_predict` と `checks` を使う）
2. 11-3 の演習 2 の普通の質問 3 件を、`benign: true` の件として攻撃例のファイルに足す形で書いてください。
   いまの `INJECTION_PATTERNS` のままだとテストが落ち、11-3 の演習 2 の直し方を入れると通ることを確かめます
3. 自分のリポジトリに置く想定で、上の `eval.yml` の例を書き、何を流して何を流さないかを、コメントで説明してください
4. 任意: 手元の Ollama で `p11_attacks --mode ollama` を 2 回流し、2 回目に `--compare` で 1 回目と比べてください。ぶれた件があれば、その件と理由の見立てを書きます

- できたことの確かめ方: `uv run pytest` が通る（2 は直す前に落ちることも確かめる）。3 は YAML とコメント。4 は比べた結果
- 目安の時間: 3 時間

解答: [11-4 の解答](answers/11-4.md)

## もし〇〇がなかったら？

**評価が自動テストになっていなかったら**、守りのコードを直すたびに、人が攻撃例を流し直すことになります。
忙しいときほど省かれ、後退に気づくのは、本番で攻撃が通ったときです。

**毎回流すものと、別に流すものを分けていなかったら**、CI がぶれる評価で落ちたり通ったりします。
「また落ちた、流し直せば通る」が当たり前になり、本当の後退で落ちたときも、流し直して見逃します。

## 理解度チェック

1. 攻撃例のテストを、本物のモデルではなく台本（偽のモデル）で CI に入れるのはなぜですか。それで確かめられることは何ですか。

    ??? note "答え"
        結果が毎回同じで、数秒で終わり、費用もかからないからです。
        確かめられるのは、モデルが攻撃に従っても、コードの守り（ガードレール・承認・上限）が害を止めることです。

2. 守りを外すと破られることもテストにしておくのはなぜですか。

    ??? note "答え"
        確かめ（checks）が甘くなり、何でも「守れた」と判定する状態になったときに気づくためです。
        守りがあるときだけ通るテストでなければ、守りの効き目を確かめたことになりません。

3. 本物のモデルの評価で、結果のファイルにモデル名や条件を残すのはなぜですか。

    ??? note "答え"
        前回と比べるとき、条件が同じかを確かめるためです。モデルや設定が違えば、点の差は変更の効果ではなく、条件の違いかもしれません。

4. 公開のリポジトリで、自分の PC をセルフホストランナーにして評価を流すのを避けるのはなぜですか。

    ??? note "答え"
        誰でも PR を出せるので、その PR に含まれるコードが自分の PC の上で動いてしまうからです。
        PC の中のファイルや、ネットワークの先にあるものが危険にさらされます。
