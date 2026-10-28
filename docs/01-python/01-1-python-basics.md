---
render_macros: true
---

# 01-1 他の言語との違いで覚える Python

!!! abstract "到達目標"
    - Java などで書いてきた処理（クラス・集計・null の扱い・後始末）を、Python らしい書き方に置き換えられる
    - 型ヒントを書き、それが実行時には確かめられないことを説明できる
    - uv で仮想環境を作り、依存を固定した版で入れて、コマンドを実行できる

## 何か

Python は、書いたコードをその場で実行する言語です。Java のようにコンパイルしてから動かす手順はありません。
変数に型を書かなくても動きます（動的型付け）。そのかわり、**型ヒント** を書いておくと、エディタや検査の道具が誤りを見つけてくれます。

業務の経験がある人は、新しい概念より「書き方の違い」でつまずきます。主な違いを表にします。

| やりたいこと | Java | Python |
|---|---|---|
| ブロック | `{ }` | 字下げ（スペース 4 つ）。字下げを間違えると動きが変わる |
| 値だけを持つクラス | `record` | `@dataclass(frozen=True)` |
| 値が無い | `null`・`Optional<T>` | `None`。型ヒントは `int \| None` |
| 型の引数（ジェネリクス） | `<T> List<T> f(...)` | `def f[T](...) -> list[T]` |
| インターフェース | `interface` と `implements` | `Protocol`（同じメソッドがあれば、宣言しなくてよい。01-4） |
| 後始末 | try-with-resources | `with` |
| 依存の管理 | Maven・Gradle（`pom.xml`） | uv（`pyproject.toml` と `uv.lock`） |
| 実行環境の分離 | プロジェクトごとの JDK とクラスパス | 仮想環境（`.venv`） |

この教材の演習は、Python {{ versions.python }} と uv {{ versions.uv }} で確かめています。

## なぜ必要か

Java の書き方のまま Python を書くと、動きはしますが、長くて読みにくいコードになります。
読みにくいコードは、レビューで誤りが見つかりにくくなります。

- 数えるだけの処理に、`if key in counts:` の分岐を毎回書く（標準ライブラリに数える道具がある）
- `None` を返すか例外を投げるかが、関数ごとにばらばらになる
- ファイルを閉じ忘れる（`with` を使えば、例外が出ても閉じる）

環境の分け方も違います。Python は、入れたパッケージを環境ごとに 1 つの場所へ置きます。
プロジェクトごとに環境を分けないと、あるプロジェクトのために上げた版が、別のプロジェクトを壊します。

## どうやるか

### uv で環境を作る

uv は、Python 本体・仮想環境・依存の版の固定を、まとめて扱う道具です（[公式ドキュメント: プロジェクトの作り方](https://docs.astral.sh/uv/guides/projects/)）。
Maven や Gradle に近い役割です。

```bash
uv init inquiry-tool        # pyproject.toml などの雛形を作る
cd inquiry-tool
uv add pandas               # 依存を足す（pyproject.toml と uv.lock が更新される）
uv add --dev pytest ruff    # 開発のときだけ使う依存
uv run python main.py       # .venv の中の Python で実行する（無ければ作る）
uv sync --locked            # uv.lock のとおりに入れ直す（CI や別の PC で）
```

| ファイル | 中身 | Java での近いもの |
|---|---|---|
| `pyproject.toml` | 依存の名前と、許す版の範囲（`pandas>=3.0` など） | `pom.xml`・`build.gradle` |
| `uv.lock` | 実際に入れた版とハッシュ（自動で作る。手で書かない） | Gradle の lockfile |
| `.venv/` | 仮想環境（入れたパッケージの置き場所。コミットしない） | ローカルのリポジトリ |

この教材の演習のコードも、同じ形です（`examples/pyproject.toml` と `examples/uv.lock`）。

!!! failure "悪い例"
    `pip install pandas` を、仮想環境に入らずに実行する。どの Python に入ったか分からず、版もどこにも残らない。

!!! success "良い例"
    `uv add pandas` で入れる。版は `uv.lock` に残り、別の PC でも `uv sync --locked` で同じ版がそろう。

### 型ヒントを書く

型ヒントは、引数と戻り値に型を書く仕組みです（[公式ドキュメント: typing](https://docs.python.org/3/library/typing.html)）。
ただし、**Python は実行時に型ヒントを確かめません**。間違った型を渡しても、その場では動いてしまいます。

```pycon
>>> def double(n: int) -> int:
...     return n * 2
>>> double("ab")  # str を渡しても止まらない
'abab'
```

型ヒントは、読む人とエディタへの「約束」です。約束を破っていないかは、エディタや型の検査の道具に確かめさせます。

型の引数（ジェネリクス）は、関数名の後ろに `[T]` と書きます（[公式ドキュメント: 型引数のリスト](https://docs.python.org/3/reference/compound_stmts.html#type-params)）。
Java の `<T>` と同じ考え方です。

### 値だけを持つクラスは dataclass で書く

Java の `record` のように、項目を並べるだけで、コンストラクタ・比較（`equals` に当たる `==`）・表示が付きます
（[公式ドキュメント: dataclasses](https://docs.python.org/3/library/dataclasses.html)）。

<!-- src: examples/src/roadmap_examples/p01_pythonic.py -->
```python title="examples/src/roadmap_examples/p01_pythonic.py"
@dataclass(frozen=True)
class Inquiry:
    """問い合わせ 1 件. frozen=True で、作った後に書き換えられない（Java の record に近い）."""

    id: str
    text: str
    category: str
    minutes: int | None = None  # 対応にかかった分. 分からなければ None
```

`frozen=True` を付けると、作った後に項目を書き換えようとしたとき例外になります。テストで確かめています。

<!-- src: examples/tests/test_p01_pythonic.py -->
```python title="examples/tests/test_p01_pythonic.py"
def test_dataclass_is_frozen_and_compared_by_value() -> None:
    assert Inquiry("q1", "a", "other") == Inquiry("q1", "a", "other")  # equals を書かなくても値で比べる
    with pytest.raises(dataclasses.FrozenInstanceError):
        ITEMS[0].category = "bug"  # type: ignore[misc]
```

### 数える・まとめるは collections に任せる

分類ごとの件数を数える処理を、Java の書き方のまま写すと、次のようになります。

!!! failure "悪い例"
    ```python
    counts = {}
    for q in inquiries:
        if q.category in counts:
            counts[q.category] = counts[q.category] + 1
        else:
            counts[q.category] = 1
    ```

標準ライブラリの `Counter` と `defaultdict` を使うと、分岐が消えます（[公式ドキュメント: collections](https://docs.python.org/3/library/collections.html)）。

!!! success "良い例"
    <!-- src: examples/src/roadmap_examples/p01_pythonic.py -->
    ```python title="examples/src/roadmap_examples/p01_pythonic.py"
    def count_by[T, K](items: Iterable[T], key: Callable[[T], K]) -> Counter[K]:
        """key ごとの件数. Java の Map<K, Integer> に getOrDefault で足していく処理を 1 行で書く."""
        return Counter(key(item) for item in items)


    def group_by[T, K](items: Iterable[T], key: Callable[[T], K]) -> dict[K, list[T]]:
        """key ごとの一覧. 無いキーは空の list から始まる（Java の computeIfAbsent に当たる）."""
        groups: defaultdict[K, list[T]] = defaultdict(list)
        for item in items:
            groups[key(item)].append(item)
        return dict(groups)
    ```

`Counter(key(item) for item in items)` の `( ... for ... in ...)` は、ジェネレーター式です。
Java の Stream の `map` に近く、要素を 1 つずつ作って渡します。`[ ... ]` で囲むと、すべてを並べた list になります（内包表記）。

関数を引数に渡すときは、`lambda` で短く書けます（Java のラムダ式と同じ考え方です）。

```python
by_category = count_by(inquiries, lambda q: q.category)
```

### None と例外: 「やってみて、だめなら」

Python では、「先に確かめてから処理する」より「処理してみて、失敗したら例外を受ける」書き方がよく使われます（EAFP と呼びます）。
数かどうかを先に正規表現で調べるより、`int()` に渡して `ValueError` を受けるほうが、漏れがありません。

<!-- src: examples/src/roadmap_examples/p01_pythonic.py -->
```python title="examples/src/roadmap_examples/p01_pythonic.py"
def parse_minutes(text: str) -> int | None:
    """「15」「15分」「 15 」を 15 に. 数でなければ None（例外を外に出さない）."""
    cleaned = text.strip().removesuffix("分")
    try:
        return int(cleaned)
    except ValueError:
        return None
```

戻り値の型 `int | None` は、「数か、値が無いか」を表します。呼ぶ側は `None` の場合を必ず考えます。
`None` と比べるときは、`==` ではなく `is None` と書きます。

### 後始末は with で

`with` を抜けると、例外が出ても、ファイルは閉じます。Java の try-with-resources と同じ役割です。

<!-- src: examples/src/roadmap_examples/p01_pythonic.py -->
```python title="examples/src/roadmap_examples/p01_pythonic.py"
def read_inquiries(path: Path) -> list[Inquiry]:
    """id,text,category,minutes の CSV を読む. with を抜けると、例外が出てもファイルは閉じる."""
    with path.open(encoding="utf-8", newline="") as f:
        return [
            Inquiry(row["id"], row["text"], row["category"], parse_minutes(row.get("minutes") or ""))
            for row in csv.DictReader(f)
        ]
```

- `encoding="utf-8"` は必ず書きます。省くと、OS の既定の文字コードで読み、Windows で日本語が化けることがあります
- `row.get("minutes") or ""` は、列が無いか空のとき `""` にします。`or` は、左が空（`None`・`""`・`0` など）なら右を返します

## 演習

`examples/src/roadmap_examples/p01_pythonic.py` の部品を使って、分類ごとの平均の対応時間を返す関数を書いてください。

```python
def average_minutes_by_category(inquiries: list[Inquiry]) -> dict[str, float | None]: ...
```

1. `uv sync --locked` で演習の環境を作り、`uv run pytest tests/test_p01_pythonic.py` が通ることを確かめる
2. 関数を自分のファイル（例: `examples/tests/test_my_average.py` の中）に書く。`group_by` と `average_minutes` を使う
3. 分が 1 件も分からない分類は `None` にする。テストを 2 つ以上書く（ふつうの場合と、`None` になる場合）

- できたことの確かめ方: `uv run pytest` が通る。Java で書くなら何行になるかと比べる
- 目安の時間: 2 時間

解答: [01-1 の解答](answers/01-1.md)

## もし〇〇がなかったら？

**仮想環境と uv.lock がなかったら**、パッケージは PC 全体の 1 か所に入ります。
あるプロジェクトのために pandas の版を上げると、別のプロジェクトが動かなくなります。
しかも、どの版で動いていたかの記録が無いので、元に戻せません。CI と手元で結果が違う、という調査に時間を取られます。

**型ヒントがなかったら**、関数が何を受け取り、何を返すかを、中身を読んで推測することになります。
`None` が返る場合を見落とし、本番で初めて `AttributeError` が出る、ということが起きやすくなります。

## 理解度チェック

1. 型ヒントで `n: int` と書いた関数に文字列を渡すと、どうなりますか。

    ??? note "答え"
        そのまま実行されます。Python は実行時に型ヒントを確かめないからです。
        誤りを見つけるのは、エディタや型の検査の道具の役割です。

2. `pyproject.toml` と `uv.lock` の違いを説明してください。

    ??? note "答え"
        `pyproject.toml` は、使う依存の名前と、許す版の範囲を人が書くファイルです。
        `uv.lock` は、実際に入れた版とハッシュを uv が書くファイルで、どの PC でも同じ版をそろえるために使います。

3. `Counter` を使うと、Java の `Map` で数える書き方と比べて、何が要らなくなりますか。

    ??? note "答え"
        「キーがまだ無ければ 0 から始める」分岐です。`Counter` は無いキーを 0 として扱います。

4. `parse_minutes` が、数でない文字列に対して例外を投げず `None` を返すのはなぜですか。

    ??? note "答え"
        汚れたデータ（「約15」や空欄）は珍しくなく、1 行の誤りで全体を止めたくないからです。
        戻り値の型を `int | None` にして、呼ぶ側に「値が無い場合」を考えさせます。
