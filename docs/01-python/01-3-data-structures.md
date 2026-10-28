# 01-3 データ構造と計算量

!!! abstract "到達目標"
    - list・dict・set・deque・heapq の主な操作の計算量を知り、処理に合うものを選べる
    - 件数が増えたときに遅くなる書き方（二重のループ・list の `in`）を見つけ、索引（dict・set）で直せる
    - 速さを timeit で測り、測った条件と一緒に比べられる

## 何か

**計算量** は、データの件数 n が増えたとき、処理の手間がどう増えるかの目安です。O 記法（オーダー）で書きます。

| 書き方 | 意味 | n が 10 倍になると |
|---|---|---|
| O(1) | 件数によらず一定 | 変わらない |
| O(log n) | 半分ずつ絞り込む | 少し増える |
| O(n) | 全件を 1 回ずつ見る | 10 倍 |
| O(n log n) | 並べ替え | 10 倍より少し多い |
| O(n²) | 全件の組み合わせを見る | 100 倍 |

Python の組み込みのデータ構造の、主な操作の計算量です（平均の場合。[Python Wiki: TimeComplexity](https://wiki.python.org/moin/TimeComplexity)）。

| データ構造 | 速い操作 | 遅い操作 | Java での近いもの |
|---|---|---|---|
| `list` | 末尾に足す・末尾を取る・番号で引く O(1) | `x in list` O(n)、先頭に足す・先頭を取る O(n) | `ArrayList` |
| `dict` | キーで引く・足す・`in` O(1) | ― | `HashMap`（`LinkedHashMap` のように足した順を保つ） |
| `set` | `in`・足す O(1) | ― | `HashSet` |
| `collections.deque` | 両端に足す・両端から取る O(1) | 真ん中を番号で引く O(n) | `ArrayDeque` |
| `heapq`（list をヒープとして使う） | 最小を取る・足す O(log n) | ― | `PriorityQueue` |

## なぜ必要か

AI の仕事では、データの件数が大きく変わります。演習では 100 件でも、本番では 100 万件の問い合わせを扱います。

100 件で 0.01 秒の O(n²) の処理は、10 万件で約 1 万秒（3 時間近く）かかります。
件数を増やして初めて遅さに気づき、原因を探すことになります。書くときに「件数が 100 倍になったら？」と考える習慣が要ります。

Java では `List.contains` の遅さを意識していても、Python では `x in items` と短く書けるので、見落としやすくなります。

## どうやるか

### list の `in` を set に替える

重複を除く処理を、2 つの書き方で比べます。結果は同じです。

<!-- src: examples/src/roadmap_examples/p01_structures.py -->
```python title="examples/src/roadmap_examples/p01_structures.py"
def slow_dedupe[H: Hashable](items: Iterable[H]) -> list[H]:
    """重複を除く（最初の順を保つ）. `x not in seen` が list の線形探索なので、全体で O(n^2)."""
    seen: list[H] = []
    for x in items:
        if x not in seen:
            seen.append(x)
    return seen


def dedupe[H: Hashable](items: Iterable[H]) -> list[H]:
    """重複を除く（最初の順を保つ）. set の `in` は平均 O(1) なので、全体で O(n)."""
    seen: set[H] = set()
    out: list[H] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
```

`measure`（下の「速さを測る」）で測った結果です。n 件の 0〜n の乱数の重複を除きました。

| 件数 n | `slow_dedupe`（list） | `dedupe`（set） |
|---|---|---|
| 1,000 | 約 3.6 ミリ秒 | 約 0.07 ミリ秒 |
| 10,000 | 約 350 ミリ秒 | 約 1.3 ミリ秒 |

測った条件: 手元の Linux（2 コアの仮想マシン）・演習の環境（`uv run`）・5 回測った中で最も速い値。数字は環境で変わります。
件数が 10 倍になると、list の版は約 100 倍、set の版は約 10〜20 倍になりました。O(n²) と O(n) の違いです。

`H: Hashable` は、「set や dict のキーにできる型」という意味の型ヒントです。
`list` は中身を書き換えられるので、キーにできません。キーにしたい組は `tuple` にします。

### 何度も探すなら、先に索引（dict）を作る

「id で行を探す」を、id の数だけ繰り返す処理です。SQL で索引の無い列を何度も検索するのと同じ遅さになります。

!!! failure "悪い例"
    <!-- src: examples/src/roadmap_examples/p01_structures.py -->
    ```python title="examples/src/roadmap_examples/p01_structures.py"
    def slow_lookup(rows: Sequence[dict[str, str]], ids: Iterable[str]) -> list[dict[str, str] | None]:
        """id ごとに行を探す. 毎回 rows を頭から見るので O(len(rows) * len(ids))."""
        return [next((r for r in rows if r["id"] == i), None) for i in ids]
    ```

!!! success "良い例"
    <!-- src: examples/src/roadmap_examples/p01_structures.py -->
    ```python title="examples/src/roadmap_examples/p01_structures.py"
    def lookup(rows: Sequence[dict[str, str]], ids: Iterable[str]) -> list[dict[str, str] | None]:
        """先に id → 行 の dict（索引）を 1 回作り、あとは O(1) で引く. 全体で O(len(rows) + len(ids))."""
        index = {r["id"]: r for r in rows}
        return [index.get(i) for i in ids]
    ```

1 万行から 1,000 個の id を探すと、同じ条件で約 190 ミリ秒と約 0.8 ミリ秒でした。

ただし、索引にすると **同じ id の行は後のもので上書き** されます。どちらの行が欲しいのかで、書き方を選びます。
テストで違いを確かめています。

<!-- src: examples/tests/test_p01_structures.py -->
```python title="examples/tests/test_p01_structures.py"
def test_lookup_keeps_last_row_for_duplicate_ids() -> None:
    # dict は同じキーを後から上書きする. slow_lookup（最初に見つかった行）とは結果が違う点に注意
    rows = [{"id": "q1", "text": "古い"}, {"id": "q1", "text": "新しい"}]
    assert lookup(rows, ["q1"]) == [{"id": "q1", "text": "新しい"}]
    assert slow_lookup(rows, ["q1"]) == [{"id": "q1", "text": "古い"}]
```

### 上位 k 件は heapq、直近 k 件は deque

08 の検索では「点の高い順に 5 件」を何度も求めます。全件を並べ替えなくても、上位だけを取り出せます。

<!-- src: examples/src/roadmap_examples/p01_structures.py -->
```python title="examples/src/roadmap_examples/p01_structures.py"
def top_k(scores: Iterable[tuple[str, float]], k: int) -> list[tuple[str, float]]:
    """点の高い順に k 件. 全部を並べ替える（O(n log n)）代わりに、ヒープで O(n log k)."""
    return heapq.nlargest(k, scores, key=lambda item: item[1])
```

`heapq.nlargest` の結果は、`sorted(..., reverse=True)[:k]` と同じです（[公式ドキュメント: heapq](https://docs.python.org/3/library/heapq.html)）。
k が n に近いときは、`sorted` のほうが速いこともあります。

「直近 3 件の問い合わせ」のように、古いものから捨てたいときは `deque(maxlen=...)` を使います。
list の `pop(0)` は、残りを 1 つずつ前にずらすので O(n) です。

<!-- src: examples/src/roadmap_examples/p01_structures.py -->
```python title="examples/src/roadmap_examples/p01_structures.py"
def sliding_windows[T](items: Iterable[T], size: int) -> Iterator[tuple[T, ...]]:
    """直近 size 件の窓を 1 件ずつずらして返す. deque(maxlen=) は古いものを O(1) で捨てる."""
    window: deque[T] = deque(maxlen=size)
    for x in items:
        window.append(x)
        if len(window) == size:
            yield tuple(window)
```

`yield` を使う関数（ジェネレーター）は、値を 1 つずつ返します。全部を list に貯めないので、大きなデータでもメモリを使いすぎません。

### 速さを測る

速さは、推測せずに測ります。標準ライブラリの `timeit` は、同じ処理を何度も実行して時間を測ります
（[公式ドキュメント: timeit](https://docs.python.org/3/library/timeit.html)）。

<!-- src: examples/src/roadmap_examples/p01_structures.py -->
```python title="examples/src/roadmap_examples/p01_structures.py"
def measure(func: Callable[[], object], repeat: int = 5, number: int = 1) -> float:
    """func を number 回呼ぶのにかかった秒数を repeat 回測り、いちばん速い値を返す（ほかの処理の影響を減らす）."""
    return min(timeit.repeat(func, repeat=repeat, number=number))
```

```python
import random

from roadmap_examples.p01_structures import dedupe, measure, slow_dedupe

items = [random.randrange(10_000) for _ in range(10_000)]
print(measure(lambda: slow_dedupe(items), repeat=3), measure(lambda: dedupe(items), repeat=3))
```

- 1 回だけ測らず、何度か測って最も速い値（または中央値）を使う。ほかの処理の影響を減らすため
- 件数を 2〜3 通り変えて測る。1 つの件数では、O(n) か O(n²) かが分からない
- 数字は、測った条件（件数・PC・Python の版）と一緒に残す

テストでは速さを確かめません。CI の環境で速さが変わり、たまに失敗するテストになるからです。テストでは「結果が同じこと」だけを確かめます。

## 演習

今日の問い合わせの顧客 ID の一覧 `today` と、過去に問い合わせた顧客 ID の一覧 `known` があります。
今日はじめて問い合わせた顧客 ID を、`today` に出てきた順に、重複なしで返す関数を書いてください。

```python
def new_customers(today: list[str], known: list[str]) -> list[str]: ...
```

1. まず二重のループ（または list の `in`）で書き、テストを書く
2. set を使って O(len(today) + len(known)) に直し、同じテストが通ることを確かめる
3. `today` と `known` を 1,000 件・10,000 件にして、2 つの版を `measure` で測り、表にする

- できたことの確かめ方: テストが両方の版で通る。表に、測った条件を書く
- 目安の時間: 2 時間

解答: [01-3 の解答](answers/01-3.md)

## もし〇〇がなかったら？

**計算量を考えずに書いていたら**、開発中の 100 件のデータでは問題なく動き、本番の 100 万件で初めて止まります。
夜間のバッチが朝までに終わらない、画面の応答が数十秒かかる、といった形で表に出ます。
原因が「list の `in` を 1 か所使っていた」だけでも、見つけるまでに時間がかかります。

**測らずに速さを判断していたら**、遅くない場所を一生懸命に直し、本当に遅い場所が残ります。
「速くなったはず」は、件数を変えて測るまで分かりません。

## 理解度チェック

1. `x in items` は、`items` が list のときと set のときで、計算量がどう違いますか。

    ??? note "答え"
        list は先頭から順に比べるので O(n) です。set はハッシュで場所を求めるので、平均 O(1) です。

2. 件数が 10 倍になったとき、処理の時間がおよそ 100 倍になりました。計算量はどれくらいだと考えられますか。

    ??? note "答え"
        O(n²) です。二重のループや、ループの中の list の `in` を探します。

3. id で行を引く dict（索引）を作るとき、同じ id の行が 2 つあると、どうなりますか。

    ??? note "答え"
        後から入れた行で上書きされ、前の行は消えます。どちらの行を残したいのかを決めてから作ります
        （03-3 の重複の扱いでも同じ判断をします）。

4. 速さのテストを CI で行わないのはなぜですか。

    ??? note "答え"
        CI の環境では、ほかの処理の影響で速さが毎回変わり、たまに失敗するテストになるからです。
        テストでは結果が同じことを確かめ、速さは手元で、条件を決めて測ります。
