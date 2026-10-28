"""01-3 データ構造と計算量: list・dict・set・deque・heapq の使い分けを、同じ処理の 2 つの書き方で比べる.

遅い書き方（slow_*）と速い書き方を並べ、結果が同じことをテストで、速さの違いを `measure` で確かめる.
"""

from __future__ import annotations

import heapq
import timeit
from collections import deque
from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence


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


def slow_lookup(rows: Sequence[dict[str, str]], ids: Iterable[str]) -> list[dict[str, str] | None]:
    """id ごとに行を探す. 毎回 rows を頭から見るので O(len(rows) * len(ids))."""
    return [next((r for r in rows if r["id"] == i), None) for i in ids]


def lookup(rows: Sequence[dict[str, str]], ids: Iterable[str]) -> list[dict[str, str] | None]:
    """先に id → 行 の dict（索引）を 1 回作り、あとは O(1) で引く. 全体で O(len(rows) + len(ids))."""
    index = {r["id"]: r for r in rows}
    return [index.get(i) for i in ids]


def top_k(scores: Iterable[tuple[str, float]], k: int) -> list[tuple[str, float]]:
    """点の高い順に k 件. 全部を並べ替える（O(n log n)）代わりに、ヒープで O(n log k)."""
    return heapq.nlargest(k, scores, key=lambda item: item[1])


def sliding_windows[T](items: Iterable[T], size: int) -> Iterator[tuple[T, ...]]:
    """直近 size 件の窓を 1 件ずつずらして返す. deque(maxlen=) は古いものを O(1) で捨てる."""
    window: deque[T] = deque(maxlen=size)
    for x in items:
        window.append(x)
        if len(window) == size:
            yield tuple(window)


def measure(func: Callable[[], object], repeat: int = 5, number: int = 1) -> float:
    """func を number 回呼ぶのにかかった秒数を repeat 回測り、いちばん速い値を返す（ほかの処理の影響を減らす）."""
    return min(timeit.repeat(func, repeat=repeat, number=number))
