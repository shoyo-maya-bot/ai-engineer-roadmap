"""01-1 他の言語との違いで覚える Python: Java などで書いてきた処理を、Python らしく書いた例.

題材は、架空の会員制サービスに届く問い合わせ（03・04・07 でも同じ題材を使う）.

- クラスの代わりに dataclass（Java の record に近い）
- for 文で数える代わりに Counter / defaultdict
- null の代わりに None と `X | None` の型ヒント
- try-with-resources の代わりに with
- 「先に確かめる」より「やってみて、だめなら例外」（EAFP）
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Inquiry:
    """問い合わせ 1 件. frozen=True で、作った後に書き換えられない（Java の record に近い）."""

    id: str
    text: str
    category: str
    minutes: int | None = None  # 対応にかかった分. 分からなければ None


def count_by[T, K](items: Iterable[T], key: Callable[[T], K]) -> Counter[K]:
    """key ごとの件数. Java の Map<K, Integer> に getOrDefault で足していく処理を 1 行で書く."""
    return Counter(key(item) for item in items)


def group_by[T, K](items: Iterable[T], key: Callable[[T], K]) -> dict[K, list[T]]:
    """key ごとの一覧. 無いキーは空の list から始まる（Java の computeIfAbsent に当たる）."""
    groups: defaultdict[K, list[T]] = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    return dict(groups)


def parse_minutes(text: str) -> int | None:
    """「15」「15分」「 15 」を 15 に. 数でなければ None（例外を外に出さない）."""
    cleaned = text.strip().removesuffix("分")
    try:
        return int(cleaned)
    except ValueError:
        return None


def read_inquiries(path: Path) -> list[Inquiry]:
    """id,text,category,minutes の CSV を読む. with を抜けると、例外が出てもファイルは閉じる."""
    with path.open(encoding="utf-8", newline="") as f:
        return [
            Inquiry(row["id"], row["text"], row["category"], parse_minutes(row.get("minutes") or ""))
            for row in csv.DictReader(f)
        ]


def average_minutes(inquiries: Iterable[Inquiry]) -> float | None:
    """分が分かっている件だけの平均. 1 件も無ければ None（0 で割らない）."""
    known = [q.minutes for q in inquiries if q.minutes is not None]
    return sum(known) / len(known) if known else None
