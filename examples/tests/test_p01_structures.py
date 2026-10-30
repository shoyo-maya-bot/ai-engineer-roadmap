"""p01_structures のテスト. 速さは環境で変わるので、結果が同じことだけを確かめる."""

from roadmap_examples.p01_structures import (
    dedupe,
    lookup,
    measure,
    sliding_windows,
    slow_dedupe,
    slow_lookup,
    top_k,
)


def test_dedupe_keeps_first_order() -> None:
    items = ["c", "a", "c", "b", "a"]
    assert dedupe(items) == slow_dedupe(items) == ["c", "a", "b"]


def test_lookup_matches_slow_version() -> None:
    rows = [{"id": f"q{i}", "text": str(i)} for i in range(100)]
    ids = ["q5", "q99", "q100", "q0"]
    assert lookup(rows, ids) == slow_lookup(rows, ids)
    assert lookup(rows, ids)[2] is None  # 無い id は None


def test_lookup_keeps_last_row_for_duplicate_ids() -> None:
    # dict は同じキーを後から上書きする. slow_lookup（最初に見つかった行）とは結果が違う点に注意
    rows = [{"id": "q1", "text": "古い"}, {"id": "q1", "text": "新しい"}]
    assert lookup(rows, ["q1"]) == [{"id": "q1", "text": "新しい"}]
    assert slow_lookup(rows, ["q1"]) == [{"id": "q1", "text": "古い"}]


def test_top_k_returns_highest_scores() -> None:
    scores = [("a", 0.2), ("b", 0.9), ("c", 0.5), ("d", 0.7)]
    assert top_k(scores, 2) == [("b", 0.9), ("d", 0.7)]
    assert top_k(scores, 10) == sorted(scores, key=lambda s: s[1], reverse=True)


def test_sliding_windows() -> None:
    assert list(sliding_windows([1, 2, 3, 4], 3)) == [(1, 2, 3), (2, 3, 4)]
    assert list(sliding_windows([1, 2], 3)) == []


def test_measure_returns_seconds() -> None:
    seconds = measure(lambda: dedupe(range(1000)), repeat=2)
    assert 0 <= seconds < 1
