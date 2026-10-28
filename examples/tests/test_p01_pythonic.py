"""p01_pythonic のテスト."""

import dataclasses
from pathlib import Path

import pytest

from roadmap_examples.p01_pythonic import Inquiry, average_minutes, count_by, group_by, parse_minutes, read_inquiries

ITEMS = [
    Inquiry("q1", "請求書がほしい", "billing", 10),
    Inquiry("q2", "パスワードを忘れた", "account", None),
    Inquiry("q3", "領収書がほしい", "billing", 20),
]


def test_count_by_counts_each_key() -> None:
    counts = count_by(ITEMS, lambda q: q.category)
    assert counts == {"billing": 2, "account": 1}
    assert counts["bug"] == 0  # Counter は無いキーを 0 として返す（KeyError にならない）


def test_group_by_keeps_order_within_group() -> None:
    groups = group_by(ITEMS, lambda q: q.category)
    assert [q.id for q in groups["billing"]] == ["q1", "q3"]
    assert "bug" not in groups  # dict に戻しているので、無いキーは増えない


def test_dataclass_is_frozen_and_compared_by_value() -> None:
    assert Inquiry("q1", "a", "other") == Inquiry("q1", "a", "other")  # equals を書かなくても値で比べる
    with pytest.raises(dataclasses.FrozenInstanceError):
        ITEMS[0].category = "bug"  # type: ignore[misc]


@pytest.mark.parametrize(("text", "want"), [("15", 15), (" 15分 ", 15), ("", None), ("約15", None)])
def test_parse_minutes(text: str, want: int | None) -> None:
    assert parse_minutes(text) == want


def test_read_inquiries_and_average(tmp_path: Path) -> None:
    path = tmp_path / "inquiries.csv"
    path.write_text("id,text,category,minutes\nq1,請求書,billing,10分\nq2,ログイン,account,\n", encoding="utf-8")
    items = read_inquiries(path)
    assert items[1] == Inquiry("q2", "ログイン", "account", None)
    assert average_minutes(items) == 10.0
    assert average_minutes([]) is None
