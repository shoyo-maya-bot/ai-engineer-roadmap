"""p11_numbers.py のテスト（正解率の幅・符号検定・数字の裏付け）."""

import io
from pathlib import Path

import pytest

from roadmap_examples.evalset import CaseResult, Report, save_results
from roadmap_examples.p11_numbers import main, paired_changes, sign_test, unsupported_numbers, wilson_interval


def test_wilson_interval() -> None:
    low, high = wilson_interval(24, 30)
    assert low == pytest.approx(0.627, abs=0.001) and high == pytest.approx(0.905, abs=0.001)
    assert wilson_interval(0, 10)[0] == 0.0 and wilson_interval(10, 10)[1] == 1.0
    with pytest.raises(ValueError):
        wilson_interval(3, 2)


def test_sign_test() -> None:
    assert sign_test(0, 0) == 1.0
    assert sign_test(10, 0) == pytest.approx(2 / 1024)
    assert sign_test(5, 2) == pytest.approx(0.453125)


def report(scores: dict[str, float]) -> Report:
    return Report([CaseResult(i, s, None) for i, s in scores.items()])


def test_paired_changes_ignores_unchanged_and_missing() -> None:
    before = report({"a": 0, "b": 1, "c": 0, "d": 1})
    after = report({"a": 1, "b": 0, "c": 0, "e": 1})
    assert paired_changes(before, after) == (1, 1)


def test_unsupported_numbers() -> None:
    contexts = ["アクセストークンの有効期限は 30 分", "リフレッシュトークンは 7 日"]
    assert unsupported_numbers("有効期限は 60 分です [1]。リフレッシュは 7 日 [2]。", contexts) == ["60"]
    assert unsupported_numbers("上限は１，０００件", ["上限は 1000 件"]) == []


def test_main(tmp_path: Path) -> None:
    before, after = tmp_path / "before.jsonl", tmp_path / "after.jsonl"
    save_results(report({"a": 0, "b": 0, "c": 1}), before)
    save_results(report({"a": 1, "b": 1, "c": 1}), after)
    stdout = io.StringIO()
    assert (
        main(["--correct", "24", "--total", "30", "--before", str(before), "--after", str(after)], stdout=stdout) == 0
    )
    text = stdout.getvalue()
    assert "95% の区間: 62.7%〜90.5%" in text
    assert "良くなった件 2 / 悪くなった件 0" in text
