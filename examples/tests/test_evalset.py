"""evalset.py のテスト（評価セットの読み書き・採点・前回との比較）."""

from pathlib import Path

import pytest

from roadmap_examples.evalset import (
    EvalCase,
    EvalSetError,
    Prediction,
    compare,
    exact_match,
    load_cases,
    load_results,
    run_eval,
    save_cases,
    save_results,
)

CASES = [
    EvalCase("c1", "請求書を再発行してほしい", {"category": "billing"}, ("基本",)),
    EvalCase("c2", "ログインできない", {"category": "account"}, ("基本",)),
    EvalCase("c3", "画面が真っ白になる", {"category": "bug"}, ("難しい",)),
]


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    save_cases(CASES, path)
    assert load_cases(path) == CASES
    assert path.read_text(encoding="utf-8").count("\n") == 3


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('{"id": "a", "input": "x"}\n', "expected"),
        ("not json\n", "JSON として読めません"),
        ('{"id": "a", "input": "x", "expected": {}}\n{"id": "a", "input": "y", "expected": {}}\n', "重複"),
        ('{"id": "a", "input": "x", "expected": {}, "tags": "基本"}\n', "tags"),
        ("[1, 2]\n", "オブジェクト"),
        ("\n\n", "空です"),
    ],
)
def test_load_rejects_bad_files(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(EvalSetError, match=message):
        load_cases(path)


def test_load_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text('{"id": "a", "input": "x", "expected": {}}\n\n{"id": 1}\n', encoding="utf-8")
    with pytest.raises(EvalSetError, match="eval.jsonl:3"):
        load_cases(path)


def test_run_eval_scores_and_sums_tokens() -> None:
    answers = {"c1": "billing", "c2": "billing", "c3": "bug"}
    report = run_eval(CASES, lambda c: Prediction({"category": answers[c.id]}, 10, 2), exact_match("category"))
    assert [r.score for r in report.results] == [1.0, 0.0, 1.0]
    assert report.mean == pytest.approx(2 / 3)
    assert (report.input_tokens, report.output_tokens) == (30, 6)
    assert report.by_tag() == {"基本": 0.5, "難しい": 1.0}


def test_run_eval_records_failure_and_continues() -> None:
    def predict(case: EvalCase) -> Prediction:
        if case.id == "c2":
            raise RuntimeError("応答が JSON ではありません")
        return Prediction({"category": case.expected["category"]})

    report = run_eval(CASES, predict, exact_match("category"))
    assert [r.score for r in report.results] == [1.0, 0.0, 1.0]
    assert report.errors == 1
    assert "RuntimeError" in report.results[1].error


def test_run_eval_rejects_score_out_of_range() -> None:
    with pytest.raises(ValueError, match="0.0〜1.0"):
        run_eval(CASES[:1], lambda c: Prediction("x"), lambda c, o: 2.0)


def test_exact_match_accepts_plain_output() -> None:
    assert exact_match("category")(CASES[0], "billing") == 1.0
    assert exact_match("category")(CASES[0], {"category": "bug"}) == 0.0


def test_save_load_results_and_compare(tmp_path: Path) -> None:
    before = run_eval(CASES, lambda c: Prediction(c.expected), exact_match("category"), meta={"model": "基準"})
    after = run_eval(
        CASES,
        lambda c: Prediction({"category": "bug"} if c.id != "c1" else c.expected),
        exact_match("category"),
    )
    path = tmp_path / "before.jsonl"
    save_results(before, path)
    loaded = load_results(path)
    assert loaded.mean == before.mean
    assert loaded.meta == {"model": "基準"}
    assert loaded.results[0].tags == ("基本",)
    # c2 は 1 → 0 に下がった（後退）. c3 は変わらない
    assert compare(loaded, after) == [("c2", 1.0, 0.0)]


def test_run_eval_stops_on_listed_exceptions() -> None:
    class Limit(RuntimeError):
        pass

    def predict(case: EvalCase) -> Prediction:
        raise Limit("上限")

    with pytest.raises(Limit):
        run_eval(CASES, predict, exact_match("category"), stop_on=(Limit,))
