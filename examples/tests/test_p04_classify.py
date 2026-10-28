"""04 の成果物のテスト（問い合わせの分類モデルと評価レポート. 07 と同じ評価セットで測る）."""

import io
import json
from collections import Counter
from pathlib import Path

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from roadmap_examples.evalset import exact_match, load_cases, load_results, run_eval
from roadmap_examples.llm import FakeLLM
from roadmap_examples.p04_classify import (
    DEFAULT_EVAL,
    DEFAULT_TRAIN,
    LABELS,
    cross_validate,
    main,
    model_predict,
    overlap,
    top_features,
    train,
)
from roadmap_examples.p07_classify import DEFAULT_EVAL as P07_EVAL
from roadmap_examples.p07_classify import main as p07_main
from roadmap_examples.p07_prompt import CATEGORIES, DEFAULT_SPEC


@pytest.fixture(scope="module")
def model():
    return train(load_cases(DEFAULT_TRAIN))


def test_same_labels_and_eval_set_as_07() -> None:
    assert set(LABELS) == set(CATEGORIES)
    assert DEFAULT_EVAL == P07_EVAL


def test_train_data_is_balanced_and_separate_from_the_eval_set() -> None:
    train_cases = load_cases(DEFAULT_TRAIN)
    assert Counter(c.expected["category"] for c in train_cases) == {label: 20 for label in LABELS}
    assert overlap(train_cases, load_cases(DEFAULT_EVAL)) == []
    # 07-1 の例示の文も、学習データに入れない
    assert {e.text for e in DEFAULT_SPEC.examples}.isdisjoint(c.input for c in train_cases)


def test_model_beats_the_07_rule_on_the_eval_set(model) -> None:
    report = run_eval(load_cases(DEFAULT_EVAL), model_predict(model), exact_match("category"))
    assert sum(r.score for r in report.results) == 29  # 07 の仮のルールベースは 20 件
    assert [r.id for r in report.results if r.score < 1] == ["c003"]


def test_default_word_analyzer_fails_on_japanese() -> None:
    """04-1 の悪い例: 既定の分け方では、空白の無い日本語の文がまるごと 1 語になる."""
    vectorizer = TfidfVectorizer().fit(
        ["先月の請求書を再発行してもらえますか。", "請求書の送付先の住所を変えたいです。"]
    )
    assert len(vectorizer.get_feature_names_out()) == 2
    model = train(load_cases(DEFAULT_TRAIN), make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000)))
    report = run_eval(load_cases(DEFAULT_EVAL), model_predict(model), exact_match("category"))
    # 語がほとんど一致しないので、予測はほぼ当て推量になる（点数がほぼ同じで、環境によって数件変わる）
    assert sum(r.score for r in report.results) < 10


def test_top_features_show_what_the_model_uses(model) -> None:
    assert "請求" in top_features(model, "billing", 5)
    assert "契約" in top_features(model, "contract", 5)


def test_cross_validation_uses_only_the_train_data() -> None:
    cv = cross_validate(load_cases(DEFAULT_TRAIN))
    assert len(cv.scores) == 5
    assert cv.mean == pytest.approx(0.67)


def test_main_writes_report_and_results(tmp_path: Path) -> None:
    out, report = tmp_path / "results-p04.jsonl", tmp_path / "report.md"
    stdout = io.StringIO()
    assert main(["--out", str(out), "--report", str(report)], stdout=stdout) == 0
    text = report.read_text(encoding="utf-8")
    assert "| 04 のモデル | 96.7% |" in text
    assert "| 07 の仮のルールベース | 66.7% |" in text
    assert "c003" in text
    assert load_results(out).meta == {"mode": "p04-model"}


def test_results_can_replace_the_07_baseline(tmp_path: Path) -> None:
    """04 の結果のファイルを、07 の --compare に渡して比べられる（07 の基準の差し替え）."""
    base = tmp_path / "results-p04.jsonl"
    main(["--out", str(base)], stdout=io.StringIO())
    replies = [json.dumps({"category": c.expected["category"], "reason": "テスト"}) for c in load_cases(P07_EVAL)]
    stdout = io.StringIO()
    code = p07_main(
        ["--mode", "ollama", "--model", "m", "--compare", str(base)], llm=FakeLLM.of(replies), stdout=stdout
    )
    assert code == 0
    assert "前回からの変化: 1 件" in stdout.getvalue()  # 04 のモデルが誤った c003 だけ


def test_main_refuses_overlapping_train_and_eval(tmp_path: Path) -> None:
    leaky = tmp_path / "train.jsonl"
    leaky.write_text(DEFAULT_TRAIN.read_text(encoding="utf-8") + P07_EVAL.read_text(encoding="utf-8"), encoding="utf-8")
    assert main(["--train", str(leaky)], stdout=io.StringIO()) == 1


def test_main_reports_bad_file(tmp_path: Path) -> None:
    assert main(["--train", str(tmp_path / "missing.jsonl")], stdout=io.StringIO()) == 1
