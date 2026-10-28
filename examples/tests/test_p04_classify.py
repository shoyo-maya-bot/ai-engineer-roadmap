"""04 の成果物のテスト（問い合わせの分類モデルと評価レポート. 03 の成果物で学習し、07 と同じ評価セットで測る）."""

import io
import json
from collections import Counter
from pathlib import Path

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from roadmap_examples.evalset import EvalCase, exact_match, load_cases, load_results, run_eval
from roadmap_examples.llm import FakeLLM
from roadmap_examples.p04_classify import (
    DEFAULT_EVAL,
    DEFAULT_TRAIN,
    LABELS,
    cross_validate,
    load_train,
    main,
    model_predict,
    nearest_pairs,
    overlap,
    top_features,
    train,
)
from roadmap_examples.p07_classify import DEFAULT_EVAL as P07_EVAL
from roadmap_examples.p07_classify import main as p07_main
from roadmap_examples.p07_prompt import CATEGORIES, DEFAULT_SPEC


@pytest.fixture(scope="module")
def model():
    return train(load_train(DEFAULT_TRAIN))


def test_same_labels_and_eval_set_as_07() -> None:
    assert set(LABELS) == set(CATEGORIES)
    assert DEFAULT_EVAL == P07_EVAL


def test_train_data_is_balanced_and_separate_from_the_eval_set() -> None:
    train_cases = load_train(DEFAULT_TRAIN)
    assert Counter(c.expected["category"] for c in train_cases) == {label: 26 for label in LABELS}
    assert overlap(train_cases, load_cases(DEFAULT_EVAL)) == []
    assert all(c.input and set(c.expected) == {"category"} for c in train_cases)  # 本文と分類だけ（対応時間は使わない）
    # 07-1 の例示の文も、学習データに入れない
    assert {e.text for e in DEFAULT_SPEC.examples}.isdisjoint(c.input for c in train_cases)


def test_model_beats_the_07_rule_on_the_eval_set(model) -> None:
    report = run_eval(load_cases(DEFAULT_EVAL), model_predict(model), exact_match("category"))
    assert sum(r.score for r in report.results) == 24  # 07 の仮のルールベースは 20 件
    assert [r.id for r in report.results if r.score < 1] == ["c002", "c009", "c011", "c023", "c025", "c026"]


def test_default_word_analyzer_fails_on_japanese() -> None:
    """04-1 の悪い例: 既定の分け方では、空白の無い日本語の文がまるごと 1 語になる."""
    vectorizer = TfidfVectorizer().fit(
        ["先月の請求書を再発行してもらえますか。", "請求書の送付先の住所を変えたいです。"]
    )
    assert len(vectorizer.get_feature_names_out()) == 2
    model = train(load_train(DEFAULT_TRAIN), make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000)))
    report = run_eval(load_cases(DEFAULT_EVAL), model_predict(model), exact_match("category"))
    # 語がほとんど一致しないので、予測はほぼ当て推量になる（点数がほぼ同じで、環境によって数件変わる）
    assert sum(r.score for r in report.results) < 10


def test_top_features_show_what_the_model_uses(model) -> None:
    assert "請求" in top_features(model, "billing", 5)
    assert "契約" in top_features(model, "contract", 5)


def test_cross_validation_uses_only_the_train_data() -> None:
    cv = cross_validate(load_train(DEFAULT_TRAIN))
    assert len(cv.scores) == 5
    assert cv.mean == pytest.approx(0.738, abs=0.001)


def test_main_writes_report_and_results(tmp_path: Path) -> None:
    out, report = tmp_path / "results-p04.jsonl", tmp_path / "report.md"
    stdout = io.StringIO()
    assert main(["--out", str(out), "--report", str(report)], stdout=stdout) == 0
    text = report.read_text(encoding="utf-8")
    assert "| 04 のモデル | 80.0% |" in text
    assert "| 07 の仮のルールベース | 66.7% |" in text
    assert "c026" in text
    assert "評価セットに近い学習データ" in text
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
    assert "前回からの変化: 6 件" in stdout.getvalue()  # 04 のモデルが誤った 6 件


def test_main_refuses_overlapping_train_and_eval(tmp_path: Path) -> None:
    leaky = tmp_path / "train.jsonl"
    leaky.write_text(P07_EVAL.read_text(encoding="utf-8"), encoding="utf-8")
    assert main(["--train", str(leaky)], stdout=io.StringIO()) == 1


def test_main_reports_bad_file(tmp_path: Path) -> None:
    assert main(["--train", str(tmp_path / "missing.jsonl")], stdout=io.StringIO()) == 1


# 04-4 の悪い例: 評価セットで誤った 6 件を読んで、その言い換えを学習データに足してしまう
PARAPHRASES = [
    EvalCase("x1", "同じ日にカードの決済が二重にされています。", {"category": "billing"}),
    EvalCase("x2", "機種を変えるときの二段階認証の移し方を知りたいです。", {"category": "account"}),
    EvalCase("x3", "スマホを買い替えたら、確認コードが前の端末に届いてログインできません。", {"category": "account"}),
    EvalCase("x4", "支払いが負担なので、安いプランへの変更を考えています。", {"category": "contract"}),
    EvalCase("x5", "いつも便利に使わせてもらっています。ありがとうございます。", {"category": "other"}),
    EvalCase("x6", "ダークモードがあるとうれしいという要望です。", {"category": "other"}),
]


def test_paraphrase_leak_inflates_only_the_eval_score() -> None:
    eval_cases = load_cases(DEFAULT_EVAL)
    base = load_train(DEFAULT_TRAIN)
    leaky = base + PARAPHRASES
    assert overlap(leaky, eval_cases) == []  # 完全一致の確かめでは見つからない
    before = run_eval(eval_cases, model_predict(train(base)), exact_match("category")).mean
    after = run_eval(eval_cases, model_predict(train(leaky)), exact_match("category")).mean
    assert (round(before * 30), round(after * 30)) == (24, 29)  # 評価セットだけが大きく上がる
    assert abs(cross_validate(leaky).mean - cross_validate(base).mean) < 0.05  # 交差検証はほとんど変わらない
    found = {t.id for _, _, t in nearest_pairs(leaky, eval_cases, n=6)}
    assert {"x5", "x6", "x3"} <= found  # ほぼ同じ文は、近い組の上位に出る
    assert not {"x1", "x2", "x4"} & found  # 言い回しを変えた文は、ほかの近い文に埋もれて出ない
