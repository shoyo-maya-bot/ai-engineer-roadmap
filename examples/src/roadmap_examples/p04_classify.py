"""04 の成果物: 問い合わせの自動分類モデル（scikit-learn）と評価レポート.

- 学習データ: `data/p04/train.jsonl`（架空の問い合わせ 100 件. 07-5 の評価セットと同じ形式・同じ分類）
- 評価セット: `data/p07/inquiries.jsonl`（07 と同じ 30 件）. 07 で LLM と比べるとき、この結果を基準にする
- モデル: 文字の 1〜2 文字の組（TF-IDF）+ ロジスティック回帰. 日本語を単語に分けずに扱える

使い方（examples/ で）:
  uv run python -m roadmap_examples.p04_classify --out results-p04.jsonl --report report-p04.md

07 の比べる基準を差し替えるときは、この結果のファイルを 07 の `--compare` に渡す（04-4 の「07 の基準を差し替える」）.

終了コード: 0 成功 / 1 データの誤り（読めない・学習データと評価セットに同じ文がある）/ 2 引数の誤り
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from roadmap_examples.evalset import (
    EvalCase,
    EvalSetError,
    Predict,
    Prediction,
    Report,
    compare,
    exact_match,
    load_cases,
    load_results,
    run_eval,
    save_results,
)
from roadmap_examples.p07_classify import baseline_predict  # 07 の仮のルールベース分類（レポートで並べる）

DATA = Path(__file__).resolve().parents[2] / "data"
DEFAULT_TRAIN = DATA / "p04" / "train.jsonl"
DEFAULT_EVAL = DATA / "p07" / "inquiries.jsonl"  # 07 と同じ評価セット
# 分類の種類（07 の CATEGORIES のキーと同じ. テストで確かめる）
LABELS = ("billing", "account", "bug", "contract", "other")


def texts_and_labels(cases: Sequence[EvalCase]) -> tuple[list[str], list[str]]:
    return [c.input for c in cases], [c.expected["category"] for c in cases]


def build_model(c: float = 1.0, ngram_range: tuple[int, int] = (1, 2)) -> Pipeline:
    """文字の組の TF-IDF と、ロジスティック回帰をつないだモデル（まだ学習していない）."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=ngram_range)),
            ("clf", LogisticRegression(C=c, max_iter=1000)),
        ]
    )


def train(cases: Sequence[EvalCase], model: Pipeline | None = None) -> Pipeline:
    texts, labels = texts_and_labels(cases)
    return (model or build_model()).fit(texts, labels)


def model_predict(model: Pipeline) -> Predict:
    """学習したモデルを、07 の `baseline_predict` と同じ形（EvalCase → Prediction）の関数にする."""

    def predict(case: EvalCase) -> Prediction:
        return Prediction({"category": str(model.predict([case.input])[0])})

    return predict


def top_features(model: Pipeline, label: str, n: int = 5) -> list[str]:
    """その分類の重み（係数）が大きい文字の組. モデルが何を手がかりにしたかを読む."""
    names = model.named_steps["tfidf"].get_feature_names_out()
    clf = model.named_steps["clf"]
    row = clf.coef_[list(clf.classes_).index(label)]
    return [str(names[i]) for i in row.argsort()[::-1][:n]]


def overlap(train_cases: Sequence[EvalCase], eval_cases: Sequence[EvalCase]) -> list[str]:
    """学習データと評価セットの両方にある問い合わせ文（あると、答えを見せた試験になる）."""
    seen = {c.input.strip() for c in train_cases}
    return sorted({c.input for c in eval_cases if c.input.strip() in seen})


@dataclass(frozen=True)
class CVResult:
    scores: list[float]

    @property
    def mean(self) -> float:
        return statistics.fmean(self.scores)

    @property
    def std(self) -> float:
        return statistics.pstdev(self.scores)


def cross_validate(cases: Sequence[EvalCase], model: Pipeline | None = None, folds: int = 5) -> CVResult:
    """層化 k 分割の交差検証（分類ごとの割合をそろえて分ける. 並びを混ぜる乱数は固定）."""
    texts, labels = texts_and_labels(cases)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    scores = cross_val_score(model or build_model(), texts, labels, cv=cv, scoring="accuracy")
    return CVResult([float(s) for s in scores])


def majority_predict(train_cases: Sequence[EvalCase]) -> Predict:
    """いちばん多い分類を常に答える基準（これより悪いモデルは、何も学んでいない）."""
    texts, labels = texts_and_labels(train_cases)
    dummy = DummyClassifier(strategy="most_frequent").fit(texts, labels)

    def predict(case: EvalCase) -> Prediction:
        return Prediction({"category": str(dummy.predict([case.input])[0])})

    return predict


def predicted_labels(report: Report) -> list[str]:
    return [str(r.output.get("category")) if isinstance(r.output, dict) else "(失敗)" for r in report.results]


def per_class(cases: Sequence[EvalCase], report: Report) -> dict[str, tuple[float, float, float, int]]:
    """分類ごとの（適合率, 再現率, F1, 件数）と、全体の macro 平均（キー「macro 平均」）."""
    _, truth = texts_and_labels(cases)
    pred = predicted_labels(report)
    p, r, f, n = precision_recall_fscore_support(truth, pred, labels=list(LABELS), zero_division=0)
    rows = {label: (float(p[i]), float(r[i]), float(f[i]), int(n[i])) for i, label in enumerate(LABELS)}
    mp, mr, mf, _ = precision_recall_fscore_support(truth, pred, labels=list(LABELS), average="macro", zero_division=0)
    rows["macro 平均"] = (float(mp), float(mr), float(mf), len(truth))
    return rows


def render_report(
    train_cases: Sequence[EvalCase],
    eval_cases: Sequence[EvalCase],
    cv: CVResult,
    reports: dict[str, Report],
    model_name: str,
) -> str:
    """評価レポート（Markdown）. reports は {基準やモデルの名前: 評価セットでの結果}."""
    model = reports[model_name]
    lines = [
        "# 問い合わせの自動分類: 評価レポート",
        "",
        "## 条件",
        "",
        f"- 学習データ: {len(train_cases)} 件 / 評価セット: {len(eval_cases)} 件（07 と同じ）",
        "- モデル: 文字の 1〜2 文字の組の TF-IDF + ロジスティック回帰（C=1.0）",
        "",
        "## 交差検証（学習データの中だけで測る）",
        "",
        f"- 正解率: 平均 {cv.mean:.1%}（標準偏差 {cv.std:.1%}、{len(cv.scores)} 分割）",
        "",
        "## 評価セットでの比較",
        "",
        "| 基準・モデル | 正解率 | macro F1 | " + " | ".join(f"正解率（{t}）" for t in model.by_tag()) + " |",
        "|---|---:|---:|" + "---:|" * len(model.by_tag()),
    ]
    for name, rep in reports.items():
        f1 = per_class(eval_cases, rep)["macro 平均"][2]
        tags = " | ".join(f"{s:.1%}" for s in rep.by_tag().values())
        lines.append(f"| {name} | {rep.mean:.1%} | {f1:.3f} | {tags} |")
    lines += [
        "",
        f"## 分類ごとの指標（{model_name}）",
        "",
        "| 分類 | 適合率 | 再現率 | F1 | 件数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, (p, r, f, n) in per_class(eval_cases, model).items():
        lines.append(f"| {label} | {p:.2f} | {r:.2f} | {f:.2f} | {n} |")
    _, truth = texts_and_labels(eval_cases)
    matrix = confusion_matrix(truth, predicted_labels(model), labels=list(LABELS))
    lines += ["", "## 混同行列（行: 正解 / 列: 予測）", "", "| 正解＼予測 | " + " | ".join(LABELS) + " |"]
    lines.append("|---|" + "---:|" * len(LABELS))
    lines += [f"| {label} | " + " | ".join(str(v) for v in matrix[i]) + " |" for i, label in enumerate(LABELS)]
    wrong = [(c, r) for c, r in zip(eval_cases, model.results, strict=True) if r.score < 1.0]
    lines += ["", "## 誤った件", ""]
    lines += [
        f"- {c.id}（{'・'.join(c.tags)}）: {c.input} 正解 {c.expected['category']} → 予測 "
        f"{predicted_labels(Report([r]))[0]}"
        for c, r in wrong
    ] or ["- なし"]
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="p04_classify", description="問い合わせの分類モデルを学習し、評価セットで測る")
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN, help="学習データ（JSONL）")
    p.add_argument("--eval", type=Path, default=DEFAULT_EVAL, help="評価セット（JSONL. 07 と同じ）")
    p.add_argument("--out", type=Path, help="評価セットでの結果を書く JSONL（07 の --compare に渡せる）")
    p.add_argument("--report", type=Path, help="評価レポートを書く Markdown")
    p.add_argument("--compare", type=Path, help="比べる前回の結果（JSONL）")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None, stdout: TextIO | None = None) -> int:
    args = parse_args(argv)
    stdout = stdout or sys.stdout
    try:
        train_cases, eval_cases = load_cases(args.train), load_cases(args.eval)
    except (OSError, EvalSetError) as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    if dup := overlap(train_cases, eval_cases):
        print(f"エラー: 学習データと評価セットに同じ文があります（{len(dup)} 件）: {dup[0]}", file=sys.stderr)
        return 1
    model = train(train_cases)
    score = exact_match("category")
    name = "04 のモデル"
    reports = {
        "多数派（常に同じ分類）": run_eval(eval_cases, majority_predict(train_cases), score),
        "07 の仮のルールベース": run_eval(eval_cases, baseline_predict, score),
        name: run_eval(eval_cases, model_predict(model), score, meta={"mode": "p04-model"}),
    }
    text = render_report(train_cases, eval_cases, cross_validate(train_cases), reports, name)
    print(text, end="", file=stdout)
    if args.out:
        save_results(reports[name], args.out)
    if args.report:
        args.report.write_text(text, encoding="utf-8")
    if args.compare:
        changed = compare(load_results(args.compare), reports[name])
        print(f"前回からの変化: {len(changed)} 件", file=stdout)
        for case_id, before, after in changed:
            print(f"  {case_id}: {before:.0f} → {after:.0f}{'（後退）' if after < before else ''}", file=stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
