"""04-4 指標・過学習.

- 正解率の落とし穴: 分類の件数が偏っていると、多い分類だけを答えても正解率は高くなる. 再現率と F1 で見抜く
- 過学習: 決定木を深くするほど、学習データの正解率は上がる. 交差検証の正解率は、ある深さから上がらなくなる
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from roadmap_examples.evalset import EvalCase
from roadmap_examples.p04_classify import build_model, cross_validate, texts_and_labels


@dataclass(frozen=True)
class Scores:
    accuracy: float
    recall_bug: float  # 不具合（bug）のうち、bug と当てた割合
    macro_f1: float  # 分類ごとの F1 を、件数に関係なく平均したもの


def imbalanced_scores(truth: Sequence[str], pred: Sequence[str]) -> Scores:
    return Scores(
        accuracy=float(accuracy_score(truth, pred)),
        recall_bug=float(recall_score(truth, pred, labels=["bug"], average="macro", zero_division=0)),
        macro_f1=float(f1_score(truth, pred, average="macro", zero_division=0)),
    )


def accuracy_trap() -> Scores:
    """100 件のうち bug が 5 件. 「いつも other」と答えるだけのモデルの指標."""
    truth = ["other"] * 95 + ["bug"] * 5
    return imbalanced_scores(truth, ["other"] * 100)


@dataclass(frozen=True)
class DepthScore:
    max_depth: int | None
    train_accuracy: float  # 学習に使ったデータでの正解率
    cv_accuracy: float  # 交差検証（学習に使っていない分割）での正解率の平均


def depth_curve(cases: Sequence[EvalCase], depths: Sequence[int | None]) -> list[DepthScore]:
    """決定木の深さの上限ごとに、学習データと交差検証の正解率を並べる."""
    texts, labels = texts_and_labels(cases)
    tfidf = build_model().named_steps["tfidf"]
    rows = []
    for depth in depths:
        model = Pipeline([("tfidf", tfidf), ("clf", DecisionTreeClassifier(max_depth=depth, random_state=0))])
        cv = cross_validate(cases, model)
        train_acc = model.fit(texts, labels).score(texts, labels)
        rows.append(DepthScore(depth, float(train_acc), cv.mean))
    return rows
