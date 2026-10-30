"""04-2 決定木・ランダムフォレスト.

- 表のデータ（利用月数と問い合わせの数）で、解約しそうかを予測する. 条件の組み合わせで決まる規則は、
  直線で分けるロジスティック回帰では表せないが、決定木は「もし〜なら」の分かれ道で表せる
- 問い合わせ文の分類（04 の成果物と同じデータ）で、ロジスティック回帰・決定木・ランダムフォレストを交差検証で比べる
"""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

from roadmap_examples.evalset import EvalCase
from roadmap_examples.p04_classify import CVResult, build_model, cross_validate, train

FEATURES = ["利用月数", "月の問い合わせ数"]
MONTHS = [1, 2, 3, 4, 5, 8, 12, 18, 24, 36]
INQUIRIES = [0, 1, 2, 3, 4, 5]


def churn_rule(months: int, inquiries: int) -> int:
    """架空の規則（1 = 解約しそう）: 始めたばかりで問い合わせが多い人と、長く使っていて問い合わせが無い人."""
    return int(inquiries >= 3) if months < 6 else int(inquiries == 0)


def churn_table() -> tuple[list[list[int]], list[int]]:
    x = [[m, q] for m in MONTHS for q in INQUIRIES]
    return x, [churn_rule(m, q) for m, q in x]


def fit_churn(max_depth: int = 3) -> tuple[DecisionTreeClassifier, LogisticRegression]:
    x, y = churn_table()
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=0).fit(x, y)
    linear = LogisticRegression(max_iter=1000).fit(x, y)
    return tree, linear


def tree_rules(tree: DecisionTreeClassifier) -> str:
    """学習した木を「もし〜なら」の字下げの文にする."""
    return export_text(tree, feature_names=FEATURES)


def text_models(seed: int = 0) -> dict[str, Pipeline]:
    """同じ特徴量（文字の組の TF-IDF）に、違う分類器をつないだモデル."""
    base = build_model()
    tfidf = base.named_steps["tfidf"]
    return {
        "ロジスティック回帰": base,
        "決定木": Pipeline([("tfidf", tfidf), ("clf", DecisionTreeClassifier(random_state=seed))]),
        "ランダムフォレスト": Pipeline(
            [("tfidf", tfidf), ("clf", RandomForestClassifier(n_estimators=200, random_state=seed))]
        ),
    }


def compare_text_models(cases: Sequence[EvalCase]) -> dict[str, CVResult]:
    return {name: cross_validate(cases, model) for name, model in text_models().items()}


def forest_importances(cases: Sequence[EvalCase], n: int = 5) -> list[tuple[str, float]]:
    """ランダムフォレストの特徴量の重要度（不純度の減り方の平均）が大きい文字の組."""
    forest = train(cases, text_models()["ランダムフォレスト"])
    names = forest.named_steps["tfidf"].get_feature_names_out()
    importances = forest.named_steps["clf"].feature_importances_
    order = importances.argsort()[::-1][:n]
    return [(str(names[i]), float(importances[i])) for i in order]
