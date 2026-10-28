"""p04_trees.py のテスト（04-2 決定木・ランダムフォレスト）."""

from roadmap_examples.evalset import load_cases
from roadmap_examples.p04_classify import DEFAULT_TRAIN
from roadmap_examples.p04_trees import churn_table, compare_text_models, fit_churn, forest_importances, tree_rules


def test_tree_learns_combined_conditions_that_a_line_cannot() -> None:
    x, y = churn_table()
    tree, linear = fit_churn(max_depth=3)
    assert tree.score(x, y) == 1.0
    assert linear.score(x, y) < 0.9
    assert fit_churn(max_depth=2)[0].score(x, y) < 1.0  # 浅すぎると表せない


def test_tree_rules_are_readable() -> None:
    rules = tree_rules(fit_churn()[0])
    assert "月の問い合わせ数 <= 2.50" in rules
    assert "利用月数 <= 6.50" in rules


def test_compare_text_models() -> None:
    results = compare_text_models(load_cases(DEFAULT_TRAIN))
    assert results["決定木"].mean < results["ランダムフォレスト"].mean <= results["ロジスティック回帰"].mean


def test_forest_importances() -> None:
    top = forest_importances(load_cases(DEFAULT_TRAIN), n=5)
    assert len(top) == 5
    assert all(a[1] >= b[1] > 0 for a, b in zip(top, top[1:], strict=False))
