"""p04_cluster.py のテスト（04-3 クラスタリング）."""

from roadmap_examples.evalset import load_cases
from roadmap_examples.p04_classify import DEFAULT_TRAIN, texts_and_labels
from roadmap_examples.p04_cluster import agreement, cluster_members, cluster_texts, silhouette_by_k

GROUPS = ["多い"] * 6 + ["問い合わせ多い"] * 6 + ["使っていない"] * 6


def test_scaled_kmeans_finds_the_three_groups() -> None:
    assert agreement(cluster_members(3), GROUPS) == 1.0


def test_without_scaling_the_large_column_wins() -> None:
    # 利用時間（数百〜千の単位）だけで分かれ、問い合わせの多さで分けられない
    assert agreement(cluster_members(3, scale=False), GROUPS) < 0.5


def test_silhouette_picks_three() -> None:
    scores = silhouette_by_k(range(2, 7))
    assert max(scores, key=scores.get) == 3


def test_text_clusters_do_not_match_human_categories() -> None:
    texts, truth = texts_and_labels(load_cases(DEFAULT_TRAIN))
    labels, terms = cluster_texts(texts, 5)
    assert len(set(labels)) == 5
    assert all(len(t) == 5 for t in terms)
    assert agreement(labels, truth) < 0.3  # 文末の形などでまとまり、窓口の分類とは一致しにくい
