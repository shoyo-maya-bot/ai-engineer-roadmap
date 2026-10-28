"""p02_vectors のテスト."""

import numpy as np
import pytest

from roadmap_examples import p06_tokens
from roadmap_examples.p02_vectors import cosine, cosine_matrix, dot, norm, normalize_rows, top_k


def test_dot_and_norm() -> None:
    assert dot([1, 2, 3], [4, 5, 6]) == 32
    assert norm([3, 4]) == 5.0
    with pytest.raises(ValueError, match="長さ"):
        dot([1, 2], [1, 2, 3])


@pytest.mark.parametrize(
    ("a", "b", "want"),
    [
        ([1, 0], [2, 0], 1.0),  # 同じ向き（長さは関係ない）
        ([1, 0], [0, 3], 0.0),  # 直交
        ([1, 0], [-1, 0], -1.0),  # 逆向き
        ([0, 0], [1, 1], 0.0),  # 長さ 0
    ],
)
def test_cosine_cases(a, b, want) -> None:
    assert cosine(a, b) == pytest.approx(want)


def test_cosine_matches_06_and_numpy() -> None:
    a, b = [0.2, 0.5, 0.1], [0.4, 0.1, 0.3]
    assert cosine(a, b) == pytest.approx(p06_tokens.cosine(a, b))
    assert cosine_matrix([a], [b])[0, 0] == pytest.approx(cosine(a, b))


def test_normalize_rows_keeps_zero_rows() -> None:
    m = normalize_rows([[3, 4], [0, 0]])
    assert np.allclose(m, [[0.6, 0.8], [0, 0]])


def test_cosine_matrix_shape_and_values() -> None:
    queries = np.array([[1.0, 0.0], [0.0, 1.0]])
    docs = np.array([[1.0, 1.0], [2.0, 0.0], [0.0, -1.0]])
    sims = cosine_matrix(queries, docs)
    assert sims.shape == (2, 3)
    want = [[cosine(q, d) for d in docs.tolist()] for q in queries.tolist()]
    assert np.allclose(sims, want)


def test_top_k_orders_by_similarity() -> None:
    docs = [[1.0, 0.0], [0.7, 0.7], [0.0, 1.0], [1.0, 0.0]]
    result = top_k([1.0, 0.1], docs, k=3)
    assert [i for i, _ in result] == [0, 3, 1]  # 同じ値（0 と 3）は行番号の小さい順
    assert result[0][1] == pytest.approx(cosine([1.0, 0.1], [1.0, 0.0]))
