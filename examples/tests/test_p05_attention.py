"""p05_attention.py のテスト（05-3 RNN からアテンションへ）."""

import numpy as np
import pytest

from roadmap_examples.p05_attention import attention, causal_mask, first_token_influence


def test_rnn_forgets_the_first_token_over_long_inputs() -> None:
    assert first_token_influence(5) > 100 * first_token_influence(20)


def test_attention_weights_are_probabilities() -> None:
    rng = np.random.default_rng(0)
    q, k, v = rng.normal(size=(3, 5, 4))
    out, weights = attention(q, k, v)
    assert out.shape == (5, 4)
    assert weights.sum(axis=-1) == pytest.approx(np.ones(5))


def test_causal_mask_hides_the_future() -> None:
    rng = np.random.default_rng(0)
    q, k, v = rng.normal(size=(3, 4, 2))
    out, weights = attention(q, k, v, causal_mask(4))
    assert np.allclose(np.triu(weights, k=1), 0.0)
    # 未来（最後の位置）の値を変えても、それより前の出力は変わらない
    v2 = v.copy()
    v2[3] += 10.0
    out2, _ = attention(q, k, v2, causal_mask(4))
    assert np.allclose(out[:3], out2[:3])


def test_attention_looks_up_the_matching_key() -> None:
    keys = np.eye(3) * 10  # 3 つの鍵. 内積が大きい鍵に重みが集まる
    values = np.array([[1.0], [2.0], [3.0]])
    out, _ = attention(np.array([[0.0, 1.0, 0.0]]) * 10, keys, values)
    assert out[0, 0] == pytest.approx(2.0, abs=1e-6)
