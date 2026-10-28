"""p06_context.py のテスト（06-2 次のトークンの選び方と、コンテキストに収める）."""

import random

import pytest

from roadmap_examples.llm import Message
from roadmap_examples.p06_context import ContextOverflow, fit_history, sample, softmax

LOGITS = [2.0, 1.0, 0.1]


def test_softmax_sums_to_one_and_keeps_order() -> None:
    probs = softmax(LOGITS)
    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] > probs[1] > probs[2]


def test_low_temperature_sharpens_high_temperature_flattens() -> None:
    assert softmax(LOGITS, 0.5)[0] > softmax(LOGITS, 1.0)[0] > softmax(LOGITS, 2.0)[0]


def test_temperature_zero_is_greedy() -> None:
    assert softmax(LOGITS, 0) == [1.0, 0.0, 0.0]


def test_softmax_handles_large_logits() -> None:
    assert softmax([1000.0, 999.0])[0] == pytest.approx(0.731, abs=0.001)


def test_sample_is_reproducible_with_seed() -> None:
    probs = softmax(LOGITS, 1.0)
    first = [sample(probs, random.Random(42)) for _ in range(5)]
    second = [sample(probs, random.Random(42)) for _ in range(5)]
    assert first == second
    assert sample([0.0, 1.0, 0.0], random.Random(0)) == 1


def count_chars(text: str) -> int:
    return len(text)


HISTORY = [
    Message("system", "s" * 10),
    Message("user", "a" * 30),
    Message("assistant", "b" * 30),
    Message("user", "c" * 20),
]


def test_fit_history_drops_oldest_first_and_keeps_system() -> None:
    kept = fit_history(HISTORY, max_context=100, reserve_output=30, count=count_chars)
    # 入力に使えるのは 70. system 10 + 最新 20 + その前 30 = 60（さらに前の 30 は入らない）
    assert [m.content[0] for m in kept] == ["s", "b", "c"]


def test_fit_history_keeps_everything_when_it_fits() -> None:
    assert fit_history(HISTORY, max_context=200, reserve_output=0, count=count_chars) == HISTORY


def test_fit_history_raises_when_latest_message_does_not_fit() -> None:
    with pytest.raises(ContextOverflow, match="入りません"):
        fit_history(HISTORY, max_context=40, reserve_output=20, count=count_chars)
