"""p04_metrics.py のテスト（04-4 指標・過学習）."""

import pytest

from roadmap_examples.evalset import load_cases
from roadmap_examples.p04_classify import DEFAULT_TRAIN
from roadmap_examples.p04_metrics import accuracy_trap, depth_curve


def test_accuracy_trap() -> None:
    scores = accuracy_trap()
    assert scores.accuracy == 0.95
    assert scores.recall_bug == 0.0
    assert scores.macro_f1 == pytest.approx(0.487, abs=0.001)


def test_deep_trees_overfit() -> None:
    rows = {r.max_depth: r for r in depth_curve(load_cases(DEFAULT_TRAIN), [1, 4, None])}
    assert rows[None].train_accuracy == 1.0
    assert rows[None].cv_accuracy < rows[4].cv_accuracy  # 深くしすぎると、新しいデータでは悪くなる
    assert rows[1].train_accuracy < rows[4].train_accuracy
