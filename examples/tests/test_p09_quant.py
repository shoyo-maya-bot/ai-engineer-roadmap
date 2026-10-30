"""p09_quant.py のテスト（量子化のずれと、メモリの見積もり）."""

import io

import numpy as np
import pytest

from roadmap_examples.p09_quant import (
    dequantize,
    kv_cache_gb,
    lora_params,
    main,
    mean_abs_error,
    quantize,
    training_gb,
)


def weights_with_outlier() -> np.ndarray:
    w = np.random.default_rng(0).normal(0, 0.02, (32, 512))
    w[0, 0] = 1.0  # 1 つだけ大きな値（外れ値）
    return w


def test_quantize_round_trip_shape_and_range() -> None:
    w = weights_with_outlier()
    qz = quantize(w, bits=4, group_size=32)
    assert dequantize(qz).shape == w.shape
    assert qz.q.min() >= -7 and qz.q.max() <= 7
    assert qz.n_bytes < w.nbytes / 10


def test_fewer_bits_mean_larger_error() -> None:
    w = weights_with_outlier()
    assert mean_abs_error(w, 8) < mean_abs_error(w, 4) < mean_abs_error(w, 2)


def test_groups_limit_the_effect_of_an_outlier() -> None:
    w = weights_with_outlier()
    assert mean_abs_error(w, 4, group_size=32) < mean_abs_error(w, 4) / 5


def test_quantize_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        quantize(np.zeros(10), bits=1)
    with pytest.raises(ValueError):
        quantize(np.zeros(10), bits=4, group_size=3)


def test_training_memory() -> None:
    assert training_gb(1, "full") == 18
    assert training_gb(1, "lora", trainable=0) == 2
    assert training_gb(1, "qlora", trainable=10**9) == pytest.approx(0.5 + 18)
    with pytest.raises(ValueError):
        training_gb(1, "unknown")


def test_lora_params_and_kv_cache() -> None:
    assert lora_params(layers=2, hidden=10, rank=1, targets=1) == 40
    assert kv_cache_gb(layers=1, kv_heads=1, head_dim=1000, context=1000, bits=16) == pytest.approx(0.004)


def test_main_prints_both_tables() -> None:
    stdout = io.StringIO()
    assert main([], stdout=stdout) == 0
    text = stdout.getvalue()
    assert "| 2 ビット・全体で 1 つの倍率 |" in text
    assert "| QLoRA |" in text
