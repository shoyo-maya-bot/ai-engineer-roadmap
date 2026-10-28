"""p09_lora.py のテスト（NumPy だけの小さなモデルで、LoRA の性質を確かめる. 数秒で終わる大きさ）."""

import io

import numpy as np

from roadmap_examples.evalset import load_cases
from roadmap_examples.p07_classify import DEFAULT_EVAL
from roadmap_examples.p09_lora import (
    BASE_TRAIN,
    SFT_TRAIN,
    encode,
    features,
    init_lora,
    init_model,
    main,
    merge,
    run_all,
    train_full,
    train_lora,
)


def base_model():
    return train_full(init_model(0), *encode(load_cases(BASE_TRAIN)), epochs=50)


def test_features_are_stable_and_normalized() -> None:
    a, b = features("請求書を再発行してほしい"), features("請求書を再発行してほしい")
    assert np.array_equal(a, b)
    assert np.isclose(np.linalg.norm(a), 1.0)
    assert not features("").any()


def test_lora_starts_as_the_base_model() -> None:
    base = base_model()
    lora = init_lora(base, r=4, alpha=8.0)
    assert not lora.delta().any()  # B が 0 なので、差分も 0
    x, _ = encode(load_cases(DEFAULT_EVAL))
    assert np.array_equal(base.forward(x)[1], base.forward(x, base.w1 + lora.delta())[1])


def test_lora_trains_only_a_and_b() -> None:
    base = base_model()
    w1_before = base.w1.copy()
    x, y = encode(load_cases(SFT_TRAIN))
    lora = train_lora(base, init_lora(base, r=4), x, y, epochs=30)
    assert np.array_equal(base.w1, w1_before)  # 元の重みは変わらない
    assert lora.delta().any()
    assert np.linalg.matrix_rank(lora.delta()) <= 4  # 差分のランクは r 以下
    assert lora.n_params == 4 * (512 + 32)
    assert lora.n_params < base.n_params / 7


def test_merged_model_gives_the_same_answers() -> None:
    base = base_model()
    x, y = encode(load_cases(SFT_TRAIN))
    lora = train_lora(base, init_lora(base), x, y, epochs=30)
    merged = merge(base, lora)
    assert merged.n_params == base.n_params  # 大きさは元のまま
    assert np.allclose(merged.forward(x)[1], base.forward(x, base.w1 + lora.delta())[1])


def test_lora_improves_the_07_eval_set() -> None:
    run = run_all(load_cases(BASE_TRAIN), load_cases(SFT_TRAIN), load_cases(DEFAULT_EVAL))
    assert run.reports["lora"].mean > run.reports["base"].mean + 0.1
    assert run.reports["lora"].mean >= 0.8


def test_main_prints_the_table() -> None:
    stdout = io.StringIO()
    assert main(["--epochs", "20"], stdout=stdout) == 0
    text = stdout.getvalue()
    assert "| LoRA（r=4） |" in text and "| 2,176 |" in text
