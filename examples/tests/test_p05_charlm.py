"""05 の成果物のテスト（小さな文字単位の言語モデル. NumPy・CPU で数秒）."""

import io
from pathlib import Path

import numpy as np
import pytest

from roadmap_examples.p05_charlm import (
    Config,
    Vocab,
    forward,
    generate,
    init_params,
    load,
    loss_and_grads,
    main,
    save,
    template_corpus,
    train,
)
from roadmap_examples.p05_mlp import numerical_grad

SMALL = Config(context=8, dim=16, hidden=32, batch=16, lr=0.01, seed=0)


def test_vocab_round_trip_and_unknown_chars() -> None:
    vocab = Vocab.build("請求書\n")
    assert vocab.decode(vocab.encode("請求")) == "請求"
    with pytest.raises(KeyError):
        vocab.encode("契約")


def test_backprop_matches_numerical_gradient() -> None:
    cfg = Config(context=5, dim=4, hidden=6, batch=2, seed=1)
    params = init_params(7, cfg)
    rng = np.random.default_rng(0)
    x, y = rng.integers(0, 7, (2, 5)), rng.integers(0, 7, (2, 5))
    _, grads = loss_and_grads(params, x, y)
    for name, p in params.items():
        expected = numerical_grad(lambda: loss_and_grads(params, x, y)[0], p)
        assert grads[name] == pytest.approx(expected, abs=1e-7), name


def test_future_tokens_do_not_change_earlier_predictions() -> None:
    params = init_params(10, SMALL)
    a = np.array([[1, 2, 3, 4, 5]])
    b = np.array([[1, 2, 3, 9, 9]])
    assert np.allclose(forward(params, a)[0][0, :3], forward(params, b)[0][0, :3])


@pytest.fixture(scope="module")
def trained():
    return train(template_corpus(400), steps=300, cfg=SMALL, eval_every=100)


def test_training_beats_guessing(trained) -> None:
    first, last = trained.log[0], trained.log[-1]
    assert last[1] < first[1]
    assert last[2] < np.log(len(trained.vocab.chars)) - 2  # 何も学んでいない（一様）より、ずっと小さい


def test_generate_is_reproducible(trained) -> None:
    text = generate(trained.params, trained.vocab, trained.cfg, "請求", 20, seed=1)
    assert text.startswith("請求") and len(text) == 22
    assert text == generate(trained.params, trained.vocab, trained.cfg, "請求", 20, seed=1)


def test_save_and_load(trained, tmp_path: Path) -> None:
    path = tmp_path / "charlm.npz"
    save(path, trained)
    params, vocab, cfg = load(path)
    assert vocab == trained.vocab and cfg.context == trained.cfg.context
    a = generate(params, vocab, cfg, "請求", 10)
    assert a == generate(trained.params, trained.vocab, trained.cfg, "請求", 10)


def test_main_trains_and_generates() -> None:
    stdout = io.StringIO()
    assert main(["--steps", "50", "--prompt", "解約", "--length", "5"], stdout=stdout) == 0
    assert "ステップ 50" in stdout.getvalue()


def test_main_rejects_unknown_prompt_chars(tmp_path: Path, trained) -> None:
    path = tmp_path / "charlm.npz"
    save(path, trained)
    assert main(["--load", str(path), "--prompt", "猫"], stdout=io.StringIO()) == 1
