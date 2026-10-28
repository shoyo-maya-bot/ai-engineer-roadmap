"""p05_loop.py のテスト（05-2 学習ループ）."""

import numpy as np

from roadmap_examples.p05_loop import SGD, Adam, batches, train_moons


def test_batches_cover_every_index_once() -> None:
    parts = list(batches(10, 4, np.random.default_rng(0)))
    assert [len(p) for p in parts] == [4, 4, 2]
    assert sorted(np.concatenate(parts).tolist()) == list(range(10))


def test_optimizers_reduce_a_simple_loss() -> None:
    # (p - 3)^2 の最小は p = 3. 勾配は 2(p - 3)
    for make in (lambda ps: SGD(ps, lr=0.1), lambda ps: Adam(ps, lr=0.1)):
        params = {"p": np.array([0.0])}
        opt = make(params)
        for _ in range(200):
            opt.step({"p": 2 * (params["p"] - 3.0)})
        assert abs(params["p"][0] - 3.0) < 0.05


def test_training_loop_learns_moons_and_stops_early() -> None:
    history, accuracy = train_moons("adam", epochs=300)
    assert history.train_losses[-1] < history.train_losses[0]
    assert len(history.train_losses) < 300  # 検証の損失が下がらなくなって止まった
    assert history.best_epoch == int(np.argmin(history.val_losses))
    assert accuracy > 0.9
