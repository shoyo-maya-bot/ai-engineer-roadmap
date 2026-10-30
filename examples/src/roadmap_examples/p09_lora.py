"""09-2 LoRA を小さく実装する: CPU で数秒で動く小さな分類モデルで、LoRA の仕組みと効き目を確かめる.

本物の LLM の LoRA（PEFT など）と同じ考え方を、NumPy だけで書いている:
- 学習済みの重み W は固定し、学習するのは小さな 2 つの行列 A（r × 入力）と B（出力 × r）だけ
- 使う重みは W + (alpha / r) × B @ A. B を 0 で始めるので、学習の前は元のモデルとまったく同じ答えになる
- 学習の後は B @ A を W に足し込める（マージ）. マージしたモデルは、元と同じ大きさ・同じ速さで動く

題材: 07 の問い合わせの分類（5 つ）. 「事前学習」の代わりに汎用の例（data/p09/base_train.jsonl）で
小さなモデルを作り、業務の例（data/p09/sft_train.jsonl）で LoRA の調整をして、07 の評価セットで前後を比べる.
学習の例は、07 の評価セットを見ていない人が、分類の定義だけから書いた（評価セットに寄せていない）.
r などの設定は検証用の例（data/p09/sft_val.jsonl）で選び、07 の評価セットは最後に 1 回だけ使う.

使い方（examples/ で. LLM もネットワークも使わない）:
  uv run python -m roadmap_examples.p09_lora --ranks 1,2,4,8 --out results-lora.jsonl
"""

from __future__ import annotations

import argparse
import sys
import time
import unicodedata
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TextIO

import numpy as np

from roadmap_examples.evalset import (
    EvalCase,
    Predict,
    Prediction,
    Report,
    exact_match,
    load_cases,
    run_eval,
    save_results,
)
from roadmap_examples.p07_classify import DEFAULT_EVAL
from roadmap_examples.p07_prompt import CATEGORIES

DATA = Path(__file__).resolve().parents[2] / "data" / "p09"
BASE_TRAIN = DATA / "base_train.jsonl"
SFT_TRAIN = DATA / "sft_train.jsonl"
SFT_VAL = DATA / "sft_val.jsonl"  # 設定（r など）を選ぶための検証用. 07 の評価セットとは別
LABELS = tuple(CATEGORIES)
DIM = 512  # 入力のベクトルの長さ（文字と 2 文字の組を、この数の箱に振り分ける）
HIDDEN = 32  # 中間の層の大きさ


def features(text: str, dim: int = DIM) -> np.ndarray:
    """文章を、文字と 2 文字の組の出現の数のベクトルにする（長さ 1 にそろえる）. 同じ文章なら毎回同じベクトル."""
    text = unicodedata.normalize("NFKC", text).lower()
    vec = np.zeros(dim)
    grams = list(text) + [text[i : i + 2] for i in range(len(text) - 1)]
    for g in grams:
        vec[zlib.crc32(g.encode("utf-8")) % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def encode(cases: Sequence[EvalCase]) -> tuple[np.ndarray, np.ndarray]:
    """（入力の行列, 正解の番号）."""
    x = np.stack([features(c.input) for c in cases])
    y = np.array([LABELS.index(c.expected["category"]) for c in cases])
    return x, y


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class TinyModel:
    """入力 → 中間の層（tanh）→ 5 つの分類、の小さなモデル. 本物の LLM の線形の層の代わり."""

    w1: np.ndarray  # (HIDDEN, DIM)
    b1: np.ndarray  # (HIDDEN,)
    w2: np.ndarray  # (分類の数, HIDDEN)
    b2: np.ndarray  # (分類の数,)

    @property
    def n_params(self) -> int:
        return self.w1.size + self.b1.size + self.w2.size + self.b2.size

    def forward(self, x: np.ndarray, w1: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """（中間の層の値, 分類ごとの確率）. w1 を渡すと、その重みで計算する（LoRA で使う）."""
        h = np.tanh(x @ (self.w1 if w1 is None else w1).T + self.b1)
        return h, softmax(h @ self.w2.T + self.b2)


def init_model(seed: int = 0, dim: int = DIM, hidden: int = HIDDEN) -> TinyModel:
    rng = np.random.default_rng(seed)
    return TinyModel(
        rng.normal(0, 1 / np.sqrt(dim), (hidden, dim)),
        np.zeros(hidden),
        rng.normal(0, 1 / np.sqrt(hidden), (len(LABELS), hidden)),
        np.zeros(len(LABELS)),
    )


@dataclass
class Adam:
    """よく使われる最適化の方法（Adam）. 重みの名前ごとに、勾配の平均と 2 乗の平均を持つ."""

    lr: float
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    t: int = 0

    def __post_init__(self) -> None:
        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self.t += 1
        out = {}
        for name, g in grads.items():
            m = self.beta1 * self.m.get(name, np.zeros_like(g)) + (1 - self.beta1) * g
            v = self.beta2 * self.v.get(name, np.zeros_like(g)) + (1 - self.beta2) * g * g
            self.m[name], self.v[name] = m, v
            m_hat, v_hat = m / (1 - self.beta1**self.t), v / (1 - self.beta2**self.t)
            out[name] = params[name] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return out


def cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12)))


def backward(model: TinyModel, x: np.ndarray, y: np.ndarray, w1: np.ndarray) -> dict[str, np.ndarray]:
    """交差エントロピーの勾配（誤差逆伝播. 05-1）. w1 は実際に使う中間の層の重み."""
    h, probs = model.forward(x, w1)
    d_logits = probs.copy()
    d_logits[np.arange(len(y)), y] -= 1.0
    d_logits /= len(y)
    d_z1 = (d_logits @ model.w2) * (1 - h**2)
    return {"w1": d_z1.T @ x, "b1": d_z1.sum(axis=0), "w2": d_logits.T @ h, "b2": d_logits.sum(axis=0)}


def train_full(model: TinyModel, x: np.ndarray, y: np.ndarray, epochs: int = 300, lr: float = 0.01) -> TinyModel:
    """すべての重みを学習する（事前学習の代わりと、比べるためのフルのファインチューニング）."""
    opt = Adam(lr)
    params = {"w1": model.w1, "b1": model.b1, "w2": model.w2, "b2": model.b2}
    for _ in range(epochs):
        grads = backward(TinyModel(**params), x, y, params["w1"])
        params = opt.step(params, grads)
    return TinyModel(**params)


@dataclass(frozen=True)
class LoRA:
    """中間の層の重み w1 に足す差分を、B @ A の形（ランク r）で持つ."""

    a: np.ndarray  # (r, DIM)
    b: np.ndarray  # (HIDDEN, r)
    alpha: float

    @property
    def r(self) -> int:
        return self.a.shape[0]

    @property
    def n_params(self) -> int:
        return self.a.size + self.b.size

    def delta(self) -> np.ndarray:
        """w1 に足す差分（alpha / r 倍する）."""
        return (self.alpha / self.r) * self.b @ self.a


def init_lora(model: TinyModel, r: int = 4, alpha: float = 8.0, seed: int = 0) -> LoRA:
    """A は小さな乱数、B は 0 で始める（学習の前は差分が 0 なので、元のモデルと同じ答えになる）."""
    rng = np.random.default_rng(seed)
    hidden, dim = model.w1.shape
    return LoRA(rng.normal(0, 1 / np.sqrt(dim), (r, dim)), np.zeros((hidden, r)), alpha)


def train_lora(model: TinyModel, lora: LoRA, x: np.ndarray, y: np.ndarray, epochs: int = 300, lr: float = 0.01) -> LoRA:
    """A と B だけを学習する. 元のモデル（model）の重みは変えない."""
    opt = Adam(lr)
    params = {"a": lora.a, "b": lora.b}
    scale = lora.alpha / lora.r
    for _ in range(epochs):
        current = replace(lora, a=params["a"], b=params["b"])
        g_w1 = backward(model, x, y, model.w1 + current.delta())["w1"]
        # w1 + scale × B @ A の、A と B についての勾配（行列の積の微分）
        grads = {"a": scale * params["b"].T @ g_w1, "b": scale * g_w1 @ params["a"].T}
        params = opt.step(params, grads)
    return replace(lora, a=params["a"], b=params["b"])


def merge(model: TinyModel, lora: LoRA) -> TinyModel:
    """差分を w1 に足し込んだ、ふつうのモデルにする（大きさも速さも元と同じ）."""
    return replace(model, w1=model.w1 + lora.delta())


def predictor(model: TinyModel, lora: LoRA | None = None) -> Predict:
    w1 = model.w1 if lora is None else model.w1 + lora.delta()

    def predict(case: EvalCase) -> Prediction:
        _, probs = model.forward(features(case.input)[None, :], w1)
        return Prediction({"category": LABELS[int(probs[0].argmax())]})

    return predict


def evaluate(model: TinyModel, cases: Sequence[EvalCase], lora: LoRA | None = None, name: str = "") -> Report:
    return run_eval(cases, predictor(model, lora), exact_match("category"), meta={"model": name})


@dataclass(frozen=True)
class Run:
    base: TinyModel
    lora: LoRA
    full: TinyModel
    reports: dict[str, Report]
    seconds: dict[str, float]


def run_all(
    base_cases: Sequence[EvalCase],
    sft_cases: Sequence[EvalCase],
    eval_cases: Sequence[EvalCase],
    r: int = 4,
    alpha: float = 8.0,
    epochs: int = 300,
    seed: int = 0,
    val_cases: Sequence[EvalCase] = (),
) -> Run:
    """事前学習の代わり → LoRA の調整 → 比べるためのフルの調整、を順に行い、検証用と評価セットで測る."""
    seconds = {}
    started = time.perf_counter()
    base = train_full(init_model(seed), *encode(base_cases), epochs=epochs)
    seconds["base"] = time.perf_counter() - started
    started = time.perf_counter()
    lora = train_lora(base, init_lora(base, r, alpha, seed), *encode(sft_cases), epochs=epochs)
    seconds["lora"] = time.perf_counter() - started
    started = time.perf_counter()
    full = train_full(base, *encode(sft_cases), epochs=epochs)
    seconds["full"] = time.perf_counter() - started
    reports = {
        "base": evaluate(base, eval_cases, name="base"),
        "lora": evaluate(base, eval_cases, lora, name=f"lora-r{r}"),
        "full": evaluate(full, eval_cases, name="full"),
    }
    if val_cases:
        reports |= {
            "base_val": evaluate(base, val_cases, name="base"),
            "lora_val": evaluate(base, val_cases, lora, name=f"lora-r{r}"),
            "full_val": evaluate(full, val_cases, name="full"),
        }
    return Run(base, lora, full, reports, seconds)


def choose_rank(
    base_cases: Sequence[EvalCase],
    sft_cases: Sequence[EvalCase],
    val_cases: Sequence[EvalCase],
    ranks: Sequence[int],
    alpha: float = 8.0,
    epochs: int = 300,
    seed: int = 0,
) -> tuple[int, dict[int, float]]:
    """検証用の例の正解率で r を選ぶ（同じ点なら小さい r）. 07 の評価セットは使わない."""
    base = train_full(init_model(seed), *encode(base_cases), epochs=epochs)
    x, y = encode(sft_cases)
    scores = {
        r: evaluate(base, val_cases, train_lora(base, init_lora(base, r, alpha, seed), x, y, epochs)).mean
        for r in ranks
    }
    best = max(sorted(scores), key=lambda r: (scores[r], -r))
    return best, scores


def score_text(report: Report) -> str:
    correct = sum(r.score for r in report.results)
    return f"{report.mean:.1%}（{correct:.0f} / {len(report.results)}）"


def summary(run: Run) -> str:
    trained = {"base": run.base.n_params, "lora": run.lora.n_params, "full": run.full.n_params}
    names = {"base": "調整の前（汎用の例だけで学習）", "lora": f"LoRA（r={run.lora.r}）", "full": "フルの調整"}
    has_val = "lora_val" in run.reports
    lines = ["| モデル | 検証用 | 評価セット（07） | 学習した重みの数 | 学習の秒数 |", "|---|---|---|---|---|"]
    for key in ("base", "lora", "full"):
        val = score_text(run.reports[f"{key}_val"]) if has_val else "—"
        lines.append(
            f"| {names[key]} | {val} | {score_text(run.reports[key])} | {trained[key]:,} | {run.seconds[key]:.2f} |"
        )
    lines.append("")
    lines.append(
        "タグごとの正解率（LoRA・評価セット）: "
        + ", ".join(f"{t} {s:.0%}" for t, s in run.reports["lora"].by_tag().items())
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="p09_lora", description="小さなモデルを LoRA で調整し、07 の評価セットで前後を比べる"
    )
    p.add_argument("--rank", type=int, default=4, help="LoRA のランク r（--ranks を渡さないとき）")
    p.add_argument("--ranks", help="r の候補（例: 1,2,4,8）. 検証用の例の正解率で選び、選んだ r だけを評価セットで測る")
    p.add_argument("--alpha", type=float, default=8.0, help="LoRA の alpha（差分は alpha / r 倍する）")
    p.add_argument("--epochs", type=int, default=300, help="学習の回数")
    p.add_argument("--seed", type=int, default=0, help="乱数の種（同じ種なら毎回同じ結果）")
    p.add_argument("--val", type=Path, default=SFT_VAL, help="設定を選ぶための検証用の例（07 の形式）")
    p.add_argument("--eval", type=Path, default=DEFAULT_EVAL, help="評価セット（07 の形式. 最後に 1 回だけ使う）")
    p.add_argument("--out", type=Path, help="LoRA の結果を書く JSONL（07 の --compare で比べられる）")
    args = p.parse_args(argv)
    try:
        args.ranks = [int(r) for r in args.ranks.split(",")] if args.ranks else []
    except ValueError:
        p.error("--ranks は 1,2,4 のように、数をカンマで区切って書いてください")
    if args.rank < 1 or args.epochs < 1 or any(r < 1 for r in args.ranks):
        p.error("r と --epochs は 1 以上にしてください")
    return args


def main(argv: Sequence[str] | None = None, stdout: TextIO | None = None) -> int:
    args = parse_args(argv)
    stdout = stdout or sys.stdout
    base_cases, sft_cases, val_cases = load_cases(BASE_TRAIN), load_cases(SFT_TRAIN), load_cases(args.val)
    rank = args.rank
    if args.ranks:
        rank, scores = choose_rank(base_cases, sft_cases, val_cases, args.ranks, args.alpha, args.epochs, args.seed)
        print(
            "検証用の例での正解率（r を選ぶ）: " + ", ".join(f"r={r} {s:.0%}" for r, s in scores.items()), file=stdout
        )
        print(f"選んだ r: {rank}（評価セットは、選んだ r だけで 1 回測る）", file=stdout)
        print("", file=stdout)
    run = run_all(base_cases, sft_cases, load_cases(args.eval), rank, args.alpha, args.epochs, args.seed, val_cases)
    print(summary(run), file=stdout)
    if args.out:
        save_results(run.reports["lora"], args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
