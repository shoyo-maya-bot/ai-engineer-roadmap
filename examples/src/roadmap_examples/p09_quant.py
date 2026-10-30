"""09-3 量子化と GPU のメモリ: 重みを少ないビット数で持つと、どれだけ小さくなり、どれだけずれるかを確かめる.

- `quantize` / `dequantize`: 対称な量子化（いちばん大きい絶対値を、使える整数の端に合わせる）.
  group_size を渡すと、重みを小さな塊（グループ）に分け、塊ごとに倍率（scale）を持つ. 外れ値の影響が塊の中だけで済む
- `quantize_model`: 09-2 の小さなモデルの重みを量子化して、07 の評価セットの正解率がどう変わるかを測る
- `training_gb` / `kv_cache_gb`: 学習と推論に要るメモリの見積もり（重み・勾配・最適化の状態・KV キャッシュ）

使い方（examples/ で. LLM もネットワークも使わない）:
  uv run python -m roadmap_examples.p09_quant
  uv run python -m roadmap_examples.p09_quant --params 8 --layers 32 --hidden 4096 --rank 16
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TextIO

import numpy as np

from roadmap_examples.evalset import EvalCase, load_cases
from roadmap_examples.p06_compare import weights_gb
from roadmap_examples.p07_classify import DEFAULT_EVAL
from roadmap_examples.p09_lora import BASE_TRAIN, SFT_TRAIN, TinyModel, evaluate, merge, run_all


@dataclass(frozen=True)
class Quantized:
    q: np.ndarray  # 整数（-qmax〜qmax）
    scale: np.ndarray  # 塊ごとの倍率（元の値 ≒ q × scale）
    shape: tuple[int, ...]
    bits: int
    group_size: int | None

    @property
    def n_bytes(self) -> float:
        """整数の分と、倍率（16 ビットで持つとする）の分の合計."""
        return self.q.size * self.bits / 8 + self.scale.size * 2


def quantize(w: np.ndarray, bits: int = 8, group_size: int | None = None) -> Quantized:
    """対称な量子化. group_size が None なら行列全体で 1 つの倍率、渡すと塊ごとに倍率を持つ."""
    if not 2 <= bits <= 8:
        raise ValueError("bits は 2〜8 にしてください")
    qmax = 2 ** (bits - 1) - 1
    flat = w.reshape(-1)
    if group_size is None:
        groups = flat[None, :]
    else:
        if flat.size % group_size:
            raise ValueError(f"重みの数（{flat.size}）が group_size（{group_size}）で割り切れません")
        groups = flat.reshape(-1, group_size)
    absmax = np.abs(groups).max(axis=1, keepdims=True)
    scale = np.where(absmax > 0, absmax / qmax, 1.0)
    q = np.clip(np.round(groups / scale), -qmax, qmax).astype(np.int8)
    return Quantized(q, scale, w.shape, bits, group_size)


def dequantize(qz: Quantized) -> np.ndarray:
    return (qz.q * qz.scale).reshape(qz.shape)


def mean_abs_error(w: np.ndarray, bits: int, group_size: int | None = None) -> float:
    return float(np.mean(np.abs(w - dequantize(quantize(w, bits, group_size)))))


def quantize_model(model: TinyModel, bits: int, group_size: int | None = None) -> TinyModel:
    """重み（w1・w2）を量子化してから戻したモデル（量子化で生じたずれだけを確かめる. 計算は元の精度のまま）."""
    return replace(
        model,
        w1=dequantize(quantize(model.w1, bits, group_size)),
        w2=dequantize(quantize(model.w2, bits, group_size)),
    )


# 学習で 1 つの重みあたりに要るバイト数（16 ビットと 32 ビットを混ぜる学習 + AdamW）.
# 重み 6（16 ビットの写し 2 + 32 ビットの元 4）+ 最適化の状態 8 + 勾配 4. 活性化（途中の計算の値）は含まない
# https://huggingface.co/docs/transformers/model_memory_anatomy
TRAIN_BYTES_PER_PARAM = 18


def lora_params(layers: int, hidden: int, rank: int, targets: int = 4) -> int:
    """各層の hidden × hidden の行列 targets 個に LoRA を付けたときの、学習する重みの数（r × (入力 + 出力) ずつ）."""
    return layers * targets * rank * (hidden + hidden)


def training_gb(params_billion: float, method: str, trainable: int = 0) -> float:
    """学習に要るメモリの目安（GB. 活性化を除く）. trainable は LoRA で学習する重みの数.

    full:  全部の重みを学習する（1 つあたり 18 バイト）
    lora:  元の重みは 16 ビットで固定（2 バイト）+ LoRA の重みだけ 18 バイト
    qlora: 元の重みは 4 ビットで固定（0.5 バイト）+ LoRA の重みだけ 18 バイト
    """
    frozen_bits = {"full": None, "lora": 16, "qlora": 4}
    if method not in frozen_bits:
        raise ValueError(f"method は {', '.join(frozen_bits)} のどれかにしてください")
    if method == "full":
        return params_billion * TRAIN_BYTES_PER_PARAM
    return weights_gb(params_billion, frozen_bits[method]) + trainable * TRAIN_BYTES_PER_PARAM / 1e9


def kv_cache_gb(layers: int, kv_heads: int, head_dim: int, context: int, bits: int = 16, batch: int = 1) -> float:
    """推論の KV キャッシュの目安（GB）. 層ごと・トークンごとに、キー（K）とバリュー（V）を 1 組ずつ持つ."""
    return 2 * layers * kv_heads * head_dim * context * batch * bits / 8 / 1e9


QUANT_ROWS: list[tuple[str, int | None, int | None]] = [
    ("64 ビット（元のまま）", None, None),
    ("8 ビット", 8, None),
    ("4 ビット・32 個ずつの塊", 4, 32),
    ("4 ビット・全体で 1 つの倍率", 4, None),
    ("2 ビット・32 個ずつの塊", 2, 32),
    ("2 ビット・全体で 1 つの倍率", 2, None),
]


def accuracy_table(model: TinyModel, cases: Sequence[EvalCase]) -> str:
    lines = ["| 重みの持ち方 | w1 の大きさ（バイト） | w1 の平均のずれ | 正解率 |", "|---|---|---|---|"]
    for name, bits, group in QUANT_ROWS:
        if bits is None:
            size, error, m = model.w1.nbytes, 0.0, model
        else:
            size, error = quantize(model.w1, bits, group).n_bytes, mean_abs_error(model.w1, bits, group)
            m = quantize_model(model, bits, group)
        lines.append(f"| {name} | {size:,.0f} | {error:.4f} | {evaluate(m, cases).mean:.1%} |")
    return "\n".join(lines)


def memory_table(params_billion: float, trainable: int) -> str:
    lines = ["| やり方 | 学習のメモリの目安（GB. 活性化を除く） |", "|---|---|"]
    for name, method in (("フルの調整", "full"), ("LoRA", "lora"), ("QLoRA", "qlora")):
        lines.append(f"| {name} | {training_gb(params_billion, method, trainable):.1f} |")
    lines.append(f"| （参考）推論の重みだけ・4 ビット | {weights_gb(params_billion, 4):.1f} |")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="p09_quant", description="量子化のずれと、学習のメモリの見積もり")
    p.add_argument("--eval", type=Path, default=DEFAULT_EVAL, help="評価セット（07 の形式）")
    p.add_argument("--params", type=float, default=8.0, help="見積もるモデルのパラメータ数（B. 10 億単位）")
    p.add_argument("--layers", type=int, default=32, help="見積もるモデルの層の数")
    p.add_argument("--hidden", type=int, default=4096, help="見積もるモデルの隠れ層の大きさ")
    p.add_argument("--rank", type=int, default=16, help="LoRA のランク")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None, stdout: TextIO | None = None) -> int:
    args = parse_args(argv)
    stdout = stdout or sys.stdout
    cases = load_cases(args.eval)
    run = run_all(load_cases(BASE_TRAIN), load_cases(SFT_TRAIN), cases)
    print("09-2 の LoRA を足し込んだモデルを量子化したときの正解率:", file=stdout)
    print(accuracy_table(merge(run.base, run.lora), cases), file=stdout)
    trainable = lora_params(args.layers, args.hidden, args.rank)
    print("", file=stdout)
    print(f"{args.params:g}B のモデル（{args.layers} 層・隠れ層 {args.hidden}・LoRA r={args.rank}）:", file=stdout)
    print(f"LoRA で学習する重み: {trainable:,} 個（全体の {trainable / (args.params * 1e9):.2%}）", file=stdout)
    print(memory_table(args.params, trainable), file=stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
