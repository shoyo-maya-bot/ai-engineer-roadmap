"""05 の成果物: CPU で学習できる小さな文字単位の言語モデル（トランスフォーマーの 1 ブロック. NumPy で手書き）.

次の 1 文字を予測するモデル. 直前の最大 `context` 文字を読み、次に来る文字の確率を出す.

  文字の番号 → エンベディング + 位置のエンベディング
            → 自己アテンション（未来を見ない. 05-3）→ 足し戻す（残差接続）
            → 全結合 → ReLU → 全結合 → 足し戻す
            → 全結合 → 次の文字の確率（softmax）

逆伝播（勾配の計算）も手で書き、数値微分と一致することをテストで確かめる. PyTorch で書く場合は 05-4 の本文の例.
本物のトランスフォーマーにある、層の正規化（LayerNorm）・複数のヘッド・ブロックの積み重ねは省いている.

使い方（examples/ で. 学習は手元の CPU で 10 秒ほど）:
  uv run python -m roadmap_examples.p05_charlm --steps 2000 --save charlm.npz
  uv run python -m roadmap_examples.p05_charlm --load charlm.npz --prompt 請求 --length 40

終了コード: 0 成功 / 1 プロンプトに学習していない文字がある・ファイルを読めない / 2 引数の誤り
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import numpy as np

from roadmap_examples.p05_loop import Adam
from roadmap_examples.p05_mlp import Params, softmax

P03_CLEAN = Path(__file__).resolve().parents[2] / "data" / "p03" / "inquiries_clean.csv"

# 架空の問い合わせの文を、部品の組み合わせで作る（03 の 130 件だけでは、すぐに丸暗記してしまうため）
TARGETS = ("請求書", "領収書", "パスワード", "メールアドレス", "プラン", "契約", "画面", "通知", "アプリ", "検索")
TARGETS += ("支払い方法", "二段階認証", "利用人数", "更新日", "ファイル", "会員ページ", "解約", "見積書")
PATTERNS = (
    "{t}を変更したいです。",
    "{t}の確認方法を教えてください。",
    "{t}について質問があります。",
    "{t}が表示されません。",
    "{t}の画面でエラーが出ます。",
    "{t}はどこから変えられますか。",
    "{t}の手続きはいつまでにすればよいですか。",
    "{t}の使い方が分かりません。",
)
OPENINGS = ("", "すみません、", "お世話になっております。", "急ぎではありませんが、", "昨日から")
CLOSINGS = ("", "よろしくお願いします。", "至急対応をお願いします。", "ご確認ください。")


def template_corpus(n_lines: int = 1500, seed: int = 0) -> str:
    """部品を乱数で組み合わせた架空の問い合わせを n_lines 行作り、改行でつなぐ（seed が同じなら毎回同じ）."""
    rng = np.random.default_rng(seed)
    lines = []
    for _ in range(n_lines):
        t, p = TARGETS[rng.integers(len(TARGETS))], PATTERNS[rng.integers(len(PATTERNS))]
        o, c = OPENINGS[rng.integers(len(OPENINGS))], CLOSINGS[rng.integers(len(CLOSINGS))]
        lines.append(o + p.format(t=t) + c)
    return "\n".join(lines) + "\n"


def file_corpus(path: Path = P03_CLEAN) -> str:
    """CSV（03 の成果物など）か JSONL（評価セットの形式）の input の文を、改行でつなぐ."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".csv":
        lines = [row["input"] for row in csv.DictReader(io.StringIO(text))]
    else:
        lines = [json.loads(line)["input"] for line in text.splitlines() if line.strip()]
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class Vocab:
    chars: tuple[str, ...]

    @classmethod
    def build(cls, text: str) -> Vocab:
        return cls(tuple(sorted(set(text))))

    def encode(self, text: str) -> np.ndarray:
        index = {c: i for i, c in enumerate(self.chars)}
        unknown = sorted({c for c in text if c not in index})
        if unknown:
            raise KeyError(f"学習していない文字です: {''.join(unknown)}")
        return np.array([index[c] for c in text], dtype=np.int64)

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.chars[i] for i in ids)


@dataclass(frozen=True)
class Config:
    context: int = 16  # 一度に読む文字数の上限（コンテキストの長さ）
    dim: int = 32  # エンベディングの次元
    hidden: int = 64  # 全結合の中間の次元
    batch: int = 16
    lr: float = 0.01
    seed: int = 0


def init_params(vocab_size: int, cfg: Config) -> Params:
    rng = np.random.default_rng(cfg.seed)

    def w(*shape: int) -> np.ndarray:
        return rng.normal(0.0, 1.0 / np.sqrt(shape[0]), shape)

    d, h = cfg.dim, cfg.hidden
    return {
        "tok": rng.normal(0.0, 0.1, (vocab_size, d)),  # 文字のエンベディング
        "pos": rng.normal(0.0, 0.1, (cfg.context, d)),  # 位置のエンベディング
        "wq": w(d, d),
        "wk": w(d, d),
        "wv": w(d, d),
        "wo": w(d, d) * 0.5,
        "w1": w(d, h),
        "b1": np.zeros(h),
        "w2": w(h, d) * 0.5,
        "b2": np.zeros(d),
        "out": w(d, vocab_size),
        "out_b": np.zeros(vocab_size),
    }


def forward(params: Params, idx: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """idx は文字の番号の（バッチ, 長さ）. 各位置の「次の文字」のロジット（バッチ, 長さ, 語彙の数）を返す."""
    t = idx.shape[1]
    x0 = params["tok"][idx] + params["pos"][:t]
    q, k, v = x0 @ params["wq"], x0 @ params["wk"], x0 @ params["wv"]
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(q.shape[-1])
    scores = np.where(np.tril(np.ones((t, t), dtype=bool)), scores, -np.inf)  # 未来の文字を見ない
    att = softmax(scores)
    mixed = att @ v
    x1 = x0 + mixed @ params["wo"]  # 残差接続: 入力に、アテンションの結果を足す
    z = x1 @ params["w1"] + params["b1"]
    r = np.maximum(z, 0.0)
    x2 = x1 + r @ params["w2"] + params["b2"]
    logits = x2 @ params["out"] + params["out_b"]
    cache = {"idx": idx, "x0": x0, "q": q, "k": k, "v": v, "att": att, "mixed": mixed, "x1": x1, "z": z, "r": r}
    cache["x2"] = x2
    return logits, cache


def loss_and_grads(params: Params, idx: np.ndarray, targets: np.ndarray) -> tuple[float, Params]:
    """交差エントロピー（全部の位置の平均）と勾配. 05-1 と同じく、出力の側から順に連鎖律で戻る."""
    logits, c = forward(params, idx)
    b, t, n_vocab = logits.shape
    probs = softmax(logits).reshape(b * t, n_vocab)
    flat = targets.reshape(-1)
    loss = float(-np.log(probs[np.arange(b * t), flat] + 1e-12).mean())

    def rows(a: np.ndarray) -> np.ndarray:  # （バッチ, 長さ, 次元）→（バッチ×長さ, 次元）
        return a.reshape(b * t, -1)

    g: Params = {}
    dlogits = probs.copy()
    dlogits[np.arange(b * t), flat] -= 1.0
    dlogits /= b * t
    g["out"], g["out_b"] = rows(c["x2"]).T @ dlogits, dlogits.sum(axis=0)
    dx2 = (dlogits @ params["out"].T).reshape(b, t, -1)
    # x2 = x1 + r W2 + b2
    g["w2"], g["b2"] = rows(c["r"]).T @ rows(dx2), rows(dx2).sum(axis=0)
    dz = (dx2 @ params["w2"].T) * (c["z"] > 0)  # ReLU の微分は、正なら 1、そうでなければ 0
    g["w1"], g["b1"] = rows(c["x1"]).T @ rows(dz), rows(dz).sum(axis=0)
    dx1 = dx2 + dz @ params["w1"].T
    # x1 = x0 + mixed Wo,  mixed = att v
    g["wo"] = rows(c["mixed"]).T @ rows(dx1)
    dmixed = dx1 @ params["wo"].T
    datt = dmixed @ c["v"].transpose(0, 2, 1)
    dv = c["att"].transpose(0, 2, 1) @ dmixed
    # softmax の微分: ds = att * (datt - Σ(datt * att)). 隠した位置は att = 0 なので勾配も 0
    dscores = c["att"] * (datt - (datt * c["att"]).sum(axis=-1, keepdims=True))
    dscores /= np.sqrt(c["q"].shape[-1])
    dq, dk = dscores @ c["k"], dscores.transpose(0, 2, 1) @ c["q"]
    x0 = rows(c["x0"])
    g["wq"], g["wk"], g["wv"] = x0.T @ rows(dq), x0.T @ rows(dk), x0.T @ rows(dv)
    dx0 = dx1 + dq @ params["wq"].T + dk @ params["wk"].T + dv @ params["wv"].T
    g["tok"] = np.zeros_like(params["tok"])
    np.add.at(g["tok"], c["idx"].reshape(-1), rows(dx0))  # 同じ文字が何度も出たら、勾配を足し合わせる
    g["pos"] = np.zeros_like(params["pos"])
    g["pos"][:t] = dx0.sum(axis=0)
    return loss, g


def sample_batch(data: np.ndarray, cfg: Config, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """文字列のランダムな位置から context 文字を取り出す. 正解は 1 文字ずらしたもの（次の文字）."""
    starts = rng.integers(0, len(data) - cfg.context - 1, cfg.batch)
    x = np.stack([data[s : s + cfg.context] for s in starts])
    y = np.stack([data[s + 1 : s + cfg.context + 1] for s in starts])
    return x, y


def mean_loss(params: Params, data: np.ndarray, cfg: Config) -> float:
    """data を context 文字ずつに区切って、損失の平均を出す（乱数を使わない. 検証用）."""
    n = (len(data) - 1) // cfg.context
    x = data[: n * cfg.context].reshape(n, cfg.context)
    y = data[1 : n * cfg.context + 1].reshape(n, cfg.context)
    return loss_and_grads(params, x, y)[0]


@dataclass
class TrainResult:
    params: Params
    vocab: Vocab
    cfg: Config
    log: list[tuple[int, float, float]] = field(default_factory=list)  # （ステップ, 学習の損失, 検証の損失）


def split_lines(text: str) -> tuple[str, str]:
    """同じ行を 1 つにまとめてから、10 行に 1 行を検証に回す.

    同じ行が学習と検証の両方に入ると、検証が「覚えた行」の試験になる. 末尾だけを検証にすると、行の並びの偏りが入る.
    """
    lines = list(dict.fromkeys(text.strip("\n").split("\n")))  # 出てきた順を保ったまま、重複を除く
    train = [line for i, line in enumerate(lines) if i % 10 != 9]
    val = [line for i, line in enumerate(lines) if i % 10 == 9]
    return "\n".join(train) + "\n", "\n".join(val) + "\n"


def train(text: str, steps: int = 2000, cfg: Config | None = None, eval_every: int = 500) -> TrainResult:
    """同じ行を除いて 9 割の行で学習し、残りの行で検証の損失を測る. 最適化は 05-2 の Adam."""
    cfg = cfg or Config()
    vocab = Vocab.build(text)
    train_text, val_text = split_lines(text)
    train_data, val_data = vocab.encode(train_text), vocab.encode(val_text)
    params = init_params(len(vocab.chars), cfg)
    optimizer = Adam(params, lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed)
    result = TrainResult(params, vocab, cfg)
    recent: list[float] = []
    for step in range(1, steps + 1):
        loss, grads = loss_and_grads(params, *sample_batch(train_data, cfg, rng))
        optimizer.step(grads)
        recent.append(loss)
        if step % eval_every == 0 or step == steps:
            result.log.append((step, float(np.mean(recent)), mean_loss(params, val_data, cfg)))
            recent = []
    return result


def generate(
    params: Params, vocab: Vocab, cfg: Config, prompt: str, length: int, temperature: float = 0.8, seed: int = 0
) -> str:
    """prompt に続けて、1 文字ずつ確率に従って選ぶ. temperature が小さいほど、確率の高い文字に偏る（06-2）."""
    rng = np.random.default_rng(seed)
    ids = list(vocab.encode(prompt))
    for _ in range(length):
        logits, _ = forward(params, np.array([ids[-cfg.context :]]))
        probs = softmax(logits[0, -1] / temperature)
        ids.append(int(rng.choice(len(probs), p=probs)))
    return vocab.decode(ids)


def save(path: Path, result: TrainResult) -> None:
    """パラメーターと語彙と設定を .npz に保存する（pickle を使わない形式）."""
    cfg = result.cfg
    np.savez(
        path,
        **result.params,
        vocab=np.array(result.vocab.chars),
        cfg=np.array([cfg.context, cfg.dim, cfg.hidden, cfg.batch, cfg.seed]),
    )


def load(path: Path) -> tuple[Params, Vocab, Config]:
    with np.load(path, allow_pickle=False) as f:
        context, dim, hidden, batch, seed = (int(v) for v in f["cfg"])
        params = {k: f[k] for k in f.files if k not in ("vocab", "cfg")}
        return params, Vocab(tuple(str(c) for c in f["vocab"])), Config(context, dim, hidden, batch, seed=seed)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="p05_charlm", description="小さな文字単位の言語モデルを学習し、文を作る")
    p.add_argument("--corpus", type=Path, help="学習に使う CSV か JSONL（input の文）. 省略すると部品を組み合わせた文")
    p.add_argument("--steps", type=int, default=2000, help="学習のステップ数")
    p.add_argument("--save", type=Path, help="学習したモデルを保存する .npz")
    p.add_argument("--load", type=Path, help="学習せずに、保存したモデルを読む")
    p.add_argument("--prompt", default="\n", help="書き出しの文字列（学習した文字だけ）")
    p.add_argument("--length", type=int, default=60, help="作る文字数")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None, stdout: TextIO | None = None) -> int:
    args = parse_args(argv)
    stdout = stdout or sys.stdout
    try:
        if args.load:
            params, vocab, cfg = load(args.load)
        else:
            text = file_corpus(args.corpus) if args.corpus else template_corpus()
            result = train(text, args.steps, Config(seed=args.seed))
            params, vocab, cfg = result.params, result.vocab, result.cfg
            train_text, val_text = split_lines(text)
            print(
                f"語彙: {len(vocab.chars)} 文字 / 学習: {len(train_text)} 文字・検証: {len(val_text)} 文字", file=stdout
            )
            print("（同じ行を 1 つにまとめてから、10 行に 1 行を検証に回す）", file=stdout)
            print(f"何も学んでいないときの損失（log 語彙数）: {np.log(len(vocab.chars)):.2f}", file=stdout)
            for step, train_loss, val_loss in result.log:
                print(f"ステップ {step}: 学習の損失 {train_loss:.2f} / 検証の損失 {val_loss:.2f}", file=stdout)
            if args.save:
                save(args.save, result)
        print(generate(params, vocab, cfg, args.prompt, args.length, args.temperature, args.seed), file=stdout)
    except (OSError, KeyError, ValueError) as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
