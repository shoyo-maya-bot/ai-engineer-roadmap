"""06-1 トークンとエンベディング: 文字がモデルに渡るまで.

- `train_bpe` と `tokenize`: 小さな BPE（Byte Pair Encoding の考え方）. よく並ぶ 2 つを 1 つのトークンにまとめる操作を
  決めた回数だけ繰り返す. 本物のトークナイザーはモデルごとに学習済みで、語彙も数万あるが、考え方は同じ
- `cosine`: 2 つのベクトルの近さ（コサイン類似度）. エンベディングを比べるときに使う

本物のトークン数は、モデルの応答に書かれた数で数える（Ollama なら prompt_eval_count と eval_count. 06-4）.
ここでのトークナイザーは、仕組みを確かめるための教材用のもの.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

Pair = tuple[str, str]


def merge(tokens: Sequence[str], pair: Pair) -> list[str]:
    """並んだ 2 つが pair に一致するところを、1 つのトークンにまとめる（左から順に）."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
            out.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def train_bpe(corpus: Sequence[str], num_merges: int) -> list[Pair]:
    """1 文字ずつから始め、いちばん多く並ぶ組を 1 つにまとめる、を num_merges 回まで繰り返す.

    2 回以上出てくる組が無くなったら、そこで止める. 同じ回数の組は、文字の順で先のものを選ぶ（毎回同じ結果にする）.
    """
    texts = [list(text) for text in corpus]
    merges: list[Pair] = []
    for _ in range(num_merges):
        counts: Counter[Pair] = Counter()
        for tokens in texts:
            counts.update(zip(tokens, tokens[1:], strict=False))
        if not counts:
            break
        best, n = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if n < 2:
            break
        merges.append(best)
        texts = [merge(tokens, best) for tokens in texts]
    return merges


def tokenize(text: str, merges: Sequence[Pair]) -> list[str]:
    """学習したまとめ方を、学習した順に当てはめる. 学習で見なかった並びは、1 文字ずつのまま残る."""
    tokens = list(text)
    for pair in merges:
        tokens = merge(tokens, pair)
    return tokens


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """コサイン類似度（-1〜1. 向きが同じほど 1 に近い）. 長さ 0 のベクトルは 0 とする."""
    if len(a) != len(b):
        raise ValueError(f"ベクトルの長さが違います（{len(a)} と {len(b)}）")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
