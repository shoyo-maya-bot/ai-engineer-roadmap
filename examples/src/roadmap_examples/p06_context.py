"""06-2 LLM の仕組みとコンテキストウィンドウ.

- `softmax` と `sample`: LLM は「次のトークンの確率」を出し、そこから 1 つ選ぶ、を繰り返して文章を作る.
  temperature は、確率の偏りを強めたり弱めたりする設定（0 なら、いつもいちばん確率の高いトークン）
- `fit_history`: コンテキストウィンドウ（1 回に読めるトークン数の上限）に収まるように、古い会話から削る.
  出力のトークンも同じ上限に入るので、出力の分（reserve_output）を先に空けておく
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence

from roadmap_examples.llm import Message, rough_tokens


def softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    """点数（logits）を確率にする. temperature が小さいほど偏り、0 以下なら最大の 1 つだけが 1 になる."""
    if not logits:
        raise ValueError("点数が空です")
    if temperature <= 0:
        best = max(range(len(logits)), key=lambda i: logits[i])
        return [1.0 if i == best else 0.0 for i in range(len(logits))]
    scaled = [x / temperature for x in logits]
    top = max(scaled)
    exps = [math.exp(x - top) for x in scaled]  # 最大を引いてから exp（大きな数で溢れないように）
    total = sum(exps)
    return [e / total for e in exps]


def sample(probs: Sequence[float], rng: random.Random) -> int:
    """確率に従って 1 つ選び、その番号を返す（rng を固定すれば毎回同じ結果になる）."""
    return rng.choices(range(len(probs)), weights=probs, k=1)[0]


class ContextOverflow(ValueError):
    """いちばん新しいメッセージだけでも、コンテキストに入らない."""


def fit_history(
    messages: Sequence[Message],
    max_context: int,
    reserve_output: int,
    count: Callable[[str], int] = rough_tokens,
) -> list[Message]:
    """system と、新しい順に入るだけのメッセージを残す（古いものから削る）.

    入力に使えるのは max_context - reserve_output トークン. 最後のメッセージ（いまの質問）が入らなければ止める.
    """
    budget = max_context - reserve_output
    system = [m for m in messages if m.role == "system"]
    rest = [m for m in messages if m.role != "system"]
    used = sum(count(m.content) for m in system)
    kept: list[Message] = []
    for m in reversed(rest):
        size = count(m.content)
        if used + size > budget:
            break
        kept.append(m)
        used += size
    if rest and not kept:
        raise ContextOverflow(f"最後のメッセージが、入力に使える {budget} トークンに入りません")
    return system + kept[::-1]
