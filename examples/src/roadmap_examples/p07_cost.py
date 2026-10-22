"""07-4 費用の最適化.

同じ入力をもう一度払わない（キャッシュ）、安いモデルを先に使う（使い分け）、上限で止める.

- `Price` と `cost`: トークン数から費用を見積もる. 単価は変わるので、コードに固定せず、
  使う API の公式の料金ページで確かめて渡す（ローカルのモデルは 0）
- `BudgetedLLM`: 見積もった費用の合計が上限に達したら、呼ばずに止める
  （`LimitedLLM` の回数・トークンの上限と組み合わせる）
- `CachedLLM`: まったく同じメッセージには、前の応答を返す（呼ばないので費用もかからない）
- `cascade`: 安いモデルで先に試し、決めた条件のときだけ高いモデルに回す
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from roadmap_examples.evalset import Prediction
from roadmap_examples.llm import Completion, LLMClient, Message


@dataclass(frozen=True)
class Price:
    """100 万トークンあたりの単価（入力・出力）. 通貨は使う API に合わせる."""

    input_per_million: float
    output_per_million: float


FREE = Price(0.0, 0.0)  # ローカルのモデル（電気代などは別）


def cost(input_tokens: int, output_tokens: int, price: Price) -> float:
    return (input_tokens * price.input_per_million + output_tokens * price.output_per_million) / 1_000_000


class BudgetExceeded(RuntimeError):
    """見積もった費用の合計が上限に達した."""


@dataclass
class BudgetedLLM:
    """費用（見積もり）の合計に上限を設ける. 上限に達したら、次の呼び出しを止める.

    1 回の呼び出しの費用は、呼んだ後でないと分からない. そのため、合計は上限を 1 回分だけ超えることがある.
    超えて困る場合は、上限を「1 回の最大の費用」だけ低くしておく.
    """

    inner: LLMClient
    price: Price
    max_cost: float
    spent: float = 0.0

    def complete(self, messages: Sequence[Message]) -> Completion:
        if self.spent >= self.max_cost:
            raise BudgetExceeded(f"費用の見積もりが上限（{self.max_cost}）に達しました（{self.spent:.4f}）")
        result = self.inner.complete(messages)
        self.spent += cost(result.input_tokens, result.output_tokens, self.price)
        return result


def cache_key(messages: Sequence[Message]) -> str:
    """メッセージの中身から作る鍵（1 文字でも違えば別の鍵）."""
    raw = json.dumps([[m.role, m.content] for m in messages], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class CachedLLM:
    """同じメッセージには前の応答を返す. キャッシュから返したときのトークン数は 0（呼んでいないので）."""

    inner: LLMClient
    store: dict[str, str] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def complete(self, messages: Sequence[Message]) -> Completion:
        key = cache_key(messages)
        if key in self.store:
            self.hits += 1
            return Completion(self.store[key], 0, 0)
        self.misses += 1
        result = self.inner.complete(messages)
        self.store[key] = result.text
        return result


def cascade(
    cheap: Callable[[str], Prediction],
    strong: Callable[[str], Prediction],
    escalate: Callable[[Prediction], bool],
) -> Callable[[str], Prediction]:
    """安いモデル（cheap）で先に試し、失敗したときか escalate が真のときだけ、高いモデル（strong）に回す."""

    def predict(text: str) -> Prediction:
        try:
            first = cheap(text)
        except Exception:  # noqa: BLE001 — 安いモデルの失敗（形の崩れなど）は、高いモデルに回す理由になる
            return strong(text)
        if not escalate(first):
            return first
        second = strong(text)
        return Prediction(
            second.output, first.input_tokens + second.input_tokens, first.output_tokens + second.output_tokens
        )

    return predict
