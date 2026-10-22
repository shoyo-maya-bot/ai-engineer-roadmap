"""LLM を呼ぶ部分のインターフェースと、テスト用の偽の実装.

演習のコードは、LLM を直接呼ばずに `LLMClient` を受け取る. 本物（ローカルのモデルなど）と
`FakeLLM`（決まった応答を返す）を差し替えられるので、テストと CI は LLM を呼ばずに、毎回同じ結果で確かめられる.
`LimitedLLM` は、呼び出しの回数とトークン数に上限を設ける（上限を超えたら止める. 費用とループの暴走の対策）.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int


class LLMClient(Protocol):
    """LLM の呼び出し. 実装は本物のモデルでも偽物でもよい."""

    def complete(self, messages: Sequence[Message]) -> Completion: ...


def rough_tokens(text: str) -> int:
    """トークン数のおおよその見積もり（テスト用. 本物の数え方はモデルごとに違う. 06 で扱う）."""
    return max(1, len(text) // 4) if text else 0


@dataclass
class FakeLLM:
    """決まった応答を順に返す偽の実装. 受け取ったメッセージを `calls` に残す."""

    replies: list[str]
    calls: list[list[Message]] = field(default_factory=list)

    @classmethod
    def of(cls, replies: Iterable[str]) -> FakeLLM:
        return cls(list(replies))

    def complete(self, messages: Sequence[Message]) -> Completion:
        if len(self.calls) >= len(self.replies):
            raise AssertionError(f"FakeLLM の応答が足りません（用意 {len(self.replies)} 件）")
        self.calls.append(list(messages))
        text = self.replies[len(self.calls) - 1]
        prompt = "\n".join(m.content for m in messages)
        return Completion(text, rough_tokens(prompt), rough_tokens(text))


class LimitExceeded(RuntimeError):
    """呼び出しの回数やトークン数が上限を超えた."""


@dataclass
class LimitedLLM:
    """呼び出しの回数とトークン数（入力 + 出力）に上限を設ける. 上限に達したら呼ばずに止める."""

    inner: LLMClient
    max_calls: int
    max_tokens: int
    calls: int = 0
    tokens: int = 0

    def complete(self, messages: Sequence[Message]) -> Completion:
        if self.calls >= self.max_calls:
            raise LimitExceeded(f"呼び出しの回数が上限（{self.max_calls} 回）に達しました")
        if self.tokens >= self.max_tokens:
            raise LimitExceeded(f"トークン数が上限（{self.max_tokens}）に達しました")
        result = self.inner.complete(messages)
        self.calls += 1
        self.tokens += result.input_tokens + result.output_tokens
        return result
