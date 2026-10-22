"""llm.py のテスト（偽の実装と上限）."""

import pytest

from roadmap_examples.llm import FakeLLM, LimitedLLM, LimitExceeded, LLMClient, Message, rough_tokens


def ask(llm: LLMClient, question: str) -> str:
    """LLMClient を受け取る業務コードの例（本物か偽物かを知らない）."""
    return llm.complete([Message("system", "短く答える"), Message("user", question)]).text


def test_fake_returns_replies_in_order_and_records_calls() -> None:
    fake = FakeLLM.of(["はい", "いいえ"])
    assert ask(fake, "1 つ目") == "はい"
    assert ask(fake, "2 つ目") == "いいえ"
    assert [call[-1].content for call in fake.calls] == ["1 つ目", "2 つ目"]


def test_fake_fails_when_replies_run_out() -> None:
    fake = FakeLLM.of(["はい"])
    ask(fake, "1 つ目")
    with pytest.raises(AssertionError, match="応答が足りません"):
        ask(fake, "2 つ目")


def test_limited_stops_at_max_calls_without_calling() -> None:
    fake = FakeLLM.of(["a", "b", "c"])
    limited = LimitedLLM(fake, max_calls=2, max_tokens=10_000)
    ask(limited, "1")
    ask(limited, "2")
    with pytest.raises(LimitExceeded, match="回数"):
        ask(limited, "3")
    assert len(fake.calls) == 2  # 上限を超えた呼び出しは中まで届かない


def test_limited_stops_at_max_tokens() -> None:
    limited = LimitedLLM(FakeLLM.of(["x" * 40, "y"]), max_calls=10, max_tokens=10)
    ask(limited, "長めの答えを返す")
    assert limited.tokens >= 10
    with pytest.raises(LimitExceeded, match="トークン"):
        ask(limited, "もう 1 回")


def test_rough_tokens() -> None:
    assert rough_tokens("") == 0
    assert rough_tokens("abc") == 1
    assert rough_tokens("a" * 40) == 10
