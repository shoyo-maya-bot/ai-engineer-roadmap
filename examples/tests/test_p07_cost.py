"""p07_cost.py のテスト（07-4 費用の見積もり・上限・キャッシュ・使い分け）."""

import pytest

from roadmap_examples.evalset import Prediction
from roadmap_examples.llm import FakeLLM, Message
from roadmap_examples.p07_cost import FREE, BudgetedLLM, BudgetExceeded, CachedLLM, Price, cascade, cost

PRICE = Price(input_per_million=100.0, output_per_million=400.0)  # テスト用の架空の単価


def test_cost() -> None:
    assert cost(1_000_000, 0, PRICE) == 100.0
    assert cost(2_000, 500, PRICE) == pytest.approx(0.4)
    assert cost(2_000, 500, FREE) == 0.0


def test_budgeted_llm_stops_when_budget_is_used() -> None:
    fake = FakeLLM.of(["x" * 400] * 5)  # 1 回の出力はおよそ 100 トークン → 0.04
    llm = BudgetedLLM(fake, PRICE, max_cost=0.05)
    llm.complete([Message("user", "1")])
    llm.complete([Message("user", "2")])  # 上限に達していないので呼ぶ（合計は上限を 1 回分超える）
    with pytest.raises(BudgetExceeded, match="上限"):
        llm.complete([Message("user", "3")])
    assert len(fake.calls) == 2


def test_cached_llm_returns_previous_reply_without_calling() -> None:
    fake = FakeLLM.of(["billing"])
    llm = CachedLLM(fake)
    first = llm.complete([Message("user", "請求書")])
    second = llm.complete([Message("user", "請求書")])
    assert first.text == second.text == "billing"
    assert (second.input_tokens, second.output_tokens) == (0, 0)
    assert (llm.hits, llm.misses, len(fake.calls)) == (1, 1, 1)
    with pytest.raises(AssertionError):  # 1 文字でも違えば、キャッシュは使わない（呼びに行く）
        llm.complete([Message("user", "請求書。")])


def test_cascade_uses_strong_model_only_when_needed() -> None:
    strong_calls: list[str] = []

    def cheap(text: str) -> Prediction:
        if "壊れ" in text:
            raise ValueError("形が崩れた")
        return Prediction({"category": "other" if "?" in text else "billing"}, 10, 2)

    def strong(text: str) -> Prediction:
        strong_calls.append(text)
        return Prediction({"category": "bug"}, 100, 5)

    predict = cascade(cheap, strong, escalate=lambda p: p.output["category"] == "other")
    assert predict("請求書").output == {"category": "billing"}
    assert strong_calls == []
    escalated = predict("よく分からない?")
    assert escalated.output == {"category": "bug"}
    assert (escalated.input_tokens, escalated.output_tokens) == (110, 7)
    assert predict("壊れた出力").output == {"category": "bug"}
    assert len(strong_calls) == 2
