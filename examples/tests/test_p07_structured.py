"""p07_structured.py のテスト（07-2 構造化出力の検証と頼み直し）."""

import json

import pytest

from roadmap_examples.llm import FakeLLM, Message
from roadmap_examples.p07_structured import CLASSIFY_SCHEMA, StructuredOutputError, ask_json, parse_json, validate

GOOD = {"category": "billing", "reason": "請求書の再発行"}


def test_validate_accepts_good_value() -> None:
    assert validate(GOOD, CLASSIFY_SCHEMA) == []


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"category": "billing"}, "「reason」がありません"),
        ({"category": "refund", "reason": "x"}, "選べる値"),
        ({**GOOD, "score": 1}, "決めていない項目"),
        ({"category": 1, "reason": "x"}, "$.category: 型が string"),
        ({"category": "billing", "reason": "あ" * 81}, "80 文字以下"),
        (["billing"], "$: 型が object"),
    ],
)
def test_validate_reports_problems(value: object, message: str) -> None:
    assert any(message in p for p in validate(value, CLASSIFY_SCHEMA))


def test_validate_numbers_arrays_and_booleans() -> None:
    schema = {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": 5}}
    assert validate([1, 5], schema) == []
    assert validate([0, 6], schema) == ["$[0]: 1 以上にしてください", "$[1]: 5 以下にしてください"]
    assert validate([True], schema) == ["$[0]: 型が integer ではありません"]  # JSON では true は数ではない
    assert validate(None, {"type": "null"}) == []
    assert validate("", {"type": "string", "minLength": 1}) == ["$: 1 文字以上にしてください"]


def test_parse_json_strips_code_fence() -> None:
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json(' {"a": 1} ') == {"a": 1}


def test_ask_json_returns_value_on_first_try() -> None:
    fake = FakeLLM.of([json.dumps(GOOD, ensure_ascii=False)])
    result = ask_json(fake, [Message("user", "分類して")], CLASSIFY_SCHEMA)
    assert result.value == GOOD
    assert result.attempts == 1
    assert result.input_tokens > 0


def test_ask_json_retries_with_the_problems() -> None:
    fake = FakeLLM.of(["カテゴリは billing です", json.dumps(GOOD, ensure_ascii=False)])
    result = ask_json(fake, [Message("user", "分類して")], CLASSIFY_SCHEMA)
    assert result.attempts == 2
    retry = fake.calls[1]
    assert [m.role for m in retry] == ["user", "assistant", "user"]
    assert "JSON として読めません" in retry[-1].content


def test_ask_json_gives_up_after_max_attempts() -> None:
    fake = FakeLLM.of(['{"category": "refund", "reason": "x"}'] * 2)
    with pytest.raises(StructuredOutputError, match="2 回") as info:
        ask_json(fake, [Message("user", "分類して")], CLASSIFY_SCHEMA, max_attempts=2)
    assert "refund" in info.value.last_text
    assert len(fake.calls) == 2
