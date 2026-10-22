"""07-2 構造化出力とスキーマ検証: LLM の出力を JSON で受け、形を確かめ、崩れていたら直させる.

- `validate`: JSON スキーマのうち、よく使う一部（type・enum・required・properties・additionalProperties・items・
  minLength・maxLength・minimum・maximum）だけを確かめる小さな実装（依存を増やさないため. 本格的に使うなら
  jsonschema や Pydantic などのライブラリを使う）
- `ask_json`: 出力を JSON として読み、スキーマで確かめる. 崩れていたら、誤りを伝えて決まった回数まで頼み直す
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from roadmap_examples.llm import LLMClient, Message
from roadmap_examples.p07_prompt import CATEGORIES

# 07 の分類の出力の形. Ollama には format にこのまま渡せる（07-2 の本文）
CLASSIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "reason": {"type": "string", "maxLength": 80},
    },
    "required": ["category", "reason"],
    "additionalProperties": False,
}

_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
}


def _is_type(value: Any, name: str) -> bool:
    if name == "null":
        return value is None
    if name in ("integer", "number") and isinstance(value, bool):
        return False  # Python では True も int の仲間だが、JSON では別の型
    return isinstance(value, _TYPES[name])


def validate(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """スキーマに合わない点の一覧（空なら合っている）. 場所は $.category のように書く."""
    errors: list[str] = []
    kind = schema.get("type")
    if kind and not _is_type(value, kind):
        return [f"{path}: 型が {kind} ではありません"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} は選べる値（{', '.join(map(str, schema['enum']))}）にありません")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path}: {schema['minLength']} 文字以上にしてください")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: {schema['maxLength']} 文字以下にしてください")
    if isinstance(value, int | float) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {schema['minimum']} 以上にしてください")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {schema['maximum']} 以下にしてください")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        errors += [f"{path}: 「{key}」がありません" for key in schema.get("required", []) if key not in value]
        if schema.get("additionalProperties") is False:
            errors += [f"{path}: 「{key}」は決めていない項目です" for key in value if key not in props]
        for key, sub in props.items():
            if key in value:
                errors += validate(value[key], sub, f"{path}.{key}")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors += validate(item, schema["items"], f"{path}[{i}]")
    return errors


def parse_json(text: str) -> Any:
    """出力を JSON として読む. ```json で囲まれていたら外す. 読めなければ json.JSONDecodeError."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        s = s.rsplit("```", 1)[0]
    return json.loads(s)


@dataclass(frozen=True)
class StructuredResult:
    value: dict[str, Any]
    attempts: int
    input_tokens: int
    output_tokens: int


class StructuredOutputError(RuntimeError):
    """決まった回数まで頼み直しても、スキーマに合う JSON が返らなかった."""

    def __init__(self, message: str, last_text: str) -> None:
        super().__init__(message)
        self.last_text = last_text


def ask_json(
    llm: LLMClient, messages: Sequence[Message], schema: dict[str, Any], max_attempts: int = 2
) -> StructuredResult:
    """JSON で答えさせ、スキーマで確かめる. 合わなければ誤りを伝えて頼み直す（合計 max_attempts 回まで）."""
    history = list(messages)
    tokens_in = tokens_out = 0
    text = ""
    for attempt in range(1, max_attempts + 1):
        result = llm.complete(history)
        tokens_in += result.input_tokens
        tokens_out += result.output_tokens
        text = result.text
        try:
            value = parse_json(text)
            problems = validate(value, schema)
        except json.JSONDecodeError as e:
            problems = [f"JSON として読めません（{e.msg}）"]
        if not problems:
            return StructuredResult(value, attempt, tokens_in, tokens_out)
        history += [
            Message("assistant", text),
            Message(
                "user", "出力が決めた形に合いません。直して JSON だけを出力してください:\n- " + "\n- ".join(problems)
            ),
        ]
    raise StructuredOutputError(f"{max_attempts} 回試しても、形に合う JSON が返りませんでした", text)
