"""ollama_client.py のテスト（Ollama は呼ばない. 送る JSON の形と、応答の読み取りを確かめる）."""

from typing import Any

import pytest

from roadmap_examples.llm import Message
from roadmap_examples.ollama_client import OllamaChat, OllamaEmbedder, UrllibTransport


class FakeTransport:
    """送った内容を残し、決まった応答を返す."""

    def __init__(self, reply: dict[str, Any]) -> None:
        self.reply = reply
        self.sent: list[tuple[str, dict[str, Any]]] = []

    def post_json(self, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        self.sent.append((url, payload))
        return self.reply


def test_complete_sends_messages_and_reads_tokens() -> None:
    fake = FakeTransport(
        {"message": {"role": "assistant", "content": "はい"}, "prompt_eval_count": 12, "eval_count": 3}
    )
    llm = OllamaChat("example-model", transport=fake, options={"temperature": 0})
    result = llm.complete([Message("system", "短く"), Message("user", "質問")])
    assert (result.text, result.input_tokens, result.output_tokens) == ("はい", 12, 3)
    url, payload = fake.sent[0]
    assert url == "http://localhost:11434/api/chat"
    assert payload["model"] == "example-model"
    assert payload["stream"] is False
    assert payload["options"] == {"temperature": 0}
    assert payload["messages"] == [{"role": "system", "content": "短く"}, {"role": "user", "content": "質問"}]
    assert "format" not in payload and "tools" not in payload


def test_complete_sends_json_schema_as_format() -> None:
    schema = {"type": "object", "properties": {"category": {"type": "string"}}, "required": ["category"]}
    fake = FakeTransport({"message": {"content": '{"category": "請求"}'}})
    OllamaChat("m", format=schema, transport=fake).complete([Message("user", "x")])
    assert fake.sent[0][1]["format"] == schema


def test_chat_reads_tool_calls() -> None:
    reply = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "get_order_status", "arguments": {"order_id": "A-1001"}}}],
        },
        "prompt_eval_count": 50,
        "eval_count": 10,
    }
    fake = FakeTransport(reply)
    tools = [{"type": "function", "function": {"name": "get_order_status", "parameters": {"type": "object"}}}]
    turn = OllamaChat("m", base_url="http://127.0.0.1:11434/", transport=fake).chat(
        [{"role": "user", "content": "注文 A-1001 は？"}], tools
    )
    assert turn.tool_calls[0].name == "get_order_status"
    assert turn.tool_calls[0].arguments == {"order_id": "A-1001"}
    assert (turn.input_tokens, turn.output_tokens) == (50, 10)
    assert fake.sent[0][0] == "http://127.0.0.1:11434/api/chat"
    assert fake.sent[0][1]["tools"] == tools


def test_chat_without_tool_calls() -> None:
    turn = OllamaChat("m", transport=FakeTransport({"message": {"content": "答え"}})).chat([])
    assert turn.text == "答え"
    assert turn.tool_calls == []


def test_embedder_sends_list_and_reads_vectors() -> None:
    fake = FakeTransport({"embeddings": [[0.1, 0.2], [0.3, 0.4]]})
    vectors = OllamaEmbedder("example-embed", transport=fake).embed(["a", "b"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    url, payload = fake.sent[0]
    assert url == "http://localhost:11434/api/embed"
    assert payload == {"model": "example-embed", "input": ["a", "b"]}


def test_embedder_checks_count_and_skips_empty() -> None:
    assert OllamaEmbedder("m", transport=FakeTransport({})).embed([]) == []
    with pytest.raises(ValueError, match="数が合いません"):
        OllamaEmbedder("m", transport=FakeTransport({"embeddings": [[1.0]]})).embed(["a", "b"])


def test_urllib_transport_rejects_other_schemes() -> None:
    with pytest.raises(ValueError, match="http"):
        UrllibTransport().post_json("file:///etc/passwd", {}, 1.0)
