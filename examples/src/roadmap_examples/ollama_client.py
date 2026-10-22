"""ローカルの LLM（Ollama）を HTTP で呼ぶ本物の実装（07・08 で使う）.

- `OllamaChat`: 会話（POST /api/chat）. `LLMClient` として使える（`complete`）.
  構造化出力（`format`）とツール使用（`chat`）にも対応
- `OllamaEmbedder`: エンベディング（POST /api/embed）

API の形は公式ドキュメントに合わせている:
  https://docs.ollama.com/api/chat
  https://docs.ollama.com/api/embed
HTTP で送る部分は `Transport`（インターフェース）にしてあり、テストでは偽の実装に差し替える
（テストと CI は Ollama を呼ばない）.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from roadmap_examples.llm import Completion, Message

DEFAULT_BASE_URL = "http://localhost:11434"


class Transport(Protocol):
    """JSON を POST して、JSON の応答を受け取る部品（本物は HTTP、テストでは偽の実装）."""

    def post_json(self, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]: ...


class UrllibTransport:
    """標準ライブラリ（urllib）で POST する本物の実装."""

    def post_json(self, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"http(s) の URL を指定してください: {url}")
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        # スキームは上で http(s) に絞っている
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
            return json.load(response)


@dataclass(frozen=True)
class ToolCall:
    """モデルが「この関数をこの引数で呼んでほしい」と返したもの."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ChatTurn:
    """ツールを渡した会話の 1 往復の結果. ツールを呼ぶときは `tool_calls` が空でない."""

    text: str
    tool_calls: list[ToolCall]
    input_tokens: int
    output_tokens: int


@dataclass
class OllamaChat:
    """Ollama の /api/chat を呼ぶ.

    format: None（ふつうの文章）・"json"・JSON スキーマ（dict）. スキーマを渡すと、その形の JSON を返すよう制約される.
    options: temperature などの生成の設定（例: {"temperature": 0}）.
    """

    model: str
    base_url: str = DEFAULT_BASE_URL
    format: str | dict[str, Any] | None = None
    options: dict[str, Any] = field(default_factory=dict)
    timeout: float = 120.0
    transport: Transport = field(default_factory=UrllibTransport)

    def _post(self, messages: list[dict[str, Any]], tools: Sequence[dict[str, Any]] | None) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False}
        if self.format is not None:
            payload["format"] = self.format
        if self.options:
            payload["options"] = self.options
        if tools:
            payload["tools"] = list(tools)
        return self.transport.post_json(f"{self.base_url.rstrip('/')}/api/chat", payload, self.timeout)

    def complete(self, messages: Sequence[Message]) -> Completion:
        data = self._post([{"role": m.role, "content": m.content} for m in messages], None)
        return Completion(
            data.get("message", {}).get("content", ""),
            int(data.get("prompt_eval_count", 0)),
            int(data.get("eval_count", 0)),
        )

    def chat(self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]] | None = None) -> ChatTurn:
        """メッセージを Ollama の形（dict）のまま送る. ツールの結果（role が tool）を含む会話に使う."""
        data = self._post(list(messages), tools)
        message = data.get("message", {})
        calls = [
            ToolCall(c["function"]["name"], dict(c["function"].get("arguments") or {}))
            for c in message.get("tool_calls") or []
        ]
        return ChatTurn(
            message.get("content", ""), calls, int(data.get("prompt_eval_count", 0)), int(data.get("eval_count", 0))
        )


@dataclass
class OllamaEmbedder:
    """Ollama の /api/embed を呼び、文章ごとのベクトルを返す."""

    model: str
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 120.0
    transport: Transport = field(default_factory=UrllibTransport)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        data = self.transport.post_json(
            f"{self.base_url.rstrip('/')}/api/embed", {"model": self.model, "input": list(texts)}, self.timeout
        )
        vectors = data.get("embeddings", [])
        if len(vectors) != len(texts):
            raise ValueError(f"エンベディングの数が合いません（送った {len(texts)} 件 / 返った {len(vectors)} 件）")
        return [[float(x) for x in v] for v in vectors]
