"""07-3 関数呼び出しとツール使用: モデルに「どの関数を、どの引数で呼ぶか」を選ばせ、実行はコードがする.

流れ: ツールの説明（名前・説明・引数の JSON スキーマ）を渡す → モデルが tool_calls を返す →
コードが引数を確かめて関数を実行し、結果を role が tool のメッセージで返す → モデルが結果を使って答える.
モデルが関数を直接実行するわけではない. 実行するか・何を許すかは、いつもコードの側で決める.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from roadmap_examples.ollama_client import ChatTurn, ToolCall
from roadmap_examples.p07_structured import validate


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # 引数の JSON スキーマ
    func: Callable[..., Any]

    def spec(self) -> dict[str, Any]:
        """モデルに渡すツールの説明（Ollama の tools の形）."""
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


class ToolChat(Protocol):
    """ツールを渡して会話する部品（本物は OllamaChat、テストでは FakeToolChat）."""

    def chat(self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]] | None = None) -> ChatTurn: ...


@dataclass
class FakeToolChat:
    """決まった応答（ChatTurn）を順に返す偽の実装. 受け取ったメッセージを `calls` に残す."""

    turns: list[ChatTurn]
    calls: list[list[dict[str, Any]]] = field(default_factory=list)

    def chat(self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]] | None = None) -> ChatTurn:
        if len(self.calls) >= len(self.turns):
            raise AssertionError(f"FakeToolChat の応答が足りません（用意 {len(self.turns)} 件）")
        self.calls.append([dict(m) for m in messages])
        return self.turns[len(self.calls) - 1]


class ToolLoopError(RuntimeError):
    """決めた回数の往復で答えが出なかった（ツールを呼び続けている）."""


@dataclass
class ToolRun:
    text: str
    messages: list[dict[str, Any]]
    steps: int
    input_tokens: int
    output_tokens: int


def call_tool(tools: dict[str, Tool], call: ToolCall) -> str:
    """1 回分のツールを実行し、結果を JSON の文字列にする. 誤りもモデルに返す（直して呼び直せるように）."""
    tool = tools.get(call.name)
    if tool is None:
        return json.dumps({"error": f"ツール「{call.name}」はありません"}, ensure_ascii=False)
    problems = validate(call.arguments, tool.parameters)
    if problems:
        return json.dumps({"error": "引数が正しくありません", "details": problems}, ensure_ascii=False)
    try:
        result = tool.func(**call.arguments)
    except (KeyError, ValueError) as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    return json.dumps(result, ensure_ascii=False)


def run_with_tools(
    chat: ToolChat, messages: Sequence[dict[str, Any]], tools: Sequence[Tool], max_steps: int = 4
) -> ToolRun:
    """モデルがツールを呼ばずに答えるまで、往復を繰り返す（max_steps 回まで）."""
    registry = {t.name: t for t in tools}
    specs = [t.spec() for t in tools]
    history = [dict(m) for m in messages]
    tokens_in = tokens_out = 0
    for step in range(1, max_steps + 1):
        turn = chat.chat(history, specs)
        tokens_in += turn.input_tokens
        tokens_out += turn.output_tokens
        if not turn.tool_calls:
            history.append({"role": "assistant", "content": turn.text})
            return ToolRun(turn.text, history, step, tokens_in, tokens_out)
        history.append(
            {
                "role": "assistant",
                "content": turn.text,
                "tool_calls": [{"function": {"name": c.name, "arguments": c.arguments}} for c in turn.tool_calls],
            }
        )
        for call in turn.tool_calls:
            history.append({"role": "tool", "tool_name": call.name, "content": call_tool(registry, call)})
    raise ToolLoopError(f"{max_steps} 回の往復で答えが出ませんでした")


# --- 演習の題材: 架空の注文と、よくある質問 ---

ORDERS: dict[str, dict[str, str]] = {
    "A-1001": {"status": "発送済み", "shipped_on": "2026-04-03"},
    "A-1002": {"status": "準備中", "shipped_on": ""},
    "A-1003": {"status": "キャンセル済み", "shipped_on": ""},
}

FAQ: dict[str, str] = {
    "領収書": "領収書は、会員ページの「ご利用履歴」から PDF で発行できます。",
    "解約": "解約は、会員ページの「契約内容」から手続きできます。月の途中で解約しても、その月の末日まで使えます。",
    "パスワード": "パスワードを忘れた場合は、ログイン画面の「パスワードを忘れた方」から再設定できます。",
}


def get_order_status(order_id: str) -> dict[str, str]:
    if order_id not in ORDERS:
        raise KeyError(f"注文番号 {order_id} は見つかりません")
    return {"order_id": order_id, **ORDERS[order_id]}


def search_faq(keyword: str) -> dict[str, list[str]]:
    return {"answers": [text for key, text in FAQ.items() if key in keyword or keyword in key]}


SUPPORT_TOOLS = [
    Tool(
        "get_order_status",
        "注文番号から、注文の状態（発送済み・準備中など）と発送日を調べる",
        {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "注文番号（例: A-1001）"}},
            "required": ["order_id"],
            "additionalProperties": False,
        },
        get_order_status,
    ),
    Tool(
        "search_faq",
        "よくある質問から、キーワードに合う回答を探す",
        {
            "type": "object",
            "properties": {"keyword": {"type": "string", "description": "探す語（例: 領収書）", "minLength": 1}},
            "required": ["keyword"],
            "additionalProperties": False,
        },
        search_faq,
    ),
]
