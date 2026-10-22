"""p07_tools.py のテスト（07-3 ツール使用の往復. モデルは FakeToolChat）."""

import json

import pytest

from roadmap_examples.ollama_client import ChatTurn, ToolCall
from roadmap_examples.p07_tools import SUPPORT_TOOLS, FakeToolChat, ToolLoopError, call_tool, run_with_tools

TOOLS = {t.name: t for t in SUPPORT_TOOLS}
QUESTION = [{"role": "user", "content": "注文 A-1001 はもう届きますか？"}]


def test_tool_spec_has_ollama_shape() -> None:
    spec = TOOLS["get_order_status"].spec()
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "get_order_status"
    assert spec["function"]["parameters"]["required"] == ["order_id"]


def test_run_with_tools_executes_call_and_returns_answer() -> None:
    chat = FakeToolChat(
        [
            ChatTurn("", [ToolCall("get_order_status", {"order_id": "A-1001"})], 100, 20),
            ChatTurn("4 月 3 日に発送済みです。", [], 150, 15),
        ]
    )
    run = run_with_tools(chat, QUESTION, SUPPORT_TOOLS)
    assert run.text == "4 月 3 日に発送済みです。"
    assert (run.steps, run.input_tokens, run.output_tokens) == (2, 250, 35)
    # 2 回目の呼び出しには、モデルの tool_calls と、ツールの結果（role: tool）が入っている
    second = chat.calls[1]
    assert second[1]["tool_calls"][0]["function"]["name"] == "get_order_status"
    assert second[2]["role"] == "tool"
    assert second[2]["tool_name"] == "get_order_status"
    assert json.loads(second[2]["content"])["status"] == "発送済み"


def test_call_tool_returns_errors_to_the_model() -> None:
    unknown = json.loads(call_tool(TOOLS, ToolCall("delete_order", {"order_id": "A-1001"})))
    assert "ありません" in unknown["error"]
    bad_args = json.loads(call_tool(TOOLS, ToolCall("get_order_status", {"id": "A-1001"})))
    assert bad_args["error"] == "引数が正しくありません"
    missing = json.loads(call_tool(TOOLS, ToolCall("get_order_status", {"order_id": "Z-9"})))
    assert "見つかりません" in missing["error"]


def test_search_faq() -> None:
    result = json.loads(call_tool(TOOLS, ToolCall("search_faq", {"keyword": "領収書"})))
    assert len(result["answers"]) == 1


def test_run_with_tools_stops_at_max_steps() -> None:
    loop = ChatTurn("", [ToolCall("search_faq", {"keyword": "解約"})], 10, 5)
    chat = FakeToolChat([loop] * 3)
    with pytest.raises(ToolLoopError, match="3 回"):
        run_with_tools(chat, QUESTION, SUPPORT_TOOLS, max_steps=3)
    assert len(chat.calls) == 3
