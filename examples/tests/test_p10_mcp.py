"""p10_mcp.py のテスト（10-3 MCP の最小の実装. サーバーとクライアントを、同じプロセスと子プロセス（stdio）でつなぐ）."""

import io
import json
import sys
from pathlib import Path

import pytest

from roadmap_examples.ollama_client import ChatTurn, ToolCall
from roadmap_examples.p07_tools import FakeToolChat
from roadmap_examples.p10_agent import Decision, run_agent, start_state
from roadmap_examples.p10_mcp import (
    INVALID_PARAMS,
    META_CAPABILITIES,
    META_VERSION,
    METHOD_NOT_FOUND,
    PROTOCOL_VERSION,
    UNSUPPORTED_PROTOCOL_VERSION,
    InProcessTransport,
    McpClient,
    McpError,
    StdioTransport,
    build_server,
    remote_tools,
    serve,
)

META = {META_VERSION: PROTOCOL_VERSION, META_CAPABILITIES: {}}


@pytest.fixture
def server(tmp_path: Path):
    return build_server(tmp_path / "issues.jsonl")


def test_discover_returns_versions_and_capabilities(server) -> None:
    result = McpClient(InProcessTransport(server)).discover()
    assert result["supportedVersions"] == [PROTOCOL_VERSION]
    assert result["capabilities"] == {"tools": {}}
    assert result["resultType"] == "complete"
    assert result["_meta"]["io.modelcontextprotocol/serverInfo"]["name"] == "roadmap-qa"


def test_list_tools_has_schema_and_annotations(server) -> None:
    tools = {t["name"]: t for t in McpClient(InProcessTransport(server)).list_tools()}
    assert set(tools) == {"search_docs", "create_issue"}
    assert tools["search_docs"]["inputSchema"]["required"] == ["query"]
    assert tools["search_docs"]["annotations"]["readOnlyHint"] is True
    assert tools["create_issue"]["annotations"]["readOnlyHint"] is False


def test_call_tool_returns_text_content(server) -> None:
    result = McpClient(InProcessTransport(server)).call_tool("search_docs", {"query": "アクセストークン"})
    assert result["isError"] is False
    assert "30 分" in json.loads(result["content"][0]["text"])["results"][0]["text"]


def test_tool_execution_error_is_a_result_not_a_protocol_error(server) -> None:
    result = McpClient(InProcessTransport(server)).call_tool("create_issue", {"title": ""})
    assert result["isError"] is True
    assert "引数" in json.loads(result["content"][0]["text"])["error"]


def test_unknown_tool_and_method_are_protocol_errors(server) -> None:
    client = McpClient(InProcessTransport(server))
    with pytest.raises(McpError) as e:
        client.call_tool("delete_all", {})
    assert e.value.code == INVALID_PARAMS
    with pytest.raises(McpError) as e:
        client.call("resources/list")
    assert e.value.code == METHOD_NOT_FOUND


def test_every_request_must_carry_version_and_capabilities(server) -> None:
    response = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    assert response["error"]["code"] == INVALID_PARAMS
    legacy = server.handle({"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {}})
    assert "error" in legacy


def test_unsupported_version_lists_supported_versions(server) -> None:
    client = McpClient(InProcessTransport(server), protocol_version="1900-01-01")
    with pytest.raises(McpError) as e:
        client.discover()
    assert e.value.code == UNSUPPORTED_PROTOCOL_VERSION
    assert e.value.data == {"supported": [PROTOCOL_VERSION], "requested": "1900-01-01"}


def test_notification_gets_no_response(server) -> None:
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"_meta": META}}) is None


def test_serve_reads_and_writes_one_message_per_line(server) -> None:
    request = {"jsonrpc": "2.0", "id": 7, "method": "tools/list", "params": {"_meta": META}}
    stdin = io.StringIO(json.dumps(request) + "\n\nnot json\n")
    stdout = io.StringIO()
    serve(server, stdin, stdout)
    lines = stdout.getvalue().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["id"] == 7
    assert json.loads(lines[1])["error"]["code"] == -32700


def test_agent_uses_remote_tools_and_still_asks_for_approval(server, tmp_path: Path) -> None:
    tools = remote_tools(McpClient(InProcessTransport(server)))
    issue = {"title": "購入の承認の流れが無い", "body": "質問: 購入の承認"}
    chat = FakeToolChat([ChatTurn("", [ToolCall("create_issue", issue)], 10, 1), ChatTurn("見送りました", [], 10, 1)])

    class Deny:
        def review(self, request):
            return Decision(False)

    state = run_agent(chat, start_state("起票して"), tools, Deny())
    assert state.events[0]["status"] == "denied"
    assert not (tmp_path / "issues.jsonl").exists()


def test_stdio_subprocess_end_to_end(tmp_path: Path) -> None:
    issues = tmp_path / "issues.jsonl"
    transport = StdioTransport([sys.executable, "-m", "roadmap_examples.p10_mcp", "--issues", str(issues)])
    try:
        client = McpClient(transport)
        assert client.discover()["supportedVersions"] == [PROTOCOL_VERSION]
        result = client.call_tool("create_issue", {"title": "対応ブラウザの記載が無い", "body": "質問: 対応ブラウザ"})
        assert json.loads(result["content"][0]["text"]) == {"id": "ISSUE-001", "created": True}
    finally:
        transport.close()
    assert transport.process.returncode == 0
    assert "対応ブラウザ" in issues.read_text(encoding="utf-8")
