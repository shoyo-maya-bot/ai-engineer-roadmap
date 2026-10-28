"""10-3 MCP でツールをつなぐ: MCP（Model Context Protocol）のツールの部分だけを、標準ライブラリで小さく実装する.

仕組みを確かめるための最小の実装で、仕様の全部（リソース・プロンプト・取り消し・購読など）は扱わない.
実際に使うときは、公式の SDK を使う（10-3 の本文）. ここで合わせた仕様は 2026-07-28 版:
  https://modelcontextprotocol.io/specification/2026-07-28

- メッセージは JSON-RPC 2.0. 要求（id あり）には応答を返し、通知（id なし）には返さない
- 接続ごとの状態を持たない. 要求ごとに _meta で、プロトコルの版とクライアントのできること（capabilities）を送る
- server/discover（版とできることを返す）・tools/list（ツールの一覧）・tools/call（ツールの実行）
- stdio: サーバーは標準入力から 1 行に 1 つのメッセージを読み、標準出力に 1 行に 1 つ書く（ログは標準エラーへ）

使い方（examples/ で. LLM もネットワークも使わない. ホストがこのコマンドを子プロセスとして起動する）:
  uv run python -m roadmap_examples.p10_mcp --issues issues.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TextIO

from roadmap_examples.ollama_client import ToolCall
from roadmap_examples.p07_tools import Tool, call_tool
from roadmap_examples.p08_chunk import load_chunks
from roadmap_examples.p08_rag import DEFAULT_DOCS
from roadmap_examples.p08_search import BM25
from roadmap_examples.p10_agent import FileIssueTracker, has_error, qa_tools

PROTOCOL_VERSION = "2026-07-28"
SUPPORTED_VERSIONS = [PROTOCOL_VERSION]
META_VERSION = "io.modelcontextprotocol/protocolVersion"
META_CAPABILITIES = "io.modelcontextprotocol/clientCapabilities"
META_CLIENT = "io.modelcontextprotocol/clientInfo"
META_SERVER = "io.modelcontextprotocol/serverInfo"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
UNSUPPORTED_PROTOCOL_VERSION = -32022

# ツールの振る舞いの注釈（ヒント）. クライアントは、信頼できるサーバーのもの以外は信じない（仕様の Tools の注意）
QA_ANNOTATIONS: dict[str, dict[str, bool]] = {
    "search_docs": {"readOnlyHint": True, "openWorldHint": False},
    "create_issue": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
}


def error(msg_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": msg_id, "error": err}


# --- サーバー ---


@dataclass
class McpServer:
    name: str
    version: str
    tools: list[Tool]
    annotations: dict[str, dict[str, bool]] = field(default_factory=dict)
    instructions: str = ""

    def describe(self, tool: Tool) -> dict[str, Any]:
        spec: dict[str, Any] = {"name": tool.name, "description": tool.description, "inputSchema": tool.parameters}
        if tool.name in self.annotations:
            spec["annotations"] = self.annotations[tool.name]
        return spec

    def handle(self, message: Any) -> dict[str, Any] | None:
        """1 つのメッセージを処理し、応答を返す（通知には None）. 前の要求の内容には頼らない."""
        if (
            not isinstance(message, dict)
            or message.get("jsonrpc") != "2.0"
            or not isinstance(message.get("method"), str)
        ):
            return error(message.get("id") if isinstance(message, dict) else None, INVALID_REQUEST, "Invalid request")
        if "id" not in message:
            return None
        msg_id, method = message["id"], message["method"]
        params = message.get("params") or {}
        meta = params.get("_meta") if isinstance(params, dict) else None
        if not isinstance(meta, dict) or META_VERSION not in meta or META_CAPABILITIES not in meta:
            return error(msg_id, INVALID_PARAMS, f"_meta に {META_VERSION} と {META_CAPABILITIES} が要ります")
        if meta[META_VERSION] not in SUPPORTED_VERSIONS:
            data = {"supported": SUPPORTED_VERSIONS, "requested": meta[META_VERSION]}
            return error(msg_id, UNSUPPORTED_PROTOCOL_VERSION, "Unsupported protocol version", data)
        if method == "server/discover":
            result: dict[str, Any] = {
                "supportedVersions": SUPPORTED_VERSIONS,
                "capabilities": {"tools": {}},
                "instructions": self.instructions,
                "ttlMs": 0,
                "cacheScope": "public",
            }
        elif method == "tools/list":
            result = {"tools": [self.describe(t) for t in self.tools], "ttlMs": 0, "cacheScope": "public"}
        elif method == "tools/call":
            registry = {t.name: t for t in self.tools}
            name = params.get("name")
            if name not in registry:
                return error(msg_id, INVALID_PARAMS, f"Unknown tool: {name}")
            text = call_tool(registry, ToolCall(name, dict(params.get("arguments") or {})))
            result = {"content": [{"type": "text", "text": text}], "isError": has_error(text)}
        else:
            return error(msg_id, METHOD_NOT_FOUND, f"Method not found: {method}")
        result["resultType"] = "complete"
        result["_meta"] = {META_SERVER: {"name": self.name, "version": self.version}}
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def serve(server: McpServer, stdin: TextIO, stdout: TextIO) -> None:
    """stdio で動かす: 1 行に 1 つのメッセージを読み、応答を 1 行で書く. 標準入力が閉じたら終わる."""
    for line in stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            response: dict[str, Any] | None = error(None, PARSE_ERROR, "Parse error")
        else:
            response = server.handle(message)
        if response is not None:
            stdout.write(json.dumps(response, ensure_ascii=False) + "\n")  # json.dumps は改行を \n に置き換える
            stdout.flush()


# --- クライアント ---


class McpTransport(Protocol):
    def request(self, message: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class InProcessTransport:
    """同じプロセスのサーバーを直接呼ぶ（テスト用）."""

    server: McpServer

    def request(self, message: dict[str, Any]) -> dict[str, Any]:
        response = self.server.handle(json.loads(json.dumps(message)))  # 送る前に JSON にできるかも確かめる
        if response is None:
            raise ValueError("要求に応答がありません")
        return response


class StdioTransport:
    """サーバーを子プロセスとして起動し、標準入出力で 1 行ずつやり取りする."""

    def __init__(self, command: Sequence[str]) -> None:
        # 起動するコマンドは呼び出し側が決める. シェルは通さない（引数の一覧のまま渡す）
        self.process = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )

    def request(self, message: dict[str, Any]) -> dict[str, Any]:
        if self.process.stdin is None or self.process.stdout is None:
            raise ConnectionError("サーバーの標準入出力がありません")
        self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise ConnectionError("サーバーが応答せずに終わりました")
        return json.loads(line)

    def close(self, timeout: float = 5.0) -> None:
        """標準入力を閉じて、終わるのを待つ（終わらなければ止める）."""
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        if self.process.stdout is not None:
            self.process.stdout.close()


class McpError(RuntimeError):
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code, self.data = code, data


@dataclass
class McpClient:
    transport: McpTransport
    name: str = "roadmap-examples"
    version: str = "0.1.0"
    protocol_version: str = PROTOCOL_VERSION
    next_id: int = 1

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """要求を送る. 要求ごとに _meta で版・クライアント・できることを送る（接続の状態に頼らない）."""
        meta = {
            META_VERSION: self.protocol_version,
            META_CLIENT: {"name": self.name, "version": self.version},
            META_CAPABILITIES: {},
        }
        message = {"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": {**(params or {}), "_meta": meta}}
        self.next_id += 1
        response = self.transport.request(message)
        if "error" in response:
            e = response["error"]
            raise McpError(e["code"], e["message"], e.get("data"))
        return response["result"]

    def discover(self) -> dict[str, Any]:
        return self.call("server/discover")

    def list_tools(self) -> list[dict[str, Any]]:
        return self.call("tools/list")["tools"]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.call("tools/call", {"name": name, "arguments": arguments})


def remote_tools(client: McpClient) -> list[Tool]:
    """MCP サーバーのツールを、エージェント（p10_agent）が使える Tool にする. 実行すると tools/call を送る."""

    def make_func(name: str):
        def func(**arguments: Any) -> Any:
            result = client.call_tool(name, arguments)
            text = "\n".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
            if result.get("isError"):
                raise ValueError(text)  # call_tool（07-3）が受け止め、誤りとしてモデルに返す
            return json.loads(text) if text else {}

        return func

    return [
        Tool(t["name"], t.get("description", ""), t["inputSchema"], make_func(t["name"])) for t in client.list_tools()
    ]


def build_server(issues: Path, docs: Path = DEFAULT_DOCS) -> McpServer:
    """08 の設計書のキーワード検索（BM25. LLM を使わない）と、手元のファイルへの起票を公開するサーバー."""
    tools = qa_tools(BM25(load_chunks(docs)), FileIssueTracker(issues))
    note = "設計書の検索（読むだけ）と、課題の起票（書き換える. ホストの側で人が承認してから呼ぶ）"
    return McpServer("roadmap-qa", "0.1.0", tools, QA_ANNOTATIONS, note)


def main(argv: Sequence[str] | None = None, stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="p10_mcp", description="設計書の検索と起票を、MCP のサーバー（stdio）として公開する"
    )
    p.add_argument("--issues", type=Path, required=True, help="起票先の JSONL ファイル（手元のファイル）")
    p.add_argument("--docs", type=Path, default=DEFAULT_DOCS, help="設計書（Markdown）のフォルダ")
    args = p.parse_args(argv)
    # Windows でも UTF-8 で読み書きする（標準入出力の既定の文字コードに頼らない）
    stdin = stdin or io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")
    stdout = stdout or io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", newline="\n")
    print("roadmap-qa: 標準入力を待っています", file=sys.stderr)  # ログは標準エラーへ（標準出力はメッセージだけ）
    serve(build_server(args.issues, args.docs), stdin, stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
