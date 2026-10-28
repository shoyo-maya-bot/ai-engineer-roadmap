"""12-1 エージェントを HTTP の API にする: 11 のガードレールつきエージェントを、ほかのシステムから呼べる形にする.

標準ライブラリの WSGI（wsgiref）だけで書いている（依存を増やさず、テストでは関数として呼べる）.
FastAPI で書く場合の形は、12-1 の本文で比べる.

- GET  /healthz                 動いているか（モデルは呼ばない）. プロンプトの版も返す
- POST /ask                     {"question": "..."} → 答え（200）か、承認待ち（202. 承認の内容を返す）
- POST /approvals/<request_id>  {"approved": true/false, "comment": "..."} → 承認して続きを動かす.
                                承認の係だけが知る鍵（環境変数 APPROVER_TOKEN）を Authorization: Bearer で付ける

承認待ちの状態は、data-dir にファイルで残す（10-2. 承認が数時間後でも続きから動く）.
1 回の呼び出しごとに、応答時間・トークン・費用の見積もり・プロンプトの版を、JSONL の記録に 1 行足す
（12-4 の監視で読む）.

使い方（examples/ で. 手元の Ollama を呼ぶ）:
  APPROVER_TOKEN=<承認の係の鍵> uv run python -m roadmap_examples.p12_api --model <ツールに対応したモデル名>
  curl -s localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "備品は何日間借りられますか？"}'
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import hmac
import json
import os
import re
import sys
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from wsgiref.simple_server import make_server

from roadmap_examples.ollama_client import DEFAULT_BASE_URL, ChatTurn, OllamaChat
from roadmap_examples.p07_cost import FREE, Price, cost
from roadmap_examples.p07_tools import Tool, ToolChat
from roadmap_examples.p08_chunk import load_chunks
from roadmap_examples.p08_rag import DEFAULT_DOCS
from roadmap_examples.p08_search import BM25
from roadmap_examples.p10_agent import AgentLimits, ApprovalRequest, Decision, FileIssueTracker, qa_tools
from roadmap_examples.p10_state import WAITING, AgentState
from roadmap_examples.p11_guard import GuardConfig, GuardedRun, guarded_system, run_guarded

ERROR = "error"  # モデルを呼べなかった
MAX_BODY = 16 * 1024  # 受け取る本文の上限（バイト）
APPROVAL_PATH = re.compile(r"^/approvals/(?P<id>[0-9a-f]{32})$")
StartResponse = Callable[[str, list[tuple[str, str]]], Any]


def prompt_version(tools: Sequence[Tool], config: GuardConfig) -> str:
    """system とツールの説明から作る版（中身が 1 文字でも変われば、別の版になる）. 記録に残し、結果を版ごとに比べる."""
    raw = json.dumps([guarded_system(config), [t.spec() for t in tools]], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


@dataclass
class TimedChat:
    """モデルの呼び出しごとに、かかった時間とトークン数を残す（追跡. どの往復が遅いかを見る）."""

    inner: ToolChat
    clock: Callable[[], float] = time.perf_counter
    spans: list[dict[str, Any]] = field(default_factory=list)

    def chat(self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]] | None = None) -> ChatTurn:
        started = self.clock()
        turn = self.inner.chat(messages, tools)
        self.spans.append(
            {
                "kind": "model",
                "ms": round((self.clock() - started) * 1000, 1),
                "input_tokens": turn.input_tokens,
                "output_tokens": turn.output_tokens,
                "tool_calls": [c.name for c in turn.tool_calls],
            }
        )
        return turn


@dataclass
class StateStore:
    """承認待ちの状態を、1 件 1 ファイルで残す."""

    folder: Path

    def _path(self, request_id: str) -> Path:
        return self.folder / f"{request_id}.json"

    def save(self, request_id: str, question: str, state: AgentState) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        data = {"question": question, "state": json.loads(state.to_json())}
        self._path(request_id).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def load(self, request_id: str) -> tuple[str, AgentState] | None:
        path = self._path(request_id)
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["question"], AgentState(**data["state"])

    def delete(self, request_id: str) -> None:
        self._path(request_id).unlink(missing_ok=True)


@dataclass
class RequestLog:
    """1 回の呼び出しを JSONL に 1 行で残す（12-4 の監視で読む）. 質問と答えの本文は残さない（個人情報を広げない）."""

    path: Path

    def write(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class Deferred:
    """承認を後で決める係（API では、その場で人に聞けない）. いつも「まだ決めていない」を返す."""

    def review(self, request: ApprovalRequest) -> Decision | None:
        return None


@dataclass
class Fixed:
    """API で受け取った承認の答えを、1 回だけ返す係."""

    decision: Decision
    used: bool = False

    def review(self, request: ApprovalRequest) -> Decision | None:
        if self.used:
            return None
        self.used = True
        return self.decision


@dataclass
class AgentApp:
    """WSGI のアプリ（app(environ, start_response) の形で呼ばれる）."""

    make_chat: Callable[[], ToolChat]
    tools: Sequence[Tool]
    store: StateStore
    log: RequestLog
    model: str
    approver_token: str = ""  # 空なら、承認の API は使えない（403）
    limits: AgentLimits = field(default_factory=AgentLimits)
    config: GuardConfig = field(default_factory=GuardConfig)
    price: Price = FREE
    clock: Callable[[], float] = time.perf_counter
    now: Callable[[], datetime] = lambda: datetime.now(UTC)

    def __post_init__(self) -> None:
        self.version = prompt_version(self.tools, self.config)

    def __call__(self, environ: dict[str, Any], start_response: StartResponse) -> Iterable[bytes]:
        method, path = environ.get("REQUEST_METHOD", "GET"), environ.get("PATH_INFO", "/")
        if method == "GET" and path == "/healthz":
            status, body = 200, {"status": "ok", "model": self.model, "prompt_version": self.version}
        elif method == "POST" and path == "/ask":
            status, body = self.ask(environ)
        elif method == "POST" and APPROVAL_PATH.match(path):
            status, body = self.approve(environ, APPROVAL_PATH.match(path).group("id"))
        else:
            status, body = 404, {"error": "そのパスはありません"}
        reasons = {200: "OK", 202: "Accepted", 400: "Bad Request", 401: "Unauthorized", 403: "Forbidden"}
        reasons |= {404: "Not Found", 409: "Conflict", 413: "Payload Too Large", 503: "Service Unavailable"}
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = [("Content-Type", "application/json; charset=utf-8"), ("Content-Length", str(len(data)))]
        start_response(f"{status} {reasons[status]}", headers)
        return [data]

    def read_json(self, environ: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """（エラーの HTTP の番号か 0, 本文の JSON）."""
        try:
            length = int(environ.get("CONTENT_LENGTH") or 0)
        except ValueError:
            return 400, {}
        if length < 0:  # 負の値を通すと、read(-1) で本文を全部読んでしまう（上限をすり抜ける）
            return 400, {}
        if length > MAX_BODY:
            return 413, {}
        try:
            data = json.loads(environ["wsgi.input"].read(length) or b"null")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return 400, {}
        return (0, data) if isinstance(data, dict) else (400, {})

    def ask(self, environ: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        code, data = self.read_json(environ)
        if code:
            return code, {"error": "本文は JSON のオブジェクトにしてください（16 KB まで）"}
        question = data.get("question")
        if not isinstance(question, str) or not question.strip():
            return 400, {"error": "question（文字列）を入れてください"}
        return self.run(uuid.uuid4().hex, "/ask", question, None, Deferred())

    def approve(self, environ: dict[str, Any], request_id: str) -> tuple[int, dict[str, Any]]:
        if not self.approver_token:
            return 403, {"error": "承認の API は無効です（APPROVER_TOKEN が設定されていません）"}
        scheme, _, given = environ.get("HTTP_AUTHORIZATION", "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(
            given.encode("utf-8"), self.approver_token.encode("utf-8")
        ):
            return 401, {"error": "承認の係の鍵が違います"}
        code, data = self.read_json(environ)
        if code or not isinstance(data.get("approved"), bool):
            return 400, {"error": 'approved（true か false）を入れてください. 例: {"approved": true}'}
        saved = self.store.load(request_id)
        if saved is None:
            return 404, {"error": "承認待ちの依頼が見つかりません"}
        question, state = saved
        if state.status != WAITING:
            return 409, {"error": "この依頼は承認待ちではありません"}
        decision = Decision(data["approved"], str(data.get("comment", "")))
        return self.run(request_id, "/approvals", question, state, Fixed(decision))

    def run(
        self, request_id: str, endpoint: str, question: str, state: AgentState | None, approver: Any
    ) -> tuple[int, dict[str, Any]]:
        started = self.clock()
        chat = TimedChat(self.make_chat(), self.clock)
        before = (state.input_tokens, state.output_tokens) if state else (0, 0)
        try:
            result = run_guarded(chat, question, self.tools, approver, self.limits, self.config, state)
        except OSError as e:  # モデルに繋がらない・時間切れ（urllib の誤りは OSError の仲間）
            result = GuardedRun(ERROR, "", [f"モデルを呼べませんでした（{type(e).__name__}）"], state)
        body: dict[str, Any] = {"request_id": request_id, "status": result.status, "answer": result.answer}
        status = 200
        if result.status == ERROR:  # 承認待ちの状態は消さずに残す（繋がるようになったら、承認をやり直せる）
            status = 503
            body["error"] = "いまは答えられません。しばらくしてから、もう一度送ってください"
        elif result.status == WAITING and result.state is not None:
            self.store.save(request_id, question, result.state)
            pending = result.state.pending[0]
            body["approval"] = {"tool": pending["name"], "arguments": pending["arguments"]}
            status = 202
        else:
            self.store.delete(request_id)
        final = result.state
        tokens_in = (final.input_tokens - before[0]) if final else 0
        tokens_out = (final.output_tokens - before[1]) if final else 0
        self.log.write(
            {
                "time": self.now().isoformat(timespec="seconds"),
                "request_id": request_id,
                "endpoint": endpoint,
                "status": result.status,
                "latency_ms": round((self.clock() - started) * 1000, 1),
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "cost": cost(tokens_in, tokens_out, self.price),
                "model": self.model,
                "prompt_version": self.version,
                "findings": result.findings,
                "trace": chat.spans
                + [{"kind": "tool", "name": e["name"], "status": e["status"]} for e in (final.events if final else [])],
            }
        )
        return status, body


def build_app(args: argparse.Namespace, make_chat: Callable[[], ToolChat] | None = None) -> AgentApp:
    tools = qa_tools(BM25(load_chunks(args.docs)), FileIssueTracker(args.data_dir / "issues.jsonl"))
    price = Price(args.price_in, args.price_out)
    chat = OllamaChat(args.model, args.base_url, options={"temperature": 0})
    return AgentApp(
        make_chat or (lambda: chat),
        tools,
        StateStore(args.data_dir / "pending"),
        RequestLog(args.data_dir / "requests.jsonl"),
        args.model,
        approver_token=os.environ.get("APPROVER_TOKEN", ""),
        limits=AgentLimits(args.max_steps, max_tokens=args.max_tokens, price=price, max_cost=args.max_cost),
        price=price,
    )


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    env = os.environ
    p = argparse.ArgumentParser(prog="p12_api", description="ガードレールつきのエージェントを HTTP の API にする")
    p.add_argument(
        "--model", default=env.get("AGENT_MODEL"), help="ツールに対応した Ollama のモデル名（環境変数 AGENT_MODEL）"
    )
    p.add_argument(
        "--base-url", default=env.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL), help="Ollama の場所（OLLAMA_BASE_URL）"
    )
    p.add_argument(
        "--data-dir", type=Path, default=Path(env.get("AGENT_DATA_DIR", "var")), help="記録と承認待ちの置き場所"
    )
    p.add_argument("--docs", type=Path, default=DEFAULT_DOCS, help="設計書（Markdown）のフォルダ")
    p.add_argument("--host", default="127.0.0.1", help="待ち受ける場所（コンテナの中では 0.0.0.0）")
    p.add_argument("--port", type=int, default=8000, help="待ち受けるポート")
    p.add_argument("--max-steps", type=int, default=6, help="1 回の依頼で、モデルを呼ぶ回数の上限")
    p.add_argument("--max-tokens", type=int, default=30_000, help="1 回の依頼の、トークン数の上限")
    p.add_argument("--price-in", type=float, default=0.0, help="入力 100 万トークンあたりの単価（費用の見積もり用）")
    p.add_argument("--price-out", type=float, default=0.0, help="出力 100 万トークンあたりの単価（費用の見積もり用）")
    p.add_argument("--max-cost", type=float, help="1 回の依頼の、費用の見積もりの上限（単価を渡したときだけ効く）")
    args = p.parse_args(argv)
    if not args.model:
        p.error("--model か、環境変数 AGENT_MODEL を指定してください")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    app = build_app(args)
    print(
        f"http://{args.host}:{args.port} で待ち受けます（プロンプトの版 {app.version}）. 止めるには Ctrl+C", flush=True
    )
    with make_server(args.host, args.port, app) as server, contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
