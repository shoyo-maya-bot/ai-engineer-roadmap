"""10-2 メモリと状態の管理: エージェントの会話と途中経過を 1 つの状態にまとめ、保存して続きから動かせるようにする.

- `AgentState`: 会話（Ollama の形のメッセージ）・往復とツールの回数・トークン数・ツールの記録・承認待ちの呼び出し・結果.
  JSON にして保存し、読み戻せる（承認を待つ間にプロセスを止めても、続きから動かせる）
- `trim_messages`: 会話が長くなったら、system と最初の依頼を残して、古いやり取りから削る.
  ツールの呼び出し（assistant の tool_calls）とその結果（role が tool）の組を、途中で切らない

状態はモデルの外（コードの側）に持つ. モデルは前の呼び出しを覚えていないので、毎回、必要な会話を渡し直す.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

RUNNING = "running"
ANSWERED = "answered"
WAITING = "waiting_approval"
STOPPED = "stopped"


@dataclass
class AgentState:
    messages: list[dict[str, Any]]
    steps: int = 0  # モデルを呼んだ回数（往復の回数）
    tool_calls: int = 0  # 実行（または却下）したツールの呼び出しの回数
    input_tokens: int = 0
    output_tokens: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)  # ツールの記録（名前・引数・結果の種類・承認）
    pending: list[dict[str, Any]] = field(default_factory=list)  # まだ実行していないツールの呼び出し（先頭から順に）
    status: str = RUNNING
    answer: str = ""
    stop_reason: str = ""

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> AgentState:
        return cls(**json.loads(text))


def save_state(state: AgentState, path: Path) -> None:
    path.write_text(state.to_json() + "\n", encoding="utf-8")


def load_state(path: Path) -> AgentState:
    return AgentState.from_json(path.read_text(encoding="utf-8"))


def trim_messages(messages: list[dict[str, Any]], keep_last: int) -> list[dict[str, Any]]:
    """system と最初の user（依頼）を残し、残りは新しい keep_last 件だけにする.

    切れ目がツールの結果（role が tool）から始まると、どの呼び出しの結果か分からなくなるので、その結果も削る.
    """
    size = 2 if messages and messages[0]["role"] == "system" else 1
    head, rest = list(messages[:size]), messages[size:]
    if len(rest) <= keep_last:
        return list(messages)
    tail = rest[len(rest) - keep_last :]
    while tail and tail[0]["role"] == "tool":
        tail = tail[1:]
    return head + tail
