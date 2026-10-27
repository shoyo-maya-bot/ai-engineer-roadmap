"""06-3 事前学習・指示調整・ファインチューニングの違い: チャットのテンプレート.

事前学習だけのモデル（ベースモデル）は、渡された文章の続きを書く. 指示調整したモデル（指示モデル）は、
「system・user・assistant の区切りが入った文章」の続きとして、assistant の答えを書くように学習している.
会話（メッセージの一覧）は、モデルに渡す前に、この区切りを入れた 1 つの文章に組み立てられる（チャットのテンプレート）.

ここの区切り（<|user|> など）は、仕組みを見るための架空のもの. 本物の区切りはモデルごとに違い、
Ollama では /api/show の template で確かめられる（06-3）.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from roadmap_examples.llm import Message

END = "<|end|>"


def render_chat(messages: Sequence[Message]) -> str:
    """メッセージを、区切りを入れた 1 つの文章にする. 最後は assistant の始まりで終え、その続きをモデルに書かせる."""
    parts = [f"<|{m.role}|>\n{m.content}{END}\n" for m in messages]
    return "".join(parts) + "<|assistant|>\n"


def overhead_tokens(messages: Sequence[Message], count: Callable[[str], int]) -> int:
    """区切りのぶん増えるトークン数（組み立てた文章のトークン数 - 中身だけのトークン数）."""
    return count(render_chat(messages)) - sum(count(m.content) for m in messages)
