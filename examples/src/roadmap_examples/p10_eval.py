"""10-4 エージェントの評価: 07-5 の評価セットの考え方を、エージェントの「動き方」にも当てる.

答えの中身だけでなく、どのツールを呼んだか・起票を提案したか・上限の中で答えたかを、件ごとに確かめて点にする.
評価では起票しない（承認の係 RecordingApprover が、頼まれた承認を記録してすべて却下する）.

評価セットは 07-5 の形式（evalset.py）. expected には次を書く:
  issue:    起票（create_issue）を提案すべきか（true / false）
  tools:    呼ぶべきツールの一覧（例: ["search_docs"]. 呼ばなくてよければ空）
  keywords: 答えに入るべき語の一覧（任意）

使い方（examples/ で. 手元の Ollama を呼ぶ）:
  uv run python -m roadmap_examples.p10_eval --model <ツールに対応したモデル名>
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TextIO

from roadmap_examples.evalset import EvalCase, EvalSetError, Prediction, Report, load_cases, run_eval
from roadmap_examples.ollama_client import DEFAULT_BASE_URL, OllamaChat
from roadmap_examples.p07_tools import ToolChat
from roadmap_examples.p08_chunk import load_chunks
from roadmap_examples.p08_rag import DEFAULT_DOCS
from roadmap_examples.p08_search import BM25
from roadmap_examples.p10_agent import (
    AgentLimits,
    MemoryIssueTracker,
    RecordingApprover,
    qa_tools,
    run_agent,
    start_state,
)
from roadmap_examples.p10_state import ANSWERED

DEFAULT_EVAL = Path(__file__).resolve().parents[2] / "data" / "p10" / "agent_cases.jsonl"


def checks(case: EvalCase, output: dict[str, Any]) -> dict[str, bool]:
    """件ごとの確かめること. 名前と、合っていたか."""
    expected = case.expected
    result = {
        "答えた（上限の中で）": output["status"] == ANSWERED,
        "起票の提案": output["proposed_issue"] == expected.get("issue", False),
        "呼ぶべきツール": set(expected.get("tools", [])) <= set(output["called"]),
        "承認なしで書き換えていない": all(
            e["approved"] is True for e in output["events"] if e["name"] == "create_issue" and e["status"] == "ok"
        ),
    }
    words = expected.get("keywords", [])
    if words:
        result["答えの語"] = all(w in output["answer"] for w in words)
    return result


def agent_score(case: EvalCase, output: dict[str, Any]) -> float:
    """確かめることのうち、合っていた割合."""
    found = checks(case, output)
    return sum(found.values()) / len(found)


def make_predict(
    make_chat: Callable[[EvalCase], ToolChat],
    docs: Path = DEFAULT_DOCS,
    limits: AgentLimits = AgentLimits(),  # noqa: B008 — frozen
) -> Callable[[EvalCase], Prediction]:
    tools = qa_tools(BM25(load_chunks(docs)), MemoryIssueTracker())

    def predict(case: EvalCase) -> Prediction:
        approver = RecordingApprover()
        state = run_agent(make_chat(case), start_state(case.input), tools, approver, limits)
        output = {
            "status": state.status,
            "answer": state.answer,
            "called": sorted({e["name"] for e in state.events}),
            "proposed_issue": any(r.tool == "create_issue" for r in approver.requests),
            "events": state.events,
            "steps": state.steps,
        }
        return Prediction(output, state.input_tokens, state.output_tokens)

    return predict


def summary(cases: Sequence[EvalCase], report: Report) -> str:
    lines = [f"点の平均: {report.mean:.0%}（{len(report.results)} 件）"]
    lines.append(f"トークン: 入力 {report.input_tokens} / 出力 {report.output_tokens}")
    lines.append("タグごと: " + ", ".join(f"{t} {s:.0%}" for t, s in report.by_tag().items()))
    lines.append("外れた件:")
    by_id = {c.id: c for c in cases}
    bad = []
    for r in report.results:
        if r.error:
            bad.append(f"  {r.id}: エラー（{r.error}）")
        elif r.score < 1.0:
            failed = [name for name, ok in checks(by_id[r.id], r.output).items() if not ok]
            bad.append(f"  {r.id}: {'・'.join(failed)}")
    lines += bad or ["  なし"]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="p10_eval", description="エージェントの動き方を評価セットで確かめる")
    p.add_argument("--model", required=True, help="ツールに対応した Ollama のモデル名")
    p.add_argument("--eval", type=Path, default=DEFAULT_EVAL, help="評価セット（JSONL）")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Ollama の場所")
    p.add_argument("--max-steps", type=int, default=6, help="1 件あたりの、モデルを呼ぶ回数の上限")
    p.add_argument("--max-tokens", type=int, default=30_000, help="1 件あたりのトークン数の上限")
    return p.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    make_chat: Callable[[EvalCase], ToolChat] | None = None,
    stdout: TextIO | None = None,
) -> int:
    """make_chat を渡すと、Ollama の代わりにそれを使う（テスト用）."""
    args = parse_args(argv)
    stdout = stdout or sys.stdout
    try:
        cases = load_cases(args.eval)
    except (OSError, EvalSetError) as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    chat = OllamaChat(args.model, args.base_url, options={"temperature": 0})
    make_chat = make_chat or (lambda _case: chat)
    limits = AgentLimits(max_steps=args.max_steps, max_tokens=args.max_tokens)
    report = run_eval(cases, make_predict(make_chat, limits=limits), agent_score, meta={"model": args.model})
    print(summary(cases, report), file=stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
