"""06 の成果物: ローカルのモデルで、同じ質問の答えとトークン数を比べる表.

同じ質問（評価セットの形式. 07-5）を、モデルごとに 1 回ずつ聞き、答え・入力と出力のトークン数・かかった秒数・
答えに入るべき語の割合を、Markdown の表にする. モデルの大きさ・量子化・コンテキスト長も Ollama の /api/show で添える.

使い方（examples/ で. 手元の Ollama を呼ぶ）:
  uv run python -m roadmap_examples.p06_compare --model <モデル名 1> --model <モデル名 2> --out compare.md

終了コード: 0 成功 / 1 上限に達した・ファイルが読めない / 2 引数の誤り
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from roadmap_examples.evalset import EvalCase, EvalSetError, load_cases
from roadmap_examples.llm import LimitedLLM, LimitExceeded, LLMClient, Message
from roadmap_examples.ollama_client import DEFAULT_BASE_URL, OllamaChat, Transport, UrllibTransport

DEFAULT_QUESTIONS = Path(__file__).resolve().parents[2] / "data" / "p06" / "questions.jsonl"
DIGIT_COMMA = re.compile(r"(?<=\d),(?=\d{3})")


@dataclass(frozen=True)
class Row:
    """1 つのモデルに 1 つの質問をした結果. score は語の割合（語を決めていない質問は None）."""

    model: str
    case_id: str
    answer: str
    input_tokens: int
    output_tokens: int
    seconds: float
    score: float | None


def normalize(text: str) -> str:
    """全角・半角をそろえ、数字の桁区切りのカンマを外す（「１，６８０」も「1680」として探せるように）."""
    return DIGIT_COMMA.sub("", unicodedata.normalize("NFKC", text))


def keyword_score(case: EvalCase, text: str) -> float | None:
    """expected の keywords のうち、答えに入っている割合. keywords が無ければ None（人が読んで判断する）."""
    words = case.expected.get("keywords", [])
    if not words:
        return None
    got = normalize(text)
    return sum(1 for w in words if normalize(w) in got) / len(words)


def run_compare(
    models: Sequence[str],
    cases: Sequence[EvalCase],
    make_llm: Callable[[str], LLMClient],
    max_tokens: int,
    clock: Callable[[], float] = time.perf_counter,
) -> list[Row]:
    """モデルごとに、全部の質問を 1 回ずつ聞く. 呼び出しの回数（質問の数）とトークン数に、モデルごとの上限を設ける."""
    rows = []
    for model in models:
        llm = LimitedLLM(make_llm(model), max_calls=len(cases), max_tokens=max_tokens)
        for case in cases:
            started = clock()
            result = llm.complete([Message("user", case.input)])
            seconds = clock() - started
            text = result.text.strip()
            rows.append(
                Row(model, case.id, text, result.input_tokens, result.output_tokens, seconds, keyword_score(case, text))
            )
    return rows


def model_details(model: str, base_url: str, transport: Transport, timeout: float = 30.0) -> dict[str, str]:
    """Ollama の /api/show から、大きさ・量子化・コンテキスト長・できること（capabilities）を取り出す."""
    data: dict[str, Any] = transport.post_json(f"{base_url.rstrip('/')}/api/show", {"model": model}, timeout)
    details = data.get("details", {})
    info = data.get("model_info", {})
    context = next((str(v) for k, v in info.items() if k.endswith(".context_length")), "")
    return {
        "parameter_size": str(details.get("parameter_size", "")),
        "quantization": str(details.get("quantization_level", "")),
        "context_length": context,
        "capabilities": ", ".join(data.get("capabilities", [])),
    }


def weights_gb(params_billion: float, bits: float) -> float:
    """重み（パラメータ）だけのメモリの目安（GB）. 実際は、これにコンテキストの分（06-2）と実行の分が足される."""
    return params_billion * bits / 8


def shorten(text: str, width: int) -> str:
    """表の 1 マスに入れる形（1 行・| を使わない・width 文字まで）."""
    one = " ".join(text.split()).replace("|", "／")
    return one if len(one) <= width else one[: width - 1] + "…"


def to_markdown(rows: Sequence[Row], details: dict[str, dict[str, str]] | None = None, width: int = 40) -> str:
    """質問ごとの表と、モデルごとのまとめの表."""
    details = details or {}
    lines = [
        "## 質問ごと",
        "",
        "| 質問 | モデル | 答え（先頭） | 入力トークン | 出力トークン | 秒 | 語 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda r: (r.case_id, r.model)):
        score = "—" if r.score is None else f"{r.score:.0%}"
        lines.append(
            f"| {r.case_id} | {r.model} | {shorten(r.answer, width)} | {r.input_tokens} | {r.output_tokens} "
            f"| {r.seconds:.1f} | {score} |"
        )
    lines += [
        "",
        "## モデルごと",
        "",
        "| モデル | 大きさ | 量子化 | コンテキスト長 | 入力の平均 | 出力の平均 | 秒の合計 | 語の平均 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in dict.fromkeys(r.model for r in rows):
        mine = [r for r in rows if r.model == model]
        scored = [r.score for r in mine if r.score is not None]
        d = details.get(model, {})
        mean = f"{sum(scored) / len(scored):.0%}" if scored else "—"
        avg_in = sum(r.input_tokens for r in mine) / len(mine)
        avg_out = sum(r.output_tokens for r in mine) / len(mine)
        lines.append(
            f"| {model} | {d.get('parameter_size', '')} | {d.get('quantization', '')} | {d.get('context_length', '')} "
            f"| {avg_in:.0f} | {avg_out:.0f} | {sum(r.seconds for r in mine):.1f} | {mean} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="p06_compare", description="同じ質問の答えとトークン数を、モデルごとに比べる")
    p.add_argument("--model", action="append", required=True, help="比べる Ollama のモデル名（くり返して指定）")
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS, help="質問（評価セットの形式の JSONL）")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Ollama の場所")
    p.add_argument("--num-predict", type=int, default=256, help="1 回の答えの出力トークン数の上限")
    p.add_argument("--max-tokens", type=int, default=50_000, help="モデルごとのトークン数（入力 + 出力）の上限")
    p.add_argument("--out", type=Path, help="表を書き出すファイル（省くと画面に出す）")
    return p.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    make_llm: Callable[[str], LLMClient] | None = None,
    transport: Transport | None = None,
    stdout: TextIO | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> int:
    """make_llm・transport を渡すと、Ollama の代わりにそれを使う（テスト用）."""
    args = parse_args(argv)
    stdout = stdout or sys.stdout
    try:
        cases = load_cases(args.questions)
    except (OSError, EvalSetError) as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    transport = transport or UrllibTransport()
    options = {"temperature": 0, "num_predict": args.num_predict}
    make_llm = make_llm or (lambda m: OllamaChat(m, args.base_url, options=options, transport=transport))
    try:
        rows = run_compare(args.model, cases, make_llm, args.max_tokens, clock)
    except LimitExceeded as e:
        print(f"上限に達したので止めました: {e}", file=sys.stderr)
        return 1
    details = {m: model_details(m, args.base_url, transport) for m in args.model}
    table = to_markdown(rows, details)
    if args.out:
        args.out.write_text(table, encoding="utf-8")
        print(f"{args.out} に書き出しました（{len(rows)} 行）", file=stdout)
    else:
        print(table, file=stdout, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
