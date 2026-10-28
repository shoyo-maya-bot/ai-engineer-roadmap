"""07 の成果物: 問い合わせの分類を LLM でやり直し、比べる基準と同じ評価セットで、精度と費用を比べる.

比べる基準は 04 の分類モデル（04 の結果のファイルを --compare に渡す. 差し替え方は 04-4）.
--mode baseline は、04 を飛ばした人のための「仮のルールベース分類」（キーワードで振り分ける）.

使い方（examples/ で）:
  uv run python -m roadmap_examples.p07_classify --mode baseline --out results-baseline.jsonl
  uv run python -m roadmap_examples.p07_classify --mode ollama --model <モデル名> --out results-llm.jsonl \
      --compare results-baseline.jsonl

- ollama モードは、手元の Ollama（http://localhost:11434）を呼ぶ. 呼び出しの回数とトークン数に上限を設ける
- --price-in / --price-out（100 万トークンあたりの単価）を渡すと、費用の見積もりも出す（ローカルのモデルは 0 のまま）

終了コード: 0 成功 / 1 評価セットの誤り・上限に達した / 2 引数の誤り
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from roadmap_examples.evalset import (
    EvalCase,
    EvalSetError,
    Predict,
    Prediction,
    Report,
    compare,
    exact_match,
    load_cases,
    load_results,
    run_eval,
    save_results,
)
from roadmap_examples.llm import LimitedLLM, LimitExceeded, LLMClient
from roadmap_examples.ollama_client import DEFAULT_BASE_URL, OllamaChat
from roadmap_examples.p07_cost import Price, cost
from roadmap_examples.p07_prompt import DEFAULT_SPEC, PromptSpec, build_messages
from roadmap_examples.p07_structured import CLASSIFY_SCHEMA, ask_json

DEFAULT_EVAL = Path(__file__).resolve().parents[2] / "data" / "p07" / "inquiries.jsonl"

# 仮のルールベース分類（04 を飛ばした人の基準. 04 のモデルへの差し替え方は 04-4）. 上から順に、最初に当たったものにする
RULES: list[tuple[str, tuple[str, ...]]] = [
    ("bug", ("エラー", "動かない", "表示されない", "真っ白", "落ちる", "固まる")),
    ("billing", ("請求", "支払", "領収書", "決済", "引き落とし", "返金")),
    ("account", ("ログイン", "パスワード", "メールアドレス", "認証", "アカウント")),
    ("contract", ("解約", "プラン", "契約", "更新", "退会")),
]


def rule_based(text: str) -> str:
    for category, words in RULES:
        if any(w in text for w in words):
            return category
    return "other"


def baseline_predict(case: EvalCase) -> Prediction:
    return Prediction({"category": rule_based(case.input)})


def llm_predict(llm: LLMClient, spec: PromptSpec = DEFAULT_SPEC, max_attempts: int = 2) -> Predict:
    """LLM で分類する予測の関数を作る（プロンプトは 07-1、検証と頼み直しは 07-2）."""

    def predict(case: EvalCase) -> Prediction:
        result = ask_json(llm, build_messages(spec, case.input), CLASSIFY_SCHEMA, max_attempts)
        return Prediction(result.value, result.input_tokens, result.output_tokens)

    return predict


def confusion(cases: Sequence[EvalCase], report: Report) -> Counter[tuple[str, str]]:
    """（正解, 予測）の組ごとの件数. 失敗した件の予測は「(失敗)」."""
    expected = {c.id: c.expected["category"] for c in cases}
    pairs: Counter[tuple[str, str]] = Counter()
    for r in report.results:
        got = r.output.get("category") if isinstance(r.output, dict) else "(失敗)"
        pairs[(expected[r.id], str(got))] += 1
    return pairs


def summary(cases: Sequence[EvalCase], report: Report, price: Price) -> str:
    lines = [
        f"正解率: {report.mean:.1%}（{sum(r.score for r in report.results):.0f} / {len(report.results)} 件）",
        f"失敗: {report.errors} 件",
        f"トークン: 入力 {report.input_tokens} / 出力 {report.output_tokens}",
        f"費用の見積もり: {cost(report.input_tokens, report.output_tokens, price):.4f}",
        f"時間: {report.seconds:.1f} 秒",
        "タグごとの正解率: " + ", ".join(f"{t} {s:.0%}" for t, s in report.by_tag().items()),
        "誤り（正解 → 予測: 件数）:",
    ]
    wrong = sorted((k, n) for k, n in confusion(cases, report).items() if k[0] != k[1])
    lines += [f"  {e} → {g}: {n}" for (e, g), n in wrong] or ["  なし"]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="p07_classify", description="問い合わせの分類を評価セットで比べる")
    p.add_argument("--mode", choices=["baseline", "ollama"], required=True, help="baseline: 仮のルール / ollama: LLM")
    p.add_argument("--eval", type=Path, default=DEFAULT_EVAL, help="評価セット（JSONL）")
    p.add_argument("--model", help="Ollama のモデル名（ollama モードで必須）")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Ollama の場所")
    p.add_argument("--max-calls", type=int, help="呼び出しの回数の上限（既定: 件数の 2 倍）")
    p.add_argument("--max-tokens", type=int, default=200_000, help="トークン数（入力 + 出力）の上限")
    p.add_argument("--price-in", type=float, default=0.0, help="入力 100 万トークンあたりの単価（見積もり用）")
    p.add_argument("--price-out", type=float, default=0.0, help="出力 100 万トークンあたりの単価（見積もり用）")
    p.add_argument("--out", type=Path, help="結果を書く JSONL")
    p.add_argument("--compare", type=Path, help="比べる前回の結果（JSONL）")
    args = p.parse_args(argv)
    if args.mode == "ollama" and not args.model:
        p.error("ollama モードでは --model を指定してください")
    return args


def main(argv: Sequence[str] | None = None, llm: LLMClient | None = None, stdout: TextIO | None = None) -> int:
    """llm を渡すと、Ollama の代わりにそれを使う（テスト用）."""
    args = parse_args(argv)
    stdout = stdout or sys.stdout
    try:
        cases = load_cases(args.eval)
    except (OSError, EvalSetError) as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    if args.mode == "baseline":
        report = run_eval(cases, baseline_predict, exact_match("category"), meta={"mode": "baseline"})
    else:
        inner = llm or OllamaChat(args.model, args.base_url, format=CLASSIFY_SCHEMA, options={"temperature": 0})
        limited = LimitedLLM(inner, args.max_calls or 2 * len(cases), args.max_tokens)
        meta = {"mode": "ollama", "model": args.model}
        try:
            report = run_eval(cases, llm_predict(limited), exact_match("category"), meta, stop_on=(LimitExceeded,))
        except LimitExceeded as e:
            print(f"上限に達したので止めました: {e}", file=sys.stderr)
            return 1
    print(summary(cases, report, Price(args.price_in, args.price_out)), file=stdout)
    if args.out:
        save_results(report, args.out)
    if args.compare:
        changed = compare(load_results(args.compare), report)
        print(f"前回からの変化: {len(changed)} 件", file=stdout)
        for case_id, before, after in changed:
            print(f"  {case_id}: {before:.0f} → {after:.0f}{'（後退）' if after < before else ''}", file=stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
