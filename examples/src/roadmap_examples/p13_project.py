"""13 実践プロジェクトの道具: 提出物のひな形を作り、費用を見積もり、提出の前に足りないものを確かめる.

- init:     提出物のひな形（設計書・費用の見積もり・配備の手順・振り返り）をフォルダに作る
- estimate: 評価の結果（07-5 の save_results の JSONL）の 1 件あたりのトークン数から、月の費用を見積もる
- check:    提出物がそろっているか（ファイル・設計書の見出し・評価の結果・記入漏れ）を確かめる

使い方（examples/ で）:
  uv run python -m roadmap_examples.p13_project init my-project
  uv run python -m roadmap_examples.p13_project estimate results.jsonl --per-day 200 --price-in <単価> \
      --price-out <単価>
  uv run python -m roadmap_examples.p13_project check my-project

終了コード: 0 問題なし / 1 足りないものがある・読めない / 2 引数の誤り
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from roadmap_examples.evalset import Report, load_results
from roadmap_examples.p07_cost import Price, cost

TODO = "（記入）"

DESIGN_HEADINGS = [
    "目的と要件",
    "構成",
    "RAG か学習か",
    "人の承認と上限",
    "評価の方法と結果",
    "安全",
    "費用",
    "配備と監視",
]

TEMPLATES: dict[str, str] = {
    "design.md": "# 設計書: （課題の名前）\n\n" + "".join(f"## {h}\n\n{TODO}\n\n" for h in DESIGN_HEADINGS),
    "cost.md": "# 費用の見積もり\n\n"
    "`p13_project estimate` の表を貼り、前提（1 日の件数・単価を確かめた日と、料金ページ）を書く。\n\n"
    f"{TODO}\n",
    "deploy.md": "# 配備の手順\n\n## 用意するもの\n\n"
    f"{TODO}\n\n## 手順\n\n{TODO}\n\n## 止め方と消し方\n\n{TODO}\n\n## 監視\n\n{TODO}\n",
    "retro.md": f"# 振り返り\n\n## うまくいったこと\n\n{TODO}\n\n## 次に直すこと\n\n{TODO}\n\n"
    f"## 数字で見た変化\n\n{TODO}\n",
}
EVAL_DIR = "eval"


def init_project(folder: Path) -> list[Path]:
    """ひな形を作る. 既にあるファイルは上書きしない（作ったファイルの一覧を返す）."""
    (folder / EVAL_DIR).mkdir(parents=True, exist_ok=True)
    made = []
    for name, text in TEMPLATES.items():
        path = folder / name
        if not path.exists():
            path.write_text(text, encoding="utf-8")
            made.append(path)
    return made


@dataclass(frozen=True)
class Estimate:
    cases: int
    input_per_case: float
    output_per_case: float
    per_day: int
    days: int

    def monthly_tokens(self) -> tuple[float, float]:
        n = self.per_day * self.days
        return self.input_per_case * n, self.output_per_case * n

    def monthly_cost(self, price: Price) -> float:
        tokens_in, tokens_out = self.monthly_tokens()
        return cost(round(tokens_in), round(tokens_out), price)


def estimate_from(report: Report, per_day: int, days: int) -> Estimate:
    """評価の結果から、1 件あたりのトークン数を出す（失敗した件も、使ったトークンは数える）."""
    n = len(report.results)
    if n == 0:
        raise ValueError("評価の結果が空です")
    return Estimate(n, report.input_tokens / n, report.output_tokens / n, per_day, days)


def estimate_table(est: Estimate, price: Price) -> str:
    tokens_in, tokens_out = est.monthly_tokens()
    return "\n".join(
        [
            "| 項目 | 値 |",
            "|---|---|",
            f"| 評価の件数 | {est.cases} |",
            f"| 1 件のトークン（入力 / 出力） | {est.input_per_case:.0f} / {est.output_per_case:.0f} |",
            f"| 月の件数 | {est.per_day} 件 × {est.days} 日 = {est.per_day * est.days} 件 |",
            f"| 月のトークン（入力 / 出力） | {tokens_in:,.0f} / {tokens_out:,.0f} |",
            "| ローカルのモデル | API の料金は 0（PC・電気代・保守の手間は別） |",
            f"| クラウドの API（100 万トークンあたり 入力 {price.input_per_million}"
            f" / 出力 {price.output_per_million}） | {est.monthly_cost(price):,.2f} |",
        ]
    )


HEADING = re.compile(r"^## (.+?)\s*$", re.M)


def check_project(folder: Path) -> list[str]:
    """足りないもの・記入漏れの一覧（空なら提出できる）."""
    problems = []
    for name in TEMPLATES:
        path = folder / name
        if not path.is_file():
            problems.append(f"{name} がありません")
            continue
        text = path.read_text(encoding="utf-8")
        if TODO in text:
            problems.append(f"{name} に記入していないところがあります（「{TODO}」が残っています）")
        if name == "design.md":
            heads = HEADING.findall(text)
            problems += [f"design.md に見出し「{h}」がありません" for h in DESIGN_HEADINGS if h not in heads]
    results = sorted((folder / EVAL_DIR).glob("*.jsonl")) if (folder / EVAL_DIR).is_dir() else []
    if not results:
        problems.append(f"{EVAL_DIR}/ に評価の結果（*.jsonl）がありません")
    for path in results:
        try:
            report = load_results(path)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as e:
            problems.append(f"{EVAL_DIR}/{path.name} を評価の結果として読めません（{type(e).__name__}）")
            continue
        if not report.results:
            problems.append(f"{EVAL_DIR}/{path.name} に件がありません")
    return problems


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="p13_project", description="13 実践プロジェクトの提出物の道具")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("init", help="提出物のひな形を作る")
    s.add_argument("folder", type=Path)
    s = sub.add_parser("estimate", help="評価の結果から月の費用を見積もる")
    s.add_argument("results", type=Path, help="評価の結果（07-5 の save_results の JSONL）")
    s.add_argument("--per-day", type=int, required=True, help="1 日の件数の見込み")
    s.add_argument("--days", type=int, default=20, help="月に動かす日数（既定: 営業日の 20 日）")
    s.add_argument(
        "--price-in", type=float, default=0.0, help="入力 100 万トークンあたりの単価（公式の料金ページで確かめる）"
    )
    s.add_argument("--price-out", type=float, default=0.0, help="出力 100 万トークンあたりの単価")
    s = sub.add_parser("check", help="提出物がそろっているかを確かめる")
    s.add_argument("folder", type=Path)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None, stdout: TextIO | None = None) -> int:
    args = parse_args(argv)
    stdout = stdout or sys.stdout
    if args.command == "init":
        made = init_project(args.folder)
        print(
            f"ひな形を作りました: {', '.join(p.name for p in made) or '（既にあるので作りませんでした）'}", file=stdout
        )
        return 0
    if args.command == "estimate":
        try:
            est = estimate_from(load_results(args.results), args.per_day, args.days)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"エラー: 評価の結果を読めません（{e}）", file=sys.stderr)
            return 1
        print(estimate_table(est, Price(args.price_in, args.price_out)), file=stdout)
        return 0
    problems = check_project(args.folder)
    for problem in problems:
        print(f"NG {problem}", file=stdout)
    print("提出できます" if not problems else f"足りないもの: {len(problems)} 件", file=stdout)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
