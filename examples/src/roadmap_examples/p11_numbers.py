"""11-1 幻覚とベンチマークの読み方: 点数の幅を見積もり、2 つの結果の差が偶然かを確かめ、答えの数字の裏付けを探す.

- `wilson_interval`: 正解率の幅（ウィルソンの区間）. 30 件で 80% でも、本当の正解率はかなり幅がある
- `paired_changes` / `sign_test`: 同じ評価セットの 2 つの結果を件ごとに比べ、良くなった件と悪くなった件の数から、
  差が偶然で出る確率を出す
- `unsupported_numbers`: 答えに出てくる数字のうち、渡した資料のどこにも無いもの（幻覚の疑い. 08 の出典の確かめを補う）

使い方（examples/ で）:
  uv run python -m roadmap_examples.p11_numbers --correct 24 --total 30
  uv run python -m roadmap_examples.p11_numbers --before results-baseline.jsonl --after results-llm.jsonl
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from roadmap_examples.evalset import Report, load_results

NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """正解率の区間（既定は 95%）. 件数が少ないほど、また 0% や 100% に近いほど、素直な計算より当てになる."""
    if total <= 0 or not 0 <= correct <= total:
        raise ValueError("0 <= correct <= total かつ total > 0 にしてください")
    p = correct / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def paired_changes(before: Report, after: Report) -> tuple[int, int]:
    """両方にある件で（良くなった件の数, 悪くなった件の数）. 点が同じ件は数えない."""
    old = {r.id: r.score for r in before.results}
    better = sum(1 for r in after.results if r.id in old and r.score > old[r.id])
    worse = sum(1 for r in after.results if r.id in old and r.score < old[r.id])
    return better, worse


def sign_test(better: int, worse: int) -> float:
    """良くなった件と悪くなった件の数が、偶然（五分五分）でこれ以上偏る確率（両側の符号検定. 02-3）."""
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def numbers_in(text: str) -> set[str]:
    """文章の中の数字（全角は半角に、桁区切りのカンマは外す）."""
    text = unicodedata.normalize("NFKC", text)
    return {n.replace(",", "") for n in NUMBER.findall(text)}


def unsupported_numbers(answer: str, contexts: Sequence[str], ignore: Sequence[str] = ()) -> list[str]:
    """答えの数字のうち、資料（contexts）に無いもの. [1] のような出典の番号は数えない. ignore の数字も外す."""
    known = set().union(*(numbers_in(c) for c in contexts)) if contexts else set()
    found = numbers_in(re.sub(r"\[\d+\]", "", answer))
    return sorted(found - known - set(ignore))


def describe(correct: int, total: int) -> str:
    low, high = wilson_interval(correct, total)
    return f"正解率 {correct / total:.1%}（{correct} / {total} 件）. 95% の区間: {low:.1%}〜{high:.1%}"


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="p11_numbers", description="正解率の幅と、2 つの結果の差を確かめる")
    p.add_argument("--correct", type=int, help="正解した件の数")
    p.add_argument("--total", type=int, help="全部の件の数")
    p.add_argument("--before", type=Path, help="前の結果（07-5 の save_results の JSONL）")
    p.add_argument("--after", type=Path, help="後の結果（同じ評価セット）")
    args = p.parse_args(argv)
    if (args.correct is None) != (args.total is None):
        p.error("--correct と --total は一緒に指定してください")
    if (args.before is None) != (args.after is None):
        p.error("--before と --after は一緒に指定してください")
    if args.correct is None and args.before is None:
        p.error("--correct と --total か、--before と --after を指定してください")
    return args


def main(argv: Sequence[str] | None = None, stdout: TextIO | None = None) -> int:
    args = parse_args(argv)
    stdout = stdout or sys.stdout
    if args.correct is not None:
        try:
            print(describe(args.correct, args.total), file=stdout)
        except ValueError as e:
            print(f"エラー: {e}", file=sys.stderr)
            return 1
    if args.before is not None:
        before, after = load_results(args.before), load_results(args.after)
        for name, report in (("前", before), ("後", after)):
            correct = round(sum(r.score for r in report.results))
            print(f"{name}: {describe(correct, len(report.results))}", file=stdout)
        better, worse = paired_changes(before, after)
        print(
            f"良くなった件 {better} / 悪くなった件 {worse}. 偶然でこれ以上偏る確率: {sign_test(better, worse):.3f}",
            file=stdout,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
