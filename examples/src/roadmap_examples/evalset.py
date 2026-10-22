"""評価セット（07-5 で決める形式. 08 以降でも同じ形式を使う）.

評価セットは JSONL（1 行に 1 件の JSON）で書く:

    {"id": "c001", "input": "請求書を再発行してほしい", "expected": {"category": "billing"}, "tags": ["基本"]}

- id: 件ごとに一意な名前（結果を前回と比べるときの目印）
- input: モデルに渡す入力（問い合わせ文・質問など）
- expected: 正解. 中身は課題ごとに決める（分類なら category、RAG なら sources など）
- tags: 任意. 「難しい」「答えが資料に無い」のように分けて集計したいときに付ける

採点は「1 件ごとに 0.0〜1.0 の点を付け、平均する」に統一する. 1 件の点の付け方（scorer）は課題ごとに決める.
結果も JSONL で残し（`save_results`）、前回の結果と比べて、点が下がった件（後退）を見つける（`compare`）.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalCase:
    id: str
    input: str
    expected: dict[str, Any]
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Prediction:
    """モデル（や比べる基準）の出力と、そのときに使ったトークン数."""

    output: Any
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class CaseResult:
    id: str
    score: float
    output: Any
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""
    tags: tuple[str, ...] = ()


@dataclass
class Report:
    results: list[CaseResult]
    seconds: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return sum(r.score for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.results)

    @property
    def output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.results)

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.error)

    def by_tag(self) -> dict[str, float]:
        """タグごとの平均点（タグの無い件は「(なし)」）."""
        groups: dict[str, list[float]] = defaultdict(list)
        for r in self.results:
            for tag in r.tags or ("(なし)",):
                groups[tag].append(r.score)
        return {tag: sum(s) / len(s) for tag, s in sorted(groups.items())}


class EvalSetError(ValueError):
    """評価セットのファイルの誤り（行番号つき）."""


def parse_case(obj: Any, where: str) -> EvalCase:
    if not isinstance(obj, dict):
        raise EvalSetError(f"{where}: 1 行は JSON のオブジェクトにしてください")
    for key, kind in (("id", str), ("input", str), ("expected", dict)):
        if not isinstance(obj.get(key), kind):
            raise EvalSetError(f"{where}: 「{key}」が無いか、型が違います（{kind.__name__}）")
    tags = obj.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise EvalSetError(f"{where}: 「tags」は文字列の配列にしてください")
    return EvalCase(obj["id"], obj["input"], obj["expected"], tuple(tags))


def load_cases(path: Path) -> list[EvalCase]:
    """JSONL の評価セットを読む. 空行は飛ばす. 形の誤りと id の重複は行番号つきで止める."""
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        where = f"{path.name}:{lineno}"
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise EvalSetError(f"{where}: JSON として読めません（{e.msg}）") from e
        case = parse_case(obj, where)
        if case.id in seen:
            raise EvalSetError(f"{where}: id「{case.id}」が重複しています")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise EvalSetError(f"{path.name}: 評価セットが空です")
    return cases


def save_cases(cases: Iterable[EvalCase], path: Path) -> None:
    lines = [
        json.dumps({"id": c.id, "input": c.input, "expected": c.expected, "tags": list(c.tags)}, ensure_ascii=False)
        for c in cases
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


Predict = Callable[[EvalCase], Prediction]
Scorer = Callable[[EvalCase, Any], float]


def run_eval(
    cases: Sequence[EvalCase],
    predict: Predict,
    score: Scorer,
    meta: dict[str, Any] | None = None,
    stop_on: tuple[type[Exception], ...] = (),
) -> Report:
    """全件を順に予測して採点する. 1 件の失敗（例外）は 0 点として記録し、残りを続ける.

    stop_on に書いた例外（費用や回数の上限など）だけは、記録せずにそのまま投げて評価を止める.
    """
    started = time.perf_counter()
    results = []
    for case in cases:
        try:
            p = predict(case)
        except stop_on:
            raise
        except Exception as e:  # noqa: BLE001 — 1 件の失敗で評価全体を止めない（失敗も結果として残す）
            results.append(CaseResult(case.id, 0.0, None, error=f"{type(e).__name__}: {e}", tags=case.tags))
            continue
        s = float(score(case, p.output))
        if not 0.0 <= s <= 1.0:
            raise ValueError(f"{case.id}: 点は 0.0〜1.0 にしてください（{s}）")
        results.append(CaseResult(case.id, s, p.output, p.input_tokens, p.output_tokens, tags=case.tags))
    return Report(results, time.perf_counter() - started, dict(meta or {}))


def exact_match(key: str) -> Scorer:
    """expected[key] と出力（dict なら出力[key]）が同じなら 1 点."""

    def score(case: EvalCase, output: Any) -> float:
        got = output.get(key) if isinstance(output, dict) else output
        return 1.0 if got == case.expected.get(key) else 0.0

    return score


def save_results(report: Report, path: Path) -> None:
    """結果を JSONL で残す（1 行目はまとめ、2 行目からは 1 件ずつ）."""
    head = {
        "summary": {
            "mean": report.mean,
            "count": len(report.results),
            "errors": report.errors,
            "input_tokens": report.input_tokens,
            "output_tokens": report.output_tokens,
            "seconds": round(report.seconds, 3),
            "meta": report.meta,
        }
    }
    lines = [json.dumps(head, ensure_ascii=False)]
    lines += [json.dumps(asdict(r), ensure_ascii=False) for r in report.results]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_results(path: Path) -> Report:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = json.loads(lines[0])["summary"]
    results = []
    for line in lines[1:]:
        obj = json.loads(line)
        obj["tags"] = tuple(obj.get("tags", ()))
        results.append(CaseResult(**obj))
    return Report(results, float(summary.get("seconds", 0.0)), summary.get("meta", {}))


def compare(before: Report, after: Report) -> list[tuple[str, float, float]]:
    """両方にある件のうち、点が変わったもの（id, 前, 後）. 点が下がった件（後退）を先に並べる."""
    old = {r.id: r.score for r in before.results}
    changed = [(r.id, old[r.id], r.score) for r in after.results if r.id in old and old[r.id] != r.score]
    return sorted(changed, key=lambda c: (c[2] - c[1], c[0]))
