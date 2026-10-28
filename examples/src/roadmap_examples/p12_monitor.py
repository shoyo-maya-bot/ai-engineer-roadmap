"""12-4 監視: 12-1 の API の記録（requests.jsonl）を集計し、監視の画面（HTML 1 枚）と、しきい値の確かめをする.

見るもの:
- 件数と状態ごとの割合（答えた・承認待ち・止めた・入力で止めた）. 止めた（stopped）が増えたら、上限かプロンプトを見直す
- 応答時間の中央値と 95 パーセンタイル（遅いほうから 5% の境目. 平均だと、まれに遅い依頼が隠れる）
- トークンと費用の見積もり（合計と 1 件あたり）
- プロンプトの版ごとの内訳（版を変えた前後で、応答時間・トークン・止めた割合を比べる）
- ガードレールが見つけたこと（指示の上書き・個人情報など）の件数

使い方（examples/ で）:
  uv run python -m roadmap_examples.p12_monitor var/requests.jsonl --html monitor.html --max-p95-ms 20000

終了コード: 0 しきい値の中 / 1 しきい値を超えた・記録が読めない / 2 引数の誤り
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO


def percentile(values: Sequence[float], q: float) -> float:
    """最も近い順位の方法: 小さい順に並べ、全体の q% の位置（切り上げ）の値."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def load_records(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@dataclass(frozen=True)
class Stats:
    count: int
    statuses: dict[str, int]
    p50_ms: float
    p95_ms: float
    input_tokens: int
    output_tokens: int
    cost: float
    findings: dict[str, int]

    def rate(self, status: str) -> float:
        return self.statuses.get(status, 0) / self.count if self.count else 0.0


def stats(records: Sequence[dict[str, Any]]) -> Stats:
    latencies = [float(r["latency_ms"]) for r in records]
    findings: Counter[str] = Counter()
    for r in records:
        findings.update(f.split("（")[0] for f in r.get("findings", []))
    return Stats(
        len(records),
        dict(Counter(r["status"] for r in records)),
        percentile(latencies, 50),
        percentile(latencies, 95),
        sum(int(r["input_tokens"]) for r in records),
        sum(int(r["output_tokens"]) for r in records),
        sum(float(r["cost"]) for r in records),
        dict(findings),
    )


def by_version(records: Sequence[dict[str, Any]]) -> dict[str, Stats]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[r.get("prompt_version", "?")].append(r)
    return {v: stats(rs) for v, rs in groups.items()}


@dataclass(frozen=True)
class Thresholds:
    max_p95_ms: float | None = None
    max_stopped_rate: float | None = None  # 上限で止まった割合
    max_cost: float | None = None  # 費用の見積もりの合計


def alerts(s: Stats, t: Thresholds) -> list[str]:
    found = []
    if t.max_p95_ms is not None and s.p95_ms > t.max_p95_ms:
        found.append(f"応答時間の 95 パーセンタイルが {s.p95_ms:.0f} ms です（しきい値 {t.max_p95_ms:.0f} ms）")
    if t.max_stopped_rate is not None and s.rate("stopped") > t.max_stopped_rate:
        found.append(f"上限で止まった割合が {s.rate('stopped'):.0%} です（しきい値 {t.max_stopped_rate:.0%}）")
    if t.max_cost is not None and s.cost > t.max_cost:
        found.append(f"費用の見積もりの合計が {s.cost:.4f} です（しきい値 {t.max_cost}）")
    return found


def summary(s: Stats, versions: dict[str, Stats]) -> str:
    per = s.count or 1
    lines = [
        f"件数: {s.count}",
        "状態: " + ", ".join(f"{k} {v} 件（{v / per:.0%}）" for k, v in sorted(s.statuses.items())),
        f"応答時間: 中央値 {s.p50_ms:.0f} ms / 95 パーセンタイル {s.p95_ms:.0f} ms",
        f"トークン: 入力 {s.input_tokens}（1 件 {s.input_tokens / per:.0f}）"
        f" / 出力 {s.output_tokens}（1 件 {s.output_tokens / per:.0f}）",
        f"費用の見積もり: {s.cost:.4f}",
        "ガードレールが見つけたこと: " + (", ".join(f"{k} {v} 件" for k, v in sorted(s.findings.items())) or "なし"),
        "プロンプトの版ごと:",
    ]
    for version, v in sorted(versions.items()):
        lines.append(
            f"  {version}: {v.count} 件 / 95 パーセンタイル {v.p95_ms:.0f} ms / 止めた {v.rate('stopped'):.0%}"
        )
    return "\n".join(lines)


def to_html(s: Stats, versions: dict[str, Stats], records: Sequence[dict[str, Any]], found: Sequence[str]) -> str:
    """監視の画面（HTML 1 枚. 外部のファイルや通信を使わない）. 応答時間は、新しい 50 件を棒で並べる."""
    esc = html.escape
    recent = list(records)[-50:]
    top = max((float(r["latency_ms"]) for r in recent), default=1.0) or 1.0
    bars = "".join(
        f'<div class="bar {esc(r["status"])}" style="height:{float(r["latency_ms"]) / top * 100:.0f}%"'
        f' title="{esc(r["time"])} {esc(r["status"])} {float(r["latency_ms"]):.0f} ms"></div>'
        for r in recent
    )
    cards = [
        ("件数", f"{s.count}"),
        ("95 パーセンタイル", f"{s.p95_ms:.0f} ms"),
        ("止めた割合", f"{s.rate('stopped'):.0%}"),
        ("トークン（入力 / 出力）", f"{s.input_tokens} / {s.output_tokens}"),
        ("費用の見積もり", f"{s.cost:.4f}"),
    ]
    rows = "".join(
        f"<tr><td>{esc(v)}</td><td>{x.count}</td><td>{x.p50_ms:.0f}</td><td>{x.p95_ms:.0f}</td>"
        f"<td>{x.rate('stopped'):.0%}</td><td>{(x.input_tokens + x.output_tokens) / (x.count or 1):.0f}</td></tr>"
        for v, x in sorted(versions.items())
    )
    alert_html = "".join(f"<li>{esc(a)}</li>" for a in found) or "<li>なし</li>"
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>エージェントの監視</title>
<style>
body{{font-family:sans-serif;margin:24px;color:#222}} .cards{{display:flex;gap:12px;flex-wrap:wrap}}
.card{{border:1px solid #ccc;border-radius:6px;padding:8px 12px}} .card b{{display:block;font-size:20px}}
.chart{{display:flex;align-items:flex-end;gap:2px;height:120px;border-bottom:1px solid #999;margin:8px 0}}
.bar{{width:10px;background:#4a7}} .bar.stopped{{background:#d84}} .bar.waiting_approval{{background:#48c}}
.bar.blocked{{background:#999}} table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:4px 8px}}
</style></head><body>
<h1>エージェントの監視</h1>
<div class="cards">{"".join(f"<div class='card'>{esc(k)}<b>{esc(v)}</b></div>" for k, v in cards)}</div>
<h2>しきい値を超えたもの</h2><ul>{alert_html}</ul>
<h2>応答時間（新しい {len(recent)} 件. 緑: 答えた / 青: 承認待ち / 橙: 上限で止めた / 灰: 入力で止めた）</h2>
<div class="chart">{bars}</div>
<h2>プロンプトの版ごと</h2>
<table><tr><th>版</th><th>件数</th><th>中央値 ms</th><th>95 パーセンタイル ms</th><th>止めた</th>
<th>1 件のトークン</th></tr>{rows}</table>
</body></html>
"""


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="p12_monitor", description="API の記録を集計し、監視の画面を作る")
    p.add_argument("log", type=Path, help="12-1 の API の記録（requests.jsonl）")
    p.add_argument("--html", type=Path, help="監視の画面を書き出す HTML")
    p.add_argument("--max-p95-ms", type=float, help="応答時間の 95 パーセンタイルのしきい値（ms）")
    p.add_argument("--max-stopped-rate", type=float, help="上限で止まった割合のしきい値（0〜1）")
    p.add_argument("--max-cost", type=float, help="費用の見積もりの合計のしきい値")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None, stdout: TextIO | None = None) -> int:
    args = parse_args(argv)
    stdout = stdout or sys.stdout
    try:
        records = load_records(args.log)
    except (OSError, json.JSONDecodeError) as e:
        print(f"エラー: 記録を読めません（{e}）", file=sys.stderr)
        return 1
    s, versions = stats(records), by_version(records)
    found = alerts(s, Thresholds(args.max_p95_ms, args.max_stopped_rate, args.max_cost))
    print(summary(s, versions), file=stdout)
    for a in found:
        print(f"しきい値を超えました: {a}", file=stdout)
    if args.html:
        args.html.write_text(to_html(s, versions, records, found), encoding="utf-8")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
