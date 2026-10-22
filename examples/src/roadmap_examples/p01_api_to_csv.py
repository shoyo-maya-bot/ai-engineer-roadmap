"""01 の成果物の雛形: 公開 API から JSON を取り、指定した項目を CSV にする CLI.

使い方:
  api-to-csv --url https://api.example.com/items --fields id,name,owner.name -o items.csv
  api-to-csv --url https://api.example.com/search --items-key data.items --fields id,title

- 応答は「オブジェクトの配列」か、配列を含むオブジェクト（--items-key で場所を指定）
- 項目は `owner.name` のようにドットで入れ子を指定できる. 無い項目は空欄にする
- HTTP で取る部分は `JsonGetter`（インターフェース）にしてあり、テストでは偽の実装に差し替える

終了コード: 0 成功 / 1 取得やデータの誤り / 2 引数の誤り
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, TextIO


class JsonGetter(Protocol):
    """URL から JSON を取ってくる部品（本物は HTTP、テストでは偽の実装）."""

    def get_json(self, url: str, timeout: float) -> Any: ...


class UrllibJsonGetter:
    """標準ライブラリ（urllib）で HTTP GET する本物の実装."""

    def get_json(self, url: str, timeout: float) -> Any:
        if not url.startswith(("https://", "http://")):
            raise ValueError(f"http(s) の URL を指定してください: {url}")
        request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "roadmap-examples"})
        # スキームは上で http(s) に絞っている
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
            return json.load(response)


class DataError(Exception):
    """応答の形が想定と違う."""


def pick(obj: Any, path: str) -> Any:
    """`owner.name` のようなドット区切りの場所の値. 無ければ None."""
    for key in path.split("."):
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj


def find_items(data: Any, items_key: str | None) -> list[dict[str, Any]]:
    """応答から、行にするオブジェクトの配列を取り出す."""
    items = pick(data, items_key) if items_key else data
    if not isinstance(items, list):
        where = f"「{items_key}」" if items_key else "応答の最上位"
        raise DataError(f"{where}が配列ではありません")
    if not all(isinstance(item, dict) for item in items):
        raise DataError("配列の要素がオブジェクトではありません")
    return items


def to_cell(value: Any) -> str:
    """CSV の 1 マス. 入れ子の値は JSON の文字列にする."""
    if value is None:
        return ""
    if isinstance(value, dict | list):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def to_rows(items: list[dict[str, Any]], fields: Sequence[str]) -> list[list[str]]:
    return [[to_cell(pick(item, field)) for field in fields] for item in items]


def write_csv(out: TextIO, fields: Sequence[str], rows: list[list[str]]) -> None:
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(fields)
    writer.writerows(rows)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="api-to-csv", description="公開 API の JSON を CSV にする")
    parser.add_argument("--url", required=True, help="取得する URL（http / https）")
    parser.add_argument("--fields", required=True, help="CSV にする項目（カンマ区切り. owner.name のように入れ子も可）")
    parser.add_argument("--items-key", help="配列の場所（例: data.items）. 省くと応答の最上位")
    parser.add_argument("-o", "--output", type=Path, help="出力する CSV ファイル. 省くと標準出力")
    parser.add_argument("--timeout", type=float, default=10.0, help="待つ秒数（既定 10）")
    args = parser.parse_args(argv)
    args.fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    if not args.fields:
        parser.error("--fields に項目を 1 つ以上書いてください")
    return args


def main(argv: Sequence[str] | None = None, getter: JsonGetter | None = None, stdout: TextIO | None = None) -> int:
    args = parse_args(argv)
    getter = getter or UrllibJsonGetter()
    stdout = stdout or sys.stdout
    try:
        items = find_items(getter.get_json(args.url, args.timeout), args.items_key)
    except (OSError, ValueError, DataError) as e:  # 接続の失敗・JSON でない応答・形の違い
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    rows = to_rows(items, args.fields)
    if args.output:
        with args.output.open("w", encoding="utf-8", newline="") as f:
            write_csv(f, args.fields, rows)
    else:
        write_csv(stdout, args.fields, rows)
    print(f"{len(rows)} 行を書きました", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
