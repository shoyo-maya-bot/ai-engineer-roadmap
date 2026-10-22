#!/usr/bin/env python3
"""本文に引用したコードが、examples/ の実物と一致するかを検査する.

引用したコードブロックの直前に、引用元をコメントで書く:

    <!-- src: examples/src/roadmap_examples/p01_api_to_csv.py#L52-L57 -->
    ```python title="examples/src/roadmap_examples/p01_api_to_csv.py" linenums="52"
    （引用したコード）
    ```

- パスはリポジトリ直下から書き、examples/ の下のファイルに限る
- `#L52-L57` の範囲を書くと、その行の中身と一致するか（行番号のずれ）まで確かめる.
  離れた行の抜粋は `#L10-L12,L20-L25` のように並べる. 範囲を省くと、ファイルのどこかにそのまま連続してあればよい
- 字下げと空行は比べない（メソッドの中身だけを字下げを外して引用してよい）
- 途中を飛ばすときは、その場所に `# ...` か `# ...（説明）` か `# （中略）` だけの行を置く
- `linenums="N"` を書いたら、範囲の始まりと同じにする
- title が examples/ で始まるブロックは、src コメントが必須（書き換えた例には examples/ の title を付けない）

ページは BOM 付き（UTF-8 with BOM）でも読める.

終了コード: 0 問題なし / 1 問題あり
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path, PurePosixPath

from mdtools import ROOT, Block, parse_blocks, read, walk

MARKER = re.compile(r"^\s*<!--\s*src:\s*(?P<path>[^\s#]+)(?:#(?P<lines>[L0-9,\-]+))?\s*-->\s*$")
MARKER_LOOSE = re.compile(r"^\s*<!--\s*src:")
RANGE = re.compile(r"^L(?P<a>\d+)(?:-L?(?P<b>\d+))?$")
ELISION = re.compile(r"^#\s*(?:\.\.\.|…)(?:\s*[（(].*[)）])?$|^#\s*（中略）$")
TITLE = re.compile(r'title="(?P<title>[^"]*)"')
LINENUMS = re.compile(r'linenums="(?P<n>\d+)"')
BASE = "examples/"


def parse_ranges(spec: str) -> list[tuple[int, int]] | None:
    """`L10-L12,L20` → [(10, 12), (20, 20)]. 書き方の誤りは None."""
    out = []
    for part in spec.split(","):
        m = RANGE.match(part.strip())
        if not m:
            return None
        a, b = int(m.group("a")), int(m.group("b") or m.group("a"))
        if a < 1 or b < a:
            return None
        out.append((a, b))
    return out


def norm(lines: list[str]) -> list[str]:
    """比べるための形（前後の空白を外し、空行を除く）."""
    return [s.strip() for s in lines if s.strip()]


def segments(body: list[str]) -> list[list[str]]:
    """引用を、飛ばした印（# ...）で区切った断片にする."""
    out: list[list[str]] = [[]]
    for line in norm(body):
        if ELISION.match(line):
            out.append([])
        else:
            out[-1].append(line)
    return [s for s in out if s]


def find(hay: list[str], needle: list[str], start: int) -> int:
    for i in range(start, len(hay) - len(needle) + 1):
        if hay[i : i + len(needle)] == needle:
            return i
    return -1


def match_in(target: list[str], segs: list[list[str]], exact: bool) -> bool:
    """断片がこの順に target にあるか. exact なら、最初の断片が先頭から、最後の断片が末尾まで（範囲を書いたとき）."""
    pos = 0
    for k, seg in enumerate(segs):
        at = find(target, seg, pos)
        if at < 0 or (exact and k == 0 and at != 0):
            return False
        pos = at + len(seg)
    return not exact or pos == len(target)


def check_block(block: Block, marker: tuple[int, str], root: Path) -> str | None:
    where = f"{block.path.relative_to(root).as_posix()}:{block.line}"
    m = MARKER.match(marker[1])
    if not m:
        return f"{where}: src コメントの書き方が違います（<!-- src: examples/...#L1-L5 -->）"
    path = m.group("path")
    if not path.startswith(BASE) or ".." in PurePosixPath(path).parts:
        return f"{where}: 引用元は {BASE} の下のファイルにしてください（{path}）"
    src = root / path
    if not src.is_file():
        return f"{where}: 引用元 {path} がありません"
    lines = read(src).split("\n")
    segs = segments(block.body)
    if not segs:
        return f"{where}: 引用したコードが空です"
    title = TITLE.search(block.info)
    if title and title.group("title") != path:
        return f"{where}: title（{title.group('title')}）と src（{path}）が違います"
    if m.group("lines") is None:
        if not match_in(norm(lines), segs, exact=False):
            return f"{where}: 引用したコードが {path} にありません（実物が変わった可能性があります）"
        return None
    ranges = parse_ranges(m.group("lines"))
    if ranges is None:
        return f"{where}: 行の範囲の書き方が違います（#L10-L12 や #L10-L12,L20-L25）"
    if ranges[-1][1] > len(lines):
        return f"{where}: {path} は {len(lines)} 行しかありません（#{m.group('lines')}）"
    linenums = LINENUMS.search(block.info)
    if linenums and int(linenums.group("n")) != ranges[0][0]:
        return f'{where}: linenums="{linenums.group("n")}" と範囲の始まり（L{ranges[0][0]}）が違います'
    target = [line for a, b in ranges for line in lines[a - 1 : b]]
    if not match_in(norm(target), segs, exact=True):
        hint = ""
        if match_in(norm(lines), segs, exact=False):
            hint = "。同じコードはファイルの別の行にあります（行番号がずれています）"
        return f"{where}: 引用したコードが {path}#{m.group('lines')} と一致しません{hint}"
    return None


def check_page(path: Path, root: Path) -> tuple[int, list[str]]:
    count = 0
    errs = []
    for block in parse_blocks(path):
        marker = block.marker(MARKER_LOOSE)
        title = TITLE.search(block.info)
        if marker is None:
            if title and title.group("title").startswith(BASE):
                rel = path.relative_to(root).as_posix()
                errs.append(f"{rel}:{block.line}: title が {BASE} で始まるのに src コメントがありません")
            continue
        count += 1
        err = check_block(block, marker, root)
        if err:
            errs.append(err)
    # コードブロックの直前に無い src コメント（書いたのに検査されない）
    text_lines = read(path).split("\n")
    used = {b.marker(MARKER_LOOSE)[0] for b in parse_blocks(path) if b.marker(MARKER_LOOSE)}
    for n, line in enumerate(text_lines, 1):
        if MARKER_LOOSE.match(line) and n not in used:
            errs.append(f"{path.relative_to(root).as_posix()}:{n}: src コメントの直後にコードブロックがありません")
    return count, errs


def run(root: Path = ROOT) -> tuple[int, list[str]]:
    total = 0
    errors: list[str] = []
    for page in walk(root / "docs", (".md",)):
        n, errs = check_page(page, root)
        total += n
        errors += errs
    return total, errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT, help="リポジトリの場所（テスト用）")
    args = ap.parse_args(argv)
    count, errors = run(args.root)
    for e in errors:
        print(f"NG {e}")
    print(f"引用 {count} 件 / 問題 {len(errors)} 件")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
