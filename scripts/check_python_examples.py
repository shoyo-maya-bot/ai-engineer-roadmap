#!/usr/bin/env python3
"""本文の Python のコード例（src コメントの無い ```python のブロック）が、構文として読めるかを検査する.

- 対象は info の最初の語が python / py / python3 のブロック. 対話の例（>>>）は ```pycon で書く（対象外）
- 構文は mkdocs.yml の extra.versions.python の版で読む（ast.parse の feature_version）.
  型・import・名前の有無は見ない（断片でよい. 中身を飛ばすときは `...` か `pass` と `# ...` を書く）
- src コメント付きのブロック（examples/ の実物の引用）は check_code_quotes.py が照合するので、ここでは見ない
- わざと構文が誤ったコードを見せるときだけ、直前に `<!-- python-syntax: skip 理由 -->` を書く（理由は必須）

使い方:
  python scripts/check_python_examples.py           # 検査
  python scripts/check_python_examples.py --list    # 対象のブロックと skip の一覧

終了コード: 0 問題なし / 1 問題あり
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import warnings
from pathlib import Path

from mdtools import ROOT, SRC_MARKER, Block, load_versions, parse_blocks, read, walk

LANGS = {"python", "py", "python3"}
SKIP = re.compile(r"^\s*<!--\s*python-syntax:\s*skip(?P<reason>[^>]*?)\s*-->\s*$")
SKIP_LOOSE = re.compile(r"^\s*<!--\s*python-syntax:")


def feature_version(version: str) -> tuple[int, int]:
    """'3.12' → (3, 12). 読めなければ ValueError."""
    m = re.fullmatch(r"(\d+)\.(\d+)", version.strip())
    if not m:
        raise ValueError(f"extra.versions.python（{version!r}）を 3.12 の形で書いてください")
    return int(m.group(1)), int(m.group(2))


def syntax_error(code: str, version: tuple[int, int]) -> str | None:
    """構文の誤り（無ければ None）."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)  # "\d" などの警告は誤りにしない
            ast.parse(code, feature_version=version)
    except SyntaxError as e:
        return f"{e.lineno} 行目: {e.msg}"
    return None


def targets(path: Path) -> list[Block]:
    return [b for b in parse_blocks(path) if b.lang in LANGS and b.marker(SRC_MARKER) is None]


def check_page(path: Path, root: Path, version: tuple[int, int]) -> tuple[int, int, list[str]]:
    """（検査した数, skip した数, 誤り）."""
    rel = path.relative_to(root).as_posix()
    checked = skipped = 0
    errs = []
    blocks = parse_blocks(path)
    marked = set()
    for block in blocks:
        mark = block.marker(SKIP_LOOSE)
        if mark:
            marked.add(mark[0])
        if block.lang not in LANGS or block.marker(SRC_MARKER):
            if mark:
                errs.append(f"{rel}:{mark[0]}: python-syntax: skip の後のブロックが Python のコード例ではありません")
            continue
        if mark:
            m = SKIP.match(mark[1])
            if not m or not m.group("reason").strip():
                errs.append(f"{rel}:{mark[0]}: python-syntax: skip には理由を書いてください（例: 誤りの例）")
            skipped += 1
            continue
        checked += 1
        err = syntax_error("\n".join(block.body), version)
        if err:
            errs.append(f"{rel}:{block.line}: Python の構文の誤り（{err}）")
    for n, line in enumerate(read(path).split("\n"), 1):
        if SKIP_LOOSE.match(line) and n not in marked:
            errs.append(f"{rel}:{n}: python-syntax: skip の直後にコードブロックがありません")
    return checked, skipped, errs


def run(root: Path = ROOT) -> tuple[int, int, list[str]]:
    try:
        version = feature_version(load_versions(root / "mkdocs.yml").get("python", ""))
    except ValueError as e:
        return 0, 0, [f"mkdocs.yml: {e}"]
    checked = skipped = 0
    errors: list[str] = []
    for page in walk(root / "docs", (".md",)):
        c, s, errs = check_page(page, root, version)
        checked += c
        skipped += s
        errors += errs
    return checked, skipped, errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT, help="リポジトリの場所（テスト用）")
    ap.add_argument("--list", action="store_true", help="対象のブロックと skip の一覧を出す")
    args = ap.parse_args(argv)
    if args.list:
        for page in walk(args.root / "docs", (".md",)):
            for b in targets(page):
                mark = b.marker(SKIP)
                state = f"skip\t{mark[1].strip()}" if mark else "check"
                print(f"{page.relative_to(args.root).as_posix()}:{b.line}\t{state}")
    checked, skipped, errors = run(args.root)
    for e in errors:
        print(f"NG {e}")
    print(f"Python のコード例 {checked} 件（skip {skipped} 件）/ 問題 {len(errors)} 件")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
