#!/usr/bin/env python3
"""教材の構成（CONTRIBUTING.md の「章の型」）と禁止語を検査する.

章（docs/<NN>-<名前>/<NN>-<N>-<英小文字と数字とハイフン>.md. 例: docs/01-python/01-1-python-basics.md）:
- 置き場所の部（フォルダの先頭の 2 桁）と、ファイル名の先頭の 2 桁が同じ
- 到達目標（!!! abstract "到達目標"）と、必須の見出し（REQUIRED）がこの順にある
- 解答ページ（同じ部の answers/<NN>-<N>.md）が存在し、本文からリンクしている
- 部の index.md の「章の一覧」から、リンクされている

部の index.md（docs/<NN>-<名前>/index.md）:
- 到達目標と、必須の見出し（INDEX_REQUIRED. 13 実践プロジェクトは「課題」「提出物」でもよい）がある
- 「章の一覧」の表の各行（先頭が NN-N）は、章のファイルがあればそこへのリンク、無ければ「準備中」と書く

全体:
- 解答ページだけがあって章が無いものは誤り
- docs/ のすべての .md が mkdocs.yml の nav にある（nav に無いページは mkdocs build --strict でも失敗しない）
- docs/・examples/・リポジトリ直下と .github/ の Markdown に、禁止語（banned-terms.txt）が無い.
  禁止語のリストが読めない・空のときも失敗にする（検査が黙って無効にならないように）

終了コード: 0 問題なし / 1 問題あり
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

from mdtools import ROOT, read, walk

REQUIRED = ["何か", "なぜ必要か", "どうやるか", "演習", "なかったら？", "理解度チェック"]
INDEX_REQUIRED: list[tuple[str, ...]] = [("前提",), ("章の一覧", "課題"), ("成果物", "提出物"), ("目安時間",)]
GOAL = '!!! abstract "到達目標"'
PART_DIR = re.compile(r"^(?P<part>\d{2})-[a-z0-9]+(?:-[a-z0-9]+)*$")
CHAPTER = re.compile(r"^(?P<num>(?P<part>\d{2})-\d{1,2})-[a-z0-9]+(?:-[a-z0-9]+)*\.md$")
ANSWER = re.compile(r"^(?P<num>(?P<part>\d{2})-\d{1,2})\.md$")
CHAPTER_ROW = re.compile(r"^\|\s*\[?(?P<num>\d{2}-\d{1,2})\s")
PENDING = "準備中"
PATH_PREFIX = "path:"
REGEX_PREFIX = "re:"
# ローカルパスが始まる位置: 行頭・空白・クォート・バッククォート・括弧・= の直後
PATH_START = r"(?:^|(?<=[\s'\"`(（「\[<=]))"
# 禁止語を探すファイル（リポジトリ直下からの場所, 拡張子）
BANNED_TARGETS = [("docs", (".md",)), ("examples", (".md", ".py", ".toml")), (".github", (".md",))]


def headings(text: str) -> list[str]:
    return re.findall(r"^## (.+)$", text, re.M)


def check_order(rel: str, heads: list[str], required: list[tuple[str, ...]]) -> list[str]:
    """必須の見出しが、この順であるか（見出しの中に語を含めばよい）."""
    errs = []
    pos = -1
    for names in required:
        idx = next((i for i, h in enumerate(heads) if any(n in h for n in names)), None)
        label = "」か「".join(names)
        if idx is None:
            errs.append(f"{rel}: 見出し「{label}」がありません")
        elif idx < pos:
            errs.append(f"{rel}: 見出し「{label}」の順番が違います")
        else:
            pos = idx
    return errs


def check_chapter(path: Path, docs: Path) -> list[str]:
    rel = path.relative_to(docs).as_posix()  # Windows でも / 区切りで出す
    text = read(path)
    m = CHAPTER.match(path.name)
    if not m:
        return [f"{rel}: 章のファイル名は <NN>-<N>-<英小文字と数字とハイフン>.md にしてください"]
    errs = []
    part = PART_DIR.match(path.parent.name)
    if path.parent.parent != docs or not part:
        errs.append(f"{rel}: 章は docs/<NN>-<名前>/ の直下に置いてください")
    elif part.group("part") != m.group("part"):
        errs.append(f"{rel}: ファイル名の部（{m.group('part')}）と置き場所の部（{part.group('part')}）が違います")
    if GOAL not in text:
        errs.append(f"{rel}: 到達目標（{GOAL}）がありません")
    errs += check_order(rel, headings(text), [(h,) for h in REQUIRED])
    num = m.group("num")
    if f"answers/{num}.md" not in text:
        errs.append(f"{rel}: 解答ページ answers/{num}.md へのリンクがありません")
    if not (path.parent / "answers" / f"{num}.md").is_file():
        errs.append(f"{rel}: 解答ページ answers/{num}.md がありません")
    return errs


def check_part_index(part_dir: Path, docs: Path) -> list[str]:
    index = part_dir / "index.md"
    rel = index.relative_to(docs).as_posix()
    if not index.is_file():
        return [f"{rel}: 部の index.md がありません"]
    text = read(index)
    errs = []
    if GOAL not in text:
        errs.append(f"{rel}: 到達目標（{GOAL}）がありません")
    errs += check_order(rel, headings(text), INDEX_REQUIRED)
    part = PART_DIR.match(part_dir.name).group("part")
    chapters = {CHAPTER.match(p.name).group("num"): p.name for p in part_dir.glob("*.md") if CHAPTER.match(p.name)}
    listed = set()
    for lineno, line in enumerate(text.split("\n"), 1):
        row = CHAPTER_ROW.match(line)
        if not row:
            continue
        num = row.group("num")
        listed.add(num)
        if not num.startswith(part + "-"):
            errs.append(f"{rel}:{lineno}: 章 {num} は部 {part} のものではありません")
        elif num in chapters:
            if f"({chapters[num]})" not in line:
                errs.append(f"{rel}:{lineno}: 章 {num} は書き上がっています。{chapters[num]} へのリンクにしてください")
        elif PENDING not in line:
            errs.append(f"{rel}:{lineno}: 章 {num} のファイルがありません（まだなら「{PENDING}」と書く）")
    for num, name in sorted(chapters.items()):
        if num not in listed:
            errs.append(f"{rel}: 章の一覧に {name} がありません")
    return errs


def check_answers(docs: Path) -> list[str]:
    errs = []
    for ans in sorted(docs.glob("*/answers/*.md")):
        m = ANSWER.match(ans.name)
        if ans.name == "index.md":
            continue
        rel = ans.relative_to(docs).as_posix()
        if not m:
            errs.append(f"{rel}: 解答ページの名前は <NN>-<N>.md にしてください")
        elif not list(ans.parent.parent.glob(f"{m.group('num')}-*.md")):
            errs.append(f"{rel}: 対応する章 {m.group('num')}-*.md がありません")
    return errs


def check_nav(docs: Path, mkdocs_yml: Path) -> list[str]:
    nav = read(mkdocs_yml).split("\nnav:", 1)
    if len(nav) < 2:
        return ["mkdocs.yml: nav がありません"]
    listed = set(re.findall(r"[\w./-]+\.md", nav[1]))
    return [
        f"{p.relative_to(docs).as_posix()}: mkdocs.yml の nav にありません"
        for p in walk(docs, (".md",))
        if p.relative_to(docs).as_posix() not in listed
    ]


def load_banned(path: Path) -> list[str]:
    """禁止語のリスト（1 行 1 語. 行頭の # と、語の後ろの「 #」から行末はコメント）. 読めない・空なら ValueError."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as e:
        raise ValueError(f"禁止語のリスト {path.name} が読めません（{e.strerror}）") from e
    terms = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        term = line.split(" #", 1)[0].strip()
        if term:
            terms.append(term)
    if not terms:
        raise ValueError(f"禁止語のリスト {path.name} が空です（検査が無効になります）")
    for term in terms:
        if term.startswith(REGEX_PREFIX):
            try:
                re.compile(term[len(REGEX_PREFIX) :])
            except re.error as e:
                raise ValueError(f"禁止語のリスト {path.name}: 正規表現 {term} が誤っています（{e}）") from e
    return terms


def nfkc(text: str) -> str:
    """全角・半角などの表記ゆれをそろえる（本文も禁止語もこれを通してから比べる）."""
    return unicodedata.normalize("NFKC", text)


def banned_matcher(term: str) -> tuple[str, re.Pattern[str]]:
    """禁止語 1 つを（表示名, 正規表現）にする（書き方は banned-terms.txt の先頭）."""
    if term.startswith(REGEX_PREFIX):
        return term, re.compile(term[len(REGEX_PREFIX) :])
    if term.startswith(PATH_PREFIX):
        word = term[len(PATH_PREFIX) :].strip()
        return word, re.compile(PATH_START + re.escape(nfkc(word)))
    return term, re.compile(re.escape(nfkc(term)), re.IGNORECASE)


def check_banned(path: Path, matchers: list[tuple[str, re.Pattern[str]]], root: Path) -> list[str]:
    rel = path.relative_to(root).as_posix()
    errs = []
    for lineno, line in enumerate(read(path).split("\n"), 1):
        norm = nfkc(line)
        errs += [f"{rel}:{lineno}: 禁止語「{name}」があります" for name, pat in matchers if pat.search(norm)]
    return errs


def banned_targets(root: Path) -> list[Path]:
    files = sorted(root.glob("*.md"))
    for sub, suffixes in BANNED_TARGETS:
        if (root / sub).is_dir():
            files += walk(root / sub, suffixes)
    return files


def run(root: Path = ROOT) -> tuple[int, list[str]]:
    """（章の数, 誤り）."""
    docs = root / "docs"
    try:
        matchers = [banned_matcher(t) for t in load_banned(root / "banned-terms.txt")]
    except ValueError as e:
        return 0, [str(e)]
    chapters = [p for p in walk(docs, (".md",)) if CHAPTER.match(p.name) and "answers" not in p.parts]
    errors = [e for p in chapters for e in check_chapter(p, docs)]
    for part_dir in sorted(p for p in docs.iterdir() if p.is_dir() and PART_DIR.match(p.name)):
        errors += check_part_index(part_dir, docs)
    errors += check_answers(docs)
    errors += check_nav(docs, root / "mkdocs.yml")
    errors += [e for p in banned_targets(root) for e in check_banned(p, matchers, root)]
    return len(chapters), errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT, help="リポジトリの場所（テスト用）")
    args = ap.parse_args(argv)
    count, errors = run(args.root)
    for e in errors:
        print(f"NG {e}")
    print(f"章 {count} / 問題 {len(errors)} 件")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
