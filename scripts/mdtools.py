"""検査スクリプトが共通で使う、教材の Markdown と設定の読み取り.

- ページは BOM 付き（UTF-8 with BOM）・CRLF でも読める
- コードブロックは ``` と ~~~ の両方. admonition の中（字下げあり）も読む
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FENCE = re.compile(r"^(?P<indent>\s*)(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
SRC_MARKER = re.compile(r"^\s*<!--\s*src:")
COMMENT = re.compile(r"^\s*<!--.*-->\s*$")
# 「どこまでが設定か」を探さずに済むよう、ディレクトリを走査するときに飛ばすもの
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", "site", "node_modules", ".cache"}


def read(path: Path) -> str:
    """BOM と CRLF をそろえて読む."""
    return path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")


def walk(base: Path, suffixes: tuple[str, ...]) -> list[Path]:
    """base の下のファイル（SKIP_DIRS を除く）. 並びは決まった順."""
    out = []
    for p in sorted(base.rglob("*")):
        if p.is_file() and p.suffix in suffixes and not SKIP_DIRS.intersection(p.relative_to(base).parts):
            out.append(p)
    return out


@dataclass
class Block:
    """コードブロック 1 つ."""

    path: Path
    line: int  # フェンスの行（1 始まり）
    lang: str  # info の最初の語（```python title="..." なら python）
    info: str
    body: list[str]  # フェンスの字下げを外した中身
    before: list[tuple[int, str]]  # 直前の HTML コメント行（近い順. 空行は飛ばす）

    def marker(self, pattern: re.Pattern[str]) -> tuple[int, str] | None:
        """直前のコメントのうち pattern に合う最初のもの（行番号, 行）."""
        return next(((n, s) for n, s in self.before if pattern.match(s)), None)


def parse_blocks(path: Path) -> list[Block]:
    lines = read(path).split("\n")
    blocks = []
    i = 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        indent, fence, info = m.group("indent"), m.group("fence"), m.group("info").strip()
        start = i
        body = []
        i += 1
        while i < len(lines):
            close = FENCE.match(lines[i])
            if close and close.group("fence").startswith(fence[0] * len(fence)) and not close.group("info").strip():
                break
            line = lines[i]
            body.append(line[len(indent) :] if line.startswith(indent) else line.lstrip())
            i += 1
        before = []
        j = start - 1
        while j >= 0 and (not lines[j].strip() or COMMENT.match(lines[j])):
            if lines[j].strip():
                before.append((j + 1, lines[j]))
            j -= 1
        lang = info.split()[0].lstrip("{.").rstrip("}") if info else ""
        blocks.append(Block(path, start + 1, lang.lower(), info, body, before))
        i += 1
    return blocks


def prose_lines(text: str) -> list[tuple[int, str]]:
    """コードブロックと HTML コメント行の外の行（行番号は 1 始まり）."""
    out = []
    fence: str | None = None
    for i, line in enumerate(text.split("\n"), 1):
        m = FENCE.match(line)
        if fence is None:
            if m:
                fence = m.group("fence")
                continue
            if not COMMENT.match(line):
                out.append((i, line))
        elif m and m.group("fence").startswith(fence[0] * len(fence)) and not m.group("info").strip():
            fence = None
    return out


def load_versions(mkdocs_yml: Path) -> dict[str, str]:
    """mkdocs.yml の extra.versions を読む（!!python/name などのタグがあるので YAML ライブラリは使わない）."""
    versions: dict[str, str] = {}
    in_extra = in_versions = False
    for line in read(mkdocs_yml).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            in_extra = line.startswith("extra:")
            in_versions = False
            continue
        if in_extra and indent == 2:
            in_versions = line.strip() == "versions:"
            continue
        if in_extra and in_versions and indent == 4:
            m = re.match(r"^\s*(\w+):\s*\"?([^\"#]*?)\"?\s*(#.*)?$", line)
            if m:
                versions[m.group(1)] = m.group(2)
    return versions
