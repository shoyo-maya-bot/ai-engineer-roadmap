#!/usr/bin/env python3
"""README・CONTRIBUTING などの説明が、リポジトリの実物とずれていないかを検査する.

対象: DOC_FILES（人や AI が読む説明）と .github/workflows（スクリプトの呼び出しだけ）.

1. 書いたスクリプト（`scripts/xxx.py`）が存在する
2. `python scripts/xxx.py --opt` のオプションが、そのスクリプトに存在する
3. コード（`...` と ``` の中）とリンク先に書いたファイルの場所（`examples/uv.lock` など）が存在する.
   `<部>`・`NN-N`・`*` のような書き方の見本は見ない. わざと無いファイル（`.env`）は MISSING_OK
4. 書いた版（「Python 3.12」「uv==0.8.17」「ruff 0.16」など）が mkdocs.yml の extra.versions と一致する
5. `uv run <コマンド>` のコマンドが、入っているツールか examples/pyproject.toml の [project.scripts] にある
6. scripts/check_*.py がどれも、README・CONTRIBUTING・AGENTS に書かれ、ci.yml で実行されている（書き漏れ・入れ漏れ）

終了コード: 0 問題なし / 1 問題あり
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

from mdtools import FENCE, ROOT, load_versions, prose_lines, read, walk

DOC_FILES = [
    "README.md",
    "CONTRIBUTING.md",
    "AGENTS.md",
    "SECURITY.md",
    ".github/copilot-instructions.md",
    ".github/pull_request_template.md",
    "examples/README.md",
]
MUST_LIST_CHECKS = ["README.md", "CONTRIBUTING.md", "AGENTS.md"]
CI = ".github/workflows/ci.yml"
MISSING_OK = {".env"}
UV_RUN_TOOLS = {"pytest", "ruff", "python", "mkdocs"}
VERSION_KEYS = ("python", "uv", "pytest", "ruff")

SCRIPT = re.compile(r"scripts/[\w./-]+\.py")
SCRIPT_CALL = re.compile(r"python\s+(?P<script>scripts/[\w-]+\.py)(?P<args>[^#&|;\n]*)")
OPTION = re.compile(r"(?<![\w-])--[a-z][\w-]*")
INLINE_CODE = re.compile(r"`([^`\n]+)`")
LINK = re.compile(r"\]\((?P<target>[^)\s]+)\)")
PATH = re.compile(
    r"(?<![\w./:@-])(?P<path>(?:\.?[\w-]+/)*\.?[\w.-]+\.(?:md|py|txt|toml|yml|yaml|ini|lock|json)|"
    r"(?:docs|scripts|tests|examples|\.github)/(?:[\w.-]+/)*)(?![\w/])"
)
PLACEHOLDER = re.compile(r"[<>*{}]|NN|xx")
# 章・解答のファイル名の見本（docs/01-python/01-4-xxx.md・answers/01-4.md）. 章の有無は check_chapters.py が見る
CHAPTER_SAMPLE = re.compile(r"(?:^|/)\d{2}-\d{1,2}(?:-[\w-]+)?\.md$")
UV_RUN = re.compile(r"\buv\s+run\s+(?:--[\w-]+(?:\s+\S+)?\s+)*(?P<cmd>[\w.-]+)")


def version_pattern(key: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![\w-]){key}\s*(?:==\s*|v)?(?P<v>\d+\.\d+(?:\.\d+)?)(?![\d.]*\d)", re.IGNORECASE)


def code_texts(text: str) -> list[str]:
    """コードとして書いた部分（インラインコードとコードブロックの中身）."""
    out = []
    fence = None
    for line in text.split("\n"):
        m = FENCE.match(line)
        if fence is None:
            if m:
                fence = m.group("fence")
                continue
            out += INLINE_CODE.findall(line)
        elif m and m.group("fence").startswith(fence[0] * len(fence)) and not m.group("info").strip():
            fence = None
        else:
            out.append(line)
    return out


def path_exists(path: str, doc: Path, root: Path) -> bool:
    p = path.rstrip("/")
    return any((base / p).exists() for base in (root, doc.parent, root / "examples", root / "scripts"))


def check_doc(doc: Path, root: Path, versions: dict[str, str], project_scripts: set[str]) -> list[str]:
    rel = doc.relative_to(root).as_posix()
    text = read(doc)
    errs = []
    for script in sorted(set(SCRIPT.findall(text))):
        if not (root / script).is_file():
            errs.append(f"{rel}: {script} がありません")
    for m in SCRIPT_CALL.finditer(text):
        src = root / m.group("script")
        if not src.is_file():
            continue
        source = read(src)
        for opt in OPTION.findall(m.group("args")):
            if f'"{opt}"' not in source:
                errs.append(f"{rel}: {m.group('script')} にオプション {opt} がありません")
    candidates = []
    for chunk in code_texts(text):
        # URL の中の「/README.md」などは PATH の先読みで外れる（直前が / や英数字のときは当てない）
        candidates += [p.group("path") for p in PATH.finditer(chunk)]
    for link in (m for _, line in prose_lines(text) for m in LINK.finditer(line)):
        target = link.group("target").split("#", 1)[0]
        if target and not re.match(r"^[a-z]+:", target):
            candidates.append(target)
    for path in sorted(set(candidates)):
        if path in MISSING_OK or PLACEHOLDER.search(path) or CHAPTER_SAMPLE.search(path) or SCRIPT.fullmatch(path):
            continue
        if not path_exists(path, doc, root):
            errs.append(f"{rel}: {path} がありません（場所が変わった・名前が変わった可能性があります）")
    for key in VERSION_KEYS:
        want = versions.get(key)
        if not want:
            continue
        for m in version_pattern(key).finditer(text):
            got = m.group("v")
            if got != want and not want.startswith(got + "."):
                errs.append(f"{rel}: {m.group(0)} と書いていますが、extra.versions.{key} は {want} です")
    for m in UV_RUN.finditer(text):
        cmd = m.group("cmd")
        if cmd not in UV_RUN_TOOLS and cmd not in project_scripts:
            errs.append(f"{rel}: uv run {cmd} のコマンドがありません（examples/pyproject.toml の [project.scripts]）")
    return errs


def project_scripts(root: Path) -> set[str]:
    try:
        return set(tomllib.loads(read(root / "examples" / "pyproject.toml")).get("project", {}).get("scripts", {}))
    except (OSError, tomllib.TOMLDecodeError):
        return set()


def check_coverage(root: Path) -> list[str]:
    errs = []
    checks = sorted(p.name for p in (root / "scripts").glob("check_*.py"))
    for name in MUST_LIST_CHECKS:
        path = root / name
        text = read(path) if path.is_file() else ""
        errs += [f"{name}: scripts/{c} の説明がありません" for c in checks if f"scripts/{c}" not in text]
    ci = root / CI
    ci_text = read(ci) if ci.is_file() else ""
    errs += [f"{CI}: scripts/{c} を実行していません" for c in checks if f"scripts/{c}" not in ci_text]
    return errs


def run(root: Path = ROOT) -> tuple[int, list[str]]:
    versions = load_versions(root / "mkdocs.yml")
    scripts = project_scripts(root)
    docs = [root / f for f in DOC_FILES if (root / f).is_file()]
    errors = [f"{f}: ありません" for f in DOC_FILES if not (root / f).is_file()]
    for doc in docs:
        errors += check_doc(doc, root, versions, scripts)
    for wf in walk(root / ".github" / "workflows", (".yml", ".yaml")):
        rel = wf.relative_to(root).as_posix()
        errors += [
            f"{rel}: {s} がありません" for s in sorted(set(SCRIPT.findall(read(wf)))) if not (root / s).is_file()
        ]
    errors += check_coverage(root)
    return len(docs), errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT, help="リポジトリの場所（テスト用）")
    args = ap.parse_args(argv)
    count, errors = run(args.root)
    for e in errors:
        print(f"NG {e}")
    print(f"説明のファイル {count} 件 / 問題 {len(errors)} 件")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
