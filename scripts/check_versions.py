#!/usr/bin/env python3
"""教材の本文に書く版（Python・uv・pytest・ruff）を検査する.

本文の版は mkdocs.yml の extra.versions の 1 か所にまとめ、本文では {{ versions.python }} のように参照する
（mkdocs-macros。参照するページは先頭に render_macros: true を書く）。このスクリプトは次を確かめる:

1. extra.versions の値が、演習のコードと CI で実際に使う版と一致する
   - python: examples/pyproject.toml の requires-python（>=X.Y）・examples/.python-version・
             .github/workflows の python-version
   - pytest: examples/pyproject.toml の dev（pytest==）・examples/uv.lock・requirements.txt・constraints.txt
   - ruff:   examples/pyproject.toml の dev（ruff==）・examples/uv.lock・.github/workflows の ruff==
   - uv:     .github/workflows の uv==・examples/pyproject.toml の uv_build>=・
             examples/Dockerfile の ghcr.io/astral-sh/uv:（あれば）
2. 本文（コードブロックと HTML コメントの外）に、版を直接書いていない（「Python 3.12」「uv 0.8」「pytest 9.1」など）
3. {{ versions.* }} を使うページに render_macros: true があり、参照する名前が extra.versions にある

コードブロックの中の版（`requires-python = ">=3.12"` の引用など）は対象外です（引用は check_code_quotes.py が照合する）.

終了コード: 0 問題なし / 1 問題あり
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

from mdtools import ROOT, load_versions, prose_lines, read, walk

KEYS = ("python", "uv", "pytest", "ruff")
MACRO = re.compile(r"\{\{\s*versions\.(\w+)\s*\}\}")
FRONT_MATTER_MACROS = re.compile(r"\A---\n(?:.*\n)*?render_macros:\s*true\n(?:.*\n)*?---\n")
# 本文に直接書いてはいけない版の書き方（コードブロックの外）
LITERAL = re.compile(
    r"\bPython\s*3\.\d+"  # 「Python 3.12」
    r"|\bpython3\.\d+"  # 「python3.12」
    r"|\buv\s*v?\d+\.\d+"  # 「uv 0.8」
    r"|\bpytest\s*v?\d+\.\d+"  # 「pytest 9.1」
    r"|\bruff\s*v?\d+\.\d+",  # 「ruff 0.16」
    re.IGNORECASE,
)


def pinned(requirements: list[str], name: str) -> str | None:
    """['pytest==9.1.1', ...] から name の版."""
    for req in requirements:
        m = re.fullmatch(rf"\s*{re.escape(name)}\s*==\s*([\w.]+)\s*", req)
        if m:
            return m.group(1)
    return None


def lock_version(lock: dict, name: str) -> str | None:
    return next((p.get("version") for p in lock.get("package", []) if p.get("name") == name), None)


def file_pins(text: str, name: str) -> list[str]:
    """`name==1.2.3` の形をすべて（requirements.txt・constraints.txt・ワークフローの pip install）."""
    return re.findall(rf"(?<![\w-]){re.escape(name)}==([\w.]+)", text)


def actual_versions(root: Path) -> tuple[dict[str, list[tuple[str, str | None]]], list[str]]:
    """{名前: [(どこ, 版)]} と、読めなかったファイルの誤り."""
    errors: list[str] = []
    found: dict[str, list[tuple[str, str | None]]] = {k: [] for k in KEYS}
    examples = root / "examples"

    try:
        project = tomllib.loads(read(examples / "pyproject.toml"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        errors.append(f"examples/pyproject.toml を読めません（{e}）")
        project = {}
    dev = project.get("dependency-groups", {}).get("dev", [])
    req_python = project.get("project", {}).get("requires-python", "")
    m = re.fullmatch(r">=\s*(\d+\.\d+)(?:\.\d+)?", req_python)
    found["python"].append(("examples/pyproject.toml の requires-python", m.group(1) if m else None))
    found["pytest"].append(("examples/pyproject.toml の dev", pinned(dev, "pytest")))
    found["ruff"].append(("examples/pyproject.toml の dev", pinned(dev, "ruff")))
    build = project.get("build-system", {}).get("requires", [])
    uv_build = next((re.search(r">=\s*([\d.]+)", r) for r in build if r.startswith("uv_build")), None)
    found["uv"].append(("examples/pyproject.toml の uv_build", uv_build.group(1) if uv_build else None))

    pyver = examples / ".python-version"
    found["python"].append(("examples/.python-version", read(pyver).strip() if pyver.is_file() else None))

    try:
        lock = tomllib.loads(read(examples / "uv.lock"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        errors.append(f"examples/uv.lock を読めません（{e}）")
        lock = {}
    found["pytest"].append(("examples/uv.lock", lock_version(lock, "pytest")))
    found["ruff"].append(("examples/uv.lock", lock_version(lock, "ruff")))

    for name in ("requirements.txt", "constraints.txt"):
        path = root / name
        if path.is_file():
            pins = file_pins(read(path), "pytest")
            found["pytest"].append((name, pins[0] if pins else None))

    dockerfile = examples / "Dockerfile"
    if dockerfile.is_file():
        for v in re.findall(r"ghcr\.io/astral-sh/uv:([\w.]+)", read(dockerfile)):
            found["uv"].append(("examples/Dockerfile の ghcr.io/astral-sh/uv", v))

    for wf in walk(root / ".github" / "workflows", (".yml", ".yaml")):
        text = read(wf)
        rel = wf.relative_to(root).as_posix()
        for v in re.findall(r"python-version:\s*\"?([\d.]+)\"?", text):
            found["python"].append((f"{rel} の python-version", v))
        for v in file_pins(text, "ruff"):
            found["ruff"].append((f"{rel} の ruff==", v))
        for v in file_pins(text, "uv"):
            found["uv"].append((f"{rel} の uv==", v))
    return found, errors


def check_sources(versions: dict[str, str], root: Path) -> list[str]:
    found, errors = actual_versions(root)
    for key in KEYS:
        want = versions.get(key)
        if not want:
            errors.append(f"mkdocs.yml: extra.versions.{key} がありません")
            continue
        for where, got in found[key]:
            if got is None:
                errors.append(f"{where}: {key} の版を読み取れません")
            elif got != want:
                errors.append(f"mkdocs.yml: extra.versions.{key} は {want} ですが、{where} は {got} です")
    return errors


def literal_pattern(versions: dict[str, str]) -> re.Pattern[str]:
    """LITERAL と、extra.versions の x.y.z の値そのもの（「9.1.1 を使う」のような書き方）."""
    values = [v for k, v in versions.items() if k in KEYS and v.count(".") >= 2]
    extra = "".join(rf"|(?<![\d.]){re.escape(v)}(?![\d.])" for v in values)
    return re.compile(LITERAL.pattern + extra, re.IGNORECASE)


def check_docs(docs: Path, versions: dict[str, str]) -> list[str]:
    errors = []
    literal = literal_pattern(versions)
    for doc in walk(docs, (".md",)):
        text = read(doc)
        rel = doc.relative_to(docs).as_posix()
        used = MACRO.findall(text)
        if used and not FRONT_MATTER_MACROS.match(text):
            errors.append(f"{rel}: {{{{ versions.* }}}} を使っているのに、先頭に render_macros: true がありません")
        for name in sorted(set(used)):
            if name not in versions:
                errors.append(f"{rel}: versions.{name} は mkdocs.yml の extra.versions にありません")
        for line_no, line in prose_lines(text):
            for m in literal.finditer(MACRO.sub("", line)):
                errors.append(
                    f"{rel}:{line_no}: 版を直接書いています（{m.group(0)}）。"
                    f"{{{{ versions.* }}}} で書いてください（値は mkdocs.yml の extra.versions）"
                )
    return errors


def run(root: Path = ROOT) -> tuple[dict[str, str], list[str]]:
    versions = load_versions(root / "mkdocs.yml")
    return versions, check_docs(root / "docs", versions) + check_sources(versions, root)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT, help="リポジトリの場所（テスト用）")
    args = ap.parse_args(argv)
    versions, errors = run(args.root)
    for e in errors:
        print(f"NG {e}")
    shown = " / ".join(f"{k} {versions.get(k, '?')}" for k in KEYS)
    print(f"版（{shown}）/ 問題 {len(errors)} 件")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
