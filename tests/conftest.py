"""検査スクリプトのテストの共通処理: scripts/ を import できるようにし、検査が通る最小のリポジトリを作る."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

HEADS = ["何か", "なぜ必要か", "どうやるか", "演習", "もし〇〇がなかったら？", "理解度チェック"]

MKDOCS = """\
site_name: テスト
markdown_extensions:
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
extra:
  versions:
    python: "3.12"
    uv: "0.8.17"
    pytest: "9.1.1"
    ruff: "0.16.10"
nav:
  - index.md
  - 01-python/index.md
  - 01-python/01-1-intro.md
  - 01-python/answers/01-1.md
"""

PYPROJECT = """\
[project]
name = "roadmap-examples"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []

[project.scripts]
api-to-csv = "roadmap_examples.p01_api_to_csv:main"

[dependency-groups]
dev = ["pytest==9.1.1", "ruff==0.16.10"]

[build-system]
requires = ["uv_build>=0.8.17,<0.9"]
build-backend = "uv_build"
"""

UV_LOCK = """\
version = 1
[[package]]
name = "pytest"
version = "9.1.1"
[[package]]
name = "ruff"
version = "0.16.10"
"""

WORKFLOW = """\
jobs:
  a:
    steps:
      - uses: actions/setup-python@0000000000000000000000000000000000000000 # v0
        with:
          python-version: "3.12"
      - run: pip install "uv==0.8.17"
"""

SAMPLE = """\
def add(a: int, b: int) -> int:
    return a + b


def greet(name: str) -> str:
    message = f"こんにちは、{name}"
    return message
"""


def chapter(heads: list[str] = HEADS, num: str = "01-1", extra: str = "") -> str:
    body = f'# {num} 章\n\n!!! abstract "到達目標"\n    - 書ける\n\n'
    body += "".join(f"## {h}\n\n本文\n\n" for h in heads)
    return body + f"解答: [{num} の解答](answers/{num}.md)\n" + extra


def part_index(rows: str = "| [01-1 入門](01-1-intro.md) | 内容 | 公開 |\n") -> str:
    return (
        '# 01 Python\n\n!!! abstract "到達目標"\n    - 書ける\n\n## 前提\n\nなし\n\n## 章の一覧\n\n'
        "| 章 | 内容 | 状態 |\n|---|---|---|\n" + rows + "\n## 成果物\n\nCLI\n\n## 目安時間\n\n16 時間\n"
    )


class Repo:
    """テスト用の最小のリポジトリ（作った時点で、どの検査も通る）."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.write("mkdocs.yml", MKDOCS)
        self.write(
            "banned-terms.txt",
            "# コメント\n株式会社\n@gmail.com  # 個人のメール\npath:/Users/\nre:\\bsk-[A-Za-z0-9]{20,}\n",
        )
        self.write("docs/index.md", "# はじめに\n")
        self.write("docs/01-python/index.md", part_index())
        self.write("docs/01-python/01-1-intro.md", chapter())
        self.write("docs/01-python/answers/01-1.md", "# 01-1 解答\n")
        self.write("examples/pyproject.toml", PYPROJECT)
        self.write("examples/.python-version", "3.12\n")
        self.write("examples/uv.lock", UV_LOCK)
        self.write("examples/src/roadmap_examples/sample.py", SAMPLE)
        self.write("requirements.txt", "mkdocs>=1.6\npytest==9.1.1\n")
        self.write(".github/workflows/ci.yml", WORKFLOW)

    def write(self, rel: str, text: str) -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        return path

    def read(self, rel: str) -> str:
        return (self.root / rel).read_text(encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    return Repo(tmp_path)
