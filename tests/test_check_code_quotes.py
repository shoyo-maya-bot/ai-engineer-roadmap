"""scripts/check_code_quotes.py のテスト（本文の引用と examples/ の実物の照合）."""

import check_code_quotes as cq
import pytest

SRC = "examples/src/roadmap_examples/sample.py"


def page(repo, body: str) -> list[str]:
    repo.write("docs/index.md", "# はじめに\n\n" + body)
    return cq.run(repo.root)[1]


def quote(code: str, lines: str = "", info: str = "") -> str:
    spec = f"#{lines}" if lines else ""
    return f"<!-- src: {SRC}{spec} -->\n```python{info}\n{code}\n```\n"


def test_whole_file_quote_matches_anywhere(repo):
    assert page(repo, quote("def greet(name: str) -> str:\n    return message", "")) != []
    assert page(repo, quote('def greet(name: str) -> str:\n    message = f"こんにちは、{name}"')) == []


def test_line_range_must_match_exactly(repo):
    assert page(repo, quote("def add(a: int, b: int) -> int:\n    return a + b", "L1-L2")) == []
    errs = page(repo, quote("def add(a: int, b: int) -> int:\n    return a + b", "L2-L3"))
    assert len(errs) == 1 and "行番号がずれています" in errs[0], errs


def test_indentation_and_blank_lines_are_ignored(repo):
    code = '    message = f"こんにちは、{name}"\n\n    return message'
    assert page(repo, quote(code.replace("    ", ""), "L6-L7")) == []


def test_elision_between_segments(repo):
    code = "def add(a: int, b: int) -> int:\n# ...（中略）\n    return message"
    assert page(repo, quote(code, "L1-L7")) == []
    # 範囲を書いたときは、最後の断片が範囲の終わりまで
    errs = page(repo, quote("def add(a: int, b: int) -> int:\n# ...\n    return a + b", "L1-L7"))
    assert len(errs) == 1, errs


def test_multiple_ranges(repo):
    code = "def add(a: int, b: int) -> int:\ndef greet(name: str) -> str:"
    assert page(repo, quote(code, "L1,L5")) == []


def test_changed_source_is_detected(repo):
    body = quote("def add(a: int, b: int) -> int:\n    return a + b", "L1-L2")
    assert page(repo, body) == []
    repo.write(SRC, repo.read(SRC).replace("return a + b", "return b + a"))
    errs = page(repo, body)
    assert len(errs) == 1 and "一致しません" in errs[0], errs


@pytest.mark.parametrize(
    ("marker", "message"),
    [
        ("<!-- src: scripts/x.py -->", "examples/ の下"),
        ("<!-- src: examples/../scripts/x.py -->", "examples/ の下"),
        ("<!-- src: examples/none.py -->", "がありません"),
        (f"<!-- src: {SRC}#L9-L99 -->", "行しかありません"),
        (f"<!-- src: {SRC}#2-3 -->", "書き方が違います"),
        (f"<!-- src: {SRC}#L3-L2 -->", "書き方が違います"),
    ],
)
def test_bad_markers(repo, marker, message):
    errs = page(repo, f"{marker}\n```python\ndef add(a: int, b: int) -> int:\n```\n")
    assert len(errs) == 1 and message in errs[0], errs


def test_linenums_and_title_must_agree(repo):
    code = "def add(a: int, b: int) -> int:"
    assert page(repo, quote(code, "L1", f' title="{SRC}" linenums="1"')) == []
    errs = page(repo, quote(code, "L1", ' linenums="2"'))
    assert len(errs) == 1 and "linenums" in errs[0], errs
    errs = page(repo, quote(code, "L1", ' title="examples/other.py"'))
    assert len(errs) == 1 and "title" in errs[0], errs


def test_examples_title_requires_marker(repo):
    errs = page(repo, f'```python title="{SRC}"\ndef add(a: int, b: int) -> int:\n```\n')
    assert len(errs) == 1 and "src コメントがありません" in errs[0], errs
    # 書き換えた例（title が examples/ で始まらない）は対象外
    assert page(repo, '```python title="例: 足し算"\ndef add(a, b): ...\n```\n') == []


def test_marker_without_block(repo):
    errs = page(repo, f"<!-- src: {SRC} -->\n\n本文\n")
    assert len(errs) == 1 and "直後にコードブロックがありません" in errs[0], errs


def test_quote_inside_admonition(repo):
    body = '!!! example "引用"\n    ' + quote("def add(a: int, b: int) -> int:", "L1").replace("\n", "\n    ")
    assert page(repo, body) == []


def test_repository_passes():
    assert cq.run(cq.ROOT)[1] == []
