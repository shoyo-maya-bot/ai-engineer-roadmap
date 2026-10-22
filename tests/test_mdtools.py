"""scripts/mdtools.py のテスト（HTML コメントの扱い. 改行をまたぐコメントを含む）.

後退: HTML コメントを「1 行で閉じるもの」だけの正規表現で探していたので、改行をまたぐコメントの中身を
本文として扱っていた（CodeQL: Bad HTML filtering regexp）.
"""

import mdtools as md
import pytest


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("<!-- src: examples/a.py -->", True),
        ("  <!-- 注記 -->  ", True),
        ("<!-- 始まりだけ", False),
        ("本文 <!-- 途中 -->", False),
        ("<!-- a --> 本文 <!-- b -->", False),
        ("<!---->", True),
    ],
)
def test_is_comment_line(line, expected):
    assert md.is_comment_line(line) is expected


def test_strip_comments_across_lines():
    assert md.strip_comments("前 <!-- 中 --> 後", False) == ("前  後", False)
    assert md.strip_comments("前 <!-- 続く", False) == ("前 ", True)
    assert md.strip_comments("まだ中", True) == ("", True)
    assert md.strip_comments("終わり --> 後 <!-- また -->!", True) == (" 後 !", False)


def test_prose_lines_skip_multiline_comments_and_code():
    text = "\n".join(
        [
            "# 見出し",  # 1
            "<!--",  # 2
            "Python 3.11 と書いたメモ",  # 3
            "-->",  # 4
            "本文 <!-- 1 行の",  # 5
            "続き --> 本文の続き",  # 6
            "```python",  # 7
            "x = 1  # <!-- コードの中は本文ではない",  # 8
            "```",  # 9
            "<!-- 1 行のコメント -->",  # 10
            "",  # 11
            "最後",  # 12
        ]
    )
    assert md.prose_lines(text) == [(1, "# 見出し"), (5, "本文 "), (6, " 本文の続き"), (11, ""), (12, "最後")]


def test_fence_inside_multiline_comment_is_not_code():
    text = "<!--\n```python\nx = (\n```\n-->\n本文\n"
    assert md.prose_lines(text) == [(6, "本文"), (7, "")]


def test_markers_before_blocks_skip_multiline_comments(tmp_path):
    page = tmp_path / "p.md"
    page.write_text(
        "本文\n\n<!-- src: examples/a.py -->\n<!--\n  説明のメモ\n-->\n\n```python\nx = 1\n```\n",
        encoding="utf-8",
    )
    (block,) = md.parse_blocks(page)
    assert block.before == [(3, "<!-- src: examples/a.py -->")]
    assert block.marker(md.SRC_MARKER) == (3, "<!-- src: examples/a.py -->")


def test_text_ending_with_comment_close_is_not_a_marker(tmp_path):
    page = tmp_path / "p.md"
    page.write_text("本文の最後が --> で終わる\n```python\nx = 1\n```\n", encoding="utf-8")
    (block,) = md.parse_blocks(page)
    assert block.before == []
