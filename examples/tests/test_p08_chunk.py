"""p08_chunk.py のテスト（見出しで分け、長い節を段落で分け、長い段落は重ねて切る）."""

import pytest

from roadmap_examples.p08_chunk import chunk_markdown, load_chunks, split_long, split_sections, split_text
from roadmap_examples.p08_rag import DEFAULT_DOCS

DOC = """# 設計書: API

前置き

## 認証

トークンの有効期限は 30 分。

```bash
# これは見出しではない
```

## 制限

1 分間に 60 回まで。
"""


def test_split_sections_keeps_heading_path_and_ignores_code() -> None:
    sections = split_sections(DOC)
    assert [h for h, _ in sections] == ["設計書: API", "設計書: API > 認証", "設計書: API > 制限"]
    assert "# これは見出しではない" in sections[1][1]


def test_chunk_markdown_prefixes_heading_and_numbers_ids() -> None:
    chunks = chunk_markdown(DOC, "api.md")
    assert [c.id for c in chunks] == ["api.md#1", "api.md#2", "api.md#3"]
    assert chunks[2].text == "設計書: API > 制限\n1 分間に 60 回まで。"
    assert chunks[2].cite == "api.md > 設計書: API > 制限"


def test_split_text_groups_paragraphs_up_to_max() -> None:
    text = "\n\n".join(["あ" * 30, "い" * 30, "う" * 30])
    assert split_text(text, max_chars=70, overlap=10) == ["あ" * 30 + "\n\n" + "い" * 30, "う" * 30]


def test_split_long_overlaps() -> None:
    pieces = split_long("0123456789" * 3, max_chars=12, overlap=4)
    assert pieces[0] == "012345678901"
    assert pieces[1].startswith("8901")  # 前のかたまりの最後の 4 文字から始まる
    assert "".join(p[4:] if i else p for i, p in enumerate(pieces)) == "0123456789" * 3


def test_overlap_must_be_smaller_than_max() -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunk_markdown(DOC, "api.md", max_chars=10, overlap=10)


def test_design_docs_chunks() -> None:
    chunks = load_chunks(DEFAULT_DOCS)
    assert {c.source for c in chunks} == {"01-overview.md", "02-api.md", "03-batch.md", "04-data.md"}
    assert all(len(c.text) <= 400 + 100 for c in chunks)  # 本文は 400 文字以下（先頭の見出しの分だけ長い）


def test_crlf_is_treated_like_lf() -> None:
    assert chunk_markdown(DOC.replace("\n", "\r\n"), "api.md") == chunk_markdown(DOC, "api.md")
