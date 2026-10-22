"""08-2 チャンク分割: Markdown を、見出しの区切りを保ったまま、検索に使う小さなかたまり（チャンク）に分ける.

1. 見出し（#）で節に分ける. 各チャンクに「どの見出しの下か」を残す（出典と、文脈の手がかりのため）
2. 節が長ければ、段落（空行）の区切りでまとめ直し、max_chars 以下にする
3. それでも長い段落は、max_chars ごとに切り、前のかたまりの終わりを overlap 文字だけ重ねる
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    id: str  # 例: 02-api.md#3
    source: str  # 文書のファイル名
    heading: str  # 例: 備品貸出システム 設計書: API > 呼び出しの回数の制限
    text: str  # 検索と生成に使う本文（見出しを先頭に付ける）

    @property
    def cite(self) -> str:
        """出典の表示（文書 > 見出し）."""
        return f"{self.source} > {self.heading}" if self.heading else self.source


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """（見出しの並び, 本文）の一覧. コードブロックの中の # は見出しとみなさない."""
    sections: list[tuple[str, str]] = []
    path: list[str] = []
    body: list[str] = []
    in_code = False

    def flush() -> None:
        text = "\n".join(body).strip()
        if text:
            sections.append((" > ".join(path), text))
        body.clear()

    for line in markdown.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        level = len(line) - len(line.lstrip("#"))
        if not in_code and 1 <= level <= 6 and line[level : level + 1] == " ":
            flush()
            path[:] = path[: level - 1] + [line[level:].strip()]
            continue
        body.append(line)
    flush()
    return sections


def split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    """max_chars ごとに切る. 次のかたまりは、前のかたまりの最後の overlap 文字から始める."""
    step = max_chars - overlap
    return [text[i : i + max_chars] for i in range(0, max(len(text) - overlap, 1), step)]


def split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """段落（空行の区切り）をまとめて max_chars 以下のかたまりにする. 長すぎる段落だけを切る."""
    pieces: list[str] = []
    current = ""
    for para in (p.strip() for p in text.split("\n\n")):
        if not para:
            continue
        if len(para) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces += split_long(para, max_chars, overlap)
        elif current and len(current) + 2 + len(para) > max_chars:
            pieces.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        pieces.append(current)
    return pieces


def chunk_markdown(markdown: str, source: str, max_chars: int = 400, overlap: int = 50) -> list[Chunk]:
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap は 0 以上、max_chars 未満にしてください")
    markdown = markdown.replace("\r\n", "\n")  # Windows の改行（CRLF）でも、空行で段落を分けられるように
    chunks: list[Chunk] = []
    for heading, body in split_sections(markdown):
        for piece in split_text(body, max_chars, overlap):
            text = f"{heading}\n{piece}" if heading else piece
            chunks.append(Chunk(f"{source}#{len(chunks) + 1}", source, heading, text))
    return chunks


def load_chunks(folder: Path, max_chars: int = 400, overlap: int = 50) -> list[Chunk]:
    """フォルダの Markdown（*.md）を全部チャンクにする（ファイル名の順）."""
    chunks: list[Chunk] = []
    for path in sorted(folder.glob("*.md")):
        chunks += chunk_markdown(path.read_text(encoding="utf-8-sig"), path.name, max_chars, overlap)
    return chunks
