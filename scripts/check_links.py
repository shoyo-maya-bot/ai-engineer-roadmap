#!/usr/bin/env python3
"""公開するサイト（mkdocs build の出力）の外部リンクが切れていないかを検査する.

外部のサイトの都合で結果が変わるため、push や PR では動かさず、週 1 回の定期実行（ci.yml の links）だけで使う。
切れていたら定期実行が Issue を起票する。

- `site/` の HTML の `<a href="http(s)://...">` を集める
  （`{{ versions.* }}` の展開後の URL を見るため、Markdown ではなく HTML を読む）.
  サイト自身へのリンク（site_url）は、公開前は切れて見えるので見ない
- localhost・`example.com` などの例示用のホスト、`<` `{` を含む書きかけの URL は見ない
- 同じ URL（`#` 以降を除く）は 1 回だけ確かめる。HEAD で断られたら GET で確かめる
- 「切れている」: 404・410 などの 4xx（401・403・429 を除く）、ホスト名が引けない、証明書の誤り
- 「確認できない」: 401・403・429（ロボットを断るサイト）、5xx・時間切れ（再試行の後も）. 失敗にはせず、一覧に出す

使い方:
  mkdocs build --strict
  python scripts/check_links.py site                       # 切れたリンクがあれば終了コード 1
  python scripts/check_links.py site --report links.md     # 結果を Markdown でも書く（Issue の本文に使う）

終了コード: 0 切れたリンクなし / 1 切れたリンクあり / 2 引数の誤り
"""

from __future__ import annotations

import argparse
import ipaddress
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urldefrag, urlsplit

USER_AGENT = (
    "Mozilla/5.0 (compatible; ai-engineer-roadmap-link-check; +https://github.com/shoyo-maya-bot/ai-engineer-roadmap)"
)
EXAMPLE_HOSTS = {"localhost"}
EXAMPLE_SUFFIXES = (".example", ".test", ".invalid", ".localhost", ".local")
EXAMPLE_DOMAINS = ("example.com", "example.org", "example.net")
UNSURE_STATUS = {401, 403, 429}
RETRIES = 2
# サイト自身（mkdocs.yml の site_url）. 公開前や Pages を有効にする前は切れて見えるので見ない
SELF_PREFIXES = ("https://shoyo-maya-bot.github.io/ai-engineer-roadmap/",)


class _Anchors(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def is_example(url: str) -> bool:
    """検査しない URL（例示用のホスト・書きかけ）."""
    if "<" in url or "{" in url:
        return True
    host = (urlsplit(url).hostname or "").lower()
    if not host or host in EXAMPLE_HOSTS or host.endswith(EXAMPLE_SUFFIXES):
        return True
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_loopback or ip.is_private or ip.is_unspecified or ip.is_link_local
    except ValueError:
        pass
    return any(host == d or host.endswith("." + d) for d in EXAMPLE_DOMAINS)


def collect(site: Path) -> dict[str, set[str]]:
    """{URL（# 以降を除く）: そのリンクがあるページ（site からの相対パス）}."""
    out: dict[str, set[str]] = {}
    for page in sorted(site.rglob("*.html")):
        parser = _Anchors()
        parser.feed(page.read_text(encoding="utf-8", errors="replace"))
        for href in parser.hrefs:
            if not href.startswith(("http://", "https://")):
                continue
            url = urldefrag(href)[0]
            if is_example(url) or url.startswith(SELF_PREFIXES):
                continue
            out.setdefault(url, set()).add(page.relative_to(site).as_posix())
    return out


@dataclass
class Result:
    url: str
    state: str  # ok / broken / unsure
    detail: str
    pages: set[str] = field(default_factory=set)


def _request(url: str, method: str, timeout: float) -> int:
    req = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    # URL は http / https だけ（collect で絞っている）
    with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec B310
        if method == "GET":
            r.read(1024)
        return r.status


def probe(
    url: str, timeout: float = 20.0, retries: int = RETRIES, sleep=time.sleep, request=_request
) -> tuple[str, str]:
    """1 つの URL を確かめ、(状態, 説明) を返す. 状態は ok / broken / unsure."""
    last = ("unsure", "")
    for attempt in range(retries + 1):
        if attempt:
            sleep(2**attempt)
        for method in ("HEAD", "GET"):
            try:
                status = request(url, method, timeout)
                return "ok", str(status)
            except urllib.error.HTTPError as e:
                if method == "HEAD":
                    continue  # HEAD を受け付けないサイトがあるので GET でも確かめる
                if e.code in UNSURE_STATUS or e.code >= 500:
                    last = ("unsure", f"HTTP {e.code}")
                    break  # 再試行する
                return "broken", f"HTTP {e.code}"
            except urllib.error.URLError as e:
                reason = e.reason
                if isinstance(reason, socket.gaierror):
                    return "broken", f"ホスト名を引けません（{reason}）"
                if isinstance(reason, ssl.SSLCertVerificationError):
                    return "broken", f"証明書の誤り（{getattr(reason, 'verify_message', None) or reason}）"
                last = ("unsure", f"接続できません（{reason}）")
                break
            except (TimeoutError, OSError) as e:
                last = ("unsure", f"接続できません（{type(e).__name__}: {e}）")
                break
    return last


def check(links: dict[str, set[str]], workers: int = 8, **kw) -> list[Result]:
    urls = sorted(links)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        states = list(pool.map(lambda u: probe(u, **kw), urls))
    return [Result(u, s, d, links[u]) for u, (s, d) in zip(urls, states, strict=True)]


def report(results: list[Result]) -> str:
    """Markdown の結果（Issue の本文・ジョブの要約に使う）."""
    broken = [r for r in results if r.state == "broken"]
    unsure = [r for r in results if r.state == "unsure"]
    lines = [f"外部リンク {len(results)} 件を確認: 切れている {len(broken)} 件・確認できない {len(unsure)} 件", ""]
    for title, items in (("切れているリンク", broken), ("確認できないリンク（失敗にはしない）", unsure)):
        if not items:
            continue
        lines += [f"### {title}", ""]
        for r in items:
            pages = ", ".join(sorted(r.pages)[:5]) + (" ほか" if len(r.pages) > 5 else "")
            lines.append(f"- <{r.url}> — {r.detail}（{pages}）")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("site", type=Path, help="mkdocs build の出力（site/）")
    ap.add_argument("--report", type=Path, help="結果を Markdown で書くファイル")
    ap.add_argument("--timeout", type=float, default=20.0)
    args = ap.parse_args(argv)
    if not args.site.is_dir():
        print(f"NG {args.site} がありません（先に mkdocs build を実行してください）")
        return 2

    results = check(collect(args.site), timeout=args.timeout)
    text = report(results)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 1 if any(r.state == "broken" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
