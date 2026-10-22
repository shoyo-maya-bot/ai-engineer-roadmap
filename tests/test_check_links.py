"""scripts/check_links.py のテスト（ネットワークには出ない. 応答は差し替える）."""

import socket
import ssl
import urllib.error

import check_links as cl
import pytest


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x.invalid/", code, "x", {}, None)


class FakeRequest:
    """(URL, メソッド) ごとの応答を順に返す. 例外なら投げる."""

    def __init__(self, answers: dict):
        self.answers = {k: list(v) for k, v in answers.items()}
        self.calls: list[tuple[str, str]] = []

    def __call__(self, url, method, timeout):
        self.calls.append((url, method))
        queue = self.answers[(url, method)]
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(answer, BaseException):
            raise answer
        return answer


def probe(answers: dict, url="https://a.org/"):
    fake = FakeRequest(answers)
    return cl.probe(url, sleep=lambda _s: None, request=fake), fake


def test_collects_external_links_once_without_examples_and_self(tmp_path):
    (tmp_path / "ch").mkdir()
    (tmp_path / "index.html").write_text(
        '<a href="https://a.org/doc#part">a</a> <a href="../ch/">内部</a> <a href="mailto:x@a.org">m</a>'
        '<a href="http://localhost:8080">l</a> <a href="https://dev.example.com/x">e</a>'
        '<a href="https://example.org">e2</a> <a href="https://&lt;host&gt;/">書きかけ</a>'
        '<a href="https://shoyo-maya-bot.github.io/ai-engineer-roadmap/guide/">自分</a>',
        encoding="utf-8",
    )
    (tmp_path / "ch" / "index.html").write_text('<p><a class="x" href="https://a.org/doc">a</a></p>', encoding="utf-8")
    assert cl.collect(tmp_path) == {"https://a.org/doc": {"index.html", "ch/index.html"}}


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:11434",
        "http://[::1]:8080/",
        "http://0.0.0.0:8080",
        "http://192.168.0.10/",
        "https://api.example.net/",
        "https://foo.test/",
        "https://{{ x }}/",
    ],
)
def test_is_example(url):
    assert cl.is_example(url)


def test_real_hosts_are_not_examples():
    assert not cl.is_example("https://docs.python.org/3/")
    assert not cl.is_example("https://notexample.com/")


def test_ok():
    (state, _), fake = probe({("https://a.org/", "HEAD"): [200]})
    assert state == "ok"
    assert fake.calls == [("https://a.org/", "HEAD")]


def test_head_rejected_falls_back_to_get():
    (state, _), _ = probe({("https://a.org/", "HEAD"): [http_error(405)], ("https://a.org/", "GET"): [200]})
    assert state == "ok"


def test_404_is_broken():
    both = {("https://a.org/", "HEAD"): [http_error(404)], ("https://a.org/", "GET"): [http_error(404)]}
    assert probe(both)[0] == ("broken", "HTTP 404")


@pytest.mark.parametrize("code", [403, 429])
def test_403_and_429_are_unsure_not_broken(code):
    both = {("https://a.org/", "HEAD"): [http_error(code)], ("https://a.org/", "GET"): [http_error(code)]}
    (state, _), fake = probe(both)
    assert state == "unsure"
    assert len(fake.calls) == 2 * (cl.RETRIES + 1)  # 再試行する


def test_5xx_retries_then_succeeds():
    get = [http_error(503), 200]
    (state, _), _ = probe({("https://a.org/", "HEAD"): [http_error(503)], ("https://a.org/", "GET"): get})
    assert state == "ok"


def test_dns_failure_and_bad_certificate_are_broken():
    dns = urllib.error.URLError(socket.gaierror(-2, "Name or service not known"))
    assert probe({("https://a.org/", "HEAD"): [dns]})[0][0] == "broken"
    cert = urllib.error.URLError(ssl.SSLCertVerificationError(1, "certificate verify failed"))
    assert probe({("https://a.org/", "HEAD"): [cert]})[0][0] == "broken"


def test_timeout_is_unsure():
    assert probe({("https://a.org/", "HEAD"): [TimeoutError("timed out")]})[0][0] == "unsure"


def test_report_lists_broken_and_unsure():
    results = [
        cl.Result("https://a.org/", "ok", "200", {"a.html"}),
        cl.Result("https://b.org/", "broken", "HTTP 404", {"x/index.html"}),
        cl.Result("https://c.org/", "unsure", "HTTP 403", {"y/index.html"}),
    ]
    text = cl.report(results)
    assert "外部リンク 3 件を確認: 切れている 1 件・確認できない 1 件" in text
    assert "- <https://b.org/> — HTTP 404（x/index.html）" in text
    assert text.index("切れているリンク") < text.index("確認できないリンク")


def test_main_exit_codes(tmp_path):
    assert cl.main([str(tmp_path / "none")]) == 2
    (tmp_path / "index.html").write_text("<p>リンクなし</p>", encoding="utf-8")
    assert cl.main([str(tmp_path)]) == 0
