"""scripts/check_python_examples.py のテスト（説明用のコード例の構文と skip の印）."""

import check_python_examples as cp
import pytest


def run(repo, body: str):
    repo.write("docs/index.md", "# はじめに\n\n" + body)
    return cp.run(repo.root)


def test_valid_fragments_pass(repo):
    body = "```python\ndef handle(item: dict) -> None:\n    ...\n```\n\n```py\nx = [i * 2 for i in range(3)]\n```\n"
    assert run(repo, body) == (2, 0, [])


def test_syntax_error_is_reported_with_line(repo):
    checked, _, errs = run(repo, "本文\n\n```python\nif True\n    pass\n```\n")
    assert checked == 1
    assert errs == ["docs/index.md:5: Python の構文の誤り（1 行目: expected ':'）"]


def test_other_languages_and_quotes_are_not_checked(repo):
    body = (
        "```bash\nif [ x ]; then\n```\n\n```pycon\n>>> 1 +\n```\n\n"
        "<!-- src: examples/src/roadmap_examples/sample.py -->\n```python\ndef add(a: int, b: int) -> int:\n```\n"
    )
    assert run(repo, body) == (0, 0, [])


def test_skip_with_reason(repo):
    body = "<!-- python-syntax: skip わざと誤った例 -->\n```python\nif True\n```\n"
    assert run(repo, body) == (0, 1, [])


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("<!-- python-syntax: skip -->\n```python\nif True\n```\n", "理由を書いてください"),
        ("<!-- python-syntax: skip 理由 -->\n```bash\nls\n```\n", "Python のコード例ではありません"),
        ("<!-- python-syntax: skip 理由 -->\n\n本文\n", "直後にコードブロックがありません"),
    ],
)
def test_bad_skip_markers(repo, body, message):
    _, _, errs = run(repo, body)
    assert len(errs) == 1 and message in errs[0], errs


def test_feature_version_follows_extra_versions(repo):
    # 3.12 で入った型パラメータの構文. 3.11 として読むと誤り
    body = "```python\ndef first[T](items: list[T]) -> T:\n    return items[0]\n```\n"
    assert run(repo, body)[2] == []
    repo.write("mkdocs.yml", repo.read("mkdocs.yml").replace('python: "3.12"', 'python: "3.11"'))
    assert len(run(repo, body)[2]) == 1


def test_unreadable_version_fails(repo):
    repo.write("mkdocs.yml", repo.read("mkdocs.yml").replace('python: "3.12"', 'python: "three"'))
    errs = run(repo, "")[2]
    assert errs and "extra.versions.python" in errs[0]


def test_list_option(repo, capsys):
    run(repo, "<!-- python-syntax: skip 誤りの例 -->\n```python\nif\n```\n\n```python\nx = 1\n```\n")
    assert cp.main(["--root", str(repo.root), "--list"]) == 0
    out = capsys.readouterr().out
    assert "docs/index.md:4\tskip" in out
    assert "docs/index.md:8\tcheck" in out


def test_repository_passes():
    assert cp.run(cp.ROOT)[2] == []
