"""scripts/check_versions.py のテスト（extra.versions と実物・本文に直接書いた版）."""

import check_versions as cv
import pytest


def errors(repo):
    return cv.run(repo.root)[1]


def test_minimal_repo_passes(repo):
    assert errors(repo) == []


@pytest.mark.parametrize(
    ("rel", "old", "new", "where"),
    [
        ("examples/pyproject.toml", ">=3.12", ">=3.13", "requires-python"),
        ("examples/.python-version", "3.12", "3.11", ".python-version"),
        ("examples/pyproject.toml", "pytest==9.1.1", "pytest==9.2.0", "examples/pyproject.toml の dev"),
        ("examples/uv.lock", '"0.16.10"', '"0.16.11"', "examples/uv.lock"),
        ("requirements.txt", "pytest==9.1.1", "pytest==9.0.0", "requirements.txt"),
        (".github/workflows/ci.yml", '"3.12"', '"3.13"', "python-version"),
        (".github/workflows/ci.yml", "uv==0.8.17", "uv==0.9.0", "uv=="),
        ("examples/pyproject.toml", "uv_build>=0.8.17", "uv_build>=0.8.0", "uv_build"),
    ],
)
def test_mismatch_with_actual_files(repo, rel, old, new, where):
    repo.write(rel, repo.read(rel).replace(old, new))
    errs = errors(repo)
    assert len(errs) == 1 and where in errs[0], errs


def test_constraints_and_workflow_ruff_are_checked_when_present(repo):
    repo.write("constraints.txt", "pytest==9.1.1 \\\n    --hash=sha256:00\n")
    repo.write(".github/workflows/lint.yml", 'jobs:\n  a:\n    steps:\n      - run: pip install "ruff==0.15.0"\n')
    errs = errors(repo)
    assert errs == [
        "mkdocs.yml: extra.versions.ruff は 0.16.10 ですが、.github/workflows/lint.yml の ruff== は 0.15.0 です"
    ]


def test_missing_key_and_unreadable_files(repo):
    repo.write("mkdocs.yml", repo.read("mkdocs.yml").replace('    uv: "0.8.17"\n', ""))
    (repo.root / "examples/uv.lock").unlink()
    errs = errors(repo)
    assert "mkdocs.yml: extra.versions.uv がありません" in errs
    assert any("examples/uv.lock を読めません" in e for e in errs), errs


@pytest.mark.parametrize(
    "line",
    [
        "Python 3.12 を使います",
        "python3.12 で動かす",
        "uv 0.8 以降",
        "pytest 9.1 の機能",
        "Ruff 0.16 で",
        "版は 9.1.1 です",
    ],
)
def test_literal_versions_in_prose(repo, line):
    repo.write("docs/index.md", f"# はじめに\n\n{line}\n")
    errs = errors(repo)
    assert len(errs) == 1 and "index.md:3: 版を直接書いています" in errs[0], errs


def test_versions_in_code_blocks_and_comments_are_allowed(repo):
    text = '# はじめに\n\n```toml\nrequires-python = ">=3.12"\n```\n\n<!-- Python 3.12 -->\n\nPython 3 系を使う\n'
    repo.write("docs/index.md", text)
    assert errors(repo) == []


def test_macros_need_front_matter_and_known_names(repo):
    repo.write("docs/index.md", "# はじめに\n\nPython {{ versions.python }}\n")
    assert errors(repo) == ["index.md: {{ versions.* }} を使っているのに、先頭に render_macros: true がありません"]
    repo.write("docs/index.md", "---\nrender_macros: true\n---\n\n# はじめに\n\n{{ versions.numpy }}\n")
    assert errors(repo) == ["index.md: versions.numpy は mkdocs.yml の extra.versions にありません"]


def test_repository_passes():
    assert cv.run(cv.ROOT)[1] == []
