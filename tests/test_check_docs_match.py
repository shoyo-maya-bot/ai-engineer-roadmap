"""scripts/check_docs_match.py のテスト（README などの説明と実物のずれ）."""

import check_docs_match as dm
import pytest

SCRIPT = 'import argparse\nap = argparse.ArgumentParser()\nap.add_argument("--root")\n'


@pytest.fixture
def docs_repo(repo):
    repo.write("scripts/check_a.py", SCRIPT)
    listing = "```bash\npython scripts/check_a.py --root .\n```\n"
    for name in ("README.md", "CONTRIBUTING.md", "AGENTS.md"):
        repo.write(name, f"# {name}\n\n{listing}")
    for name in dm.DOC_FILES[3:]:
        repo.write(name, "# 説明\n")
    repo.write(
        ".github/workflows/ci.yml", repo.read(".github/workflows/ci.yml") + "      - run: python scripts/check_a.py\n"
    )
    return repo


def errors(repo):
    return dm.run(repo.root)[1]


def test_consistent_docs_pass(docs_repo):
    assert errors(docs_repo) == []


def test_missing_script_and_option(docs_repo):
    docs_repo.write(
        "README.md",
        docs_repo.read("README.md")
        + "\n`python scripts/check_b.py`\n\n```bash\npython scripts/check_a.py --fast  # 速く\n```\n",
    )
    errs = errors(docs_repo)
    assert "README.md: scripts/check_b.py がありません" in errs
    assert "README.md: scripts/check_a.py にオプション --fast がありません" in errs
    assert len(errs) == 2, errs


def test_paths_in_code_and_links_must_exist(docs_repo):
    text = (
        "\n`examples/uv.lock` と `docs/gone.md` と `scripts/` と `.env`\n"
        "[規約](CONTRIBUTING.md) [無い](MISSING.md) [外](https://example.com/README.md)\n"
        "`docs/<部>/index.md` と `answers/01-4.md` と https://example.com/x/none.md\n"
    )
    docs_repo.write("README.md", docs_repo.read("README.md") + text)
    errs = errors(docs_repo)
    assert sorted(errs) == [
        "README.md: MISSING.md がありません（場所が変わった・名前が変わった可能性があります）",
        "README.md: docs/gone.md がありません（場所が変わった・名前が変わった可能性があります）",
    ]


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        ("Python 3.12 で動かす", True),
        ("Python 3.11 で動かす", False),
        ("uv 0.8 以降", True),
        ('pip install "uv==0.9.1"', False),
        ("ruff==0.16.10", True),
        ("pytest 8.4", False),
    ],
)
def test_versions_in_docs(docs_repo, text, ok):
    docs_repo.write("CONTRIBUTING.md", docs_repo.read("CONTRIBUTING.md") + f"\n{text}\n")
    assert (errors(docs_repo) == []) is ok


def test_uv_run_commands(docs_repo):
    commands = ["uv run pytest", "uv run api-to-csv --help", "uv run --project examples ruff check .", "uv run nothing"]
    text = "\n```bash\n" + "\n".join(commands) + "\n```\n"
    docs_repo.write("examples/README.md", "# examples\n" + text)
    assert errors(docs_repo) == [
        "examples/README.md: uv run nothing のコマンドがありません（examples/pyproject.toml の [project.scripts]）"
    ]


def test_every_check_script_is_documented_and_run_in_ci(docs_repo):
    docs_repo.write("scripts/check_new.py", SCRIPT)
    errs = errors(docs_repo)
    assert "README.md: scripts/check_new.py の説明がありません" in errs
    assert "AGENTS.md: scripts/check_new.py の説明がありません" in errs
    assert ".github/workflows/ci.yml: scripts/check_new.py を実行していません" in errs


def test_workflow_script_references(docs_repo):
    docs_repo.write(".github/workflows/x.yml", "jobs:\n  a:\n    steps:\n      - run: python scripts/gone.py\n")
    assert errors(docs_repo) == [".github/workflows/x.yml: scripts/gone.py がありません"]


def test_missing_doc_file(docs_repo):
    (docs_repo.root / "SECURITY.md").unlink()
    assert errors(docs_repo) == ["SECURITY.md: ありません"]


def test_repository_passes():
    assert dm.run(dm.ROOT)[1] == []
