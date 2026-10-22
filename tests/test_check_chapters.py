"""scripts/check_chapters.py のテスト（仮のリポジトリで、構成の誤りと禁止語を検知できるか）."""

import check_chapters as cc
import pytest
from conftest import HEADS, chapter, part_index


def errors(repo):
    return cc.run(repo.root)[1]


def test_minimal_repo_passes(repo):
    assert cc.run(repo.root) == (1, [])


def test_missing_goal_and_heading_order(repo):
    swapped = HEADS[1:2] + HEADS[:1] + HEADS[2:]
    repo.write("docs/01-python/01-1-intro.md", chapter(swapped).replace("到達目標", "目標"))
    errs = errors(repo)
    assert any("到達目標" in e for e in errs), errs
    assert any("の順番が違います" in e for e in errs), errs


def test_missing_exercise_heading(repo):
    repo.write("docs/01-python/01-1-intro.md", chapter([h for h in HEADS if h != "演習"]))
    assert errors(repo) == ["01-python/01-1-intro.md: 見出し「演習」がありません"]


def test_answer_page_must_exist_and_be_linked(repo):
    (repo.root / "docs/01-python/answers/01-1.md").unlink()
    repo.write("docs/01-python/01-1-intro.md", chapter().replace("answers/01-1.md", "answers.md"))
    errs = errors(repo)
    assert "01-python/01-1-intro.md: 解答ページ answers/01-1.md へのリンクがありません" in errs
    assert "01-python/01-1-intro.md: 解答ページ answers/01-1.md がありません" in errs


def test_orphan_answer_page(repo):
    repo.write("docs/01-python/answers/01-2.md", "# 01-2 解答\n")
    repo.write("mkdocs.yml", repo.read("mkdocs.yml") + "  - 01-python/answers/01-2.md\n")
    assert errors(repo) == ["01-python/answers/01-2.md: 対応する章 01-2-*.md がありません"]


def test_chapter_in_wrong_part(repo):
    repo.write("docs/02-math/index.md", part_index("").replace("01 Python", "02 数学"))
    repo.write("docs/02-math/01-2-x.md", chapter(num="01-2"))
    repo.write("docs/02-math/answers/01-2.md", "# 解答\n")
    nav = "  - 02-math/index.md\n  - 02-math/01-2-x.md\n  - 02-math/answers/01-2.md\n"
    repo.write("mkdocs.yml", repo.read("mkdocs.yml") + nav)
    errs = errors(repo)
    assert any("ファイル名の部（01）と置き場所の部（02）が違います" in e for e in errs), errs


def test_page_missing_from_nav(repo):
    repo.write("docs/extra.md", "# おまけ\n")
    assert errors(repo) == ["extra.md: mkdocs.yml の nav にありません"]


def test_part_index_requires_sections(repo):
    repo.write("docs/01-python/index.md", part_index().replace("## 目安時間", "## 時間"))
    assert errors(repo) == ["01-python/index.md: 見出し「目安時間」がありません"]


def test_project_part_may_use_task_and_submission_headings(repo):
    text = part_index("").replace("## 章の一覧", "## 課題").replace("## 成果物", "## 提出物")
    repo.write("docs/13-project/index.md", text)
    repo.write("mkdocs.yml", repo.read("mkdocs.yml") + "  - 13-project/index.md\n")
    assert errors(repo) == []


def test_chapter_list_rows_link_written_chapters_and_mark_pending(repo):
    rows = "| 01-1 入門 | 内容 | 準備中 |\n| 01-2 次 | 内容 | 未定 |\n| 01-3 その次 | 内容 | 準備中 |\n"
    repo.write("docs/01-python/index.md", part_index(rows))
    errs = errors(repo)
    assert any("章 01-1 は書き上がっています" in e for e in errs), errs
    assert any("章 01-2 のファイルがありません（まだなら「準備中」と書く）" in e for e in errs), errs
    assert not any("01-3" in e for e in errs), errs


def test_written_chapter_must_be_listed(repo):
    repo.write("docs/01-python/index.md", part_index(""))
    assert errors(repo) == ["01-python/index.md: 章の一覧に 01-1-intro.md がありません"]


def test_bad_chapter_file_names_are_not_chapters(repo):
    # 大文字やアンダースコアの名前は章として扱わない（nav の検査で見つかる）
    repo.write("docs/01-python/01-2-Bad_Name.md", chapter(num="01-2"))
    assert errors(repo) == ["01-python/01-2-Bad_Name.md: mkdocs.yml の nav にありません"]


def test_banned_terms_in_docs_examples_and_root(repo):
    repo.write("docs/01-python/01-1-intro.md", chapter(extra="連絡先は someone@GMAIL.com です\n"))
    repo.write("examples/src/roadmap_examples/x.py", '# 架空の株式会社サンプル\nKEY = "sk-abcdefghijklmnopqrstuvwx"\n')
    repo.write("README.md", "# 読んで\n\n保存先は /Users/taro/work です\n")
    errs = errors(repo)
    line = len(chapter().split("\n"))
    assert f"docs/01-python/01-1-intro.md:{line}: 禁止語「@gmail.com」があります" in errs
    assert "examples/src/roadmap_examples/x.py:1: 禁止語「株式会社」があります" in errs
    assert any(e.startswith("examples/src/roadmap_examples/x.py:2: 禁止語「re:") for e in errs), errs
    assert "README.md:3: 禁止語「/Users/」があります" in errs


def test_banned_terms_skip_virtualenv(repo):
    repo.write("examples/.venv/lib/x.py", "# 株式会社\n")
    assert errors(repo) == []


def test_banned_terms_match_full_width_forms(repo):
    repo.write("docs/index.md", "# はじめに\n\n連絡先は someone＠ＧＭＡＩＬ．ｃｏｍ です\nｇｍａｉｌ の話\n")
    assert errors(repo) == ["docs/index.md:3: 禁止語「@gmail.com」があります"]


def test_local_path_terms_match_only_path_starts(repo):
    repo.write("docs/index.md", "# はじめに\n\n`GET /api/users/{id}`\nhttps://example.com/Users/list\n'/Users/taro'\n")
    assert errors(repo) == ["docs/index.md:5: 禁止語「/Users/」があります"]


def test_chat_ids_and_keys_in_repository_list(repo):
    repo.write("banned-terms.txt", (cc.ROOT / "banned-terms.txt").read_text(encoding="utf-8"))
    text = "# はじめに\n\nU0ABCDEFGH1 に送る\nghp_" + "a" * 36 + "\nUTF8 と HTTP の話\n"
    repo.write("docs/index.md", text)
    lines = sorted(int(e.split(":")[1]) for e in errors(repo))
    assert lines == [3, 4]


@pytest.mark.parametrize(("text", "message"), [("# コメントだけ\n\n", "空です"), ("re:[\n", "正規表現")])
def test_empty_or_broken_banned_list_fails(repo, text, message):
    repo.write("banned-terms.txt", text)
    errs = errors(repo)
    assert errs and message in errs[0], errs


def test_missing_banned_list_fails(repo):
    (repo.root / "banned-terms.txt").unlink()
    errs = errors(repo)
    assert errs and "読めません" in errs[0], errs


def test_repository_banned_list_is_usable():
    assert len(cc.load_banned(cc.ROOT / "banned-terms.txt")) >= 10


def test_bom_and_crlf_pages(repo):
    path = repo.root / "docs/01-python/01-1-intro.md"
    path.write_bytes(b"\xef\xbb\xbf" + chapter().replace("\n", "\r\n").encode("utf-8"))
    assert errors(repo) == []


def test_main_exit_code(repo):
    assert cc.main(["--root", str(repo.root)]) == 0
    repo.write("docs/01-python/01-1-intro.md", chapter(HEADS[:-1]))
    assert cc.main(["--root", str(repo.root)]) == 1


def test_repository_passes():
    assert cc.run(cc.ROOT)[1] == []
