"""p13_project.py のテスト（ひな形・費用の見積もり・提出物の確かめ）."""

import io
from pathlib import Path

import pytest

from roadmap_examples.evalset import CaseResult, Report, save_results
from roadmap_examples.p07_cost import Price
from roadmap_examples.p13_project import DESIGN_HEADINGS, TODO, check_project, estimate_from, init_project, main


def results(tmp_path: Path) -> Path:
    path = tmp_path / "results.jsonl"
    report = Report([CaseResult("a", 1.0, None, 1000, 100), CaseResult("b", 0.0, None, 3000, 300)])
    save_results(report, path)
    return path


def test_init_makes_templates_and_does_not_overwrite(tmp_path: Path) -> None:
    folder = tmp_path / "project"
    assert len(init_project(folder)) == 4
    design = (folder / "design.md").read_text(encoding="utf-8")
    assert all(f"## {h}" in design for h in DESIGN_HEADINGS)
    (folder / "retro.md").write_text("書いた", encoding="utf-8")
    assert init_project(folder) == []
    assert (folder / "retro.md").read_text(encoding="utf-8") == "書いた"


def test_estimate(tmp_path: Path) -> None:
    from roadmap_examples.evalset import load_results

    est = estimate_from(load_results(results(tmp_path)), per_day=100, days=20)
    assert (est.input_per_case, est.output_per_case) == (2000, 200)
    assert est.monthly_tokens() == (4_000_000, 400_000)
    assert est.monthly_cost(Price(1.0, 5.0)) == pytest.approx(4 + 2)
    with pytest.raises(ValueError):
        estimate_from(Report([]), 1, 1)


def test_check_finds_missing_and_unfilled(tmp_path: Path) -> None:
    folder = tmp_path / "project"
    init_project(folder)
    problems = check_project(folder)
    assert any(TODO in p for p in problems)
    assert any("eval/" in p for p in problems)
    for path in folder.glob("*.md"):
        path.write_text(path.read_text(encoding="utf-8").replace(TODO, "書いた"), encoding="utf-8")
    (folder / "eval" / "broken.jsonl").write_text("{}\n", encoding="utf-8")
    assert check_project(folder) == ["eval/broken.jsonl を評価の結果として読めません（KeyError）"]
    (folder / "eval" / "broken.jsonl").unlink()
    save_results(Report([CaseResult("a", 1.0, None)]), folder / "eval" / "results.jsonl")
    assert check_project(folder) == []


def test_main(tmp_path: Path) -> None:
    folder = tmp_path / "project"
    assert main(["init", str(folder)], stdout=io.StringIO()) == 0
    assert main(["check", str(folder)], stdout=io.StringIO()) == 1
    stdout = io.StringIO()
    args = ["estimate", str(results(tmp_path)), "--per-day", "100", "--price-in", "1", "--price-out", "5"]
    assert main(args, stdout=stdout) == 0
    assert "| 月のトークン（入力 / 出力） | 4,000,000 / 400,000 |" in stdout.getvalue()
