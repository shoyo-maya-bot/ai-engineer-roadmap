"""07 の成果物のテスト（仮のルールベース分類と LLM の分類を、同じ評価セットで比べる. LLM は FakeLLM）."""

import io
import json
from pathlib import Path

from roadmap_examples.evalset import load_cases, load_results
from roadmap_examples.llm import FakeLLM
from roadmap_examples.p07_classify import DEFAULT_EVAL, main, rule_based
from roadmap_examples.p07_prompt import CATEGORIES


def test_eval_set_is_valid_and_covers_every_category() -> None:
    cases = load_cases(DEFAULT_EVAL)
    assert len(cases) == 30
    assert {c.expected["category"] for c in cases} == set(CATEGORIES)


def test_rule_based() -> None:
    assert rule_based("請求書を再発行してほしい") == "billing"
    assert rule_based("ログインボタンでエラーが出る") == "bug"  # 上の規則が先に当たる
    assert rule_based("ありがとうございます") == "other"


def test_main_baseline_writes_results(tmp_path: Path) -> None:
    out = tmp_path / "baseline.jsonl"
    stdout = io.StringIO()
    assert main(["--mode", "baseline", "--out", str(out)], stdout=stdout) == 0
    assert "正解率: 66.7%（20 / 30 件）" in stdout.getvalue()
    assert load_results(out).meta == {"mode": "baseline"}


def test_main_llm_mode_with_fake_and_compare(tmp_path: Path) -> None:
    cases = load_cases(DEFAULT_EVAL)
    replies = [json.dumps({"category": c.expected["category"], "reason": "テスト"}) for c in cases]
    baseline = tmp_path / "baseline.jsonl"
    main(["--mode", "baseline", "--out", str(baseline)], stdout=io.StringIO())
    stdout = io.StringIO()
    code = main(
        ["--mode", "ollama", "--model", "example-model", "--compare", str(baseline), "--price-in", "100"],
        llm=FakeLLM.of(replies),
        stdout=stdout,
    )
    text = stdout.getvalue()
    assert code == 0
    assert "正解率: 100.0%" in text
    assert "前回からの変化: 10 件" in text
    assert "後退" not in text


def test_main_stops_at_call_limit(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(DEFAULT_EVAL.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    # 形が崩れた応答だけを返す → 頼み直しで呼び出しの上限（1 回）に達する
    code = main(
        ["--mode", "ollama", "--model", "m", "--eval", str(eval_path), "--max-calls", "1"],
        llm=FakeLLM.of(["崩れた応答"] * 3),
        stdout=io.StringIO(),
    )
    assert code == 1


def test_main_reports_bad_eval_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n", encoding="utf-8")
    assert main(["--mode", "baseline", "--eval", str(bad)], stdout=io.StringIO()) == 1
