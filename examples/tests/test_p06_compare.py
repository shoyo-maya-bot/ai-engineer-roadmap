"""p06_compare.py のテスト（06 の成果物. LLM は FakeLLM、Ollama の /api/show は偽の Transport）."""

import io
import itertools
from pathlib import Path
from typing import Any

import pytest

from roadmap_examples.evalset import EvalCase, load_cases
from roadmap_examples.llm import FakeLLM, LimitExceeded
from roadmap_examples.p06_compare import (
    DEFAULT_QUESTIONS,
    Row,
    keyword_score,
    main,
    model_details,
    run_compare,
    shorten,
    to_markdown,
    weights_gb,
)

SHOW = {
    "details": {"parameter_size": "8.0B", "quantization_level": "Q4_K_M"},
    "model_info": {"general.architecture": "example", "example.context_length": 131072},
    "capabilities": ["completion", "tools"],
}


class FakeTransport:
    def __init__(self, reply: dict[str, Any]) -> None:
        self.reply = reply
        self.sent: list[tuple[str, dict[str, Any]]] = []

    def post_json(self, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        self.sent.append((url, payload))
        return self.reply


def ticking_clock(step: float = 0.5):
    """呼ぶたびに step 秒ずつ進む時計（かかった秒数を毎回同じにする）."""
    counter = itertools.count()
    return lambda: next(counter) * step


def test_questions_file_is_a_valid_eval_set() -> None:
    cases = load_cases(DEFAULT_QUESTIONS)
    assert len(cases) == 8
    assert {"日本語", "英語"} <= {t for c in cases for t in c.tags}


def test_keyword_score_normalizes_width_and_commas() -> None:
    case = EvalCase("x", "q", {"keywords": ["1680"]})
    assert keyword_score(case, "合計は１，６８０円です") == 1.0
    assert keyword_score(case, "合計は 1,600 円です") == 0.0
    assert keyword_score(EvalCase("y", "q", {"keywords": []}), "なんでも") is None


def test_run_compare_asks_every_question_to_every_model() -> None:
    cases = [EvalCase("q1", "首都は？", {"keywords": ["東京"]}), EvalCase("q2", "1+1は？", {"keywords": ["2"]})]
    fakes = {"small": FakeLLM.of(["東京です", "3"]), "large": FakeLLM.of(["東京です", "2 です"])}
    rows = run_compare(["small", "large"], cases, fakes.__getitem__, max_tokens=1000, clock=ticking_clock())
    assert [(r.model, r.case_id, r.score) for r in rows] == [
        ("small", "q1", 1.0),
        ("small", "q2", 0.0),
        ("large", "q1", 1.0),
        ("large", "q2", 1.0),
    ]
    assert all(r.seconds == 0.5 for r in rows)
    assert fakes["small"].calls[0][0].content == "首都は？"


def test_run_compare_stops_at_token_limit() -> None:
    cases = [EvalCase("q1", "x" * 400, {}), EvalCase("q2", "y", {})]
    with pytest.raises(LimitExceeded):
        run_compare(["m"], cases, lambda _: FakeLLM.of(["a", "b"]), max_tokens=10)


def test_model_details_reads_api_show() -> None:
    fake = FakeTransport(SHOW)
    d = model_details("example-model", "http://localhost:11434/", fake)
    assert d == {
        "parameter_size": "8.0B",
        "quantization": "Q4_K_M",
        "context_length": "131072",
        "capabilities": "completion, tools",
    }
    assert fake.sent[0] == ("http://localhost:11434/api/show", {"model": "example-model"})


def test_weights_gb() -> None:
    assert weights_gb(8, 4) == 4.0
    assert weights_gb(8, 16) == 16.0


def test_shorten_keeps_one_line_without_pipes() -> None:
    assert shorten("a\nb | c", 20) == "a b ／ c"
    assert shorten("あいうえおかきくけこ", 5) == "あいうえ…"


def test_to_markdown_has_rows_and_summary() -> None:
    rows = [
        Row("m1", "q1", "東京です", 20, 3, 1.0, 1.0),
        Row("m1", "q2", "わかりません", 30, 5, 2.0, None),
        Row("m2", "q1", "Tokyo", 18, 2, 0.5, 0.0),
    ]
    text = to_markdown(rows, {"m1": {"parameter_size": "3B", "quantization": "Q4_K_M", "context_length": "4096"}})
    assert "| q1 | m1 | 東京です | 20 | 3 | 1.0 | 100% |" in text
    assert "| q2 | m1 | わかりません | 30 | 5 | 2.0 | — |" in text
    assert "| m1 | 3B | Q4_K_M | 4096 | 25 | 4 | 3.0 | 100% |" in text
    assert "| m2 |  |  |  | 18 | 2 | 0.5 | 0% |" in text


def test_main_writes_table(tmp_path: Path) -> None:
    out = tmp_path / "compare.md"
    replies = ["東京", "Tokyo", "1680 円", "1680 yen", "14 日", "billing", "分かりません", "キーで引く"]
    stdout = io.StringIO()
    code = main(
        ["--model", "m1", "--model", "m2", "--out", str(out)],
        make_llm=lambda _: FakeLLM.of(replies),
        transport=FakeTransport(SHOW),
        stdout=stdout,
        clock=ticking_clock(),
    )
    assert code == 0
    assert "16 行" in stdout.getvalue()
    text = out.read_text(encoding="utf-8")
    assert "| m1 | 8.0B | Q4_K_M | 131072 |" in text
    assert text.count("| q01 |") == 2


def test_main_stops_at_limit() -> None:
    code = main(
        ["--model", "m", "--max-tokens", "1"],
        make_llm=lambda _: FakeLLM.of(["x"] * 8),
        transport=FakeTransport(SHOW),
        stdout=io.StringIO(),
    )
    assert code == 1


def test_main_reports_missing_questions(tmp_path: Path) -> None:
    assert main(["--model", "m", "--questions", str(tmp_path / "none.jsonl")], stdout=io.StringIO()) == 1
