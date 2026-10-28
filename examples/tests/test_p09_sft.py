"""p09_sft.py のテスト（SFT の会話の形と、評価セットとの重なりの検査）."""

import io
import json
from pathlib import Path

from roadmap_examples.evalset import EvalCase, load_cases
from roadmap_examples.p07_classify import DEFAULT_EVAL
from roadmap_examples.p07_structured import CLASSIFY_SCHEMA, validate
from roadmap_examples.p09_lora import SFT_TRAIN
from roadmap_examples.p09_sft import main, overlaps, to_record


def test_training_examples_do_not_overlap_the_eval_set() -> None:
    assert overlaps(load_cases(SFT_TRAIN), load_cases(DEFAULT_EVAL)) == []


def test_overlap_ignores_width_and_spaces() -> None:
    train = [EvalCase("t1", "ＣＳＶ を出力すると 文字化けします。", {})]
    assert overlaps(train, load_cases(DEFAULT_EVAL)) == [("t1", "c015")]


def test_record_uses_the_07_prompt_and_a_valid_answer() -> None:
    record = to_record(load_cases(SFT_TRAIN)[0])
    assert [m["role"] for m in record["prompt"]] == ["system", "user"]
    assert record["prompt"][1]["content"].startswith("<inquiry>")
    answer = json.loads(record["completion"][0]["content"])
    assert validate(answer, CLASSIFY_SCHEMA) == []


def test_main_writes_jsonl_and_refuses_overlap(tmp_path: Path) -> None:
    out = tmp_path / "sft.jsonl"
    assert main(["--out", str(out)], stdout=io.StringIO()) == 0
    assert len(out.read_text(encoding="utf-8").splitlines()) == 40
    leaked = tmp_path / "leaked.jsonl"
    leaked.write_text(
        json.dumps(
            {
                "id": "x1",
                "input": "パスワードを忘れてしまいました。",
                "expected": {"category": "account", "reason": "r"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert main(["--train", str(leaked), "--out", str(out)], stdout=io.StringIO()) == 1
