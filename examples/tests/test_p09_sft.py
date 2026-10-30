"""p09_sft.py のテスト（SFT の会話の形と、評価セットとの重なりの検査）."""

import io
import json
from pathlib import Path

from roadmap_examples.evalset import EvalCase, load_cases
from roadmap_examples.p07_classify import DEFAULT_EVAL
from roadmap_examples.p07_structured import CLASSIFY_SCHEMA, validate
from roadmap_examples.p09_lora import BASE_TRAIN, SFT_TRAIN, SFT_VAL
from roadmap_examples.p09_sft import main, nearest_pairs, overlaps, similarity, to_record


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


def test_similarity_finds_paraphrases() -> None:
    # 07 の c005 を少し言い換えただけの文は、完全一致ではないが、似ている度合いが高い
    assert (
        similarity(
            "解約したのに、その月の分まで引き落とされています。", "解約したはずの月の分まで引き落とされています。"
        )
        > 0.5
    )
    assert similarity("ＣＳＶ を出力", "CSVを出力") == 1.0


def test_training_examples_are_not_paraphrases_of_the_eval_set() -> None:
    """09-1 の本文に書いた結果: 似ている度合い 0.5 を超える組は、汎用の例の b007 と c008 の 1 組だけ."""
    evaluation = load_cases(DEFAULT_EVAL)
    for path in (SFT_TRAIN, SFT_VAL):
        assert nearest_pairs(load_cases(path), evaluation, top=1)[0][0] < 0.5
    close = [(t, e) for score, t, e in nearest_pairs(load_cases(BASE_TRAIN), evaluation, top=5) if score > 0.5]
    assert close == [("b007", "c008")]
