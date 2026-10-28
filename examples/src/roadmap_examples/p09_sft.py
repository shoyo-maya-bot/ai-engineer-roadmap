"""09-2 SFT（教師ありファインチューニング）のデータを作る: 07 の分類の例を、学習のライブラリが読む会話の形にする.

- 1 件 = 07 と同じプロンプト（system と問い合わせ）と、期待する答え（07 の JSON）の組
  （TRL の SFTTrainer が読む「会話の prompt-completion」の形. https://huggingface.co/docs/trl/sft_trainer ）
- 学習に使う例と、評価セット（07）に同じ文章が無いことを確かめる（混ざると、覚えた答えで点が上がって見える）
- 07 と同じプロンプトで学習すると、調整したモデルを 07 の p07_classify でそのまま評価できる

使い方（examples/ で）:
  uv run python -m roadmap_examples.p09_sft --out sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

from roadmap_examples.evalset import EvalCase, EvalSetError, load_cases
from roadmap_examples.p07_classify import DEFAULT_EVAL
from roadmap_examples.p07_prompt import DEFAULT_SPEC, PromptSpec, answer_json, system_prompt, wrap_inquiry
from roadmap_examples.p09_lora import SFT_TRAIN


def to_record(case: EvalCase, spec: PromptSpec = DEFAULT_SPEC) -> dict[str, Any]:
    """1 件を、会話の prompt-completion の形にする. 例示は入れない（例示の役目を学習が担う）."""
    expected = case.expected
    if "category" not in expected or "reason" not in expected:
        raise EvalSetError(f"{case.id}: 学習の例には expected の category と reason が要ります")
    return {
        "prompt": [
            {"role": "system", "content": system_prompt(spec)},
            {"role": "user", "content": wrap_inquiry(case.input)},
        ],
        "completion": [{"role": "assistant", "content": answer_json(expected["category"], expected["reason"])}],
    }


def normalized(text: str) -> str:
    """比べるための形（全角・半角をそろえ、空白を除く）."""
    return "".join(unicodedata.normalize("NFKC", text).split())


def overlaps(train: Sequence[EvalCase], evaluation: Sequence[EvalCase]) -> list[tuple[str, str]]:
    """学習の例と評価セットで、同じ文章の組（学習の id, 評価の id）. 空でなければ学習に使わない."""
    seen = {normalized(c.input): c.id for c in evaluation}
    return [(c.id, seen[normalized(c.input)]) for c in train if normalized(c.input) in seen]


def write_jsonl(records: Sequence[dict[str, Any]], path: Path) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="p09_sft", description="分類の例を SFT の会話の形（JSONL）にする")
    p.add_argument(
        "--train", type=Path, default=SFT_TRAIN, help="学習の例（07 の形式. expected に category と reason）"
    )
    p.add_argument("--eval", type=Path, default=DEFAULT_EVAL, help="重なりを確かめる評価セット")
    p.add_argument("--out", type=Path, required=True, help="書き出す JSONL")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None, stdout: TextIO | None = None) -> int:
    args = parse_args(argv)
    stdout = stdout or sys.stdout
    try:
        train, evaluation = load_cases(args.train), load_cases(args.eval)
        records = [to_record(c) for c in train]
    except (OSError, EvalSetError) as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    same = overlaps(train, evaluation)
    if same:
        print("エラー: 評価セットと同じ文章が学習の例にあります（学習に使えません）:", file=sys.stderr)
        for train_id, eval_id in same:
            print(f"  {train_id} = {eval_id}", file=sys.stderr)
        return 1
    write_jsonl(records, args.out)
    print(f"{len(records)} 件を書き出しました: {args.out}", file=stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
