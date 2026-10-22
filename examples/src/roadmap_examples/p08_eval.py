"""08-4 RAG の評価: 検索と生成を分けて測り、外れた件が「検索の外れ」か「生成の外れ」かを切り分ける.

評価セットは 07-5 の形式（evalset.py）. expected には次を書く:
  sources:  答えの根拠になる文書のファイル名の一覧（資料に答えが無い質問は空）
  keywords: 正しい答えに必ず入る語の一覧（資料に答えが無い質問は空）

- 検索の点（retrieval_recall）: 根拠の文書のうち、検索の上位 k 件に入った割合
- 生成の点（generation_score）: 答えの語が入っているか × 正しい出典を示したか.
  資料に無い質問は「記載がありません」と答えたら 1 点
- 生成だけを測るときは、検索を飛ばし、根拠の文書のチャンクをそのまま渡す（`oracle_hits`）

使い方（examples/ で. 手元の Ollama を呼ぶ）:
  uv run python -m roadmap_examples.p08_eval --model <モデル名> --embed-model <エンベディングのモデル名>
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

from roadmap_examples.evalset import EvalCase, EvalSetError, Prediction, Report, load_cases, run_eval
from roadmap_examples.llm import LimitedLLM, LimitExceeded, LLMClient
from roadmap_examples.ollama_client import DEFAULT_BASE_URL, OllamaChat, OllamaEmbedder
from roadmap_examples.p08_chunk import Chunk, load_chunks
from roadmap_examples.p08_rag import DEFAULT_DOCS, Answer, answer_with, build_retriever
from roadmap_examples.p08_search import Embedder, Hit, Retriever

DEFAULT_EVAL = Path(__file__).resolve().parents[2] / "data" / "p08" / "qa.jsonl"


def answerable(case: EvalCase) -> bool:
    return bool(case.expected.get("sources"))


def retrieval_recall(case: EvalCase, sources: list[str]) -> float:
    """根拠の文書のうち、検索結果（文書名の一覧）に入った割合."""
    expected = set(case.expected["sources"])
    return len(expected & set(sources)) / len(expected)


def generation_score(case: EvalCase, output: dict[str, Any]) -> float:
    """語の割合（0〜1）× 出典の点（正しい文書を示せば 1、示さなければ 0.5、資料に無い番号を示せば 0）."""
    text, cited = output["text"], set(output["cited_sources"])
    if not answerable(case):
        return 1.0 if output["no_answer"] and not cited else 0.0
    words = case.expected.get("keywords", [])
    found = sum(1 for w in words if w in text) / len(words) if words else 1.0
    if output["invalid"]:
        citation = 0.0
    elif cited & set(case.expected["sources"]):
        citation = 1.0
    else:
        citation = 0.5
    return found * citation


def answer_output(ans: Answer) -> dict[str, Any]:
    """評価の結果に残す形（JSON にできるもの）."""
    return {
        "text": ans.text,
        "cited_sources": sorted({c.source for c in ans.sources}),
        "retrieved": [h.chunk.id for h in ans.hits],
        "invalid": ans.invalid,
        "no_answer": ans.is_no_answer,
    }


def eval_retrieval(cases: Sequence[EvalCase], retriever: Retriever, k: int) -> Report:
    """検索だけを測る（LLM を呼ばない）. 資料に答えが無い質問は対象外."""

    def predict(case: EvalCase) -> Prediction:
        return Prediction([h.chunk.source for h in retriever.search(case.input, k)])

    return run_eval([c for c in cases if answerable(c)], predict, retrieval_recall, meta={"part": "retrieval", "k": k})


def oracle_hits(case: EvalCase, chunks: Sequence[Chunk]) -> list[Hit]:
    """正しい根拠の文書のチャンク（検索が完全に当たった場合の資料）."""
    return [Hit(c, 1.0) for c in chunks if c.source in case.expected.get("sources", [])]


def eval_generation(
    cases: Sequence[EvalCase],
    llm: LLMClient,
    retriever: Retriever | None,
    chunks: Sequence[Chunk],
    k: int,
) -> Report:
    """生成を測る. retriever が None なら、検索を飛ばして正しい根拠を渡す（生成だけの力を測る）.

    資料に答えが無い質問には、検索の結果（retriever が None なら全チャンクの先頭 k 件）を渡す.
    """

    def predict(case: EvalCase) -> Prediction:
        if retriever is not None:
            hits = retriever.search(case.input, k)
        elif answerable(case):
            hits = oracle_hits(case, chunks)
        else:
            hits = [Hit(c, 0.0) for c in chunks[:k]]
        ans = answer_with(case.input, hits, llm)
        return Prediction(answer_output(ans), ans.input_tokens, ans.output_tokens)

    meta = {"part": "generation", "retrieval": "oracle" if retriever is None else "search", "k": k}
    return run_eval(cases, predict, generation_score, meta, stop_on=(LimitExceeded,))


def diagnose(retrieval: Report, generation: Report, threshold: float = 1.0) -> dict[str, str]:
    """件ごとに、どこで外れたか. 検索の点が threshold 未満なら「検索」、生成の点が 1 未満なら「生成」."""
    found = {r.id: r.score for r in retrieval.results}
    out = {}
    for r in generation.results:
        retrieval_ok = found.get(r.id, 1.0) >= threshold  # 資料に無い質問は、検索の外れとはしない
        generation_ok = r.score >= 1.0
        if retrieval_ok and generation_ok:
            out[r.id] = "ok"
        elif not retrieval_ok:
            out[r.id] = "検索" if generation_ok else "検索と生成"
        else:
            out[r.id] = "生成"
    return out


def summary(retrieval: Report, search_gen: Report, oracle_gen: Report | None) -> str:
    lines = [
        f"検索（recall@{retrieval.meta.get('k')}）: {retrieval.mean:.0%}（{len(retrieval.results)} 件）",
        f"生成（検索の結果で）: {search_gen.mean:.0%}",
        f"トークン: 入力 {search_gen.input_tokens} / 出力 {search_gen.output_tokens}",
    ]
    if oracle_gen is not None:
        lines.append(f"生成（正しい根拠を渡して）: {oracle_gen.mean:.0%}")
    lines.append("タグごと（生成・検索の結果で）: " + ", ".join(f"{t} {s:.0%}" for t, s in search_gen.by_tag().items()))
    lines.append("外れた件:")
    bad = [(cid, where) for cid, where in diagnose(retrieval, search_gen).items() if where != "ok"]
    lines += [f"  {cid}: {where}" for cid, where in bad] or ["  なし"]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="p08_eval", description="RAG の検索と生成を分けて評価する")
    p.add_argument("--eval", type=Path, default=DEFAULT_EVAL, help="評価セット（JSONL）")
    p.add_argument("--docs", type=Path, default=DEFAULT_DOCS, help="設計書（Markdown）のフォルダ")
    p.add_argument("--model", required=True, help="答えを作る Ollama のモデル名")
    p.add_argument("--embed-model", required=True, help="エンベディングの Ollama のモデル名")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Ollama の場所")
    p.add_argument("--k", type=int, default=4, help="プロンプトに入れる資料の数")
    p.add_argument("--oracle", action="store_true", help="正しい根拠を渡した生成も測る（呼び出しが倍になる）")
    p.add_argument("--max-tokens", type=int, default=300_000, help="トークン数（入力 + 出力）の上限")
    return p.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    stdout: TextIO | None = None,
) -> int:
    """llm・embedder を渡すと、Ollama の代わりにそれを使う（テスト用）."""
    args = parse_args(argv)
    stdout = stdout or sys.stdout
    try:
        cases = load_cases(args.eval)
    except (OSError, EvalSetError) as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    chunks = load_chunks(args.docs)
    retriever = build_retriever(chunks, embedder or OllamaEmbedder(args.embed_model, args.base_url))
    calls = len(cases) * (2 if args.oracle else 1)
    inner = llm or OllamaChat(args.model, args.base_url, options={"temperature": 0})
    limited = LimitedLLM(inner, calls, args.max_tokens)
    try:
        retrieval = eval_retrieval(cases, retriever, args.k)
        search_gen = eval_generation(cases, limited, retriever, chunks, args.k)
        oracle_gen = eval_generation(cases, limited, None, chunks, args.k) if args.oracle else None
    except LimitExceeded as e:
        print(f"上限に達したので止めました: {e}", file=sys.stderr)
        return 1
    print(summary(retrieval, search_gen, oracle_gen), file=stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
