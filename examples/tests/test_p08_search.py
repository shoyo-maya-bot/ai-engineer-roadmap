"""p08_search.py のテスト（ベクトル検索・BM25・RRF・並べ替え. エンベディングは FakeEmbedder、LLM は FakeLLM）."""

import json

import numpy as np
import pytest

from roadmap_examples.evalset import load_cases
from roadmap_examples.llm import FakeLLM
from roadmap_examples.p08_chunk import Chunk, load_chunks
from roadmap_examples.p08_eval import DEFAULT_EVAL, eval_retrieval
from roadmap_examples.p08_rag import DEFAULT_DOCS
from roadmap_examples.p08_search import (
    BM25,
    FakeEmbedder,
    Hit,
    HybridRetriever,
    VectorIndex,
    normalize,
    rerank,
    rrf,
    tokens,
)

CHUNKS = [
    Chunk("a#1", "a.md", "認証", "アクセストークンの有効期限は 30 分"),
    Chunk("b#1", "b.md", "制限", "呼び出しは 1 分間に 60 回まで。超えたら 429"),
    Chunk("c#1", "c.md", "バッチ", "返却期限の通知メールは 2:00 に送る"),
]


def test_tokens_are_bigrams_after_normalizing() -> None:
    assert tokens("ＡＰＩ の制限") == ["ap", "pi", "の制", "制限"]  # 全角は半角に、大文字は小文字に
    assert tokens("E-101") == ["e", "10", "01"]


def test_normalize_makes_unit_rows_and_keeps_zero_rows() -> None:
    m = normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert m[0] == pytest.approx([0.6, 0.8])
    assert list(m[1]) == [0.0, 0.0]


def test_vector_index_finds_the_closest_chunk() -> None:
    index = VectorIndex(FakeEmbedder(), CHUNKS)
    hits = index.search("トークンの有効期限", k=2)
    assert hits[0].chunk.id == "a#1"
    assert hits[0].score > hits[1].score


def test_bm25_ranks_rare_words_higher_and_drops_zero_scores() -> None:
    bm25 = BM25(CHUNKS)
    hits = bm25.search("429 が返る", k=3)
    assert [h.chunk.id for h in hits] == ["b#1"]  # 語が 1 つも重ならない資料は返さない


def test_rrf_merges_by_rank() -> None:
    a, b, c = (Hit(ch, 0.0) for ch in CHUNKS)
    fused = rrf([[a, b], [b, c]], k=60)
    assert [h.chunk.id for h in fused] == ["b#1", "a#1", "c#1"]  # 両方に出た b がいちばん上
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_hybrid_retrieval_on_design_docs() -> None:
    chunks = load_chunks(DEFAULT_DOCS)
    retriever = HybridRetriever(VectorIndex(FakeEmbedder(), chunks), BM25(chunks))
    report = eval_retrieval(load_cases(DEFAULT_EVAL), retriever, k=4)
    assert report.mean == 1.0  # 12 問（資料に答えがある問い）の根拠の文書が、すべて上位 4 件に入る
    assert len(report.results) == 12


def test_rerank_orders_by_llm_scores() -> None:
    hits = [Hit(ch, 0.0) for ch in CHUNKS]
    fake = FakeLLM.of([json.dumps({"scores": [1, 9, 5]})])
    ranked = rerank(fake, "何回まで呼べる？", hits, top_n=2)
    assert [h.chunk.id for h in ranked] == ["b#1", "c#1"]
    assert "[2]" in fake.calls[0][-1].content


@pytest.mark.parametrize("reply", ["点数は 9 です", json.dumps({"scores": [1, 2]}), json.dumps({"x": 1})])
def test_rerank_keeps_original_order_when_reply_is_unusable(reply: str) -> None:
    hits = [Hit(ch, 0.0) for ch in CHUNKS]
    ranked = rerank(FakeLLM.of([reply]), "質問", hits, top_n=2)
    assert [h.chunk.id for h in ranked] == ["a#1", "b#1"]
