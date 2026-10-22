"""p08_rag.py と p08_eval.py のテスト（出典つきの答えと、検索・生成を分けた評価. LLM は FakeLLM）."""

import io

from roadmap_examples.evalset import EvalCase, load_cases
from roadmap_examples.llm import FakeLLM
from roadmap_examples.p08_chunk import Chunk, load_chunks
from roadmap_examples.p08_eval import (
    DEFAULT_EVAL,
    diagnose,
    eval_generation,
    eval_retrieval,
    generation_score,
    main,
)
from roadmap_examples.p08_rag import (
    DEFAULT_DOCS,
    NO_ANSWER,
    answer,
    answer_with,
    build_messages,
    build_retriever,
    format_answer,
    parse_citations,
)
from roadmap_examples.p08_search import FakeEmbedder, Hit

CHUNKS = [
    Chunk("02-api.md#3", "02-api.md", "API > 制限", "1 分間に 60 回まで"),
    Chunk("03-batch.md#1", "03-batch.md", "バッチ > 一覧", "2:00 に通知"),
]
HITS = [Hit(c, 1.0) for c in CHUNKS]


def test_build_messages_numbers_sources_with_citation_labels() -> None:
    messages = build_messages("何回まで？", HITS)
    assert "<資料>\n[1] 02-api.md > API > 制限\n" in messages[1].content
    assert "[2] 03-batch.md > バッチ > 一覧" in messages[1].content
    assert messages[1].content.endswith("質問: 何回まで？")
    assert NO_ANSWER in messages[0].content


def test_parse_citations_splits_valid_and_invalid() -> None:
    assert parse_citations("60 回です [1]。通知は 2:00 [2][1]。[5]", 2) == ([1, 2], [5])


def test_answer_with_maps_numbers_to_sources() -> None:
    ans = answer_with("何回まで？", HITS, FakeLLM.of(["1 分間に 60 回までです [1]。"]))
    assert [c.source for c in ans.sources] == ["02-api.md"]
    text = format_answer(ans)
    assert "出典:\n  [1] 02-api.md > API > 制限" in text


def test_format_answer_warns_about_missing_or_invalid_citations() -> None:
    no_cite = answer_with("q", HITS, FakeLLM.of(["60 回です。"]))
    assert "出典が示されていません" in format_answer(no_cite)
    invalid = answer_with("q", HITS, FakeLLM.of(["60 回です [3]。"]))
    assert "資料に無い番号" in format_answer(invalid)
    no_answer = answer_with("q", HITS, FakeLLM.of([NO_ANSWER]))
    assert format_answer(no_answer) == NO_ANSWER


def test_answer_without_hits_does_not_call_llm() -> None:
    fake = FakeLLM.of([])
    assert answer_with("q", [], fake).text == NO_ANSWER
    assert fake.calls == []


def test_answer_searches_then_generates() -> None:
    retriever = build_retriever(load_chunks(DEFAULT_DOCS), FakeEmbedder())
    fake = FakeLLM.of(["60 回までです [1]。"])
    ans = answer("API は 1 分間に何回まで呼べますか？", retriever, fake, k=3)
    assert len(ans.hits) == 3
    assert "02-api.md" in fake.calls[0][-1].content


CASE = EvalCase("q", "何回まで？", {"sources": ["02-api.md"], "keywords": ["60 回", "429"]})
NO_DOC_CASE = EvalCase("n", "ブラウザは？", {"sources": [], "keywords": []})


def output(text: str, cited: list[str], invalid: list[int] | None = None) -> dict:
    return {"text": text, "cited_sources": cited, "invalid": invalid or [], "no_answer": NO_ANSWER in text}


def test_generation_score() -> None:
    assert generation_score(CASE, output("60 回まで。超えると 429", ["02-api.md"])) == 1.0
    assert generation_score(CASE, output("60 回まで", ["02-api.md"])) == 0.5  # 語が半分
    assert generation_score(CASE, output("60 回まで。超えると 429", [])) == 0.5  # 出典なし
    assert generation_score(CASE, output("60 回まで。超えると 429", ["02-api.md"], [7])) == 0.0
    assert generation_score(NO_DOC_CASE, output(NO_ANSWER, [])) == 1.0
    assert generation_score(NO_DOC_CASE, output("Chrome です", ["01-overview.md"])) == 0.0


def test_eval_generation_with_oracle_passes_only_gold_chunks() -> None:
    chunks = load_chunks(DEFAULT_DOCS)
    fake = FakeLLM.of(["1 分間に 60 回まで。超えると 429 を返します [1]。"])
    report = eval_generation([CASE], fake, None, chunks, k=4)
    assert report.results[0].score == 1.0
    prompt = fake.calls[0][-1].content
    assert "02-api.md" in prompt and "03-batch.md" not in prompt


def test_diagnose_separates_retrieval_and_generation_failures() -> None:
    chunks = load_chunks(DEFAULT_DOCS)
    cases = [
        EvalCase("ok", "API は 1 分間に何回まで呼べますか？", {"sources": ["02-api.md"], "keywords": ["60 回"]}),
        EvalCase("gen", "API は 1 分間に何回まで呼べますか？", {"sources": ["02-api.md"], "keywords": ["600 回"]}),
        EvalCase("ret", "延滞の集計", {"sources": ["04-data.md"], "keywords": ["14 日"]}),
    ]
    retriever = build_retriever(chunks, FakeEmbedder())
    retrieval = eval_retrieval(cases, retriever, k=1)
    fake = FakeLLM.of(["60 回まで [1]", "60 回まで [1]", "14 日です [1]"])
    generation = eval_generation(cases, fake, retriever, chunks, k=1)
    assert diagnose(retrieval, generation) == {"ok": "ok", "gen": "生成", "ret": "検索と生成"}


def test_main_with_fakes() -> None:
    cases = load_cases(DEFAULT_EVAL)
    replies = [NO_ANSWER if not c.expected["sources"] else " ".join(c.expected["keywords"]) + " [1]" for c in cases]
    stdout = io.StringIO()
    code = main(["--model", "m", "--embed-model", "e"], llm=FakeLLM.of(replies), embedder=FakeEmbedder(), stdout=stdout)
    text = stdout.getvalue()
    assert code == 0
    assert "検索（recall@4）: 100%（12 件）" in text


def test_main_stops_at_limit() -> None:
    stdout = io.StringIO()
    code = main(
        ["--model", "m", "--embed-model", "e", "--max-tokens", "1"],
        llm=FakeLLM.of(["x [1]"] * 20),
        embedder=FakeEmbedder(),
        stdout=stdout,
    )
    assert code == 1
