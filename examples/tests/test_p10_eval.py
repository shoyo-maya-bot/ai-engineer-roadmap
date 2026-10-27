"""p10_eval.py のテスト（10-4 エージェントの評価. モデルは件ごとに決めた FakeToolChat）."""

import io

from roadmap_examples.evalset import EvalCase, load_cases, run_eval
from roadmap_examples.ollama_client import ChatTurn, ToolCall
from roadmap_examples.p07_tools import FakeToolChat
from roadmap_examples.p10_eval import DEFAULT_EVAL, agent_score, checks, main, make_predict

SEARCH = ChatTurn("", [ToolCall("search_docs", {"query": "q"})], 100, 10)
ISSUE = ChatTurn("", [ToolCall("create_issue", {"title": "記載なし", "body": "質問"})], 100, 10)


def say(text: str) -> ChatTurn:
    return ChatTurn(text, [], 100, 10)


def ideal_turns(case: EvalCase) -> list[ChatTurn]:
    """評価セットの期待どおりに動くモデル（の台本）."""
    turns = [SEARCH] if case.expected["tools"] else []
    if case.expected["issue"]:
        turns.append(ISSUE)
    return [*turns, say(" ".join(case.expected.get("keywords", [])) or "どういたしまして")]


def test_eval_set_is_valid() -> None:
    cases = load_cases(DEFAULT_EVAL)
    assert len(cases) == 8
    assert all({"issue", "tools"} <= set(c.expected) for c in cases)
    assert sum(c.expected["issue"] for c in cases) == 2


def test_ideal_agent_scores_full_marks_and_creates_nothing() -> None:
    cases = load_cases(DEFAULT_EVAL)
    report = run_eval(cases, make_predict(lambda c: FakeToolChat(ideal_turns(c))), agent_score)
    assert report.mean == 1.0
    proposed = [r.id for r in report.results if r.output["proposed_issue"]]
    assert proposed == ["a04", "a05"]
    # 承認の係がすべて却下するので、評価では起票されない
    assert all(
        e["status"] == "denied" for r in report.results for e in r.output["events"] if e["name"] == "create_issue"
    )


def test_unrequested_issue_and_missing_search_lose_points() -> None:
    case = EvalCase("x", "備品を買うときの承認の流れを教えてください。", {"issue": False, "tools": ["search_docs"]})
    predict = make_predict(lambda _: FakeToolChat([ISSUE, say("起票を提案しました")]))
    output = predict(case).output
    found = checks(case, output)
    assert found["起票の提案"] is False and found["呼ぶべきツール"] is False
    assert found["答えた（上限の中で）"] is True
    assert agent_score(case, output) == 0.5


def test_main_prints_summary() -> None:
    stdout = io.StringIO()
    code = main(["--model", "m"], make_chat=lambda c: FakeToolChat(ideal_turns(c)), stdout=stdout)
    assert code == 0
    assert "点の平均: 100%（8 件）" in stdout.getvalue()
    assert "外れた件:\n  なし" in stdout.getvalue()


def test_main_lists_failed_checks() -> None:
    stdout = io.StringIO()
    main(["--model", "m", "--max-steps", "1"], make_chat=lambda c: FakeToolChat(ideal_turns(c)), stdout=stdout)
    assert "a01: 答えた（上限の中で）" in stdout.getvalue()
