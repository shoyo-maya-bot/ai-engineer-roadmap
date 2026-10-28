"""p10_agent.py のテスト（10 の成果物. モデルは FakeToolChat、承認は決まった答えを返す偽の係）."""

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

from roadmap_examples.ollama_client import ChatTurn, ToolCall
from roadmap_examples.p07_cost import Price
from roadmap_examples.p07_tools import FakeToolChat
from roadmap_examples.p08_chunk import load_chunks
from roadmap_examples.p08_rag import DEFAULT_DOCS
from roadmap_examples.p08_search import BM25
from roadmap_examples.p10_agent import (
    AgentLimits,
    ApprovalRequest,
    ConsoleApprover,
    Decision,
    FileIssueTracker,
    MemoryIssueTracker,
    main,
    qa_tools,
    run_agent,
    start_state,
)
from roadmap_examples.p10_state import ANSWERED, STOPPED, WAITING, AgentState

QUESTION = "備品を買うときの承認の流れを教えてください。資料に無ければ課題にしてください。"
ISSUE = {"title": "購入の承認の流れが設計書に無い", "body": "質問: 備品を買うときの承認の流れ"}


@dataclass
class ScriptedApprover:
    """決まった答えを順に返し、頼まれた内容を残す偽の承認係. 答えが尽きたら None（まだ決めていない）."""

    decisions: list[Decision | None]
    requests: list[ApprovalRequest] = field(default_factory=list)

    def review(self, request: ApprovalRequest) -> Decision | None:
        self.requests.append(request)
        return self.decisions.pop(0) if self.decisions else None


def tools(tracker: MemoryIssueTracker | None = None):
    return qa_tools(BM25(load_chunks(DEFAULT_DOCS)), tracker if tracker is not None else MemoryIssueTracker())


def search(query: str) -> ChatTurn:
    return ChatTurn("", [ToolCall("search_docs", {"query": query})], 100, 10)


def file_issue(args: dict | None = None) -> ChatTurn:
    return ChatTurn("", [ToolCall("create_issue", args or ISSUE)], 200, 30)


def say(text: str) -> ChatTurn:
    return ChatTurn(text, [], 150, 20)


def test_read_only_tool_runs_without_approval() -> None:
    chat = FakeToolChat([search("貸出期間"), say("14 日です（04-data.md）")])
    approver = ScriptedApprover([])
    state = run_agent(chat, start_state("備品は何日借りられますか？"), tools(), approver)
    assert state.status == ANSWERED
    assert state.answer == "14 日です（04-data.md）"
    assert approver.requests == []
    assert state.events[0]["status"] == "ok" and state.events[0]["approved"] is None
    results = json.loads(chat.calls[1][-1]["content"])["results"]
    assert any("04-data.md" in r["source"] for r in results)


def test_issue_is_created_only_after_approval() -> None:
    tracker = MemoryIssueTracker()
    chat = FakeToolChat([search("購入 承認"), file_issue(), say("ISSUE-001 を起票しました")])
    approver = ScriptedApprover([Decision(True)])
    state = run_agent(chat, start_state(QUESTION), tools(tracker), approver)
    assert state.status == ANSWERED
    assert approver.requests == [ApprovalRequest("create_issue", ISSUE)]
    assert [i.title for i in tracker.issues] == [ISSUE["title"]]
    assert state.events[1] == {"name": "create_issue", "arguments": ISSUE, "approved": True, "status": "ok"}


def test_denied_issue_is_not_created_and_model_is_told() -> None:
    tracker = MemoryIssueTracker()
    chat = FakeToolChat([file_issue(), say("起票は見送りました")])
    state = run_agent(chat, start_state(QUESTION), tools(tracker), ScriptedApprover([Decision(False, "重複です")]))
    assert tracker.issues == []
    assert state.events[0]["status"] == "denied"
    told = json.loads(chat.calls[1][-1]["content"])
    assert "承認しなかった" in told["error"] and told["comment"] == "重複です"


def test_invalid_arguments_are_returned_to_model_without_asking_human() -> None:
    chat = FakeToolChat([file_issue({"title": ""}), say("やめました")])
    approver = ScriptedApprover([Decision(True)])
    state = run_agent(chat, start_state(QUESTION), tools(), approver)
    assert approver.requests == []
    assert state.events[0]["status"] == "error"


def test_unknown_tool_is_an_error_not_an_approval() -> None:
    chat = FakeToolChat([ChatTurn("", [ToolCall("delete_all", {})], 10, 1), say("できません")])
    approver = ScriptedApprover([Decision(True)])
    state = run_agent(chat, start_state("全部消して"), tools(), approver)
    assert approver.requests == []
    assert "ありません" in json.loads(state.messages[3]["content"])["error"]


def test_waiting_for_approval_saves_state_and_resumes(tmp_path: Path) -> None:
    tracker = MemoryIssueTracker()
    first = FakeToolChat([file_issue()])
    state = run_agent(first, start_state(QUESTION), tools(tracker), ScriptedApprover([]))
    assert state.status == WAITING
    assert state.pending == [{"name": "create_issue", "arguments": ISSUE}]
    assert tracker.issues == []
    # 承認を待つ間に保存して止め、後で読み戻して続きから動かす
    path = tmp_path / "state.json"
    path.write_text(state.to_json(), encoding="utf-8")
    resumed = AgentState.from_json(path.read_text(encoding="utf-8"))
    second = FakeToolChat([say("起票しました")])
    done = run_agent(second, resumed, tools(tracker), ScriptedApprover([Decision(True)]))
    assert done.status == ANSWERED
    assert len(tracker.issues) == 1
    assert done.steps == 2  # 止める前の 1 回 + 続きの 1 回
    assert second.calls[0][-1]["role"] == "tool"


def test_same_title_is_not_created_twice() -> None:
    tracker = MemoryIssueTracker()
    chat = FakeToolChat([file_issue(), file_issue(), say("済みです")])
    state = run_agent(chat, start_state(QUESTION), tools(tracker), ScriptedApprover([Decision(True), Decision(True)]))
    assert len(tracker.issues) == 1
    assert json.loads(state.messages[-2]["content"])["created"] is False


def test_stops_at_max_steps_without_calling_model_again() -> None:
    chat = FakeToolChat([search("a"), search("b"), search("c")])
    state = run_agent(chat, start_state("?"), tools(), ScriptedApprover([]), AgentLimits(max_steps=2))
    assert state.status == STOPPED
    assert "往復の回数" in state.stop_reason
    assert len(chat.calls) == 2


def test_stops_at_max_tool_calls() -> None:
    many = ChatTurn("", [ToolCall("search_docs", {"query": str(i)}) for i in range(5)], 10, 1)
    chat = FakeToolChat([many])
    state = run_agent(chat, start_state("?"), tools(), ScriptedApprover([]), AgentLimits(max_tool_calls=3))
    assert state.status == STOPPED
    assert state.tool_calls == 3
    assert "ツールの呼び出し" in state.stop_reason


def test_stops_at_token_and_cost_limits() -> None:
    chat = FakeToolChat([search("a"), search("b")])
    state = run_agent(chat, start_state("?"), tools(), ScriptedApprover([]), AgentLimits(max_tokens=100))
    assert "トークン数" in state.stop_reason and len(chat.calls) == 1
    chat = FakeToolChat([search("a"), search("b")])
    limits = AgentLimits(price=Price(1000.0, 1000.0), max_cost=0.1)  # 110 トークンで 0.11
    state = run_agent(chat, start_state("?"), tools(), ScriptedApprover([]), limits)
    assert "費用の見積もり" in state.stop_reason and len(chat.calls) == 1


def test_file_issue_tracker_appends_jsonl(tmp_path: Path) -> None:
    tracker = FileIssueTracker(tmp_path / "issues.jsonl")
    assert tracker.find("a") is None
    first = tracker.create("a", "本文")
    second = tracker.create("b", "本文")
    assert (first.id, second.id) == ("ISSUE-001", "ISSUE-002")
    assert tracker.find("b") == second
    assert len((tmp_path / "issues.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_console_approver_needs_explicit_yes() -> None:
    out = io.StringIO()
    request = ApprovalRequest("create_issue", ISSUE)
    assert ConsoleApprover(ask=lambda _: "y", out=out).review(request) == Decision(True)
    assert ConsoleApprover(ask=lambda _: "", out=out).review(request).approved is False
    assert ConsoleApprover(ask=lambda _: "yes please", out=out).review(request).approved is False
    assert ISSUE["title"] in out.getvalue()


def test_main_with_fakes(tmp_path: Path) -> None:
    issues = tmp_path / "issues.jsonl"
    chat = FakeToolChat([search("購入 承認"), file_issue(), say("ISSUE-001 を起票しました")])
    stdout = io.StringIO()
    code = main(
        ["--question", QUESTION, "--model", "m", "--issues", str(issues)],
        chat=chat,
        approver=ScriptedApprover([Decision(True)]),
        stdout=stdout,
    )
    assert code == 0
    assert "create_issue" in stdout.getvalue() and "（承認）" in stdout.getvalue()
    assert json.loads(issues.read_text(encoding="utf-8"))["id"] == "ISSUE-001"


def test_main_returns_1_when_stopped(tmp_path: Path) -> None:
    chat = FakeToolChat([search("a")])
    code = main(
        ["--question", "?", "--model", "m", "--issues", str(tmp_path / "i.jsonl"), "--max-steps", "1"],
        chat=chat,
        approver=ScriptedApprover([]),
        stdout=io.StringIO(),
    )
    assert code == 1
