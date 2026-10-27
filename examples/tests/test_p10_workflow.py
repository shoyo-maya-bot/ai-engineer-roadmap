"""p10_workflow.py のテスト（10-1 手順をコードで決めるワークフロー. LLM は FakeLLM）."""

from roadmap_examples.llm import FakeLLM
from roadmap_examples.p08_chunk import load_chunks
from roadmap_examples.p08_rag import DEFAULT_DOCS, NO_ANSWER
from roadmap_examples.p08_search import BM25
from roadmap_examples.p10_agent import ApprovalRequest, Decision, MemoryIssueTracker
from roadmap_examples.p10_workflow import answer_or_file, draft_issue

RETRIEVER = BM25(load_chunks(DEFAULT_DOCS))


class Approver:
    def __init__(self, decision: Decision | None) -> None:
        self.decision = decision
        self.requests: list[ApprovalRequest] = []

    def review(self, request: ApprovalRequest) -> Decision | None:
        self.requests.append(request)
        return self.decision


def test_answered_question_does_not_ask_for_approval() -> None:
    approver = Approver(Decision(True))
    result = answer_or_file(
        "備品は何日借りられますか？", RETRIEVER, FakeLLM.of(["14 日です [1]"]), MemoryIssueTracker(), approver
    )
    assert result.issue is None and result.decision is None
    assert approver.requests == []


def test_no_answer_files_issue_after_approval() -> None:
    tracker = MemoryIssueTracker()
    approver = Approver(Decision(True))
    question = "備品を買うときの承認の流れを教えてください。"
    result = answer_or_file(question, RETRIEVER, FakeLLM.of([NO_ANSWER]), tracker, approver)
    assert result.issue is not None and result.issue.id == "ISSUE-001"
    assert approver.requests[0].arguments == dict(zip(("title", "body"), draft_issue(question), strict=True))


def test_no_answer_without_approval_files_nothing() -> None:
    tracker = MemoryIssueTracker()
    for decision in (Decision(False), None):
        result = answer_or_file("対応ブラウザは？", RETRIEVER, FakeLLM.of([NO_ANSWER]), tracker, Approver(decision))
        assert result.issue is None
    assert tracker.issues == []


def test_draft_issue_shortens_long_questions() -> None:
    title, body = draft_issue("あ" * 100)
    assert len(title) == len("設計書に記載なし: ") + 40
    assert body.endswith("あ" * 100)
