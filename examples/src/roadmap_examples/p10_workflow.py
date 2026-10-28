"""10-1 ワークフロー: 手順をコードで決め、LLM は決まった場所（答えを作る）でだけ使う.

手順: 1. 08 の Q&A で答える → 2. 資料に記載が無ければ、課題の下書きをコードで作る → 3. 人が承認したら起票する.
LLM を呼ぶのは 1 回だけで、どのツールをいつ使うかはコードが決める（エージェントのループ p10_agent と比べる）.
"""

from __future__ import annotations

from dataclasses import dataclass

from roadmap_examples.llm import LLMClient
from roadmap_examples.p08_rag import Answer, answer
from roadmap_examples.p08_search import Retriever
from roadmap_examples.p10_agent import ApprovalRequest, Approver, Decision, Issue, IssueTracker


@dataclass(frozen=True)
class WorkflowResult:
    answer: Answer
    decision: Decision | None  # 承認を頼まなかったか、まだ決まっていなければ None
    issue: Issue | None


def draft_issue(question: str) -> tuple[str, str]:
    """課題の題名と中身を、決まった形で作る（LLM を使わない）."""
    title = "設計書に記載なし: " + (question if len(question) <= 40 else question[:39] + "…")
    body = f"次の質問に答える記載が、設計書にありませんでした。\n\n質問: {question}"
    return title, body


def answer_or_file(
    question: str, retriever: Retriever, llm: LLMClient, tracker: IssueTracker, approver: Approver
) -> WorkflowResult:
    ans = answer(question, retriever, llm)
    if not ans.is_no_answer:
        return WorkflowResult(ans, None, None)
    title, body = draft_issue(question)
    decision = approver.review(ApprovalRequest("create_issue", {"title": title, "body": body}))
    if decision is None or not decision.approved:
        return WorkflowResult(ans, decision, None)
    issue = tracker.find(title) or tracker.create(title, body)
    return WorkflowResult(ans, decision, issue)
