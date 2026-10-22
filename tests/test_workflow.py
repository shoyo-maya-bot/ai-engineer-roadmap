"""CI（.github/workflows）の約束のテスト.

- 公開（deploy）は、定期実行だけのジョブ（links）を除くすべての検査のジョブを needs に持つ（検査が落ちたら公開しない）
- 週 1 回の起票は、すべての検査のジョブを見る
- 外部リンクの検査は定期実行と手動実行だけ
- actions はコミット SHA で固定し、どのジョブにも timeout-minutes がある. ワークフローの既定の権限は読み取りだけ
- 同じ PR の古い実行は取り消し、公開は 1 つずつ順番に行う
- テストと CI は LLM を呼ばない（キーをシークレットから渡していない）
（テストのジョブは YAML のライブラリに頼らず、文字列で見る）
"""

import re
from pathlib import Path

import pytest

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github/workflows"
CI = WORKFLOWS / "ci.yml"
NOT_CHECKS = {"deploy", "report-scheduled-failure"}
SCHEDULE_ONLY = {"links"}  # 外部のサイトの都合で結果が変わる検査. push・PR では動かさない


def jobs(text: str) -> dict[str, str]:
    """ジョブ名 → そのジョブの本文（jobs: の直下、2 字下げのキー）."""
    body = text.split("\njobs:\n", 1)[1]
    parts = re.split(r"^  ([\w-]+):\s*$", body, flags=re.M)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def needs(job: str) -> set[str]:
    m = re.search(r"^    needs:\s*\[(?P<list>[^\]]*)\]", job, re.M)
    return {x.strip() for x in m.group("list").split(",")} if m else set()


@pytest.fixture
def ci_jobs():
    return jobs(CI.read_text(encoding="utf-8"))


def test_deploy_needs_every_check(ci_jobs):
    checks = set(ci_jobs) - NOT_CHECKS
    assert {"docs", "tests", "windows", "lint", "security"} <= checks
    assert needs(ci_jobs["deploy"]) == checks - SCHEDULE_ONLY


def test_scheduled_failure_report_watches_every_check(ci_jobs):
    assert needs(ci_jobs["report-scheduled-failure"]) == set(ci_jobs) - NOT_CHECKS


def test_link_check_runs_only_on_schedule_and_manual_runs(ci_jobs):
    for name in SCHEDULE_ONLY:
        m = re.search(r"(?m)^    if: (?P<cond>.+)$", ci_jobs[name])
        assert m, f"{name} に if がありません（push・PR でも動いてしまう）"
        assert m.group("cond") == "github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'"
    assert "scripts/check_links.py site" in ci_jobs["links"]
    report = ci_jobs["report-scheduled-failure"]
    assert "needs.links.outputs.report" in report
    assert "if: env.LINKS_RESULT == 'failure'" in report


def test_deploy_runs_one_at_a_time_and_only_on_push(ci_jobs):
    deploy = ci_jobs["deploy"]
    assert re.search(r"(?m)^    concurrency:\n      group: pages\n      cancel-in-progress: false$", deploy)
    assert re.search(r"(?m)^    if: github.event_name == 'push'$", deploy)
    assert "pages: write" in deploy and "id-token: write" in deploy


def test_outdated_pull_request_runs_are_cancelled():
    top = CI.read_text(encoding="utf-8").split("\njobs:\n", 1)[0]
    m = re.search(r"(?m)^concurrency:\n  group: (?P<group>.+)\n  cancel-in-progress: (?P<cancel>.+)$", top)
    assert m, "ワークフローに concurrency がありません"
    assert "github.event_name" in m.group("group")
    assert "pull_request.number" in m.group("group")
    assert m.group("cancel") == "${{ github.event_name == 'pull_request' }}"


@pytest.mark.parametrize("path", sorted(WORKFLOWS.glob("*.yml")), ids=lambda p: p.name)
def test_actions_pinned_permissions_and_timeouts(path):
    text = path.read_text(encoding="utf-8")
    for ref in re.findall(r"uses:\s*(\S+)", text):
        assert re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", ref), f"{path.name}: {ref} を SHA で固定してください"
    assert re.search(r"(?m)^permissions:\n  contents: read$", text), f"{path.name}: 既定の権限を読み取りだけに"
    assert re.search(r"(?m)^concurrency:$", text), f"{path.name}: concurrency がありません"
    for name, job in jobs(text).items():
        assert re.search(r"(?m)^    timeout-minutes: \d+$", job), f"{path.name}: {name} に timeout-minutes がありません"


@pytest.mark.parametrize("path", sorted(WORKFLOWS.glob("*.yml")), ids=lambda p: p.name)
def test_no_llm_keys_in_workflows(path):
    text = path.read_text(encoding="utf-8")
    secrets = set(re.findall(r"\$\{\{\s*secrets\.(\w+)", text))
    assert secrets <= {"GITHUB_TOKEN"}, f"{path.name}: CI にシークレット（{secrets}）を渡さない（LLM を呼ばない）"
    assert "API_KEY" not in text


def test_dependencies_are_installed_with_hashes(ci_jobs):
    for name, job in ci_jobs.items():
        for line in job.splitlines():
            if "pip install" in line and "constraints.txt" in line:
                assert "pip install --require-hashes -r constraints.txt" in line, name
            if "uv sync" in line:
                assert "--locked" in line, name


def test_parser_sees_needs():
    assert needs("    needs: [a, b]\n") == {"a", "b"}
