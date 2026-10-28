"""p10_state.py のテスト（10-2 エージェントの状態の保存と、会話の削り方）."""

from pathlib import Path

from roadmap_examples.p10_state import WAITING, AgentState, load_state, save_state, trim_messages

SYSTEM = {"role": "system", "content": "決まり"}
ASK = {"role": "user", "content": "依頼"}


def call(n: int) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "search_docs", "arguments": {}}}],
        "n": n,
    }


def result(n: int) -> dict:
    return {"role": "tool", "tool_name": "search_docs", "content": f"結果 {n}"}


def test_state_round_trips_through_json(tmp_path: Path) -> None:
    state = AgentState([SYSTEM, ASK], steps=2, input_tokens=100, output_tokens=20, status=WAITING)
    state.pending.append({"name": "create_issue", "arguments": {"title": "t", "body": "b"}})
    path = tmp_path / "state.json"
    save_state(state, path)
    loaded = load_state(path)
    assert loaded == state
    assert loaded.tokens == 120


def test_trim_keeps_system_and_request_and_recent_messages() -> None:
    messages = [SYSTEM, ASK, call(1), result(1), call(2), result(2), {"role": "assistant", "content": "答え"}]
    trimmed = trim_messages(messages, keep_last=3)
    assert trimmed[:2] == [SYSTEM, ASK]
    assert trimmed[2:] == [call(2), result(2), {"role": "assistant", "content": "答え"}]


def test_trim_does_not_start_with_an_orphan_tool_result() -> None:
    messages = [SYSTEM, ASK, call(1), result(1), call(2), result(2)]
    trimmed = trim_messages(messages, keep_last=3)  # そのまま切ると result(1) から始まる
    assert trimmed == [SYSTEM, ASK, call(2), result(2)]


def test_trim_returns_everything_when_short() -> None:
    messages = [SYSTEM, ASK, call(1)]
    assert trim_messages(messages, keep_last=5) == messages
    assert trim_messages([ASK, call(1), result(1)], keep_last=1) == [ASK]
