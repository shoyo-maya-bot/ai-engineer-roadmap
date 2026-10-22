"""テストの共通設定: テスト中はネットワークに接続させない（LLM や外部の API を本当に呼ばない）."""

import socket

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args, **_kwargs):
        raise RuntimeError("テスト中にネットワークへ接続しようとしました（偽の実装に差し替えてください）")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
