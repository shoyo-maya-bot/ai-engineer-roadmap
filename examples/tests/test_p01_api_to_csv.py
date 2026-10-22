"""p01_api_to_csv のテスト（HTTP は偽の実装に差し替える）."""

import io
import json
from typing import Any

import pytest

from roadmap_examples import p01_api_to_csv as cli


class FakeGetter:
    """決まった JSON を返す偽の実装. 呼ばれた URL を残す."""

    def __init__(self, data: Any = None, error: Exception | None = None) -> None:
        self.data = data
        self.error = error
        self.urls: list[str] = []

    def get_json(self, url: str, timeout: float) -> Any:
        self.urls.append(url)
        if self.error:
            raise self.error
        return self.data


ITEMS = [
    {"id": 1, "name": "会員 A", "owner": {"name": "営業部"}, "tags": ["新規"]},
    {"id": 2, "name": "会員, B", "owner": None},
]


def run(argv: list[str], getter: FakeGetter) -> tuple[int, str]:
    out = io.StringIO()
    code = cli.main(argv, getter=getter, stdout=out)
    return code, out.getvalue()


def test_writes_selected_fields_as_csv() -> None:
    getter = FakeGetter(ITEMS)
    code, out = run(["--url", "https://api.example.com/items", "--fields", "id,name,owner.name,tags"], getter)
    assert code == 0
    assert getter.urls == ["https://api.example.com/items"]
    assert out.splitlines() == [
        "id,name,owner.name,tags",
        '1,会員 A,営業部,"[""新規""]"',
        '2,"会員, B",,',  # カンマは引用符で囲む. 無い項目は空欄
    ]


def test_items_key_points_to_nested_array(tmp_path) -> None:
    getter = FakeGetter({"data": {"items": ITEMS[:1]}})
    dest = tmp_path / "out.csv"
    argv = ["--url", "https://api.example.com/s", "--items-key", "data.items", "--fields", "id", "-o", str(dest)]
    code, out = run(argv, getter)
    assert (code, out) == (0, "")
    assert dest.read_text(encoding="utf-8") == "id\n1\n"


@pytest.mark.parametrize(
    ("data", "items_key", "message"),
    [
        ({"items": []}, None, "応答の最上位が配列ではありません"),
        ({"data": {}}, "data.items", "「data.items」が配列ではありません"),
        ([1, 2], None, "配列の要素がオブジェクトではありません"),
    ],
)
def test_unexpected_shape_is_an_error(data, items_key, message, capsys) -> None:
    argv = ["--url", "https://api.example.com/x", "--fields", "id"] + (["--items-key", items_key] if items_key else [])
    code, out = run(argv, FakeGetter(data))
    assert (code, out) == (1, "")
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("error", [OSError("接続できません"), json.JSONDecodeError("JSON ではありません", "<html>", 0)])
def test_fetch_failure_is_an_error(error, capsys) -> None:
    code, _ = run(["--url", "https://api.example.com/x", "--fields", "id"], FakeGetter(error=error))
    assert code == 1
    assert "エラー:" in capsys.readouterr().err


def test_empty_fields_is_an_argument_error() -> None:
    with pytest.raises(SystemExit) as e:
        run(["--url", "https://api.example.com/x", "--fields", " , "], FakeGetter([]))
    assert e.value.code == 2


def test_real_getter_rejects_non_http_urls() -> None:
    with pytest.raises(ValueError, match="http"):
        cli.UrllibJsonGetter().get_json("file:///etc/passwd", timeout=1)


def test_network_is_blocked_in_tests() -> None:
    # conftest.py の no_network が効いていること（本物の getter を使うと接続の前で止まる）
    with pytest.raises(RuntimeError, match="ネットワーク"):
        cli.UrllibJsonGetter().get_json("https://api.example.com/x", timeout=1)
