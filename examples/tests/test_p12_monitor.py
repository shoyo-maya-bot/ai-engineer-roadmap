"""p12_monitor.py のテスト（記録の集計・しきい値・監視の画面）."""

import io
from pathlib import Path

from roadmap_examples.p12_monitor import Thresholds, alerts, by_version, load_records, main, percentile, stats

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "p12" / "requests-sample.jsonl"


def test_percentile_nearest_rank() -> None:
    values = list(range(1, 101))
    assert percentile(values, 50) == 50
    assert percentile(values, 95) == 95
    assert percentile([7.0], 95) == 7.0
    assert percentile([], 95) == 0.0


def test_stats_and_versions_of_sample() -> None:
    records = load_records(SAMPLE)
    s = stats(records)
    assert s.count == 40
    assert s.statuses == {"answered": 34, "stopped": 2, "waiting_approval": 2, "blocked": 2}
    assert s.findings == {"指示の上書きの形があります": 2, "電話番号": 1}
    versions = by_version(records)
    assert set(versions) == {"a1b2c3d4e5f6", "0f9e8d7c6b5a"}
    assert versions["a1b2c3d4e5f6"].rate("stopped") == 0.1


def test_alerts() -> None:
    s = stats(load_records(SAMPLE))
    assert alerts(s, Thresholds()) == []
    found = alerts(s, Thresholds(max_p95_ms=1000, max_stopped_rate=0.01, max_cost=-1))
    assert len(found) == 3


def test_main_writes_html_and_returns_1_over_threshold(tmp_path: Path) -> None:
    page = tmp_path / "monitor.html"
    assert main([str(SAMPLE), "--html", str(page)], stdout=io.StringIO()) == 0
    text = page.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>") and "a1b2c3d4e5f6" in text and "<script" not in text
    stdout = io.StringIO()
    assert main([str(SAMPLE), "--max-p95-ms", "1000"], stdout=stdout) == 1
    assert "しきい値を超えました" in stdout.getvalue()
    assert main([str(tmp_path / "missing.jsonl")], stdout=io.StringIO()) == 1
