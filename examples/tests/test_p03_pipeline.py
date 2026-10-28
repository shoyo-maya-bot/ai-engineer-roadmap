"""p03_pipeline（03 の成果物）のテスト. 入力は data/p03/ の架空のデータと、テストの中で作る小さな CSV."""

import io
from pathlib import Path

import pandas as pd
import pytest

from roadmap_examples.evalset import load_cases
from roadmap_examples.p03_pipeline import (
    CATEGORY_ALIASES,
    CHANNEL_ALIASES,
    DATA,
    OUTPUT_COLUMNS,
    PipelineError,
    main,
    map_aliases,
    normalize_customer_id,
    normalize_text,
    parse_minutes,
    parse_received_at,
    read_clean,
    run,
    validate,
    write_csv,
)
from roadmap_examples.p07_classify import DEFAULT_EVAL
from roadmap_examples.p07_prompt import CATEGORIES

CLEAN = DATA / "inquiries_clean.csv"


def test_normalize_text() -> None:
    s = pd.Series(["  ＣＳＶ を出力\nしたい　", "ﾒｰﾙ", "　", None], dtype="str")
    assert normalize_text(s).tolist()[:2] == ["CSV を出力 したい", "メール"]
    assert normalize_text(s).isna().tolist()[2:] == [True, True]  # 空白だけ・空欄は欠損


def test_map_aliases() -> None:
    s = pd.Series(["E-mail", "ﾒｰﾙ", " TEL ", "Web フォーム", "FAX", None], dtype="str")
    assert map_aliases(s, CHANNEL_ALIASES).tolist()[:4] == ["email", "email", "phone", "form"]
    assert map_aliases(s, CHANNEL_ALIASES).isna().tolist()[4:] == [True, True]  # 対応表に無い値も欠損
    assert map_aliases(pd.Series(["Billing ", "請求", "billing/contract"]), CATEGORY_ALIASES).tolist()[:2] == [
        "billing",
        "billing",
    ]


def test_parse_received_at_accepts_three_formats() -> None:
    s = pd.Series(["2026-04-03 09:15", "2026/4/3 9:15", "2026年4月3日 9:15", "不明", "2026-02-30 10:00"])
    got = parse_received_at(s)
    assert (got[:3] == pd.Timestamp("2026-04-03 09:15")).all()
    assert got[3:].isna().all()  # 読めない文字列と、無い日付（2 月 30 日）は欠損


def test_parse_minutes() -> None:
    minutes, out_of_range = parse_minutes(pd.Series(["25", "25分", "", "-1", "99999", "abc"]).replace("", None))
    assert minutes.tolist()[:2] == [25, 25]
    assert minutes[2:].isna().all()
    assert out_of_range == 2  # -1 と 99999. 「abc」は数でないだけ（範囲外には数えない）


def test_normalize_customer_id() -> None:
    s = pd.Series(["K-0003", "K0003", "k-0003", "Ｋ－０００３", "X-1", None], dtype="str")
    assert normalize_customer_id(s).tolist()[:3] == ["K-0003"] * 3
    assert normalize_customer_id(s)[4:].isna().all()


def test_run_on_sample_data_reports_every_drop() -> None:
    df, report = run()
    assert report.rows_read == 147
    assert (report.duplicates_exact, report.duplicates_updated) == (4, 2)
    assert report.dropped == {"本文が空": 3, "分類が空": 6, "分類が 5 種類のどれでもない": 2}
    assert report.fixed == {"受付日時が読めない": 1, "対応時間が範囲外": 5, "顧客の一覧に無い顧客": 2}
    assert report.rows_out == len(df) == 130
    assert report.by_category == dict.fromkeys(CATEGORIES, 26)


def test_output_shape_matches_07_eval_set() -> None:
    df, _ = run()
    assert list(df.columns) == OUTPUT_COLUMNS
    assert set(df["category"]) == set(CATEGORIES)  # 07 の評価セットと同じ 5 種類
    assert set(df["channel"]) == {"email", "phone", "chat", "form", "unknown"}
    assert set(df["plan"]) == {"free", "standard", "pro", "unknown"}
    eval_inputs = {c.input for c in load_cases(DEFAULT_EVAL)}
    assert not eval_inputs & set(df["input"])  # 評価セットと同じ文を学習に入れない（答えを見せた試験になる）


def test_later_export_wins_for_updated_rows() -> None:
    df, _ = run()
    q13 = df.loc[df["id"] == "Q0013"].iloc[0]
    assert q13["response_minutes"] == 52  # 元の行は 37 分. 後から出力された行（52 分）を残す


def test_committed_clean_csv_is_up_to_date(tmp_path: Path) -> None:
    # data/p03/inquiries_clean.csv は、このパイプラインの出力そのもの（手で直さない）
    out = tmp_path / "clean.csv"
    df, _ = run()
    write_csv(df, out)
    # Windows の checkout で改行が CRLF に変わっても比べられるように、改行をそろえてから比べる
    assert out.read_text(encoding="utf-8") == CLEAN.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_rerun_gives_identical_output(tmp_path: Path) -> None:
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    assert main(["--out", str(a)], stdout=io.StringIO()) == 0
    assert main(["--out", str(b)], stdout=io.StringIO()) == 0
    assert a.read_bytes() == b.read_bytes()


def test_read_clean_restores_types() -> None:
    df = read_clean(CLEAN)
    assert str(df["received_at"].dtype).startswith("datetime64")
    assert str(df["response_minutes"].dtype) == "Int64"
    assert df["received_at"].isna().sum() == 1 and df["response_minutes"].isna().sum() == 9
    validate(df)


def test_main_writes_cases_in_eval_set_format(tmp_path: Path) -> None:
    out, cases_path, stdout = tmp_path / "clean.csv", tmp_path / "train.jsonl", io.StringIO()
    assert main(["--out", str(out), "--cases", str(cases_path)], stdout=stdout) == 0
    cases = load_cases(cases_path)  # 07 の評価セットと同じ読み込み関数で読める
    assert len(cases) == 130 and cases[0].expected == {"category": "billing"}
    assert "捨てた行（分類が空）: 6" in stdout.getvalue()


def test_validate_rejects_broken_table() -> None:
    df = read_clean(CLEAN)
    with pytest.raises(PipelineError, match="分類が 5 種類以外"):
        validate(df.assign(category=df["category"].where(df.index > 0, "請求")))
    with pytest.raises(PipelineError, match="id が重複"):
        validate(pd.concat([df, df.head(1)], ignore_index=True))


def test_missing_column_is_an_error(tmp_path: Path, capsys) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text("受付番号,本文\nQ1,テスト\n", encoding="utf-8")
    assert main(["--raw", str(raw), "--out", str(tmp_path / "o.csv")]) == 1
    assert "列" in capsys.readouterr().err


def test_na_like_text_is_kept(tmp_path: Path) -> None:
    # 「NA」「NULL」は文字として残す（pandas の既定では欠損になる）
    raw = tmp_path / "raw.csv"
    raw.write_text(
        "受付番号,受付日時,顧客ID,窓口,本文,分類,対応時間（分）\nQ1,2026-04-01 10:00,K-0001,メール,NA,other,5\n",
        encoding="utf-8",
    )
    df, _ = run(raw, DATA / "customers.csv")
    assert df["input"].tolist() == ["NA"]
