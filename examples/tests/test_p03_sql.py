"""p03_sql のテスト. SQL と pandas の結果が同じことを確かめる."""

import pandas as pd
import pytest

from roadmap_examples.p03_pipeline import (
    CUSTOMER_COLUMNS,
    DATA,
    DEFAULT_CUSTOMERS,
    DEFAULT_RAW,
    RAW_COLUMNS,
    read_clean,
    read_csv_as_text,
)
from roadmap_examples.p03_sql import (
    CHANNEL_COUNTS,
    INQUIRIES_OF_CUSTOMER,
    JOIN_ROW_COUNTS,
    MINUTES_BY_CHANNEL,
    channel_counts,
    connect,
    minutes_by_channel,
    query,
)


@pytest.fixture
def raw_conn():
    raw = read_csv_as_text(DEFAULT_RAW, RAW_COLUMNS)
    customers = read_csv_as_text(DEFAULT_CUSTOMERS, CUSTOMER_COLUMNS)
    conn = connect({"inquiries": raw, "customers": customers})
    yield raw, conn
    conn.close()


@pytest.fixture
def clean():
    return read_clean(DATA / "inquiries_clean.csv")


def test_raw_channel_counts_show_spelling_variants(raw_conn) -> None:
    raw, conn = raw_conn
    sql = query(conn, CHANNEL_COUNTS)
    pd.testing.assert_frame_equal(sql, channel_counts(raw), check_dtype=False)
    email_like = {"メール", "mail", "E-mail", "ﾒｰﾙ"}
    assert email_like <= set(sql["channel"])  # 同じ「メール」が 4 つの行に分かれる
    assert len(sql) == 14


def test_inner_join_silently_drops_rows(raw_conn) -> None:
    _, conn = raw_conn
    counts = query(conn, JOIN_ROW_COUNTS).iloc[0].to_dict()
    assert counts == {"inquiries": 147, "inner_join": 126, "left_join": 147}


def test_minutes_by_channel_sql_equals_pandas(clean) -> None:
    conn = connect({"inquiries": clean})
    sql = query(conn, MINUTES_BY_CHANNEL, {"min_rows": 10})
    pdf = minutes_by_channel(clean, min_rows=10)
    pd.testing.assert_frame_equal(sql, pdf, check_dtype=False)
    assert sql["channel"].tolist() == ["phone", "email", "chat", "form"]  # unknown（7 件）は HAVING で落ちる
    email = sql.set_index("channel").loc["email"]
    assert (email["n"], email["n_known"]) == (61, 56)  # 平均は、対応時間がある 56 件だけで計算する


def test_placeholder_treats_input_as_a_value(clean) -> None:
    conn = connect({"inquiries": clean})
    assert len(query(conn, INQUIRIES_OF_CUSTOMER, {"customer_id": "K-0031"})) == 5
    # 文字列をつなげて SQL を作ると条件を書き換えられる入力も、プレースホルダーならただの値
    assert query(conn, INQUIRIES_OF_CUSTOMER, {"customer_id": "x' OR '1'='1"}).empty
