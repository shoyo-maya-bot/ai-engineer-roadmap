"""03-2 SQL での集計と結合: 同じ集計を SQL（SQLite）と pandas の両方で書き、結果が同じことを確かめる.

SQLite は Python の標準ライブラリ（sqlite3）で使える. サーバーを立てずに、メモリの中に表を作って試せる.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping

import pandas as pd

# 窓口ごとの件数（整える前の表で数えると、表記ゆれの数だけ行が分かれる）
CHANNEL_COUNTS = """
SELECT channel, COUNT(*) AS n
FROM inquiries
GROUP BY channel
ORDER BY n DESC, channel
"""

# 内部結合と左外部結合で、残る行の数を比べる（顧客の一覧に無い・書き方が違う顧客 ID の行が、内部結合では消える）
JOIN_ROW_COUNTS = """
SELECT
  (SELECT COUNT(*) FROM inquiries) AS inquiries,
  (SELECT COUNT(*) FROM inquiries AS i JOIN customers AS c ON i.customer_id = c.customer_id) AS inner_join,
  (SELECT COUNT(*) FROM inquiries AS i LEFT JOIN customers AS c ON i.customer_id = c.customer_id) AS left_join
"""

# 窓口ごとの件数と平均の対応時間. COUNT(*) は行の数、COUNT(列) と AVG(列) は NULL を数えない.
# 件数の少ない窓口は平均がぶれるので、HAVING で集計の後に絞る（WHERE は集計の前に行を絞る）
MINUTES_BY_CHANNEL = """
SELECT channel,
       COUNT(*) AS n,
       COUNT(response_minutes) AS n_known,
       AVG(response_minutes) AS avg_minutes
FROM inquiries
GROUP BY channel
HAVING COUNT(*) >= :min_rows
ORDER BY avg_minutes DESC, channel
"""

# 顧客 1 人の問い合わせ. 値はプレースホルダー（:customer_id）で渡し、SQL の文字列に埋め込まない
INQUIRIES_OF_CUSTOMER = """
SELECT id, category, input
FROM inquiries
WHERE customer_id = :customer_id
ORDER BY id
"""


def connect(tables: Mapping[str, pd.DataFrame]) -> sqlite3.Connection:
    """メモリの中の SQLite に、DataFrame を表として入れる（表の名前 → DataFrame）."""
    conn = sqlite3.connect(":memory:")
    for name, df in tables.items():
        df.to_sql(name, conn, index=False)
    return conn


def query(conn: sqlite3.Connection, sql: str, params: Mapping[str, object] | None = None) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def channel_counts(df: pd.DataFrame) -> pd.DataFrame:
    """CHANNEL_COUNTS の pandas 版."""
    counts = df.groupby("channel", dropna=False).size().reset_index(name="n")
    return counts.sort_values(["n", "channel"], ascending=[False, True], ignore_index=True)


def minutes_by_channel(df: pd.DataFrame, min_rows: int) -> pd.DataFrame:
    """MINUTES_BY_CHANNEL の pandas 版. count() と mean() も欠損を数えない."""
    out = (
        df.groupby("channel")
        .agg(
            n=("id", "size"),
            n_known=("response_minutes", "count"),
            avg_minutes=("response_minutes", "mean"),
        )
        .reset_index()
    )
    out = out[out["n"] >= min_rows]  # HAVING に当たる
    return out.sort_values(["avg_minutes", "channel"], ascending=[False, True], ignore_index=True)
