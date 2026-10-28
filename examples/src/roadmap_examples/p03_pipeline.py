"""03 の成果物: 架空の問い合わせデータ（汚れた CSV）を、学習に使える表に整えるパイプライン.

入力: data/p03/inquiries_raw.csv（問い合わせ）と data/p03/customers.csv（顧客とプラン）
出力: 1 行 1 件の表（OUTPUT_COLUMNS）. 分類（category）は 07 の評価セットと同じ 5 種類にそろえる.
      04 はこの表で分類モデルを学習し、07 の評価セット（data/p07/inquiries.jsonl）で測る.

使い方（examples/ で）:
  uv run python -m roadmap_examples.p03_pipeline --out inquiries_clean.csv
  uv run python -m roadmap_examples.p03_pipeline --out inquiries_clean.csv --cases inquiries_train.jsonl

- 同じ入力からは、何度実行しても同じ出力になる（並び順も含めて）. 入力のファイルは書き換えない
- 捨てた行・直した値の数を、理由ごとに表示する（黙って捨てない）

終了コード: 0 成功 / 1 入力の誤り・整えた結果が決まりを満たさない / 2 引数の誤り
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import pandas as pd

from roadmap_examples.evalset import EvalCase, save_cases
from roadmap_examples.p07_prompt import CATEGORIES

DATA = Path(__file__).resolve().parents[2] / "data" / "p03"
DEFAULT_RAW = DATA / "inquiries_raw.csv"
DEFAULT_CUSTOMERS = DATA / "customers.csv"

# 元の列名 → 整えた後の列名
RAW_COLUMNS = {
    "受付番号": "id",
    "受付日時": "received_at",
    "顧客ID": "customer_id",
    "窓口": "channel",
    "本文": "input",
    "分類": "category",
    "対応時間（分）": "response_minutes",
}
CUSTOMER_COLUMNS = {"顧客ID": "customer_id", "プラン": "plan", "利用開始日": "started_on"}
OUTPUT_COLUMNS = ["id", "received_at", "customer_id", "plan", "channel", "input", "category", "response_minutes"]

# 表記ゆれの対応表（キーは normalize_key で、全角半角・大文字小文字・前後の空白をそろえた形）
CATEGORY_ALIASES = {c: c for c in CATEGORIES} | {
    "請求": "billing",
    "アカウント": "account",
    "不具合": "bug",
    "契約": "contract",
    "その他": "other",
}
CHANNEL_ALIASES = {
    "email": "email",
    "メール": "email",
    "mail": "email",
    "e-mail": "email",
    "phone": "phone",
    "電話": "phone",
    "tel": "phone",
    "chat": "chat",
    "チャット": "chat",
    "form": "form",
    "フォーム": "form",
    "問い合わせフォーム": "form",
    "web フォーム": "form",
}
PLAN_ALIASES = {
    "free": "free",
    "フリー": "free",
    "standard": "standard",
    "スタンダード": "standard",
    "std": "standard",
    "pro": "pro",
    "プロ": "pro",
}
UNKNOWN = "unknown"
MAX_MINUTES = 7 * 24 * 60  # 対応時間は 0 分以上 7 日未満. 範囲外（未完了の既定値 99999 など）は欠損にする


class PipelineError(ValueError):
    """入力の形が想定と違う、または整えた結果が決まりを満たさない."""


@dataclass
class CleanReport:
    """どの行を捨て、どの値を直したか（理由ごとの件数）."""

    rows_read: int = 0
    duplicates_exact: int = 0
    duplicates_updated: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    fixed: dict[str, int] = field(default_factory=dict)
    rows_out: int = 0
    by_category: dict[str, int] = field(default_factory=dict)

    def lines(self) -> list[str]:
        out = [
            f"読んだ行: {self.rows_read}",
            f"重複: 同じ内容 {self.duplicates_exact} 行・"
            f"同じ番号で後から直された {self.duplicates_updated} 行（後のほうを残す）",
        ]
        out += [f"捨てた行（{reason}）: {n}" for reason, n in self.dropped.items()]
        out += [f"欠損にした値（{reason}）: {n}" for reason, n in self.fixed.items()]
        out.append(f"出力した行: {self.rows_out}")
        out.append("分類ごとの件数: " + ", ".join(f"{c} {n}" for c, n in self.by_category.items()))
        return out


def read_csv_as_text(path: Path, columns: dict[str, str]) -> pd.DataFrame:
    """CSV をすべて文字列として読み、列名を付け替える. 空欄だけを欠損にする（「NA」などの文字は残す）."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise PipelineError(f"{path.name}: 列 {missing} がありません")
    return df[list(columns)].rename(columns=columns)


def normalize_key(value: str) -> str:
    """表記ゆれの比較用: 全角半角をそろえ（NFKC）、小文字にし、前後と途中の空白をまとめる."""
    return " ".join(unicodedata.normalize("NFKC", value).lower().split())


def normalize_text(s: pd.Series) -> pd.Series:
    """本文: NFKC で全角英数字・半角カナをそろえ、改行や連続した空白を 1 つの空白にする. 空になったら欠損."""
    text = s.map(lambda v: " ".join(unicodedata.normalize("NFKC", v).split()), na_action="ignore")
    return text.mask(text == "")


def map_aliases(s: pd.Series, aliases: dict[str, str]) -> pd.Series:
    """対応表で正式な値に置き換える. 対応表に無い値と空欄は欠損（NaN）."""
    return s.map(lambda v: aliases.get(normalize_key(v)), na_action="ignore")


def parse_received_at(s: pd.Series) -> pd.Series:
    """「2026-04-03 09:15」「2026/4/3 9:15」「2026年4月3日 9:15」を日時に. 読めないものは NaT（欠損）."""
    parts = s.str.extract(r"^\s*(\d{4})\D+(\d{1,2})\D+(\d{1,2})\D*\s+(\d{1,2}):(\d{2})\s*$")
    parts.columns = ["year", "month", "day", "hour", "minute"]
    return pd.to_datetime(parts.astype(float), errors="coerce")


def parse_minutes(s: pd.Series) -> tuple[pd.Series, int]:
    """「25」「25分」を数に. 戻り値は（分. 欠損は <NA>, 範囲外で欠損にした件数）."""
    text = s.map(lambda v: normalize_key(v).removesuffix("分"), na_action="ignore")
    minutes = pd.to_numeric(text, errors="coerce")
    out_of_range = minutes.notna() & ((minutes < 0) | (minutes >= MAX_MINUTES))
    return minutes.mask(out_of_range).astype("Int64"), int(out_of_range.sum())


def normalize_customer_id(s: pd.Series) -> pd.Series:
    """「K-0003」「K0003」「k-0003」を「K-0003」に. 形が違えば欠損."""

    def one(value: str) -> str | None:
        m = re.fullmatch(r"K-?(\d{4})", unicodedata.normalize("NFKC", value).strip().upper())
        return f"K-{m.group(1)}" if m else None

    return s.map(one, na_action="ignore")


def drop_duplicates(df: pd.DataFrame, report: CleanReport) -> pd.DataFrame:
    """同じ内容の行は 1 つに. 同じ番号で内容が違う行は、後から出力されたほう（下の行）を残す."""
    exact = df.duplicated(keep="first")
    df = df[~exact]
    updated = df.duplicated(subset="id", keep="last")
    report.duplicates_exact, report.duplicates_updated = int(exact.sum()), int(updated.sum())
    return df[~updated]


def clean_inquiries(raw: pd.DataFrame, customers: pd.DataFrame, report: CleanReport) -> pd.DataFrame:
    """問い合わせと顧客の表を整え、結合して、学習に使える行だけを残す."""
    report.rows_read = len(raw)
    df = drop_duplicates(raw, report).reset_index(drop=True)  # 結合（merge）の後も、行の目印をそろえる

    df["input"] = normalize_text(df["input"])
    blank_category = df["category"].isna()
    df["category"] = map_aliases(df["category"], CATEGORY_ALIASES)
    df["channel"] = map_aliases(df["channel"], CHANNEL_ALIASES).fillna(UNKNOWN)
    df["received_at"] = parse_received_at(df["received_at"])
    df["response_minutes"], out_of_range = parse_minutes(df["response_minutes"])
    df["customer_id"] = normalize_customer_id(df["customer_id"])

    plans = customers.assign(
        customer_id=normalize_customer_id(customers["customer_id"]),
        plan=map_aliases(customers["plan"], PLAN_ALIASES),
    )[["customer_id", "plan"]]
    # 左外部結合: 顧客の一覧に無い問い合わせも残す（内部結合だと黙って消える）. 顧客は 1 行ずつのはず
    df = df.merge(plans, on="customer_id", how="left", validate="many_to_one")
    unknown_customer = df["plan"].isna() & df["customer_id"].notna()
    df["plan"] = df["plan"].fillna(UNKNOWN)

    rules = {
        "本文が空": df["input"].isna(),
        "分類が空": df["input"].notna() & blank_category,
        "分類が 5 種類のどれでもない": df["input"].notna() & ~blank_category & df["category"].isna(),
    }
    report.dropped = {reason: int(mask.sum()) for reason, mask in rules.items()}
    keep = ~(rules["本文が空"] | rules["分類が空"] | rules["分類が 5 種類のどれでもない"])
    report.fixed = {
        "受付日時が読めない": int((df["received_at"].isna() & keep).sum()),
        "対応時間が範囲外": out_of_range,
        "顧客の一覧に無い顧客": int((unknown_customer & keep).sum()),
    }

    df = df[keep].sort_values("id", kind="stable").reset_index(drop=True)[OUTPUT_COLUMNS]
    report.rows_out = len(df)
    report.by_category = {c: int((df["category"] == c).sum()) for c in CATEGORIES}
    return df


def validate(df: pd.DataFrame) -> None:
    """整えた表の決まり. 破っていたら PipelineError（次の部に壊れたデータを渡さない）."""
    problems = []
    if list(df.columns) != OUTPUT_COLUMNS:
        problems.append(f"列が違います: {list(df.columns)}")
    if df["id"].duplicated().any():
        problems.append("id が重複しています")
    if df["input"].isna().any() or (df["input"].str.strip() == "").any():
        problems.append("本文が空の行があります")
    unknown = set(df["category"].dropna()) - set(CATEGORIES)
    if df["category"].isna().any() or unknown:
        problems.append(f"分類が 5 種類以外です: {sorted(unknown)}")
    if problems:
        raise PipelineError("; ".join(problems))


def run(raw_path: Path = DEFAULT_RAW, customers_path: Path = DEFAULT_CUSTOMERS) -> tuple[pd.DataFrame, CleanReport]:
    report = CleanReport()
    raw = read_csv_as_text(raw_path, RAW_COLUMNS)
    customers = read_csv_as_text(customers_path, CUSTOMER_COLUMNS)
    df = clean_inquiries(raw, customers, report)
    validate(df)
    return df, report


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """日時は「2026-04-03T09:15」の形、欠損は空欄、改行は LF で書く（どの OS でも同じバイト列にする）."""
    df.to_csv(path, index=False, encoding="utf-8", date_format="%Y-%m-%dT%H:%M", lineterminator="\n")


def read_clean(path: Path) -> pd.DataFrame:
    """write_csv で書いた表を、型を戻して読む（04 で使う）."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8")
    df["received_at"] = pd.to_datetime(df["received_at"], format="%Y-%m-%dT%H:%M")
    df["response_minutes"] = pd.to_numeric(df["response_minutes"]).astype("Int64")
    return df


def to_cases(df: pd.DataFrame) -> list[EvalCase]:
    """07 の評価セットと同じ形式（id・input・expected.category）にする. 04 の学習データを JSONL でも渡せる."""
    return [EvalCase(row.id, row.input, {"category": row.category}) for row in df.itertuples(index=False)]


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="p03_pipeline", description="架空の問い合わせデータを、学習に使える表に整える")
    p.add_argument("--raw", type=Path, default=DEFAULT_RAW, help="問い合わせの CSV")
    p.add_argument("--customers", type=Path, default=DEFAULT_CUSTOMERS, help="顧客の CSV")
    p.add_argument("--out", type=Path, required=True, help="整えた表を書く CSV")
    p.add_argument("--cases", type=Path, help="07 の評価セットと同じ形式の JSONL も書く")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None, stdout: TextIO | None = None) -> int:
    args = parse_args(argv)
    stdout = stdout or sys.stdout
    try:
        df, report = run(args.raw, args.customers)
    except (OSError, PipelineError) as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1
    write_csv(df, args.out)
    if args.cases:
        save_cases(to_cases(df), args.cases)
    print("\n".join(report.lines()), file=stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
