"""03-4 特徴量と前処理パイプライン: 整えた表（p03_pipeline の出力）から、モデルに渡す数の表を作る.

前処理（欠損の穴埋め・標準化・カテゴリの 0/1 化・文章のベクトル化）は、学習用のデータだけで fit し、
評価用のデータには transform だけを行う. 04 では `build_preprocessor()` の後ろに分類モデルをつなぐ.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "category"
TEXT = "input"
CATEGORICAL = ["channel", "plan"]
NUMERIC = ["input_length", "hour"]
# 対応が終わってから分かる値. 分類したい時点（問い合わせが届いたとき）には無いので、特徴量にしない
LEAKY = ["response_minutes"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """元の列から計算できる特徴量を足す（行ごとに決まる計算だけ. 全体の平均などはここで使わない）."""
    return df.assign(
        input_length=df[TEXT].str.len(),
        hour=df["received_at"].dt.hour,  # 受付日時が欠損の行は NaN（後で穴埋めする）
    )


def build_preprocessor(text_only: bool = False) -> ColumnTransformer:
    """列の種類ごとの前処理をまとめる. まだ何も学んでいない（fit する前の）状態で返す.

    text_only=True は本文だけを使う. 07 の評価セットには本文（input）しか無いので、
    その評価セットで測るモデル（04）は本文だけで学習する（評価のときに無い列は、学習にも使えない）.
    """
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),  # 欠損を、学習用のデータの中央値で埋める
            ("scale", StandardScaler()),  # 平均 0・標準偏差 1 にそろえる（02-4）
        ]
    )
    # 列名を文字列 1 つで渡すと、1 次元（文章の並び）のまま渡る. 日本語は単語の区切りが無いので、1〜2 文字の並びを数える
    text = ("text", TfidfVectorizer(analyzer="char", ngram_range=(1, 2)), TEXT)
    if text_only:
        return ColumnTransformer([text])
    return ColumnTransformer(
        [
            text,
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),  # 学習で見なかった値は全部 0
            ("num", numeric, NUMERIC),
        ]
    )


def split(df: pd.DataFrame, test_size: float = 0.25, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """学習用と評価用に分ける. 分類ごとの割合をそろえ（stratify）、seed で毎回同じ分け方にする."""
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[TARGET])
    return train.sort_values("id"), test.sort_values("id")
