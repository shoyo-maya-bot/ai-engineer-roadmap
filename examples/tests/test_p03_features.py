"""p03_features のテスト. 前処理を学習用のデータだけで fit することを確かめる."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline

from roadmap_examples.p03_features import LEAKY, NUMERIC, TARGET, add_features, build_preprocessor, split
from roadmap_examples.p03_pipeline import DATA, read_clean


@pytest.fixture
def data() -> pd.DataFrame:
    return add_features(read_clean(DATA / "inquiries_clean.csv"))


def test_add_features(data) -> None:
    row = data.loc[data["id"] == "Q0001"].iloc[0]
    assert row["input_length"] == len("見積書と請求書の金額が違います。")
    assert row["hour"] == 14
    assert data["hour"].isna().sum() == 1  # 受付日時が読めなかった行


def test_split_is_stratified_and_reproducible(data) -> None:
    train, test = split(data)
    assert (len(train), len(test)) == (97, 33)
    assert set(train["id"]).isdisjoint(test["id"])
    assert test[TARGET].value_counts().between(6, 7).all()  # 分類ごとの割合がそろう
    assert split(data)[1]["id"].tolist() == test["id"].tolist()


def test_preprocessor_learns_only_from_train(data) -> None:
    train, test = split(data)
    pre = build_preprocessor().fit(train)
    medians = pre.named_transformers_["num"].named_steps["impute"].statistics_
    assert medians.tolist() == train[NUMERIC].median().tolist()  # 全体ではなく学習用のデータの中央値
    vocab = pre.named_transformers_["text"].vocabulary_
    test_only = {c for text in test["input"] for c in text} - {c for text in train["input"] for c in text}
    assert test_only and not test_only & set(vocab)  # 評価用にしか無い文字は、語彙に入らない
    assert pre.transform(test).shape == (len(test), pre.transform(train).shape[1])


def test_unknown_category_and_missing_hour_do_not_fail(data) -> None:
    train, _ = split(data)
    pre = build_preprocessor().fit(train)
    new = add_features(
        pd.DataFrame(
            {
                "input": ["請求書を送ってください。"],
                "channel": ["fax"],  # 学習で見なかった窓口
                "plan": ["pro"],
                "received_at": pd.to_datetime(pd.Series([pd.NaT])),
            }
        )
    )
    row = pre.transform(new)
    names = pre.get_feature_names_out()
    channel_cols = [i for i, n in enumerate(names) if n.startswith("cat__channel_")]
    assert np.asarray(row[:, channel_cols].todense()).sum() == 0  # 見なかった値は全部 0
    assert np.isfinite(np.asarray(row.todense())).all()  # 欠損の時刻は中央値で埋まる


def test_leaky_columns_are_not_used() -> None:
    used = {c for _, _, cols in build_preprocessor().transformers for c in ([cols] if isinstance(cols, str) else cols)}
    assert used == {"input", "channel", "plan", "input_length", "hour"}
    assert not used & set(LEAKY)
    text_only = build_preprocessor(text_only=True).transformers
    assert [cols for _, _, cols in text_only] == ["input"]  # 07 の評価セットにある列だけ


def test_preprocessor_plugs_into_a_model(data) -> None:
    # 04 では、DummyClassifier の代わりに分類モデルをつなぐ
    train, test = split(data)
    model = make_pipeline(build_preprocessor(text_only=True), DummyClassifier(strategy="most_frequent"))
    model.fit(train, train[TARGET])
    assert len(model.predict(test)) == len(test)
