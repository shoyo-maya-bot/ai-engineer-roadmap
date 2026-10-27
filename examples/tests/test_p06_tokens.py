"""p06_tokens.py のテスト（06-1 小さな BPE とコサイン類似度）."""

import pytest

from roadmap_examples.p06_tokens import cosine, merge, tokenize, train_bpe

CORPUS = ["注文の状態", "注文の取り消し", "注文番号", "請求書の再発行", "請求書の宛名"]


def test_merge_joins_matching_pairs_from_left() -> None:
    assert merge(["a", "a", "a"], ("a", "a")) == ["aa", "a"]
    assert merge(["注", "文", "の"], ("注", "文")) == ["注文", "の"]


def test_frequent_words_become_one_token() -> None:
    merges = train_bpe(CORPUS, num_merges=10)
    assert tokenize("注文の状態", merges)[0] == "注文の"
    assert "請求書の" in tokenize("請求書の控え", merges)


def test_unseen_text_stays_as_characters() -> None:
    merges = train_bpe(CORPUS, num_merges=10)
    assert tokenize("配送", merges) == ["配", "送"]


def test_text_from_the_training_domain_needs_fewer_tokens() -> None:
    merges = train_bpe(CORPUS, num_merges=10)
    seen, unseen = "注文の取り消し", "配送日の変更願"  # どちらも 7 文字
    assert len(tokenize(seen, merges)) < len(tokenize(unseen, merges)) == 7


def test_training_stops_when_no_pair_repeats() -> None:
    assert train_bpe(["abc"], num_merges=5) == []
    assert train_bpe([], num_merges=5) == []


def test_training_is_deterministic() -> None:
    assert train_bpe(CORPUS, 10) == train_bpe(list(CORPUS), 10)


def test_cosine() -> None:
    assert cosine([1.0, 0.0], [2.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 3.0]) == pytest.approx(0.0)
    assert cosine([1.0, 1.0], [-1.0, -1.0]) == pytest.approx(-1.0)
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    with pytest.raises(ValueError, match="長さ"):
        cosine([1.0], [1.0, 2.0])
