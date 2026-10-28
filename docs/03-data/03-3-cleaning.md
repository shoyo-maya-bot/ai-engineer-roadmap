# 03-3 欠損・外れ値・表記ゆれの整理

!!! abstract "到達目標"
    - データの汚れ（欠損・外れ値・表記ゆれ・重複・形式の違い）を見分け、それぞれ「捨てる・直す・欠損にする・不明として残す」を理由つきで決められる
    - Unicode の正規化（NFKC）と対応表で、表記ゆれを 1 つの値にそろえられる
    - 捨てた行・直した値の数を、理由ごとに数えて報告できる

## 何か

03-1 で見つけた汚れを、種類ごとに整理します。

| 汚れ | 03 のデータでの例 | 決めたこと |
|---|---|---|
| 表記ゆれ | 分類の `Billing`・` billing`・`請求`、窓口の `mail`・`ﾒｰﾙ` | NFKC と対応表で、決まった値にそろえる |
| 形式の違い | 受付日時の `2026-04-01 14:36`・`2026/4/4 15:48`・`2026年4月8日 17:37` | 3 つの形式を読み、日時の型にする |
| 欠損 | 窓口・分類・対応時間の空欄、全角の空白だけの本文 | 列ごとに決める（下の表） |
| 欠損の代わりの値 | 対応時間の `99999`（未完了）・`-1` | 範囲外として欠損にする |
| 重複 | 同じ内容の行が 2 回・同じ番号で対応時間だけ違う行 | 同じ内容は 1 つに。違う場合は後から出力されたほうを残す |
| 一覧に無い値 | 顧客の一覧に無い顧客 ID、5 種類のどれでもない分類 | 列ごとに決める（下の表） |

欠損と「一覧に無い値」の扱いは、その列を何に使うかで決めます。

| 列 | 欠損・不正な値のとき | 理由 |
|---|---|---|
| 本文（`input`） | 行を捨てる | 分類の材料が無い |
| 分類（`category`） | 行を捨てる | 学習の正解が無い |
| 窓口・プラン | `unknown` として残す | 本文があれば学習に使える。「分からない」こと自体も情報になりうる |
| 受付日時・対応時間 | 欠損のまま残す | 学習の前に穴埋めする（03-4。学習用のデータだけで決める） |

## なぜ必要か

- 汚れたままでは、モデルは `billing` と `請求` を別の分類として学ぶ。窓口の 14 種類も、実際は 4 種類の言い換えにすぎない
- 99999 のような「欠損の代わりの値」は、数としては正しく見えるので、平均や学習を大きく狂わせる
- 何を捨てたかを数えていないと、「学習データが 147 件から 130 件に減った理由」を、後から誰も説明できない

汚れの直し方には、正解が 1 つではない判断が含まれます。判断はコードに書き、理由をコメントとテストに残して、誰が実行しても同じ結果にします。

## どうやるか

### 全角・半角をそろえる（NFKC）

Unicode の正規化 **NFKC** は、見た目が同じ文字を 1 つの書き方にそろえます
（[公式ドキュメント: unicodedata.normalize](https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize)）。

```pycon
>>> import unicodedata
>>> unicodedata.normalize("NFKC", "ＣＳＶ　ﾒｰﾙ ①")
'CSV メール 1'
```

全角の英数字は半角に、半角カナは全角に、全角の空白は半角の空白に、丸数字は数字になります。
比べるための値（窓口や分類）は、さらに小文字にし、前後と途中の空白をまとめます。

<!-- src: examples/src/roadmap_examples/p03_pipeline.py -->
```python title="examples/src/roadmap_examples/p03_pipeline.py"
def normalize_key(value: str) -> str:
    """表記ゆれの比較用: 全角半角をそろえ（NFKC）、小文字にし、前後と途中の空白をまとめる."""
    return " ".join(unicodedata.normalize("NFKC", value).lower().split())


def normalize_text(s: pd.Series) -> pd.Series:
    """本文: NFKC で全角英数字・半角カナをそろえ、改行や連続した空白を 1 つの空白にする. 空になったら欠損."""
    text = s.map(lambda v: " ".join(unicodedata.normalize("NFKC", v).split()), na_action="ignore")
    return text.mask(text == "")
```

本文は小文字にしません。意味が変わる場合があるからです（「ID」と「id」は同じでも、固有の名前は変わりうる）。
空白だけの本文は、そろえた後に空になるので、欠損として扱えます（03-1 で見た「空欄ではないが中身が無い」本文）。

!!! warning "NFKC は元に戻せない"
    NFKC は情報を落とします（「①」と「1」の区別が消える）。元の値が要る場面（画面に表示する本文など）では、元の列を残して、別の列に入れます。

### 別名は対応表で 1 つにする

NFKC でそろわない言い換え（`mail` と `メール`、`tel` と `電話`）は、対応表で決まった値にします。

<!-- src: examples/src/roadmap_examples/p03_pipeline.py -->
```python title="examples/src/roadmap_examples/p03_pipeline.py"
CATEGORY_ALIASES = {c: c for c in CATEGORIES} | {
    "請求": "billing",
    "アカウント": "account",
    "不具合": "bug",
    "契約": "contract",
    "その他": "other",
}
```

`CATEGORIES` は、07 のプロンプトで決めた 5 種類（`billing`・`account`・`bug`・`contract`・`other`）をそのまま読み込んでいます。
分類の名前を 1 か所で決めておくと、03 の学習データと 07 の評価セットの分類が食い違いません。

対応表に無い値は欠損にします（`map_aliases`。03-1）。知らない値を勝手にどれかに寄せず、数えて報告します。

<!-- src: examples/tests/test_p03_pipeline.py -->
```python title="examples/tests/test_p03_pipeline.py"
def test_map_aliases() -> None:
    s = pd.Series(["E-mail", "ﾒｰﾙ", " TEL ", "Web フォーム", "FAX", None], dtype="str")
    assert map_aliases(s, CHANNEL_ALIASES).tolist()[:4] == ["email", "email", "phone", "form"]
    assert map_aliases(s, CHANNEL_ALIASES).isna().tolist()[4:] == [True, True]  # 対応表に無い値も欠損
```

!!! failure "悪い例"
    `df["分類"].str.contains("bill")` のような部分一致で寄せる。`billing/contract`（どちらか決まっていない値）まで billing になる。

!!! success "良い例"
    そろえた値の完全一致で、対応表から引く。対応表に無い値（`billing/contract`・`要確認`）は欠損にし、件数を報告する。

### 日時の形式をそろえる

3 つの形式の受付日時を、数字の並び（年・月・日・時・分）として取り出してから、日時にします。

<!-- src: examples/src/roadmap_examples/p03_pipeline.py -->
```python title="examples/src/roadmap_examples/p03_pipeline.py"
def parse_received_at(s: pd.Series) -> pd.Series:
    """「2026-04-03 09:15」「2026/4/3 9:15」「2026年4月3日 9:15」を日時に. 読めないものは NaT（欠損）."""
    parts = s.str.extract(r"^\s*(\d{4})\D+(\d{1,2})\D+(\d{1,2})\D*\s+(\d{1,2}):(\d{2})\s*$")
    parts.columns = ["year", "month", "day", "hour", "minute"]
    return pd.to_datetime(parts.astype(float), errors="coerce")
```

- `str.extract` は、正規表現の `( )` ごとに 1 列の表を返します
- `pd.to_datetime` に year・month・day・hour・minute の列を持つ表を渡すと、日時にまとめます。
  `errors="coerce"` で、読めない値と、無い日付（2 月 30 日）は NaT（日時の欠損）になります
  （[公式ドキュメント: to_datetime](https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html)）

「不明」と書かれた 1 件は NaT になります。行は捨てません（本文と分類があれば学習に使えるので）。

### 外れ値: 見つける方法と、決める基準を分ける

外れ値の候補は、四分位範囲（IQR）で見つけられます。下から 25% の値を Q1、75% の値を Q3 として、
Q3 + 1.5 ×（Q3 − Q1）より大きい値を候補にする方法です。

整えた後の対応時間（121 件）では、Q1 = 21 分・Q3 = 47 分で、86 分を超える 8 件（94〜135 分）が候補になりました。
しかし、中身を読むと「アカウントがロックされた」「請求書の送付先を変えたい」など、2 時間前後かかってもおかしくない件です。
**統計の目安で見つかった候補を、そのまま捨ててはいけません。**

外れ値かどうかは、業務の基準で決めます。03 では「対応時間は 0 分以上 7 日未満。範囲外は入力の誤りか未完了の印」としました。

<!-- src: examples/src/roadmap_examples/p03_pipeline.py -->
```python title="examples/src/roadmap_examples/p03_pipeline.py"
def parse_minutes(s: pd.Series) -> tuple[pd.Series, int]:
    """「25」「25分」を数に. 戻り値は（分. 欠損は <NA>, 範囲外で欠損にした件数）."""
    text = s.map(lambda v: normalize_key(v).removesuffix("分"), na_action="ignore")
    minutes = pd.to_numeric(text, errors="coerce")
    out_of_range = minutes.notna() & ((minutes < 0) | (minutes >= MAX_MINUTES))
    return minutes.mask(out_of_range).astype("Int64"), int(out_of_range.sum())
```

- 99999（未完了）と -1 は、範囲外として欠損にします。行は捨てません（対応時間は分類の学習には使わないので。03-4）
- `Int64`（大文字の I）は、欠損を持てる整数の型です。ふつうの `int64` は欠損を持てないので、欠損があると小数（float）になります
  （[公式ドキュメント: Nullable integer](https://pandas.pydata.org/docs/user_guide/integer_na.html)）

### 重複: どちらを残すかを決める

重複には 2 種類あります。同じ内容の行（出力が 2 回重なった）と、同じ番号で中身が違う行（後から直して出力し直した）です。

<!-- src: examples/src/roadmap_examples/p03_pipeline.py -->
```python title="examples/src/roadmap_examples/p03_pipeline.py"
def drop_duplicates(df: pd.DataFrame, report: CleanReport) -> pd.DataFrame:
    """同じ内容の行は 1 つに. 同じ番号で内容が違う行は、後から出力されたほう（下の行）を残す."""
    exact = df.duplicated(keep="first")
    df = df[~exact]
    updated = df.duplicated(subset="id", keep="last")
    report.duplicates_exact, report.duplicates_updated = int(exact.sum()), int(updated.sum())
    return df[~updated]
```

「後から出力されたほうが新しい」は、このデータの決まりです。更新日時の列があるデータなら、それで決めます。
テスト（`test_later_export_wins_for_updated_rows`）で、Q0013 の対応時間が、後の行の 52 分になることを確かめています。

### 何を捨て、何を直したかを数える

パイプラインは、理由ごとの件数を `CleanReport` に記録し、最後に表示します。

```bash
uv run python -m roadmap_examples.p03_pipeline --out inquiries_clean.csv
```

```text
読んだ行: 147
重複: 同じ内容 4 行・同じ番号で後から直された 2 行（後のほうを残す）
捨てた行（本文が空）: 3
捨てた行（分類が空）: 6
捨てた行（分類が 5 種類のどれでもない）: 2
欠損にした値（受付日時が読めない）: 1
欠損にした値（対応時間が範囲外）: 5
欠損にした値（顧客の一覧に無い顧客）: 2
出力した行: 130
分類ごとの件数: billing 26, account 26, bug 26, contract 26, other 26
```

147 − 4 − 2 − 3 − 6 − 2 = 130 で、件数の出入りが合います。この件数はテストでも固定しています。
処理を直して件数が変わったら、テストが失敗するので、「意図した変化か」を確かめられます。

<!-- src: examples/tests/test_p03_pipeline.py -->
```python title="examples/tests/test_p03_pipeline.py"
def test_run_on_sample_data_reports_every_drop() -> None:
    df, report = run()
    assert report.rows_read == 147
    assert (report.duplicates_exact, report.duplicates_updated) == (4, 2)
    assert report.dropped == {"本文が空": 3, "分類が空": 6, "分類が 5 種類のどれでもない": 2}
    assert report.fixed == {"受付日時が読めない": 1, "対応時間が範囲外": 5, "顧客の一覧に無い顧客": 2}
    assert report.rows_out == len(df) == 130
    assert report.by_category == dict.fromkeys(CATEGORIES, 26)
```

## 演習

1. 窓口に「ｅメール」（全角の e）と書かれた行が来るようになりました。今の処理では何になるかを確かめ、email になるように直してください。テストも足します
2. 整えた表の対応時間について、IQR で外れ値の候補を求め、候補の本文を読んで、捨てるべきかを判断してください
3. 「本文が 5 文字未満の行は、分類の材料にならないので捨てる」という決まりを足してください。捨てた件数を `CleanReport` に理由つきで記録し、テストを書きます

- できたことの確かめ方: `uv run pytest` が通る。2 は候補の一覧と判断の理由
- 目安の時間: 4 時間

解答: [03-3 の解答](answers/03-3.md)

## もし〇〇がなかったら？

**表記ゆれをそろえなかったら**、分類が 19 種類の書き方のまま学習し、同じ意味のデータが別々の分類に散らばります。
1 つの分類あたりの件数が減り、モデルはどの分類もうまく学べません。

**捨てた理由を数えていなかったら**、来月のデータで「分類が空」の行が急に 3 割に増えても、気づかずに学習します。
サポートの道具の設定が変わって分類が入らなくなった、という上流の変化を、件数の報告なら翌日に見つけられます。

## 理解度チェック

1. NFKC で正規化すると、「ＣＳＶ」「ﾒｰﾙ」「①」はそれぞれどうなりますか。

    ??? note "答え"
        「CSV」（半角）・「メール」（全角カナ）・「1」になります。見た目が同じ文字を 1 つの書き方にそろえます。

2. 対応表に無い値（`billing/contract`）を、近そうな分類に寄せずに欠損にするのはなぜですか。

    ??? note "答え"
        どちらの分類かを決めるのは、データを作った側の判断で、こちらで推測すると誤った正解を学習させるからです。
        欠損にして件数を報告し、必要なら元のデータを直してもらいます。

3. IQR で外れ値の候補になった値を、そのまま捨ててはいけないのはなぜですか。

    ??? note "答え"
        統計の目安は「珍しい値」を見つけるだけで、「誤った値」かどうかは分からないからです。
        2 時間かかった問い合わせは珍しくても本当の値です。業務の基準（0 分以上 7 日未満など）で決めます。

4. 本文が空の行は捨て、窓口が空の行は unknown として残すのはなぜですか。

    ??? note "答え"
        本文は分類の材料なので、無ければ学習に使えません。窓口は補助の情報なので、無くても本文で学習できます。
        列を何に使うかで、欠損の扱いを決めます。
