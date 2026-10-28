# 03-1 NumPy と pandas

!!! abstract "到達目標"
    - NumPy の配列で、for 文を書かずに列全体を計算・絞り込みできる
    - pandas で CSV を「すべて文字列」として読み、形・型・欠損・値の種類を調べて、データの汚れを見つけられる
    - pandas の行の目印（index）がずれると何が起きるかを説明し、防げる

## 何か

**NumPy** は、同じ型の数を並べた配列（`ndarray`）を、まとめて計算するライブラリです（02 で使いました）。
**pandas** は、名前の付いた列を並べた表（`DataFrame`）を扱うライブラリです。1 つの列は `Series` で、中身は NumPy の配列に近いものです。

| もの | 中身 | Java での近いもの |
|---|---|---|
| `np.ndarray` | 同じ型の数の並び（1 次元・2 次元…） | `double[]`・`double[][]` |
| `pd.Series` | 1 列分の値と、行の目印（index） | `Map<行の目印, 値>` |
| `pd.DataFrame` | 名前の付いた Series を並べた表 | `List<Map<String, Object>>` を列ごとに持ち直したもの |

03 の題材は、架空の会員制サービスに届いた問い合わせです。サポートの道具から書き出した、という想定の汚れた CSV を使います。

| ファイル | 中身 |
|---|---|
| `examples/data/p03/inquiries_raw.csv` | 問い合わせ 147 件（受付番号・受付日時・顧客 ID・窓口・本文・分類・対応時間（分）） |
| `examples/data/p03/customers.csv` | 顧客 40 人（顧客 ID・プラン・利用開始日） |

この部の成果物は、これを学習に使える表に整えるパイプラインです（03-4）。整えた表は、04 の分類モデルの学習に使います。

## なぜ必要か

機械学習の仕事の多くは、モデルを選ぶ前の「データを見て、整える」作業です。データが汚れていると、どんなモデルでも良い結果は出ません。

- 1 行ずつ for 文で処理すると、遅く、書く量も多くなる。列全体の演算で書くと、短く速くなる
- 最初にデータを見ずに学習を始めると、「分類の列に 19 種類の書き方が混ざっていた」ことに、結果が出てから気づく
- CSV を既定の設定で読むと、pandas が型を推測して、値を黙って書き換えることがある

## どうやるか

### NumPy: 列全体を一度に計算する

```pycon
>>> import numpy as np
>>> m = np.array([25, 40, 12, 99999, 30])  # 対応時間（分）
>>> m * 60  # 全部を秒に（for 文を書かない）
array([   1500,    2400,     720, 5999940,    1800])
>>> m[m < 10080]  # 7 日未満だけ（真偽の配列で絞り込む）
array([25, 40, 12, 30])
>>> m.mean(), np.median(m)
(np.float64(20021.2), np.float64(30.0))
```

`m < 10080` は、要素ごとに比べた真偽（True・False）の配列です。それを `m[...]` に渡すと、True の位置だけが残ります。

平均は 20,021 分（約 2 週間）ですが、中央値は 30 分です。99999 の 1 件が平均を大きく引っ張っています。
この 99999 は、サポートの道具が「未完了」のときに入れる値です。こうした値の扱いは 03-3 で決めます。

### pandas: すべて文字列として読む

`pd.read_csv` は、既定では列の型を推測し、`NA`・`NULL`・`None` のような文字も欠損（NaN）に変えます
（[公式ドキュメント: read_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html) の `keep_default_na`・`na_values`）。
汚れたデータでは、まず **すべて文字列として** 読み、空欄だけを欠損にします。型は、中身を確かめてから自分で変えます。

<!-- src: examples/src/roadmap_examples/p03_pipeline.py -->
```python title="examples/src/roadmap_examples/p03_pipeline.py"
def read_csv_as_text(path: Path, columns: dict[str, str]) -> pd.DataFrame:
    """CSV をすべて文字列として読み、列名を付け替える. 空欄だけを欠損にする（「NA」などの文字は残す）."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], encoding="utf-8")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise PipelineError(f"{path.name}: 列 {missing} がありません")
    return df[list(columns)].rename(columns=columns)
```

列名は、日本語のままだとコードで扱いにくいので、英語の名前に付け替えます（`受付番号` → `id` など。対応表は `RAW_COLUMNS`）。
本文に「NA」とだけ書かれた問い合わせが欠損にならないことを、テストで確かめています（`test_na_like_text_is_kept`）。

!!! failure "悪い例"
    `pd.read_csv(...)` と既定のまま読む。例えば ID が「0012」のような数字だけの列なら数の 12 になり、本文に「NA」とだけ書かれた行は欠損になる（このデータの顧客 ID は「K-0012」の形なので前者は起きないが、別のデータでは起きる）。

!!! success "良い例"
    `dtype=str, keep_default_na=False, na_values=[""]` で読む。型の変換と欠損の判断は、後の処理で、理由を付けて行う。

### データを見る

読んだら、学習に進む前に、形・欠損・値の種類を見ます（`examples/` で実行）。

```pycon
>>> import pandas as pd
>>> df = pd.read_csv("data/p03/inquiries_raw.csv", dtype=str, keep_default_na=False, na_values=[""])
>>> df.shape  # （行の数, 列の数）
(147, 7)
>>> df.isna().sum()  # 列ごとの欠損の数
受付番号        0
受付日時        0
顧客ID        0
窓口         10
本文          2
分類          6
対応時間（分）     6
dtype: int64
>>> df["窓口"].value_counts(dropna=False).head(5)  # 値の種類と数（欠損も数える）
窓口
メール       26
mail      15
チャット      14
E-mail    13
ﾒｰﾙ       12
Name: count, dtype: int64
```

`value_counts` で分類の列を見ると、`billing`・`Billing`・` billing`（前に空白）・`請求` のように、19 種類の書き方に分かれています（5 分類の言い換え 17 種類と、どれにも決まらない値 2 種類。空欄は別）。
窓口も同じです。学習に使う前に、これらを 1 つの書き方にそろえる必要があります（03-3）。

本文の欠損は 2 件ですが、全角の空白だけの本文がもう 1 件あります。「空欄ではないが、中身が無い」値は、`isna` では見つかりません。

### 1 行ずつではなく、列で処理する

!!! failure "悪い例"
    ```python
    for i, row in df.iterrows():  # 1 行ずつ取り出して書き換える（遅く、型も崩れやすい）
        df.loc[i, "本文"] = row["本文"].strip()
    ```

!!! success "良い例"
    ```python
    df["本文"] = df["本文"].str.strip()  # 列全体に、文字列の処理をまとめてかける
    ```

`Series.str` には、`strip`・`lower`・`replace`・`contains` など、文字列の処理がまとめてあります
（[公式ドキュメント: Working with text data](https://pandas.pydata.org/docs/user_guide/text.html)）。
`.str` に無い処理は、`Series.map` に関数を渡します。`na_action="ignore"` を付けると、欠損には関数を呼ばず、欠損のまま残します。

<!-- src: examples/src/roadmap_examples/p03_pipeline.py -->
```python title="examples/src/roadmap_examples/p03_pipeline.py"
def map_aliases(s: pd.Series, aliases: dict[str, str]) -> pd.Series:
    """対応表で正式な値に置き換える. 対応表に無い値と空欄は欠損（NaN）."""
    return s.map(lambda v: aliases.get(normalize_key(v)), na_action="ignore")
```

絞り込みは、真偽の Series を `loc` に渡します（NumPy の真偽の配列と同じ考え方です）。

```python
billing = df.loc[df["分類"] == "billing", ["受付番号", "本文"]]  # 行の条件と、残す列
```

### 行の目印（index）のずれに気をつける

pandas の Series どうしの計算は、**位置ではなく行の目印（index）で** 対応を取ります。目印がずれていると、黙って違う行どうしが計算されます。

```pycon
>>> a = pd.Series([1, 2, 3], index=[0, 1, 2])
>>> b = pd.Series([10, 20, 30], index=[1, 2, 3])
>>> (a + b).tolist()  # 目印 1 と 2 だけが対応し、ほかは欠損
[nan, 12.0, 23.0, nan]
```

03 のパイプラインを作るとき、実際にこれで誤りました。重複を除いた表（目印が飛び飛び）で「分類が空か」の真偽を作り、
顧客の表と結合（`merge`）した後の表（目印が 0 から振り直される）に使ったため、別の行の分類を見ていました。
pandas は「Boolean Series key will be reindexed」という警告を出しただけで、処理は最後まで進みました。

直し方は、結合の前に目印を 0 から振り直しておくことです。

<!-- src: examples/src/roadmap_examples/p03_pipeline.py -->
```python title="examples/src/roadmap_examples/p03_pipeline.py"
    df = drop_duplicates(raw, report).reset_index(drop=True)  # 結合（merge）の後も、行の目印をそろえる
```

警告は読み飛ばさず、テストで件数（捨てた行の数など）を確かめておくと、こうした誤りに気づけます（03-4 のテスト）。

## 演習

列ごとの様子を 1 つの表にまとめる関数を書き、`examples/data/p03/inquiries_raw.csv` に使ってください。

```python
def profile(df: pd.DataFrame) -> pd.DataFrame: ...
```

1. 列ごとに、欠損の数・値の種類の数・最も多い値・最も多い値の件数を、1 行ずつ持つ表を返す
2. 生の CSV に使い、学習の前に直すべき問題を 3 つ以上挙げる（どの列の・どんな問題か）
3. `Series.str.strip().str.lower()` で窓口の値をそろえると、種類がいくつに減るか。まだ残る表記ゆれは何か

- できたことの確かめ方: 小さな DataFrame で `profile` のテストが通る。2 と 3 は箇条書き
- 目安の時間: 3 時間

解答: [03-1 の解答](answers/03-1.md)

## もし〇〇がなかったら？

**データを見ずに学習を始めていたら**、分類の列が 19 種類の書き方に分かれたまま学習し、「billing」と「請求」を別の分類として覚えたモデルができます。
精度が低い原因を、モデルの選び方やパラメーターに探して、時間を使います。

**型を推測させて読んでいたら**、数字だけの ID の列で先頭の 0 が消えて別の ID と同じになる、本文の「NA」が欠損として捨てられる、といったことが黙って起きます。
読むときに壊れた値は、後の処理では元に戻せません。

## 理解度チェック

1. `m[m < 10080]` は、何をしていますか。

    ??? note "答え"
        `m < 10080` で、要素ごとの真偽の配列を作り、True の位置の要素だけを取り出しています。for 文を書かずに絞り込めます。

2. 汚れた CSV を `dtype=str, keep_default_na=False, na_values=[""]` で読むのはなぜですか。

    ??? note "答え"
        pandas に型を推測させると、先頭の 0 が消えたり、「NA」のような文字が欠損になったりするからです。
        すべて文字列として読み、空欄だけを欠損にして、型の変換は中身を確かめてから行います。

3. 対応時間の平均が 2 万分を超え、中央値が 30 分でした。何が考えられますか。

    ??? note "答え"
        極端に大きな値（外れ値）が平均を引っ張っています。この題材では「未完了」を表す 99999 が混ざっています。
        平均だけを見ず、中央値や値の分布も見ます。

4. 2 つの Series を足したら、思ったより欠損が増えました。何を疑いますか。

    ??? note "答え"
        行の目印（index）のずれです。pandas は目印で対応を取るので、目印が合わない行は欠損になります。
        `reset_index(drop=True)` で振り直すか、目印がそろっているかを確かめます。
