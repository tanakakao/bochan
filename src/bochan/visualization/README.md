# bochan.visualization

`BayesianOptimizer` / `CandidateResult` / `BochanStudy` から、Plotly 用の可視化データと Figure を作る補助モジュールです。

## 目的

- YY plot
- 1D 予測曲線
- 2D 獲得関数または予測値ヒートマップ
- multiclass / ordinal のクラス別予測確率・決定領域・不確かさ
- 3成分制約の三角プロット
- 2目的散布図
- study の cycle 推移

を API 本体から分離して扱います。

## 基本例

```python
from bochan.visualization import show_yyplot_from_optimizer

fig = show_yyplot_from_optimizer(
    bo,
    target="y",
    feature_cols=["x0", "x1"],
    target_cols=["y"],
)
fig.show()
```

候補点を重ねる場合は、`candidate(..., return_result=True)` の結果を渡します。

```python
result = bo.candidate(acq_config, opt_config, return_result=True)

fig = show_scatter_with_acqf_from_optimizer(
    bo,
    "x0",
    "x1",
    "y",
    feature_cols=["x0", "x1"],
    target_cols=["y"],
    candidate_result=result,
    n=40,
    show_type="acqf",
)
fig.show()
```

## Multiclass

`show_1dplot_from_optimizer` は multiclass モデルを自動判定し、全クラスの予測確率を1本ずつ描画します。

```python
from bochan.visualization import show_1dplot_from_optimizer

fig = show_1dplot_from_optimizer(
    bo,
    feature="x0",
    target="class",
    feature_cols=["x0", "x1"],
    target_cols=["class"],
    value_dict={"x1": 0.5},
    class_labels=["A", "B", "C"],  # metadata にあれば省略可能
    n=100,
)
fig.show()
```

2次元では `show_type="pred"` を指定すると multiclass 用の決定領域ヒートマップへ自動分岐します。既定の `class_confidence` は、色相で予測クラス、色の濃さで最大予測確率を表します。

```python
fig = show_scatter_with_acqf_from_optimizer(
    bo,
    "x0",
    "x1",
    "class",
    feature_cols=["x0", "x1"],
    target_cols=["class"],
    show_type="pred",
    multiclass_mode="class_confidence",
    n=60,
)
fig.show()
```

境界付近の曖昧さを見る場合は、次の表示へ切り替えられます。

- `multiclass_mode="entropy"`: 正規化予測エントロピー。大きいほどクラス判定が曖昧。
- `multiclass_mode="margin"`: 1位と2位の予測確率差。小さいほど境界に近い。
- `multiclass_mode="class"`: 予測クラスだけを離散色で表示。

専用関数 `show_multiclass_1dplot_from_optimizer` と `show_multiclass_heatmap_from_optimizer`、前処理関数 `multiclass_prediction_dataframe` / `multiclass_grid_1d` / `multiclass_grid_2d` も直接利用できます。

multi-output multiclass では `target` から出力列を推定します。明示する場合は `output_index` を指定してください。

## Ordinal

multiclass の予測表示は常にカテゴリ確率ベースです。ordinal では潜在スコア表示にも意味があるため、`ordinal_display` で次の2種類を明示的に選択します。

- `ordinal_display="latent"`: 従来の潜在スコア・予測値表示。既定値。
- `ordinal_display="probability"`: ordinal likelihood の cutpoint を通して求めた順序カテゴリ確率。

```python
fig = show_scatter_with_acqf_from_optimizer(
    bo,
    "x0",
    "x1",
    "level",
    feature_cols=["x0", "x1"],
    target_cols=["level"],
    show_type="pred",
    ordinal_display="latent",  # 省略時も latent
)
```

順序カテゴリごとの確率を表示する場合は `ordinal_display="probability"` に切り替えます。latent mean を単純にカテゴリ化するのではなく、ordinal likelihood の cutpoint を通して計算します。

```python
fig = show_scatter_with_acqf_from_optimizer(
    bo,
    "x0",
    "x1",
    "level",
    feature_cols=["x0", "x1"],
    target_cols=["level"],
    show_type="pred",
    ordinal_display="probability",
    ordinal_mode="class_confidence",
    class_labels=["low", "middle", "high"],
    n=60,
)
fig.show()
```

1次元でも同じ切替を使用します。

```python
fig = show_1dplot_from_optimizer(
    bo,
    feature="x0",
    target="level",
    feature_cols=["x0", "x1"],
    target_cols=["level"],
    value_dict={"x1": 0.5},
    ordinal_display="probability",
    n=100,
)
```

`ordinal_mode` は multiclass の `multiclass_mode` と同じ値を使用します。

- `class_confidence`: 色相が最尤カテゴリ、濃さがその最大確率。
- `class`: 最尤カテゴリだけを離散色で表示。
- `entropy`: 順序カテゴリ分布の正規化エントロピー。
- `margin`: 1位と2位のカテゴリ確率差。

専用関数 `show_ordinal_1dplot_from_optimizer` / `show_ordinal_heatmap_from_optimizer` / `show_ordinal_triscatter_from_optimizer` と、`ordinal_probabilities` / `ordinal_grid_1d` / `ordinal_grid_2d` / `ordinal_tri_grid` も利用できます。multi-output ordinal は `target` または `output_index` で対象出力を選択します。

## 三角プロット

三角プロットは、3列の和が `sum_value` になるグリッドで評価します。

```python
fig = show_triscatter_with_acqf_from_optimizer(
    bo,
    "a",
    "b",
    "c",
    "y",
    feature_cols=["a", "b", "c", "temp"],
    target_cols=["y"],
    value_dict={"temp": 1000.0},
    sum_value=1.0,
    candidate_result=result,
)
fig.show()
```

multiclass でも同じ関数を使用し、`show_type="pred"` を指定します。

```python
fig = show_triscatter_with_acqf_from_optimizer(
    bo,
    "a",
    "b",
    "c",
    "class",
    feature_cols=["a", "b", "c", "temp"],
    target_cols=["class"],
    value_dict={"temp": 1000.0},
    sum_value=1.0,
    show_type="pred",
    multiclass_mode="class_confidence",
    boundary_margin=0.08,
    n=60,
)
fig.show()
```

ordinal の三角図も、潜在スコア表示とカテゴリ確率表示を切り替えられます。

```python
fig = show_triscatter_with_acqf_from_optimizer(
    bo,
    "a",
    "b",
    "c",
    "level",
    feature_cols=["a", "b", "c", "temp"],
    target_cols=["level"],
    value_dict={"temp": 1000.0},
    sum_value=1.0,
    show_type="pred",
    ordinal_display="probability",
    ordinal_mode="class_confidence",
    boundary_margin=0.08,
    n=60,
)
fig.show()
```

三角図の `class_confidence` 表示では、色相が予測クラス、明度が最大予測確率を表します。`boundary_margin` 以下の top-2 確率差を持つ点は黒いリングで表示され、決定境界の候補を確認できます。2次元図と同様に `entropy` / `margin` / `class` へ切り替えられます。

## 補足

`plotly`, `pandas`, `numpy`, `torch` が必要です。既存の API には依存を強制せず、可視化を使う場合だけインストールする想定です。
