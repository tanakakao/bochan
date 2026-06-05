# bochan.visualization

`BayesianOptimizer` / `CandidateResult` / `BochanStudy` から、Plotly 用の可視化データと Figure を作る補助モジュールです。

## 目的

- YY plot
- 1D 予測曲線
- 2D 獲得関数または予測値ヒートマップ
- 3成分制約の三角プロット
- 2目的散布図
- study の cycle 推移

を API 本体から分離して扱います。

## 例

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

## 補足

`plotly`, `pandas`, `numpy`, `torch` が必要です。既存の API には依存を強制せず、可視化を使う場合だけインストールする想定です。
