# Composition API and tabular integration

組成式の解析・変換・逆変換・simplex repair は、組成ユーティリティと `bochan.tabular.TabularBayesianOptimizer` の canonical `composition_sites` API から利用できます。

## 通常 API

```python
from bochan.composition import (
    CompositionSearchSpace,
    CompositionTransformer,
    SimplexTransform,
    parse_formula,
)
```

```python
transformer = CompositionTransformer(
    elements=["Fe", "Co", "Ni"],
    normalization="atomic_fraction",
    representation="ilr",
    prefix="formula",
)

X_composition = transformer.fit_transform(df["formula"])
formula = transformer.inverse_transform(X_composition)
```

`CompositionTransformer` は次を扱います。

- 組成式の解析と canonical 表記
- 原子分率・重量分率
- fraction / CLR / ALR / ILR
- 元素物性の加重記述子
- 数値座標から組成式への逆変換

組成候補の制約と補正には `CompositionSearchSpace` を使います。

```python
space = CompositionSearchSpace(
    components=["Fe", "Co", "Ni", "Cr"],
    total=1.0,
    bounds={
        "Fe": (0.20, 0.80),
        "Co": (0.00, 0.40),
        "Ni": (0.00, 0.60),
        "Cr": (0.00, 0.20),
    },
    steps={name: 0.01 for name in ["Fe", "Co", "Ni", "Cr"]},
    required_components=["Fe"],
    min_active_components=2,
    max_active_components=3,
)

valid_composition = space.repair(raw_composition)
```

## TabularBayesianOptimizer

Tabular API では、単一組成・複数サイト組成のどちらも `composition_sites` に統一します。単一組成は 1 サイトだけを定義します。

```python
from bochan.tabular import TabularBayesianOptimizer

bo = TabularBayesianOptimizer(
    task_type="regression",
    model_type="base",
    input_cols=["formula", "temperature"],
    target_cols="property",
    bounds={"temperature": [850.0, 1000.0]},
    composition_sites={
        "formula": {
            "column": "formula",
            "elements": ["Fe", "Co", "Ni"],
            "representation": "ilr",
            "coordinate_bounds": [-8.0, 8.0],
            "bounds": {
                "Fe": [0.20, 0.80],
                "Co": [0.00, 0.40],
                "Ni": [0.00, 0.60],
            },
            "steps": {
                "Fe": 0.01,
                "Co": 0.01,
                "Ni": 0.01,
            },
            "required_components": ["Fe"],
            "min_components": 2,
            "max_components": 3,
        }
    },
)

bo.fit(df)
candidates, acq_value = bo.candidate(acq_name="EI", q=3)
```

学習時には組成式列がモデル空間へ変換されます。

```text
formula -> formula__ilr__1, formula__ilr__2
```

候補 DataFrame は自動的に逆変換・repair されます。

```text
formula
formula__fraction__Fe
formula__fraction__Co
formula__fraction__Ni
temperature
```

`predict()` も組成式を含む元の DataFrame をそのまま受け取ります。

```python
prediction = bo.predict(
    pd.DataFrame(
        {
            "formula": ["Fe0.5Co0.2Ni0.3"],
            "temperature": [925.0],
        }
    )
)
```

## composition_sites の主な設定

| 設定 | 内容 |
|---|---|
| `column` | 組成式列 |
| `elements` | 探索対象元素 |
| `normalization` | `atomic_fraction` / `weight_fraction` |
| `representation` | `fractions` / `clr` / `alr` / `ilr` |
| `coordinate_bounds` | CLR/ALR/ILR 座標の探索範囲 |
| `bounds` | 元素量の上下限 |
| `steps` | 元素量の刻み |
| `required_components` | 必須元素 |
| `min_components` | 最小有効元素数 |
| `max_components` | 最大有効元素数 |
| `total` | 固定サイト総量 |
| `total_bounds` | 可変サイト総量の探索範囲 |

元素ごとの数値列から組成を作る場合は `element_columns` を使用します。`column` と `element_columns` は同一サイトで同時には指定しません。

## 複数サイト

複数の独立組成は同じ mapping にサイトを追加します。

```python
composition_sites={
    "A": {
        "column": "A_formula",
        "elements": ["La", "Sr", "Ba"],
    },
    "B": {
        "column": "B_formula",
        "elements": ["Fe", "Co", "Mn"],
    },
}
```

単一サイトと複数サイトで別 optimizer や別 API は使用しません。

## 組成記述子

通常 API では `CompositionDescriptorCalculator` または `CompositionTransformer(include_descriptors=True)` を使用できます。

Tabular API ではサイト設定に `include_descriptors=True`、`descriptor_properties`、`element_properties` を指定できます。記述子は組成から決まる派生値であるため、学習・予測には使用できますが、候補生成では独立変数として最適化しません。候補生成を行うサイトでは `include_descriptors=False` を使用してください。

## API 方針

`TabularBayesianOptimizer` の組成設定は `composition_sites` が唯一の canonical entry point です。旧 `composition_col` / `formula_col` / `composition_*` 引数を保持する compatibility layer はありません。呼び出し側を `composition_sites` へ直接移行してください。
