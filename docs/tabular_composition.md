# Composition API and tabular integration

組成式の解析・変換・逆変換・simplex repair は通常 API の `bochan.composition` から利用できます。
モデル固有のクラスではなく、任意の回帰・分類・active learning 処理の前後に置く汎用 API です。

```python
from bochan.composition import (
    CompositionSearchSpace,
    CompositionTransformer,
    SimplexTransform,
    parse_formula,
)
```

## 通常 API

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

## TabularBayesianOptimizer の直接引数

Tabular API では `CompositionColumnConfig` を作る必要はありません。
元の `input_cols` には組成式列をそのまま指定します。

```python
from bochan.tabular import TabularBayesianOptimizer

bo = TabularBayesianOptimizer(
    task_type="regression",
    model_type="base",
    input_cols=["formula", "temperature"],
    target_cols="property",
    bounds={
        "temperature": [850.0, 1000.0],
    },
    composition_col="formula",
    composition_elements=["Fe", "Co", "Ni"],
    composition_representation="ilr",
    composition_coordinate_bounds=[-8.0, 8.0],
    composition_bounds={
        "Fe": [0.20, 0.80],
        "Co": [0.00, 0.40],
        "Ni": [0.00, 0.60],
    },
    composition_steps={
        "Fe": 0.01,
        "Co": 0.01,
        "Ni": 0.01,
    },
    composition_required_components=["Fe"],
    composition_min_components=2,
    composition_max_components=3,
)

bo.fit(df)
candidates, acq_value = bo.candidate(acq_name="EI", q=3)
```

学習時には内部で次の変換が行われます。

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

## 主な直接引数

| 引数 | 内容 |
|---|---|
| `composition_col` | 組成式列 |
| `composition_elements` | 探索対象元素。省略時は学習データから推定 |
| `composition_normalization` | `atomic_fraction` / `weight_fraction` |
| `composition_representation` | `fractions` / `clr` / `alr` / `ilr` |
| `composition_coordinate_bounds` | CLR/ALR/ILR 座標の探索範囲 |
| `composition_bounds` | 元素比率の上下限 |
| `composition_steps` | 元素比率の刻み |
| `composition_required_components` | 必須元素 |
| `composition_min_components` | 最小有効元素数 |
| `composition_max_components` | 最大有効元素数 |

## 組成記述子

通常 API では `CompositionDescriptorCalculator` または
`CompositionTransformer(include_descriptors=True)` を使用できます。

Tabular API でも `composition_include_descriptors=True` で学習と予測に使用できます。
ただし、記述子は組成から決まる派生値であるため、現在の連続候補最適化では独立変数として最適化しません。
候補生成時は `composition_include_descriptors=False` を使用してください。

## 互換 API

既存の `CompositionColumnConfig` と `CompositionTabularPreprocessor` は互換用途として残しています。
新規コードでは、通常 API または `TabularBayesianOptimizer` の直接引数を推奨します。
