# 組成データの tabular 最適化

このパッケージは、化学組成を数値特徴量へ変換し、既存の `bochan` の tabular モデルへ渡すための組成ユーティリティを提供します。組成専用 GP を定義するのではなく、モデルの前後で次を行います。

```text
組成を含む DataFrame
    ↓
組成の解析・正規化
    ↓
Fraction / CLR / ALR / ILR へ変換
    ↓
TabularBayesianOptimizer
    ↓
候補座標を組成へ逆変換
    ↓
組成制約・刻みを repair
```

## TabularBayesianOptimizer との統合

`TabularBayesianOptimizer` では、単一組成・複数サイト組成とも `composition_sites` が唯一の設定入口です。

```python
from bochan.tabular import TabularBayesianOptimizer

bo = TabularBayesianOptimizer(
    task_type="regression",
    model_type="base",
    input_cols=["formula", "temperature"],
    target_cols="property",
    bounds={"temperature": [850.0, 1050.0]},
    composition_sites={
        "formula": {
            "column": "formula",
            "elements": ["Fe", "Co", "Ni"],
            "normalization": "atomic_fraction",
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

学習時には `formula` がモデル座標へ変換されます。

```text
formula -> formula__ilr__1, formula__ilr__2
```

候補生成後は、既定で組成式と分率列へ戻されます。

```text
formula
formula__fraction__Fe
formula__fraction__Co
formula__fraction__Ni
temperature
```

`keep_composition_coordinates=True` を指定すると ILR 等のモデル座標も残せます。`return_composition=False` ではモデル座標の DataFrame をそのまま返します。

## 元素列を直接使う

組成式列ではなく、元素量が別々の数値列として存在する場合は `element_columns` を使います。

```python
composition_sites={
    "alloy": {
        "element_columns": {
            "Fe": "Fe_wt",
            "Co": "Co_wt",
            "Ni": "Ni_wt",
        },
        "input_basis": "weight_fraction",
        "representation": "ilr",
        "total": 100.0,
    }
}
```

1 サイトでは `column` と `element_columns` のどちらか一方だけを使用します。

## 複数サイト

A/B サイトなど複数組成も同じ API で定義します。

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

単一サイト専用 optimizer や別の互換 API は使用しません。

## 主なサイト設定

| キー | 内容 |
|---|---|
| `column` | 組成式列 |
| `element_columns` | 元素ごとの数値列 |
| `elements` | formula site の候補元素 |
| `normalization` | `atomic_fraction` / `weight_fraction` |
| `representation` | `fractions` / `clr` / `alr` / `ilr` |
| `reference_element` | ALR の基準元素 |
| `pseudocount` | log-ratio 変換用微小値 |
| `coordinate_bounds` | CLR/ALR/ILR 座標の探索範囲 |
| `bounds` | 元素量の上下限 |
| `steps` | 元素量の刻み |
| `required_components` | 必須元素 |
| `min_components` | 最小有効元素数 |
| `max_components` | 最大有効元素数 |
| `total` | 固定サイト総量 |
| `total_bounds` | 可変サイト総量の探索範囲 |
| `include_descriptors` | 組成記述子を学習・予測特徴量へ追加 |

`min_active_components`、`max_active_components`、`required_elements`、`composition_col`、`formula_col`、`composition_*` などの compatibility alias は提供しません。

## 組成ユーティリティ

モデルとは独立に、以下を直接利用できます。

```python
from bochan.tabular.composition import (
    CompositionDescriptorCalculator,
    CompositionSearchSpace,
    CompositionTransformer,
    SimplexTransform,
    close_compositions,
    format_formula,
    normalize_composition,
    parse_formula,
)
```

### CompositionTransformer

組成式の解析、原子/重量分率への正規化、Fraction/CLR/ALR/ILR 変換、逆変換、必要に応じた元素記述子の追加を担当します。

```python
transformer = CompositionTransformer(
    elements=["Fe", "Co", "Ni"],
    normalization="atomic_fraction",
    representation="ilr",
    prefix="formula",
)

X_comp = transformer.fit_transform(df["formula"])
```

### CompositionSearchSpace

候補組成の total、元素上下限、刻み、必須元素、有効元素数を一つの探索空間として扱います。

```python
space = CompositionSearchSpace(
    components=["Fe", "Co", "Ni"],
    total=1.0,
    bounds={
        "Fe": (0.20, 0.80),
        "Co": (0.00, 0.40),
        "Ni": (0.00, 0.60),
    },
    steps={"Fe": 0.01, "Co": 0.01, "Ni": 0.01},
    required_components=["Fe"],
    min_active_components=2,
    max_active_components=3,
)

valid = space.repair(raw_composition)
```

ここで `min_active_components` / `max_active_components` は `CompositionSearchSpace` 自体の正式フィールドです。`TabularBayesianOptimizer.composition_sites` の設定名は `min_components` / `max_components` を使用します。

## 記述子

`CompositionDescriptorCalculator` または `CompositionTransformer(include_descriptors=True)` で組成記述子を計算できます。Tabular API でもサイトに `include_descriptors=True` を指定すると学習・予測へ利用できます。

記述子は組成から決まる派生値なので、現在の連続候補最適化では独立変数として最適化しません。候補生成を行うサイトでは `include_descriptors=False` を使用してください。

## 設計方針

組成処理は tabular optimizer 本体への monkey patch や compatibility shim ではなく、明示的な transform / search-space component と `composition_sites` の canonical path で構成します。旧 API を維持するための forwarding module は置きません。
