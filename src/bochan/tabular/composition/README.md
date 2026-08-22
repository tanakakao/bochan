# 組成データの tabular 最適化

組成処理は、汎用 domain API と tabular 統合を明確に分離します。

```text
bochan.composition
    組成式解析 / 正規化 / simplex 変換 / 記述子 / 探索空間
        ↑
bochan.tabular.composition
    DataFrame / composition_sites / TabularBayesianOptimizer 統合
```

`bochan.composition` は pandas や `bochan.tabular` に依存しません。旧 `bochan.tabular.composition` から domain API を再公開する forwarding / compatibility layer は提供しません。

## Domain API

組成そのものを扱う canonical API は `bochan.composition` です。

```python
from bochan.composition import (
    CompositionDescriptorCalculator,
    CompositionSearchSpace,
    CompositionTransformer,
    SimplexTransform,
    TorchSimplexTransform,
    close_compositions,
    format_formula,
    normalize_composition,
    parse_formula,
)
```

`CompositionTransformer` は pandas 非依存です。入力には組成式の iterable を渡し、`transform()` / `fit_transform()` は NumPy 配列、`inverse_transform()` は組成式の `list[str]` を返します。

```python
transformer = CompositionTransformer(
    elements=["Fe", "Co", "Ni"],
    representation="ilr",
    prefix="formula",
)

X_comp = transformer.fit_transform([
    "Fe0.5Co0.2Ni0.3",
    "Fe0.4Co0.3Ni0.3",
])
formulas = transformer.inverse_transform(X_comp)
```

勾配ベースの acquisition 最適化では `TorchSimplexTransform` を使い、ILR などの
モデル座標を fraction に戻します。先頭次元を保つため、通常の batch だけでなく
BoTorch の `batch_shape x q x d` Tensor をそのまま渡せます。演算は Torch 内で完結し、
入力の dtype / device と autograd graph を維持します。

```python
import torch

from bochan.composition import TorchSimplexTransform


inverse_ilr = TorchSimplexTransform(n_components=3, method="ilr")
X_ilr = torch.tensor(
    [[[0.25, -0.15], [-0.10, 0.30]]],
    dtype=torch.double,
    requires_grad=True,
)
fractions = inverse_ilr(X_ilr)
weighted_property = (fractions * X_ilr.new_tensor([1.0, 2.0, 4.0])).sum()
weighted_property.backward()
```

`method="fractions"` は非負 Tensor を closure し、`clr` / `alr` / `ilr` は安定な
softmax によって simplex へ写します。表形式の前処理には引き続き NumPy ベースの
`SimplexTransform` / `CompositionTransformer` を使います。

`CompositionSearchSpace` は total、元素上下限、刻み、必須元素、有効元素数を domain object として扱います。

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

valid = space.repair({"Fe": 0.6, "Co": 0.2, "Ni": 0.2})
```

## Tabular API

DataFrame と `TabularBayesianOptimizer` の統合は `bochan.tabular` / `bochan.tabular.composition` が担当します。

```python
from bochan.tabular import (
    CompositionColumnConfig,
    CompositionTabularPreprocessor,
    TabularBayesianOptimizer,
)
```

単一組成・複数サイト組成とも optimizer の canonical entry point は `composition_sites` です。

```python
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

組成式列は学習時にモデル座標へ変換され、候補生成後は組成式・分率へ逆変換されます。

```text
formula -> formula__ilr__1, formula__ilr__2
```

元素量が別々の列なら `element_columns` を使用します。

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

## 設計方針

- `bochan.composition` は組成 domain logic の唯一の実体です。
- `bochan.tabular.composition` は DataFrame と optimizer 統合だけを担当します。
- `bochan.composition -> bochan.tabular` の依存は禁止します。
- monkey patch、compatibility shim、forwarding module は置きません。
- 旧 domain import path を維持するための re-export は行いません。
