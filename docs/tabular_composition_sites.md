# A/Bサイトを分けた組成最適化

`bochan.tabular.TabularBayesianOptimizer` は `composition_sites` を canonical API として、単一組成・A/Bサイトなど複数の独立した組成サイトを同じ仕組みで扱います。

各サイトでは、使用可能元素、元素数、必須元素、上下限、刻み、表現、原子分率/重量分率、固定または可変の総量を個別に設定できます。

## 入力データ

```python
import pandas as pd


df = pd.DataFrame(
    {
        "A_formula": ["La0.7Sr0.3", "La0.5Ba0.5"],
        "B_formula": ["Fe0.6Co0.4", "Fe0.5Mn0.5"],
        "temperature": [900.0, 950.0],
        "property": [10.0, 12.0],
    }
)
```

## A/Bサイトの設定

```python
from bochan.tabular import TabularBayesianOptimizer


bo = TabularBayesianOptimizer(
    input_cols=["A_formula", "B_formula", "temperature"],
    target_cols="property",
    bounds={"temperature": [850.0, 1100.0]},
    composition_sites={
        "A": {
            "column": "A_formula",
            "elements": ["La", "Sr", "Ba", "Ca"],
            "min_components": 1,
            "max_components": 2,
            "required_components": ["La"],
            "representation": "ilr",
            "bounds": {
                "La": [0.20, 1.00],
                "Sr": [0.00, 0.60],
                "Ba": [0.00, 0.50],
                "Ca": [0.00, 0.30],
            },
            "steps": {
                "La": 0.01,
                "Sr": 0.01,
                "Ba": 0.01,
                "Ca": 0.01,
            },
        },
        "B": {
            "column": "B_formula",
            "elements": ["Fe", "Co", "Mn", "Ni"],
            "min_components": 2,
            "max_components": 3,
            "required_components": ["Fe"],
            "representation": "ilr",
            "bounds": {
                "Fe": [0.20, 1.00],
                "Co": [0.00, 0.60],
                "Mn": [0.00, 0.50],
                "Ni": [0.00, 0.40],
            },
            "steps": {
                "Fe": 0.01,
                "Co": 0.01,
                "Mn": 0.01,
                "Ni": 0.01,
            },
        },
    },
)

bo.fit(df)
candidates, acq_value = bo.candidate(acq_name="EI", q=3)
```

候補にはサイトごとの組成式と分率列が返されます。

```text
A_formula
A__fraction__La
A__fraction__Sr
A__fraction__Ba
A__fraction__Ca
B_formula
B__fraction__Fe
B__fraction__Co
B__fraction__Mn
B__fraction__Ni
temperature
```

## 単一組成

単一組成も同じ `composition_sites` を 1 サイトだけ定義します。

```python
bo = TabularBayesianOptimizer(
    input_cols=["formula", "temperature"],
    target_cols="property",
    composition_sites={
        "formula": {
            "column": "formula",
            "elements": ["Fe", "Co", "Ni"],
            "min_components": 2,
            "max_components": 3,
            "required_components": ["Fe"],
        }
    },
)
```

単一サイト専用の `composition_col` / `formula_col` / `composition_*` API はありません。

## 元素数を固定する

最小値と最大値を同じにします。

```python
"A": {
    "column": "A_formula",
    "elements": ["La", "Sr", "Ba", "Ca"],
    "min_components": 2,
    "max_components": 2,
}
```

## 必須元素

```python
"required_components": ["La", "Sr"]
```

必須元素数は `max_components` 以下である必要があります。

## サイトごとの repair

候補生成後、各サイトは独立して次の制約へ repair されます。

1. 必須元素を有効化
2. 正の下限を持つ元素を有効化
3. 候補値が大きい元素から最大元素数まで選択
4. 最小元素数に足りない場合は候補元素を追加
5. 非選択元素をゼロに固定
6. サイト内の合計が `total` になるよう bounded simplex へ射影
7. 刻み幅へ丸め、残差を再配分

AサイトとBサイトでは元素集合、元素数、必須元素、上下限、刻み幅が独立です。

## 主なサイト設定

| キー | 内容 | 既定値 |
|---|---|---|
| `column` | サイト組成式の列名 | formula site では必須 |
| `element_columns` | 元素ごとの数値列 | element-column site で使用 |
| `elements` | 使用可能な元素 | formula site では必須 |
| `min_components` | 使用元素数の最小値 | `1` |
| `max_components` | 使用元素数の最大値 | 全候補元素数 |
| `required_components` | 必ず使用する元素 | なし |
| `representation` | `fractions` / `clr` / `alr` / `ilr` | `ilr` |
| `normalization` | `atomic_fraction` / `weight_fraction` | `atomic_fraction` |
| `bounds` | 元素ごとの下限・上限 | `[0, total]` |
| `steps` | 元素ごとの刻み幅 | なし |
| `total` | 固定サイト総量 | `1.0` |
| `total_bounds` | 可変サイト総量の探索範囲 | なし |
| `coordinate_bounds` | CLR/ALR/ILR座標の探索範囲 | `[-8, 8]` |

設定名は上記 canonical 名のみを使用します。`min_active_components` / `max_active_components` / `required_elements` などの compatibility alias はサポートしません。
