# A/Bサイトを分けた組成最適化

`bochan.tabular.TabularBayesianOptimizer`は、`composition_sites`を指定することで、Aサイト、Bサイトなど複数の独立した組成サイトを同時に扱えます。

各サイトについて、以下を個別に指定できます。

- 使用可能な元素
- 1候補で使用する元素数の最小値・最大値
- 必ず使用する元素
- 各元素の上下限
- 各元素の刻み幅
- Fraction、CLR、ALR、ILR表現
- 原子分率または重量分率

## 入力データ

サイトごとに組成式列を分けます。

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
    bounds={
        "temperature": [850.0, 1100.0],
    },
    composition_sites={
        "A": {
            "column": "A_formula",

            # Aサイトで使用可能な元素
            "elements": ["La", "Sr", "Ba", "Ca"],

            # 1候補あたり1～2元素
            "min_components": 1,
            "max_components": 2,

            # Laを必ず使用
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

            # Bサイトで使用可能な元素
            "elements": ["Fe", "Co", "Mn", "Ni"],

            # 1候補あたり2～3元素
            "min_components": 2,
            "max_components": 3,

            # Feを必ず使用
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

候補には、サイトごとの組成式と分率列が返されます。

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

## 元素数を固定する

Aサイトを常に2元素にする場合は、最小値と最大値を同じにします。

```python
"A": {
    "column": "A_formula",
    "elements": ["La", "Sr", "Ba", "Ca"],
    "min_components": 2,
    "max_components": 2,
}
```

## 使用元素を限定する

`elements`に含めた元素だけが、そのサイトの候補として使用されます。

```python
"elements": ["La", "Sr"]
```

この場合、BaやCaは候補に生成されません。

## 必須元素を複数指定する

```python
"required_components": ["La", "Sr"]
```

必須元素数は`max_components`以下である必要があります。

## サイトごとの独立したrepair

候補生成後、各サイトは独立して次の制約へrepairされます。

1. 必須元素を有効化
2. 正の下限を持つ元素を有効化
3. 候補値が大きい元素から最大元素数まで選択
4. 最小元素数に足りない場合は候補元素を追加
5. 非選択元素をゼロに固定
6. サイト内の合計が`total`になるようbounded simplexへ射影
7. 刻み幅へ丸め、残差を再配分

AサイトとBサイトでは、元素集合、元素数、必須元素、上下限、刻み幅が完全に独立です。

## 単一組成APIとの関係

従来の単一組成指定は引き続き利用できます。

```python
bo = TabularBayesianOptimizer(
    composition_col="formula",
    composition_elements=["Fe", "Co", "Ni"],
    composition_min_components=2,
    composition_max_components=3,
    composition_required_components=["Fe"],
)
```

`composition_sites`と`composition_col`または`formula_col`を同時に指定することはできません。

## 主なサイト設定

| キー | 内容 | 既定値 |
|---|---|---|
| `column` | サイト組成式の列名 | 必須 |
| `elements` | 使用可能な元素 | 必須 |
| `min_components` | 使用元素数の最小値 | `1` |
| `max_components` | 使用元素数の最大値 | 全候補元素数 |
| `required_components` | 必ず使用する元素 | なし |
| `representation` | `fractions`、`clr`、`alr`、`ilr` | `ilr` |
| `normalization` | `atomic_fraction`または`weight_fraction` | `atomic_fraction` |
| `bounds` | 元素ごとの下限・上限 | `[0, total]` |
| `steps` | 元素ごとの刻み幅 | なし |
| `total` | サイト内組成量の合計 | `1.0` |
| `coordinate_bounds` | CLR・ALR・ILR座標の探索範囲 | `[-8, 8]` |

`min_active_components`、`max_active_components`は、それぞれ`min_components`、`max_components`の別名としても使用できます。`required_elements`は`required_components`の別名です。
