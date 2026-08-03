# 元素列を使った組成最適化

`TabularBayesianOptimizer`は、組成式列だけでなく、元素ごとに分かれた数値列も
`composition_sites`から直接扱えます。

## 単一組成

```python
from bochan.tabular import TabularBayesianOptimizer

bo = TabularBayesianOptimizer(
    input_cols=["Fe", "Ti", "Al", "temperature"],
    target_cols="property",
    bounds={"temperature": [850.0, 1100.0]},
    composition_sites={
        "alloy": {
            "element_columns": {
                "Fe": "Fe",
                "Ti": "Ti",
                "Al": "Al",
            },
            "input_basis": "atomic_fraction",
            "representation": "ilr",
            "min_components": 2,
            "max_components": 3,
            "required_components": ["Fe"],
            "bounds": {
                "Fe": [0.20, 0.80],
                "Ti": [0.00, 0.50],
                "Al": [0.00, 0.40],
            },
            "steps": {
                "Fe": 0.01,
                "Ti": 0.01,
                "Al": 0.01,
            },
        }
    },
)

bo.fit(df)
candidates, acq_value = bo.candidate(acq_name="EI", q=3)
```

学習時には`Fe`、`Ti`、`Al`列が行ごとに合計1へ正規化され、3元素の
組成は2次元のILR座標へ変換されます。候補生成後は逆変換・repairが行われ、
候補DataFrameには元の`Fe`、`Ti`、`Al`列が返されます。

## 0～1と0～100

入力値のスケールは行ごとの閉包で吸収されるため、以下は同じ組成として
扱われます。

```text
Fe=0.5, Ti=0.3, Al=0.2
Fe=50,  Ti=30,  Al=20
```

候補出力の合計値は`total`で指定します。

```python
# 候補を0～1の分率で返す
"total": 1.0

# 候補を合計100の百分率で返す
"total": 100.0
```

`total=100.0`の場合、上下限と刻みも同じ単位で指定します。

```python
{
    "total": 100.0,
    "bounds": {
        "Fe": [20.0, 80.0],
        "Ti": [0.0, 50.0],
        "Al": [0.0, 40.0],
    },
    "steps": {
        "Fe": 1.0,
        "Ti": 1.0,
        "Al": 1.0,
    },
}
```

## 原子比と重量比

原子比、mol比、at%の場合は次を指定します。

```python
"input_basis": "atomic_fraction"
```

重量比、mass fraction、wt%の場合は次を指定します。

```python
"input_basis": "weight_fraction"
```

重量比入力は内部で原子量を使って原子量比へ変換されます。ILR座標自体は
指定された入力基準上で構成され、元素記述子を使う場合は原子分率へ変換して
計算されます。

`input_basis`の主な別名は次のとおりです。

```text
atomic, atomic_fraction, at_fraction, at%, molar, mole_fraction
weight, weight_fraction, mass_fraction, wt%
raw, amount, stoichiometric, none
```

## A/Bサイト

AサイトとBサイトの元素列を独立に指定できます。

```python
bo = TabularBayesianOptimizer(
    input_cols=[
        "A_La", "A_Sr", "A_Ba",
        "B_Fe", "B_Co", "B_Mn",
        "temperature",
    ],
    target_cols="property",
    bounds={"temperature": [850.0, 1100.0]},
    composition_sites={
        "A": {
            "element_columns": {
                "La": "A_La",
                "Sr": "A_Sr",
                "Ba": "A_Ba",
            },
            "representation": "ilr",
            "min_components": 1,
            "max_components": 2,
            "required_components": ["La"],
        },
        "B": {
            "element_columns": {
                "Fe": "B_Fe",
                "Co": "B_Co",
                "Mn": "B_Mn",
            },
            "representation": "ilr",
            "min_components": 2,
            "max_components": 3,
            "required_components": ["Fe"],
        },
    },
)
```

AサイトとBサイトはそれぞれ独立したsimplexとして変換・逆変換・repairされます。
各サイトで候補元素、使用元素数、必須元素、上下限、刻み、合計値を個別に設定
できます。

## 組成式列との併用

サイトごとに、次のどちらか一方を指定します。

```python
# 組成式列
{"column": "A_formula", "elements": ["La", "Sr", "Ba"]}

# 元素ごとの数値列
{"element_columns": {"La": "A_La", "Sr": "A_Sr", "Ba": "A_Ba"}}
```

同じサイトで`column`と`element_columns`を同時には指定できません。ただし、
Aサイトを組成式列、Bサイトを元素列とする混在構成は可能です。

## ゼロ成分

CLR、ALR、ILRでは対数を使用するため、ゼロ成分には`pseudocount`が適用されます。
候補の逆変換後は`CompositionSearchSpace.repair()`により、元素数制約に従って
非採用元素が厳密にゼロへ戻されます。
