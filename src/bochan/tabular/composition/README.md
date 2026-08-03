# 組成式を入力とする tabular 最適化

このパッケージは、化学組成式を数値特徴量へ変換し、既存の `bochan` モデルへ渡すための機能を提供します。

組成式専用の GP モデルを定義するのではなく、次の処理をモデルの外側で行います。

```text
組成式を含む DataFrame
    ↓
組成式の解析
    ↓
原子分率・重量分率への正規化
    ↓
Fraction / CLR / ALR / ILR 座標への変換
    ↓
既存の TabularBayesianOptimizer
    ↓
候補座標を組成へ逆変換
    ↓
組成制約・刻みの repair
    ↓
組成式と各元素比率を含む候補 DataFrame
```

通常の利用では `CompositionColumnConfig` を作成する必要はありません。`TabularBayesianOptimizer` に `composition_*` 引数を直接指定してください。

---

## 1. インストール

pandas を使用するため、tabular extra をインストールします。

```bash
pip install -e ".[tabular]"
```

開発用依存関係も含める場合は次の通りです。

```bash
pip install -e ".[dev,tabular,visualization,evo]"
```

---

## 2. 最小例

入力データに組成式列、プロセス条件、目的変数を用意します。

```python
import pandas as pd

from bochan.tabular import TabularBayesianOptimizer


df = pd.DataFrame(
    {
        "formula": [
            "Fe0.50Co0.20Ni0.30",
            "Fe0.45Co0.25Ni0.30",
            "Fe0.40Co0.20Ni0.40",
            "Fe0.55Co0.15Ni0.30",
            "Fe0.35Co0.30Ni0.35",
        ],
        "temperature": [900.0, 925.0, 950.0, 975.0, 1000.0],
        "property": [10.2, 11.0, 11.8, 11.4, 12.1],
    }
)
```

`input_cols` には、変換後の列ではなく元の組成式列を指定します。

```python
bo = TabularBayesianOptimizer(
    task_type="regression",
    model_type="base",
    input_cols=["formula", "temperature"],
    target_cols="property",
    bounds={
        "temperature": [850.0, 1050.0],
    },
    composition_col="formula",
    composition_elements=["Fe", "Co", "Ni"],
    composition_representation="ilr",
)

bo.fit(df)
```

候補を生成します。

```python
candidates_df, acq_value = bo.candidate(
    acq_name="EI",
    q=3,
    num_restarts=10,
    raw_samples=256,
)

print(candidates_df)
```

候補 DataFrame は、組成式、元素比率、その他の探索変数を含みます。

```text
formula
formula__fraction__Fe
formula__fraction__Co
formula__fraction__Ni
temperature
```

ILR 座標は既定では出力から除外されます。

---

## 3. 完全な組成制約付き例

元素ごとの上下限、刻み、必須元素、有効元素数を指定できます。

```python
bo = TabularBayesianOptimizer(
    task_type="regression",
    model_type="base",
    input_cols=["formula", "temperature", "time"],
    target_cols="property",
    bounds={
        "temperature": [850.0, 1050.0],
        "time": [1.0, 10.0],
    },

    # 組成式列
    composition_col="formula",

    # 探索対象とする元素とその順序
    composition_elements=["Fe", "Co", "Ni", "Cr"],

    # 組成表現
    composition_normalization="atomic_fraction",
    composition_representation="ilr",
    composition_pseudocount=1e-12,

    # ILR 座標の探索範囲
    composition_coordinate_bounds=[-8.0, 8.0],

    # 組成空間の制約
    composition_total=1.0,
    composition_bounds={
        "Fe": [0.20, 0.80],
        "Co": [0.00, 0.40],
        "Ni": [0.00, 0.60],
        "Cr": [0.00, 0.20],
    },
    composition_steps={
        "Fe": 0.01,
        "Co": 0.01,
        "Ni": 0.01,
        "Cr": 0.01,
    },
    composition_required_components=["Fe"],
    composition_min_components=2,
    composition_max_components=3,
    composition_precision=4,
)

bo.fit(df)

candidates_df, acq_value = bo.candidate(
    acq_name="EI",
    q=5,
    num_restarts=20,
    raw_samples=512,
)
```

候補生成後に、次の順序で組成を補正します。

1. ILR / ALR / CLR 座標から組成比へ逆変換
2. 非負制約と合計値制約を満たす simplex へ射影
3. 元素ごとの上下限を適用
4. 必須元素と有効元素数を適用
5. 指定刻みへ丸める
6. 合計値を再調整
7. canonical な組成式を生成

---

## 4. 主な直接引数

### 4.1 組成列と元素

| 引数 | 既定値 | 説明 |
|---|---:|---|
| `composition_col` | `None` | 組成式が格納された列名。指定すると組成処理を有効化します。 |
| `formula_col` | `None` | `composition_col` の alias。両方に異なる値を指定するとエラーです。 |
| `composition_elements` | `None` | 使用する元素と列順。省略時は学習データから原子番号順で推定します。 |
| `composition_prefix` | 組成列名 | 生成列名の接頭辞です。 |
| `composition_precision` | `6` | 逆変換した組成式の小数桁数です。 |

元素集合を固定できる場合は、`composition_elements` を明示することを推奨します。

```python
composition_elements=["Fe", "Co", "Ni"]
```

学習データから自動推定した場合、予測時に未学習の元素を含む組成式を入力するとエラーになります。

### 4.2 正規化と表現

| 引数 | 既定値 | 説明 |
|---|---:|---|
| `composition_normalization` | `"atomic_fraction"` | 組成を原子分率または重量分率へ変換します。 |
| `composition_representation` | `"ilr"` | モデルへ渡す組成座標です。 |
| `composition_reference_element` | `None` | ALR の分母に使用する元素です。 |
| `composition_pseudocount` | `1e-12` | ゼロ成分を含む log-ratio 変換用の微小値です。 |

`composition_normalization` は次を指定できます。

```text
atomic_fraction
weight_fraction
none
```

通常は `atomic_fraction` を推奨します。

`composition_representation` は次を指定できます。

```text
fractions
clr
alr
ilr
```

### 4.3 探索制約

| 引数 | 既定値 | 説明 |
|---|---:|---|
| `composition_total` | `1.0` | 組成比の合計値です。100 wt% 表記なら `100.0` も指定できます。 |
| `composition_bounds` | `{}` | 元素ごとの下限と上限です。 |
| `composition_steps` | `{}` | 元素ごとの離散刻みです。 |
| `composition_required_components` | `[]` | 候補に必ず含める元素です。 |
| `composition_min_components` | `1` | 非ゼロ元素数の下限です。 |
| `composition_max_components` | `None` | 非ゼロ元素数の上限です。 |
| `composition_coordinate_bounds` | `[-8, 8]` | CLR / ALR / ILR 座標の最適化範囲です。 |

`composition_bounds` と `composition_steps` のキーは、`composition_elements` の元素名に一致させます。

```python
composition_bounds={
    "Fe": [0.20, 0.80],
    "Co": [0.00, 0.40],
    "Ni": [0.00, 0.60],
}

composition_steps={
    "Fe": 0.01,
    "Co": 0.01,
    "Ni": 0.01,
}
```

---

## 5. Fraction / CLR / ALR / ILR の選択

### 5.1 fractions

```python
composition_representation="fractions"
```

各元素分率をそのままモデルへ渡します。

```text
formula__fraction__Fe
formula__fraction__Co
formula__fraction__Ni
```

解釈しやすい一方、合計が一定なので列は線形従属します。少数元素の単純な問題や可視化には利用できますが、GP の標準入力としては ILR を推奨します。

fraction 表現では、元素ごとの `composition_bounds` からモデル座標の bounds を自動生成します。

### 5.2 CLR

```python
composition_representation="clr"
```

全元素を対称に扱えますが、CLR 座標の合計は 0 となるため、座標自体は線形従属します。

```text
formula__clr__Fe
formula__clr__Co
formula__clr__Ni
```

### 5.3 ALR

```python
composition_representation="alr"
composition_reference_element="Ni"
```

指定元素を基準とした対数比を使用します。

```text
formula__alr__Fe_over_Ni
formula__alr__Co_over_Ni
```

出力次元は元素数より1小さくなりますが、結果は基準元素に依存します。

### 5.4 ILR

```python
composition_representation="ilr"
```

単体上の組成を直交座標へ変換します。

```text
formula__ilr__1
formula__ilr__2
```

出力次元は元素数より1小さく、通常の GP へ渡す標準表現として最も推奨します。

---

## 6. 学習時の内部変換

次の設定を例とします。

```python
bo = TabularBayesianOptimizer(
    input_cols=["formula", "temperature"],
    target_cols="property",
    composition_col="formula",
    composition_elements=["Fe", "Co", "Ni"],
    composition_representation="ilr",
    bounds={"temperature": [850.0, 1050.0]},
)
```

`fit()` 内部では、次のように `input_cols` が変換されます。

```text
ユーザー指定:
    formula
    temperature

モデル入力:
    formula__ilr__1
    formula__ilr__2
    temperature
```

学習後の実際の特徴量名は次で確認できます。

```python
bo.dataset.feature_names
```

組成変換器と探索空間も確認できます。

```python
bo.composition_transformer_
bo.composition_search_space_
```

---

## 7. 候補 DataFrame の制御

### 7.1 組成式と比率を返す

既定では、候補を組成式と各元素比率へ戻します。

```python
candidates_df, acq_value = bo.candidate(
    acq_name="EI",
    q=3,
)
```

### 7.2 ILR 等の座標も残す

```python
candidates_df, acq_value = bo.candidate(
    acq_name="EI",
    q=3,
    keep_composition_coordinates=True,
)
```

この場合は、次の両方が出力されます。

```text
formula
formula__fraction__Fe
formula__fraction__Co
formula__fraction__Ni
formula__ilr__1
formula__ilr__2
temperature
```

### 7.3 モデル座標のまま返す

```python
candidates_df, acq_value = bo.candidate(
    acq_name="EI",
    q=3,
    return_composition=False,
)
```

### 7.4 tensor のまま返す

```python
candidates, acq_value = bo.candidate(
    acq_name="EI",
    q=3,
    return_dataframe=False,
)
```

この場合、組成式への自動逆変換は行われません。

---

## 8. 予測

`predict()` には、学習時と同じ形式の組成式 DataFrame を渡せます。

```python
prediction_df = bo.predict(
    pd.DataFrame(
        {
            "formula": [
                "Fe0.50Co0.20Ni0.30",
                "Fe0.40Co0.25Ni0.35",
            ],
            "temperature": [925.0, 975.0],
        }
    )
)

print(prediction_df)
```

組成式列は内部で学習済みの変換器を使って ILR 等へ変換されます。

入力列も出力へ含める場合は、通常の tabular API と同様に `include_input=True` を指定します。

```python
prediction_df = bo.predict(
    prediction_input,
    include_input=True,
)
```

`include_input=True` の入力部分には、内部変換後のモデル特徴量が含まれる場合があります。元の組成式を必ず残したい場合は、予測入力 DataFrame をユーザー側でも保持してください。

---

## 9. 明示的な変換と逆変換

学習済み optimizer から、変換処理だけを呼び出せます。

### 9.1 組成式 DataFrame をモデル空間へ変換

```python
model_df = bo.transform_compositions(
    pd.DataFrame(
        {
            "formula": ["Fe0.5Co0.2Ni0.3"],
            "temperature": [950.0],
        }
    )
)
```

### 9.2 モデル空間から組成式へ逆変換

```python
restored_df = bo.inverse_compositions(
    model_candidate_df,
    repair=True,
)
```

repair を無効化する場合です。

```python
restored_df = bo.inverse_compositions(
    model_candidate_df,
    repair=False,
)
```

通常の実験候補には `repair=True` を推奨します。

---

## 10. CSV から使用する

```python
from bochan.tabular import TabularBayesianOptimizer


bo = TabularBayesianOptimizer.from_csv(
    "composition_data.csv",
    input_cols=["formula", "temperature"],
    target_cols="property",
    bounds={"temperature": [850.0, 1050.0]},
    composition_col="formula",
    composition_elements=["Fe", "Co", "Ni"],
    composition_representation="ilr",
    read_csv_kwargs={"encoding": "utf-8-sig"},
)

bo.fit()
```

`from_csv()` でもコンストラクタへ直接指定した組成引数が保持されます。

---

## 11. ask / tell 形式

`ask()` は `candidate()` の alias なので、組成式付き候補を返します。

```python
candidates_df, acq_value = bo.ask(
    acq_name="EI",
    q=3,
    keep_composition_coordinates=True,
)
```

現在の `tell()` はモデル空間の入力列を使用します。そのため、組成候補を `tell()` へ戻す場合は `keep_composition_coordinates=True` を指定し、ILR 等の座標列を保持してください。

```python
candidates_df["property"] = measured_values
bo.tell(candidates_df, refit=True)
```

組成式と元素分率だけを返した候補は、そのまま `tell()` へ渡さず、`transform_compositions()` でモデル空間へ変換してから登録してください。

---

## 12. 組成記述子

原子番号や原子量などの組成加重統計量を追加できます。

```python
bo = TabularBayesianOptimizer(
    input_cols=["formula", "temperature"],
    target_cols="property",
    composition_col="formula",
    composition_elements=["Fe", "Co", "Ni"],
    composition_representation="ilr",
    composition_include_descriptors=True,
    composition_descriptor_properties=[
        "atomic_number",
        "atomic_weight",
        "electronegativity",
    ],
    composition_descriptor_statistics=[
        "mean",
        "std",
        "min",
        "max",
        "range",
    ],
    composition_element_properties={
        "electronegativity": {
            "Fe": 1.83,
            "Co": 1.88,
            "Ni": 1.91,
        }
    },
)
```

生成される列の例です。

```text
formula__descriptor__atomic_number__mean
formula__descriptor__atomic_number__std
formula__descriptor__atomic_weight__range
formula__descriptor__electronegativity__mean
formula__descriptor__num_elements
formula__descriptor__mixing_entropy
```

記述子は、組成から一意に決まる派生特徴量です。

現在は、次の用途に利用できます。

- 物性予測
- 固定候補プールの評価
- active learning
- `predict()`
- feature importance

一方、連続候補最適化で記述子を独立変数として動かすことはできません。`composition_include_descriptors=True` のまま `candidate()` を呼ぶと、誤った候補を生成しないため明示的なエラーになります。

候補最適化を行う場合は次の設定を推奨します。

```python
composition_representation="ilr"
composition_include_descriptors=False
```

---

## 13. 通常の組成 API

組成処理だけをモデルと独立に使用できます。

```python
from bochan.api.composition import (
    CompositionDescriptorCalculator,
    CompositionSearchSpace,
    CompositionTransformer,
    SimplexTransform,
    format_formula,
    parse_formula,
)
```

便宜上、次の import も利用できます。

```python
from bochan.composition import CompositionTransformer
```

### 13.1 組成式を解析する

```python
from bochan.api.composition import parse_formula


composition = parse_formula("(La0.6Sr0.4)(Co0.2Fe0.8)O3")

print(composition)
```

```python
{
    "La": 0.6,
    "Sr": 0.4,
    "Co": 0.2,
    "Fe": 0.8,
    "O": 3.0,
}
```

水和物も解析できます。

```python
parse_formula("CuSO4·5H2O")
```

### 13.2 組成式を数値 DataFrame へ変換する

```python
import pandas as pd

from bochan.api.composition import CompositionTransformer


transformer = CompositionTransformer(
    elements=["Fe", "Co", "Ni"],
    representation="ilr",
    prefix="alloy",
)

model_X = transformer.fit_transform(
    pd.Series(
        [
            "Fe0.5Co0.2Ni0.3",
            "Fe0.4Co0.3Ni0.3",
        ]
    )
)
```

生成列は次の通りです。

```text
alloy__ilr__1
alloy__ilr__2
```

### 13.3 組成式へ逆変換する

```python
formula = transformer.inverse_transform(model_X)
```

逆変換は元の文字列を完全に復元する処理ではなく、正規化された canonical formula を生成する処理です。

### 13.4 組成探索空間を repair する

```python
from bochan.api.composition import CompositionSearchSpace


space = CompositionSearchSpace(
    components=["Fe", "Co", "Ni"],
    total=1.0,
    bounds={
        "Fe": [0.2, 0.8],
        "Co": [0.0, 0.4],
        "Ni": [0.0, 0.6],
    },
    steps={
        "Fe": 0.01,
        "Co": 0.01,
        "Ni": 0.01,
    },
    required_components=["Fe"],
    min_active_components=2,
    max_active_components=3,
)

repaired = space.repair(
    {
        "Fe": 0.523,
        "Co": 0.164,
        "Ni": 0.319,
    }
)
```

検証だけを行う場合です。

```python
errors = space.validate(repaired)
```

有効な組成なら空リストが返ります。

---

## 14. 対応する組成式

対応例です。

```text
Fe0.5Ni0.5
LiFePO4
(La0.6Sr0.4)(Co0.2Fe0.8)O3
CuSO4·5H2O
```

次に対応します。

- 元素記号
- 整数係数
- 小数係数
- `()` と `[]` の入れ子
- `·` 区切りの水和物

次は直接解析できません。

```text
Li1-xNaxCoO2
Fe3+
O3-delta
La(1-x)SrxMnO3
```

記号変数 `x`、電荷表記、`delta` などを扱う場合は、事前に数値へ展開した組成式を入力してください。

また、組成式だけから A サイト、B サイト、結晶構造、相、欠陥位置を一意に推定することはできません。これらは別の説明変数または明示的なサイト定義として管理してください。

---

## 15. 注意点

### 元の組成式文字列は完全には復元されない

```text
Li2Fe2P2O8
LiFePO4
```

これらを原子分率へ正規化すると同じ組成になります。逆変換では canonical な組成式を返すため、元の文字列を保存したい場合は入力列を別途保持してください。

### ゼロ成分と log-ratio 変換

CLR / ALR / ILR は対数を使用するため、ゼロ成分には `composition_pseudocount` が加えられます。逆変換時には表示精度以下の擬似成分を除去します。

### ILR 座標の bounds

組成比の上下限は ILR 座標上では単純な矩形になりません。現在は `composition_coordinate_bounds` の矩形内で獲得関数を最適化し、逆変換後に `CompositionSearchSpace` で有効組成へ repair します。

範囲が狭すぎると探索可能な組成が限定され、広すぎるとほぼ純元素に近い極端な候補が増えます。既定値 `[-8, 8]` から開始し、対象系に応じて調整してください。

### 重量分率

`composition_normalization="weight_fraction"` の場合、モデルは重量分率を入力として使用します。出力組成式は元素の原子量を用いて原子分率へ戻して表示します。

### 記述子を独立に最適化しない

組成記述子は組成から再計算される値です。組成座標と記述子を独立変数として同時最適化すると、物理的に整合しない候補が生成されます。

### 結晶構造は別入力

同一組成でも結晶構造、相、粒径、焼成条件などにより物性は変化します。必要に応じて次を別列として追加してください。

```text
structure_type
space_group
phase
calcination_temperature
calcination_time
pressure
atmosphere
```

---

## 16. 推奨設定

固定元素系で組成比とプロセス条件を最適化する場合は、次から開始することを推奨します。

```python
bo = TabularBayesianOptimizer(
    task_type="regression",
    model_type="base",
    input_cols=["formula", "temperature", "time"],
    target_cols="property",
    bounds={
        "temperature": [800.0, 1100.0],
        "time": [1.0, 20.0],
    },
    composition_col="formula",
    composition_elements=["Fe", "Co", "Ni"],
    composition_normalization="atomic_fraction",
    composition_representation="ilr",
    composition_coordinate_bounds=[-8.0, 8.0],
    composition_bounds={
        "Fe": [0.1, 0.8],
        "Co": [0.0, 0.6],
        "Ni": [0.0, 0.6],
    },
    composition_steps={
        "Fe": 0.01,
        "Co": 0.01,
        "Ni": 0.01,
    },
    composition_required_components=["Fe"],
    composition_min_components=2,
    composition_max_components=3,
    composition_include_descriptors=False,
)
```

まずこの構成で動作と候補の妥当性を確認し、必要に応じて元素物性記述子、サイト別表現、構造情報を追加してください。
