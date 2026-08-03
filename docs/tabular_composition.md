# Tabular composition preprocessing

`bochan.tabular.composition` は、組成式を通常の tabular モデルへ渡すための前処理層です。
専用 GP モデルは定義せず、組成式の解析、数値特徴量化、候補の逆変換、組成制約の repair を分離します。

## 対応範囲

- 化学式の解析
  - 小数係数
  - `()` / `[]` の入れ子
  - `CuSO4·5H2O` のような水和物
- 原子分率、重量分率への変換
- 組成式への canonical な逆変換
- 元素物性の加重統計量
  - 原子番号
  - 原子量
  - 任意のユーザー定義元素物性
  - 平均、標準偏差、最小、最大、範囲
  - 元素数、混合エントロピー
- simplex 表現
  - fraction
  - CLR
  - ALR
  - ILR
- 組成探索制約
  - 合計値
  - 元素ごとの上下限
  - 刻み
  - 必須元素
  - 最小・最大有効元素数

## 通常の tabular モデルへ渡す

```python
import pandas as pd

from bochan.tabular import (
    CompositionColumnConfig,
    CompositionTabularPreprocessor,
    TabularBayesianOptimizer,
)


df = pd.DataFrame(
    {
        "formula": [
            "Fe0.50Ni0.30Co0.20",
            "Fe0.40Ni0.40Co0.20",
            "Fe0.45Ni0.25Co0.30",
        ],
        "temperature": [900.0, 950.0, 925.0],
        "property": [10.0, 12.0, 11.5],
    }
)

preprocessor = CompositionTabularPreprocessor(
    CompositionColumnConfig(
        column="formula",
        elements=["Fe", "Co", "Ni"],
        representation="ilr",
    )
)
model_df = preprocessor.fit_transform(df)

input_cols = [
    "formula__ilr__1",
    "formula__ilr__2",
    "temperature",
]

bo = TabularBayesianOptimizer(
    task_type="regression",
    model_type="base",
    input_cols=input_cols,
    target_cols="property",
    bounds={
        "formula__ilr__1": [-5.0, 5.0],
        "formula__ilr__2": [-5.0, 5.0],
        "temperature": [850.0, 1000.0],
    },
)
bo.fit(model_df)

candidate_df, acq_value = bo.candidate(acq_name="EI", q=3)
restored = preprocessor.inverse_candidates(candidate_df)
```

ILR / ALR 空間で候補を生成すると、逆変換後の組成は自動的に非負かつ合計 1 になります。

## 組成制約を適用する

```python
from bochan.tabular import CompositionSearchSpace


search_space = CompositionSearchSpace(
    components=["Fe", "Co", "Ni", "Cr"],
    total=1.0,
    bounds={
        "Fe": (0.20, 0.80),
        "Co": (0.00, 0.40),
        "Ni": (0.00, 0.60),
        "Cr": (0.00, 0.20),
    },
    steps={
        "Fe": 0.01,
        "Co": 0.01,
        "Ni": 0.01,
        "Cr": 0.01,
    },
    required_components=["Fe"],
    min_active_components=2,
    max_active_components=3,
)

preprocessor = CompositionTabularPreprocessor(
    CompositionColumnConfig(
        column="formula",
        elements=["Fe", "Co", "Ni", "Cr"],
        representation="ilr",
    ),
    search_space=search_space,
)
```

`inverse_candidates(..., repair=True)` は、逆変換した組成に上下限、刻み、有効元素数、必須元素を適用してから組成式を生成します。

## 組成記述子

```python
from bochan.tabular import CompositionColumnConfig, CompositionTabularPreprocessor


config = CompositionColumnConfig(
    column="formula",
    representation="ilr",
    include_descriptors=True,
    descriptor_properties=["atomic_number", "atomic_weight", "electronegativity"],
    element_properties={
        "electronegativity": {
            "Fe": 1.83,
            "Co": 1.88,
            "Ni": 1.91,
        }
    },
)
```

記述子は組成から一意に決まる派生特徴量です。学習・予測・active learning にはそのまま使えますが、
通常の候補最適化で記述子列を独立した探索変数にはしないでください。候補最適化の入力列には fraction / CLR / ALR / ILR とプロセス条件だけを指定し、必要な場合は候補組成から記述子を再計算します。

## 表記上の注意

原子分率へ正規化すると全体のスケール情報は失われます。そのため、逆変換は元の文字列を完全復元する処理ではなく、
正規化された canonical formula を生成する処理です。元の入力表記が必要な場合は、元の `formula` 列を別途保存してください。

サイト占有は化学式だけから一意に判定できないため、A サイト / B サイトなどの情報は別の列または明示的なサイト設定として管理してください。
