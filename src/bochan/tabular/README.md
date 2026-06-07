# bochan.tabular

`bochan.tabular` は、既存の tensor ベース API である `bochan.api.BayesianOptimizer` の上に被せる、pandas / numpy / CSV 向けの薄いラッパーです。

内部では DataFrame / numpy 配列を `torch.Tensor` に変換し、モデル学習・獲得関数生成・候補点最適化は既存の `bochan.api` に委譲します。候補点は必要に応じて pandas `DataFrame` として返します。

## 目的

実験データや CSV をそのまま扱えるようにすることが目的です。

- pandas `DataFrame` を直接 `fit()` できる
- CSV を `from_csv()` で読み込める
- numpy 配列にも対応する
- 列名で説明変数・目的変数・カテゴリ列を指定できる
- `CandidateRepairConfig` 相当の設定も列名で指定できる
- 文字列カテゴリ列を label encoding して、候補点では元の文字列に戻せる
- 欠損値を削除または補完できる
- 既存の `ModelConfig`, `AcquisitionConfig`, `OptimizeConfig` なども引き続き使える

---

## インストール

通常の DataFrame / CSV 利用では `pandas` が必要です。

```bash
pip install -e .
```

IterativeImputer による多重代入法風の補完も使う場合は、`impute` extra を入れます。

```bash
pip install -e ".[impute]"
```

全部入りで入れる場合は次の通りです。

```bash
pip install -e ".[all]"
```

---

## 最小例: DataFrame から学習して候補点を出す

```python
import pandas as pd

from bochan.tabular import TabularBayesianOptimizer


df = pd.read_csv("data.csv")

bo = TabularBayesianOptimizer(
    task_type="regression",
    model_type="base",
    input_cols=["x1", "x2", "x3"],
    target_cols="y",
    bounds={
        "x1": [0.0, 1.0],
        "x2": [0.0, 1.0],
        "x3": [0.0, 1.0],
    },
)

bo.fit(df)

candidates_df, acq_value = bo.candidate(
    acq_name="EI",
    q=3,
    num_restarts=5,
    raw_samples=64,
)

print(candidates_df)
print(acq_value)
```

`candidates_df` は pandas `DataFrame` です。実験候補としてそのまま CSV 出力できます。

```python
candidates_df.to_csv("next_candidates.csv", index=False)
```

---

## CSV から直接使う

```python
from bochan.tabular import TabularBayesianOptimizer


bo = TabularBayesianOptimizer.from_csv(
    "data.csv",
    task_type="regression",
    model_type="base",
    input_cols=["x1", "x2", "x3"],
    target_cols="y",
    bounds={
        "x1": [0.0, 1.0],
        "x2": [0.0, 1.0],
        "x3": [0.0, 1.0],
    },
)

bo.fit()

candidates_df, acq_value = bo.candidate(
    acq_name="NIPV",
    q=5,
)
```

`read_csv()` に渡す引数は `read_csv_kwargs` で指定できます。

```python
bo = TabularBayesianOptimizer.from_csv(
    "data.csv",
    task_type="regression",
    model_type="base",
    input_cols=["x1", "x2"],
    target_cols="y",
    read_csv_kwargs={"encoding": "utf-8-sig"},
)
```

---

## numpy 配列から使う

numpy 配列の場合は列名がないため、必要に応じて `feature_names` を渡します。

```python
from bochan.tabular import TabularBayesianOptimizer


bo = TabularBayesianOptimizer(
    task_type="regression",
    model_type="base",
)

bo.fit(
    X_np,
    y_np,
    feature_names=["x1", "x2", "x3"],
    input_cols=["x1", "x2", "x3"],
    target_names=["y"],
)

candidates_df, acq_value = bo.candidate(
    acq_name="EI",
    q=3,
)
```

列名が不要な場合は、列番号ベースでも使えます。

```python
bo.fit(
    X_np,
    y_np,
    input_cols=[0, 1, 2],
)
```

---

## 直接引数 API

`bochan.tabular` では、既存の config dataclass を明示的に作らなくても使えます。

```python
bo = TabularBayesianOptimizer(
    task_type="regression",
    model_type="base",
    input_cols=["x1", "x2", "x3"],
    target_cols="y",
    lr=0.01,
    num_epochs=300,
)

bo.fit(df)

candidates_df, acq_value = bo.candidate(
    acq_name="NIPV",
    q=10,
    optimizer="evo",
)
```

内部では以下の config が自動的に組み立てられます。

- `ModelConfig`
- `FitConfig`
- `AcquisitionConfig`
- `ObjectiveConfig`
- `OptimizeConfig`
- `CandidateRepairConfig`

既存の config オブジェクトも引き続き利用できます。

```python
from bochan.api import AcquisitionConfig, ModelConfig, OptimizeConfig
from bochan.tabular import TabularBayesianOptimizer


bo = TabularBayesianOptimizer(
    model_config=ModelConfig(task_type="regression", model_type="base"),
    input_cols=["x1", "x2", "x3"],
    target_cols="y",
)

bo.fit(df)

candidates_df, acq_value = bo.candidate(
    AcquisitionConfig(name="NIPV"),
    OptimizeConfig(q=10),
)
```

---

## カテゴリ列の扱い

### 説明変数側のカテゴリ

説明変数にカテゴリ列がある場合は `categorical_cols` に指定します。

```python
bo = TabularBayesianOptimizer(
    task_type="regression",
    model_type="base",
    input_cols=["temperature", "pressure", "machine"],
    target_cols="yield",
    categorical_cols=["machine"],
)

bo.fit(df)
```

`categorical_cols` は自動的に `ModelConfig.cat_dims` に変換されます。文字列カテゴリ列は label encoding され、候補点 DataFrame では元の文字列に戻されます。

```python
candidates_df, acq_value = bo.candidate(acq_name="EI", q=3)
print(candidates_df["machine"])
```

数値カテゴリは、すでにエンコード済みとして扱います。

### 明示的なカテゴリ mapping

カテゴリの符号化を固定したい場合は `category_maps` を指定します。

```python
bo = TabularBayesianOptimizer(
    task_type="regression",
    model_type="base",
    input_cols=["x1", "machine"],
    target_cols="y",
    categorical_cols=["machine"],
    category_maps={
        "machine": {"A": 0, "B": 1, "C": 2},
    },
)
```

学習後、mapping は以下で確認できます。

```python
bo.dataset.category_maps
bo.dataset.inverse_category_maps
```

---

## 目的変数がカテゴリの場合

目的変数が文字列カテゴリの場合も label encoding されます。

```python
bo = TabularBayesianOptimizer(
    task_type="binary",
    model_type="base",
    input_cols=["x1", "x2", "machine"],
    target_cols="judgement",
    categorical_cols=["machine"],
    target_categorical_cols=["judgement"],
)

bo.fit(df)
```

`target_categorical_cols` を省略した場合でも、非数値の目的変数は自動検出されます。

明示的に mapping を固定したい場合は `target_category_maps` を使います。

```python
bo = TabularBayesianOptimizer(
    task_type="binary",
    model_type="base",
    input_cols=["x1", "machine"],
    target_cols="judgement",
    categorical_cols=["machine"],
    category_maps={"machine": {"A": 0, "B": 1}},
    target_category_maps={"judgement": {"NG": 0, "OK": 1}},
)

bo.fit(df)
```

目的変数側の mapping は次で確認できます。

```python
bo.dataset.target_category_maps
bo.dataset.inverse_target_category_maps
```

候補点 DataFrame は説明変数のみを含むため、目的変数側のカテゴリを戻す処理は候補点出力には使われません。

---

## 欠損値の扱い

### デフォルト: 欠損行を削除

デフォルトでは `dropna=True` により、説明変数または目的変数に欠損がある行を削除します。

```python
bo = TabularBayesianOptimizer(
    task_type="regression",
    model_type="base",
    input_cols=["x1", "x2"],
    target_cols="y",
)

bo.fit(df)
```

### 欠損値を補完する

`missing_strategy="impute"` を指定すると、説明変数の欠損を補完します。

```python
bo = TabularBayesianOptimizer(
    task_type="regression",
    model_type="base",
    input_cols=["x1", "x2", "machine"],
    target_cols="y",
    categorical_cols=["machine"],
    missing_strategy="impute",
    continuous_impute_strategy="mean",
    categorical_impute_strategy="mode",
)

bo.fit(df)
```

補完ルールは次の通りです。

| 列の種類 | 補完方法 |
| --- | --- |
| 連続値 | 平均値補完、または IterativeImputer |
| カテゴリ値 | 最頻値補完 |
| 目的変数 | デフォルトでは欠損行を削除 |

補完値は以下で確認できます。

```python
bo.dataset.impute_values
bo.dataset.target_impute_values
```

### 連続値を IterativeImputer で補完する

`continuous_impute_strategy="iterative"` を指定すると、scikit-learn の `IterativeImputer` を使います。

```python
bo = TabularBayesianOptimizer(
    task_type="regression",
    model_type="base",
    input_cols=["x1", "x2", "machine"],
    target_cols="y",
    categorical_cols=["machine"],
    missing_strategy="impute",
    continuous_impute_strategy="iterative",
    impute_random_state=0,
    impute_max_iter=10,
    multiple_impute_sample_posterior=True,
)

bo.fit(df)
```

`continuous_impute_strategy="iterative"` には scikit-learn が必要です。

```bash
pip install -e ".[impute]"
```

### 目的変数も補完する

目的変数の欠損は安全側で削除されます。目的変数も補完したい場合は `impute_targets=True` を指定します。

```python
bo = TabularBayesianOptimizer(
    task_type="regression",
    model_type="base",
    input_cols=["x1", "x2"],
    target_cols="y",
    missing_strategy="impute",
    impute_targets=True,
)

bo.fit(df)
```

---

## bounds の指定

`bounds` は列名の dict で指定できます。

```python
bo = TabularBayesianOptimizer(
    task_type="regression",
    model_type="base",
    input_cols=["x1", "x2", "x3"],
    target_cols="y",
    bounds={
        "x1": [0.0, 1.0],
        "x2": [0.0, 10.0],
        "x3": [-1.0, 1.0],
    },
)
```

内部では BoTorch 形式の `2 x d` tensor に変換されます。

---

## 候補点の後処理: step 丸め・k-sparse

`candidate()` の直接引数として `CandidateRepairConfig` 相当の設定を渡せます。

```python
candidates_df, acq_value = bo.candidate(
    acq_name="NIPV",
    q=10,
    numeric_indices=["x1", "x2", "x3"],
    steps={"x1": 0.1, "x2": 0.1, "x3": 0.1},
    comp_idx=["x1", "x2", "x3"],
    k=2,
    final_priority="constraints",
)
```

この例では、`x1`, `x2`, `x3` のうち最大 2 成分だけを active にし、さらに `0.1` 刻みに丸めます。

`repair_bounds` を指定した場合は、`repair_bounds[0] + n * step` に丸めます。`repair_bounds` を指定しない場合は、0 基準の `n * step` に丸めます。

```python
candidates_df, acq_value = bo.candidate(
    acq_name="NIPV",
    q=10,
    repair_bounds={
        "x1": [0.05, 1.0],
        "x2": [0.20, 1.0],
        "x3": [0.00, 1.0],
    },
    numeric_indices=["x1", "x2", "x3"],
    steps={"x1": 0.1, "x2": 0.1, "x3": 0.1},
    comp_idx=["x1", "x2", "x3"],
    k=2,
)
```

---

## 固定特徴量

`fixed_features` も列名で指定できます。

```python
candidates_df, acq_value = bo.candidate(
    acq_name="EI",
    q=3,
    fixed_features={"machine": 1},
)
```

複数の固定条件を試す場合は `fixed_features_list` を使います。

```python
candidates_df, acq_value = bo.candidate(
    acq_name="EI",
    q=3,
    fixed_features_list=[
        {"machine": 0},
        {"machine": 1},
        {"machine": 2},
    ],
)
```

カテゴリ列が文字列カテゴリの場合は、現時点では内部の数値コードを指定してください。文字列固定値を直接指定したい場合は、`bo.dataset.category_maps` を確認して対応するコードを指定します。

```python
bo.dataset.category_maps
```

---

## ask / tell 形式

候補点を出して、実験結果を追加し、再学習する ask / tell 形式も使えます。

```python
candidates_df, acq_value = bo.ask(
    acq_name="EI",
    q=3,
)

# 実験後の結果を追加
new_df = candidates_df.copy()
new_df["y"] = measured_y

bo.tell(new_df, refit=True)
```

---

## 予測

DataFrame を直接渡して予測できます。

```python
posterior = bo.predict(df[["x1", "x2", "x3"]])
mean = posterior.mean
variance = posterior.variance
```

既存の tensor API と同様に `return_type` も指定できます。

```python
mean = bo.predict(
    df[["x1", "x2", "x3"]],
    return_type="mean",
)
```

---

## 低レベル変換関数

`bochan.tabular` には、変換だけを行う関数もあります。

```python
from bochan.tabular import dataframe_to_tensors, numpy_to_tensors, tensor_to_dataframe
```

DataFrame を tensor に変換する例です。

```python
from bochan.tabular import TabularDataConfig, dataframe_to_tensors


dataset = dataframe_to_tensors(
    df,
    TabularDataConfig(
        input_cols=["x1", "x2"],
        target_cols="y",
    ),
)

train_X = dataset.X
train_Y = dataset.Y
```

`TabularDataConfig` は低レベル変換用・互換用として残しています。通常の利用では `TabularBayesianOptimizer` の直接引数 API を推奨します。

---

## よく使う属性

学習後は以下を確認できます。

```python
bo.dataset.X
bo.dataset.Y
bo.dataset.feature_names
bo.dataset.target_names
bo.dataset.cat_dims
bo.dataset.bounds

bo.dataset.category_maps
bo.dataset.inverse_category_maps
bo.dataset.target_category_maps
bo.dataset.inverse_target_category_maps

bo.dataset.impute_values
bo.dataset.target_impute_values
```

既存の tensor ベース optimizer には次からアクセスできます。

```python
bo.bo
```

---

## 注意点

- `bochan.tabular` は薄いラッパーであり、モデル構築・学習・獲得関数・候補点最適化の中核は `bochan.api` に委譲します。
- DataFrame / CSV 利用には pandas が必要です。
- `continuous_impute_strategy="iterative"` には scikit-learn が必要です。
- 文字列カテゴリの候補点は元の文字列に戻しますが、最適化中は数値コードとして扱います。
- 目的変数の文字列カテゴリは学習用に label encoding されます。候補点 DataFrame には目的変数は含まれません。
- numpy 入力で文字列カテゴリや欠損値補完を本格的に使う場合は、DataFrame に変換してから使う方が扱いやすいです。
