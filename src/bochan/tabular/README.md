# bochan.tabular

`bochan.tabular` は、既存の tensor ベース API である `bochan.api.BayesianOptimizer` の上に被せる、pandas / numpy / CSV 向けの薄いラッパーです。

内部では DataFrame / numpy 配列を `torch.Tensor` に変換し、モデル学習・獲得関数生成・候補点最適化は既存の `bochan.api` に委譲します。候補点は必要に応じて pandas `DataFrame` として返します。

---

## 1. 目的

実験データや CSV をそのまま扱えるようにすることが目的です。

- pandas `DataFrame` を直接 `fit()` できる
- CSV を `from_csv()` で読み込める
- numpy 配列にも対応する
- 列名で説明変数・目的変数・カテゴリ列を指定できる
- 文字列カテゴリ列を label encoding して、候補点では元の文字列に戻せる
- 欠損値を削除または補完できる
- `CandidateRepairConfig` 相当の step 丸め・k-sparse・固定特徴量を列名で指定できる
- `FitConfig.beta`, `OptimizeConfig.evo_method`, `OutcomeConstraintConfig` など、公開 `bochan.api` の設定を直接または config object 経由で使える
- 既存の `ModelConfig`, `FitConfig`, `AcquisitionConfig`, `OptimizeConfig` なども引き続き使える
- `bochan.visualization` の Plotly 可視化を `TabularBayesianOptimizer` のメソッドとして直接呼べる

---

## 2. インストール

通常の DataFrame / CSV 利用では `pandas` が必要です。

```bash
pip install -e ".[tabular]"
```

開発時に FastAPI や可視化もまとめて入れる場合です。

```bash
pip install -e ".[dev,api,tabular,visualization,evo]"
```

全部入りで入れる場合は次の通りです。

```bash
pip install -e ".[all]"
```

`continuous_impute_strategy="iterative"` には scikit-learn が必要です。`tabular` extra または `all` extra に含まれます。

---

## 3. 最小例: DataFrame から学習して候補点を出す

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

## 4. CSV から直接使う

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

## 5. numpy 配列から使う

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

文字列カテゴリや欠損値補完を本格的に使う場合は、numpy より DataFrame の方が扱いやすいです。

---

## 6. 直接引数 API と config object API

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
    evo_method="ga",
)
```

内部では以下の config が自動的に組み立てられます。

- `ModelConfig`
- `FitConfig`
- `AcquisitionConfig`
- `ObjectiveConfig`
- `OptimizeConfig`
- `CandidateRepairConfig`

既存の config object も引き続き利用できます。

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

直接引数と config object を混ぜる場合は、直接引数が config object に上書き適用されます。

```python
from bochan.api import OptimizeConfig


candidates_df, acq_value = bo.candidate(
    opt_config=OptimizeConfig(q=3, optimizer="evo"),
    evo_method="pso",
)
```

---

## 7. FitConfig.beta

`FitConfig.beta` は `mll_kwargs["beta"]` への便利 alias です。DeepGP / DeepKernel classifier などで ELBO の beta を調整したい場合に使います。

初期化時に指定する例です。

```python
bo = TabularBayesianOptimizer(
    task_type="multiclass",
    model_type="deepgp",
    input_cols=["x1", "x2"],
    target_cols="label",
    model_kwargs={"num_classes": 3, "num_inducing_points": 32},
    num_epochs=300,
    lr=0.03,
    fit_beta=0.5,
)

bo.fit(df)
```

`beta` という短い alias も使えます。

```python
bo = TabularBayesianOptimizer(
    task_type="multiclass",
    model_type="deepgp",
    input_cols=["x1", "x2"],
    target_cols="label",
    beta=0.5,
)
```

`fit()` 時にだけ上書きする例です。

```python
bo.fit(df, fit_beta=0.8)
```

`fit_beta` と `beta` を同時に指定すると曖昧なのでエラーになります。

---

## 8. optimizer と evo_method

候補点最適化 backend は `candidate()` の `optimizer` で指定します。

```python
candidates_df, acq_value = bo.candidate(
    acq_name="EI",
    q=3,
    optimizer="optimize_acqf",
    num_restarts=10,
    raw_samples=256,
)
```

Evolutionary backend を使う場合です。

```python
candidates_df, acq_value = bo.candidate(
    acq_name="NIPV",
    q=5,
    optimizer="evo",
    evo_method="ga",
    optimizer_kwargs={
        "population_size": 128,
        "num_generations": 80,
    },
)
```

`evo_method` は次を指定できます。

```text
ga / pso / sa / cmaes
```

`optimizer="ga"`, `optimizer="pso"`, `optimizer="sa"`, `optimizer="cmaes"` のように直接指定しても、内部では `optimizer="evo"` に正規化されます。

```python
candidates_df, acq_value = bo.candidate(
    acq_name="NIPV",
    q=3,
    optimizer="pso",
)
```

---

## 9. acquisition / objective の指定

### 9.1 EI / UCB / NIPV

```python
candidates_df, acq_value = bo.candidate(
    acq_name="EI",
    q=3,
    acqf_kwargs={"best_f": 1.0},
)
```

UCB の `beta` は `acqf_kwargs` に入れます。`FitConfig.beta` とは別物です。

```python
candidates_df, acq_value = bo.candidate(
    acq_name="UCB",
    q=3,
    acqf_kwargs={"beta": 2.0},
)
```

```python
candidates_df, acq_value = bo.candidate(
    acq_name="NIPV",
    q=10,
)
```

### 9.2 ObjectiveConfig を直接引数で指定する

`objective_*` 形式の引数は `ObjectiveConfig` に変換されます。

```python
candidates_df, acq_value = bo.candidate(
    acq_name="EI",
    q=3,
    objective_mode="scalar",
    objective_output=0,
    objective_direction="maximize",
    objective_weight=1.0,
)
```

multi-output の重み付き目的の例です。

```python
candidates_df, acq_value = bo.candidate(
    acq_name="EI",
    q=3,
    objective_mode="multi_output",
    objective_outputs=[0, 1],
    objective_directions=["maximize", "minimize"],
    objective_weights=[0.7, 0.3],
)
```

input perturbation を使う場合は、`objective_n_w`, `objective_risk_type`, `objective_alpha` を指定できます。

```python
candidates_df, acq_value = bo.candidate(
    acq_name="EI",
    q=3,
    objective_n_w=8,
    objective_risk_type="cvar",
    objective_alpha=0.2,
)
```

### 9.3 multiclass active learning

```python
candidates_df, acq_value = bo.candidate(
    acq_name="entropy",
    q=5,
)
```

```python
candidates_df, acq_value = bo.candidate(
    acq_name="BALD",
    q=5,
)
```

### 9.4 multiclass / ordinal level-set

```python
candidates_df, acq_value = bo.candidate(
    acq_name="straddle",
    q=5,
    acqf_kwargs={
        "target_class": 2,
        "threshold": 0.5,
    },
)
```

---

## 10. outcome constraints

`AcquisitionConfig` には、低レベルの `constraints` と、JSON / notebook で扱いやすい `outcome_constraint_config` があります。通常は `outcome_constraint_config` を推奨します。

### 10.1 数値出力に対する制約

```python
from bochan.api import OutcomeConstraintConfig


candidates_df, acq_value = bo.candidate(
    acq_name="NEHVI",
    q=3,
    outcome_constraint_config=OutcomeConstraintConfig(
        output_indices=[0, 1],
        operators=["ge", "le"],
        thresholds=[0.5, 1.2],
    ),
)
```

直接 dict で渡すこともできます。

```python
candidates_df, acq_value = bo.candidate(
    acq_name="NEHVI",
    q=3,
    outcome_constraint_config={
        "output_indices": [0, 1],
        "operators": ["ge", "le"],
        "thresholds": [0.5, 1.2],
    },
)
```

### 10.2 feasibility constraint spec

classification / ordinal の確率制約など、model access が必要な制約は feasibility wrapper 経由で扱います。

```python
candidates_df, acq_value = bo.candidate(
    acq_name="EI",
    q=1,
    outcome_constraint_config={
        "constraints": [
            {
                "kind": "feasibility",
                "output": "defect",
                "operator": "le",
                "threshold": 0.2,
            }
        ],
        "eta": 1e-3,
        "reduce_constraints": "prod",
        "reduce_q": "mean",
        "posterior_mode": "objective",
    },
    acqf_kwargs={"best_f": 1.0},
)
```

`constraints` と `outcome_constraint_config` は同時指定できません。

---

## 11. カテゴリ列の扱い

### 11.1 説明変数側のカテゴリ

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

### 11.2 明示的なカテゴリ mapping

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

## 12. 目的変数がカテゴリの場合

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

## 13. 欠損値の扱い

### 13.1 デフォルト: 欠損行を削除

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

### 13.2 欠損値を補完する

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

### 13.3 連続値を IterativeImputer で補完する

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

### 13.4 目的変数も補完する

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

## 14. bounds の指定

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

候補点生成時だけ bounds を上書きすることもできます。

```python
candidates_df, acq_value = bo.candidate(
    acq_name="EI",
    q=3,
    bounds={
        "x1": [0.2, 0.8],
        "x2": [1.0, 5.0],
        "x3": [-0.5, 0.5],
    },
)
```

---

## 15. 候補点の後処理: step 丸め・k-sparse

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

`steps=None` の場合は丸めません。`comp_idx=[]` または `comp_idx=None` の場合は、k-sparse ではなく丸め・制約補修だけを行います。

---

## 16. 固定特徴量

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

## 17. ask / tell 形式

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

`fit_config` を渡すと、tell 後の再学習設定を変えられます。

```python
from bochan.api import FitConfig


bo.tell(
    new_df,
    refit=True,
    fit_config=FitConfig(num_epochs=150, lr=0.03, beta=0.5),
)
```

---

## 18. 予測

`bochan.tabular` の `predict()` は、ユーザー側に `torch.Tensor` や posterior object を出さないため、既定で pandas `DataFrame` を返します。出力には期待値とばらつきの両方を含めます。

```python
pred_df = bo.predict(df[["x1", "x2", "x3"]])
print(pred_df)
```

単一目的の回帰で `target_cols="y"` の場合、列名は次のようになります。

```text
y_mean
y_variance
```

複数目的の場合は目的変数名ごとに列が作られます。

```text
yield_mean
yield_variance
defect_rate_mean
defect_rate_variance
```

multiclass / ordinal のように class 次元を持つ場合は、class index を含めた列名になります。

```text
label_class_0_mean
label_class_1_mean
label_class_2_mean
label_class_0_variance
label_class_1_variance
label_class_2_variance
```

入力列も同じ DataFrame に含めたい場合は `include_input=True` を指定します。

```python
pred_df = bo.predict(
    df[["x1", "x2", "x3"]],
    include_input=True,
)
```

出力 DataFrame の `attrs` には、予測空間や分散の意味を残します。

```python
pred_df.attrs["task_type"]
pred_df.attrs["prediction_space"]
pred_df.attrs["variance_kind"]
```

binary classification では `prediction_space="probability"` になり、`mean` はクラス1確率、`variance` は通常 `p * (1 - p)` に対応する Bernoulli 観測分散です。

従来の tensor API 相当の戻り値が必要な場合だけ、`return_type` を明示します。この場合は低レベル互換用であり、戻り値は tensor / posterior object になります。

```python
posterior = bo.predict(
    df[["x1", "x2", "x3"]],
    return_type="posterior",
)

mean, variance = bo.predict(
    df[["x1", "x2", "x3"]],
    return_type="mean_variance",
)
```

---

## 19. 可視化

`TabularBayesianOptimizer` から `bochan.visualization` の Plotly 可視化を直接呼べます。学習時の `input_cols` / `target_cols` は自動的に使われるため、通常は列名を再指定する必要はありません。

```python
fig = bo.plot_yy()
fig.show()
```

単一目的でない場合は、対象の目的変数を指定します。

```python
fig = bo.plot_yy(target="yield")
fig.show()
```

### 19.1 1D 予測曲線

1つの説明変数だけを動かし、他の変数は代表値に固定して、予測平均と不確かさを表示します。

```python
fig = bo.plot_1d(
    feature="temperature",
    target="yield",
    value_dict={"pressure": 0.5, "machine": 1},
    n=100,
)
fig.show()
```

### 19.2 2D ヒートマップ

2つの説明変数を動かして、獲得関数または予測値をヒートマップ表示します。

```python
fig = bo.plot_2d(
    "temperature",
    "pressure",
    target="yield",
    show_type="pred",
    value_dict={"machine": 1},
    n=60,
)
fig.show()
```

`plot_heatmap()` と `plot_scatter()` は `plot_2d()` の alias です。

候補点生成後に呼ぶと、直近の `CandidateResult` が自動的に使われます。別の候補点結果を重ねたい場合は `candidate_result` を渡します。

```python
result = bo.candidate(
    acq_name="EI",
    q=3,
    return_result=True,
)

fig = bo.plot_2d(
    "temperature",
    "pressure",
    target="yield",
    show_type="acqf",
    candidate_result=result,
)
```

### 19.3 三角プロット

3成分制約の組成探索では三角プロットを使えます。

```python
fig = bo.plot_tri(
    "a",
    "b",
    "c",
    target="yield",
    sum_value=1.0,
    value_dict={"temperature": 1000.0},
    show_type="pred",
    n=60,
)
fig.show()
```

`plot_ternary()` は `plot_tri()` の alias です。

### 19.4 2目的散布図

2目的の実測値と候補点予測を確認する場合は `plot_pareto()` を使います。

```python
fig = bo.plot_pareto("yield", "cost")
fig.show()
```

`target_cols` が2列以上ある場合は、先頭2列を自動的に使えます。

```python
fig = bo.plot_pareto()
```

### 19.5 multiclass / ordinal

multiclass / ordinal の可視化引数もそのまま渡せます。

```python
fig = bo.plot_2d(
    "x1",
    "x2",
    target="label",
    show_type="pred",
    multiclass_mode="class_confidence",
    n=60,
)
```

ordinal probability 表示に切り替える例です。

```python
fig = bo.plot_2d(
    "x1",
    "x2",
    target="level",
    show_type="pred",
    ordinal_display="probability",
    ordinal_mode="entropy",
    n=60,
)
```

可視化用の DataFrame だけを取り出したい場合は、次を使います。

```python
X_df, y_df = bo.visualization_training_dataframe()
df_cand = bo.visualization_candidates_dataframe()
```

---

## 20. 低レベル変換関数

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

## 21. よく使う属性

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

学習データ tensor は property からも確認できます。

```python
bo.train_X
bo.train_Y
```

---

## 22. 注意点

- `bochan.tabular` は薄いラッパーであり、モデル構築・学習・獲得関数・候補点最適化の中核は `bochan.api` に委譲します。
- DataFrame / CSV 利用には pandas が必要です。
- 可視化メソッドには `plotly` が必要です。
- `continuous_impute_strategy="iterative"` には scikit-learn が必要です。
- 文字列カテゴリの候補点は元の文字列に戻しますが、最適化中は数値コードとして扱います。
- 目的変数の文字列カテゴリは学習用に label encoding されます。候補点 DataFrame には目的変数は含まれません。
- numpy 入力で文字列カテゴリや欠損値補完を本格的に使う場合は、DataFrame に変換してから使う方が扱いやすいです。
- `FitConfig.beta` と UCB の `acqf_kwargs["beta"]` は別物です。前者は学習時の MLL / ELBO 側、後者は acquisition 側の探索パラメータです。
- `predict()` は既定で DataFrame を返します。`return_type="posterior"`, `return_type="mean"`, `return_type="variance"`, `return_type="mean_variance"` は低レベル互換用です。
