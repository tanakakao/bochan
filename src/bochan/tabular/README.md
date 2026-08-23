# bochan.tabular

`bochan.tabular` は、tensor ベースの `bochan.api.BayesianOptimizer` に DataFrame / numpy / CSV 入出力、列名解決、カテゴリ変換、組成変換を追加する tabular adapter です。

モデル学習・獲得関数・候補最適化の意味論は `bochan.api` に委譲し、tabular 固有の責務だけをこのパッケージで扱います。

## 設計方針

- public entry point は `bochan.tabular.TabularBayesianOptimizer` の 1 つだけです。
- runtime monkey patch は使用しません。
- compatibility-only optimizer subclass / shim / forwarding module は置きません。
- config は `bochan.api` の canonical field を基準にします。
- tabular の flattened direct fields は adapter boundary で明示的に nested config へ変換します。
- 単一組成と複数サイト組成は `composition_sites` に統一します。

## インストール

```bash
pip install -e ".[tabular]"
```

開発時に API / 可視化 / evolutionary optimizer も使う場合:

```bash
pip install -e ".[dev,api,tabular,visualization,evo]"
```

## DataFrame から学習

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

candidates, acq_value = bo.candidate(
    acq_name="EI",
    q=3,
    num_restarts=5,
    raw_samples=64,
)
```

候補は既定で pandas `DataFrame` として返ります。

## CSV

```python
bo = TabularBayesianOptimizer.from_csv(
    "data.csv",
    task_type="regression",
    model_type="base",
    input_cols=["x1", "x2", "x3"],
    target_cols="y",
)

bo.fit()
```

`pandas.read_csv()` の追加引数は `read_csv_kwargs` で渡します。

## numpy

```python
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
```

文字列カテゴリや欠損値処理を使う場合は DataFrame 入力を推奨します。

## Config object と direct fields

`bochan.api` の config dataclass をそのまま利用できます。

```python
from bochan.api import AcquisitionConfig, FitConfig, ModelConfig, OptimizeConfig

bo = TabularBayesianOptimizer(
    model_config=ModelConfig(
        task_type="regression",
        model_type="base",
    ),
    fit_config=FitConfig(
        lr=0.01,
        num_epochs=300,
    ),
    input_cols=["x1", "x2"],
    target_cols="y",
)

bo.fit(df)

candidates, acq_value = bo.candidate(
    acq_config=AcquisitionConfig(name="EI"),
    opt_config=OptimizeConfig(q=3),
)
```

頻繁に使う設定は direct fields でも指定できます。

```python
bo = TabularBayesianOptimizer(
    task_type="regression",
    model_type="base",
    input_cols=["x1", "x2"],
    target_cols="y",
    lr=0.01,
    num_epochs=300,
)
```

config object と direct field を同時に与えた場合、明示的な direct field がその field を上書きします。

## FitConfig.beta

DeepGP / DeepKernel classifier などの学習で ELBO の `beta` を調整する場合は canonical field `beta` を使用します。

```python
bo = TabularBayesianOptimizer(
    task_type="multiclass",
    model_type="deepgp",
    input_cols=["x1", "x2"],
    target_cols="label",
    beta=0.5,
)

bo.fit(df)
bo.fit(df, beta=0.8)
```

`fit_beta` compatibility alias はありません。

獲得関数 UCB の beta は学習用 `FitConfig.beta` とは別で、`acqf_kwargs` に指定します。

```python
candidates, _ = bo.candidate(
    acq_name="UCB",
    q=3,
    acqf_kwargs={"beta": 2.0},
)
```

## 入力 transform

入力正規化・摂動は `InputTransformConfig` または direct fields で指定できます。

```python
bo = TabularBayesianOptimizer(
    task_type="regression",
    model_type="base",
    input_cols=["x1", "x2"],
    target_cols="y",
    normalize=True,
    perturbation=True,
    n_w=8,
    std=0.1,
)
```

## Candidate optimization

```python
candidates, _ = bo.candidate(
    acq_name="EI",
    q=3,
    optimizer="optimize_acqf",
    num_restarts=10,
    raw_samples=256,
)
```

Evolutionary backend を使う場合:

```python
candidates, _ = bo.candidate(
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

## Objective

flat な `objective_*` fields は tabular adapter が `ObjectiveConfig` へ変換します。

```python
candidates, _ = bo.candidate(
    acq_name="EI",
    q=3,
    objective_mode="multi_output",
    objective_outputs=[0, 1],
    objective_directions=["maximize", "minimize"],
    objective_weights=[0.7, 0.3],
)
```

## 制約

列名ベースの candidate constraint / repair は tabular adapter で tensor index へ解決されます。

```python
candidates, _ = bo.candidate(
    acq_name="EI",
    q=3,
    constraints=[
        (["x1", "x2"], [1.0, 1.0], "<=", 1.0),
    ],
)
```

`CandidateRepairConfig` を直接渡すこともできます。

## カテゴリ列

文字列カテゴリ入力は学習時に encode され、DataFrame 候補では元のカテゴリへ decode されます。

```python
bo = TabularBayesianOptimizer(
    input_cols=["temperature", "furnace"],
    categorical_cols=["furnace"],
    target_cols="property",
)
```

binary / multiclass / ordinal target のカテゴリ metadata も tabular 側で保持し、予測ラベルや文字列 outcome constraint の解決に使用します。

## 組成

組成は `composition_sites` が canonical entry point です。単一組成も 1 site として表します。

```python
bo = TabularBayesianOptimizer(
    input_cols=["formula", "temperature"],
    target_cols="property",
    composition_sites={
        "formula": {
            "column": "formula",
            "elements": ["Fe", "Co", "Ni"],
            "representation": "ilr",
            "bounds": {
                "Fe": [0.2, 0.8],
                "Co": [0.0, 0.4],
                "Ni": [0.0, 0.6],
            },
            "required_components": ["Fe"],
            "min_components": 2,
            "max_components": 3,
        }
    },
)
```

複数サイトも同じ mapping に site を追加します。`composition_col` / `formula_col` / `composition_*` compatibility API はありません。

### CrabNet-GP / CrabNet-DKL

単一組成と連続processを使うGaussian regressionでは、凍結encoderの
`model_type="crabnet_gp"`、またはencoderを共同学習する
`model_type="crabnet_dkl"`を選べます。

```python
bo = TabularBayesianOptimizer(
    task_type="regression",
    model_type="crabnet_gp",
    input_cols=["formula", "temperature", "pressure", "holding_time"],
    target_cols="property",
    composition_sites={
        "formula": {
            "column": "formula",
            "elements": ["Ba", "Sr", "Ti", "O"],
            "representation": "ilr",
            "bounds": {
                "Ba": [0.0, 0.6],
                "Sr": [0.0, 0.6],
                "Ti": [0.1, 0.6],
                "O": [0.2, 0.8],
            },
        }
    },
    bounds={
        "temperature": [1000.0, 1400.0],
        "pressure": [0.5, 2.0],
        "holding_time": [1.0, 10.0],
    },
    model_kwargs={
        "latent_dim": 32,
        "checkpoint": "path/to/crabnet-checkpoint.pt",
    },
)
```

`crabnet_dkl`では、`model_kwargs={"encoder_training": "partial"}`が最終
Transformer層を、`{"encoder_training": "full"}`がencoder全体を学習します。
低レベルAPIの明示的な層数を使う場合は`trainable_encoder_layers`を指定します。

元素番号、組成座標の列、連続process列は`input_cols`と`composition_sites`から
自動解決されます。ILR / ALR / CLR / fraction座標はモデル内部で微分可能な原子分率へ
変換されるため、`composition_dims`や`process_dims`を指定する必要はありません。
現在の初期保証は、1つの組成site、連続process、single-output Gaussian regressionです。
カテゴリprocess、複数site、multi-outputを指定した場合は明示的なエラーになります。

詳細は以下を参照してください。

- `docs/tabular_composition.md`
- `docs/tabular_composition_sites.md`
- `src/bochan/tabular/composition/README.md`

## Prediction

```python
pred = bo.predict(df_new)
```

DataFrame return では回帰の平均・標準偏差、分類 probability、必要に応じて decoded prediction label を返します。

## Observation API

部分観測や observation state は `ObservationTabularMixin` を通じて public optimizer に統合されています。tensor 側の semantics は `bochan.api` に委譲します。

## 内部構造

主要な責務は次のように分離されています。

```text
public_optimizer.py
    ├── ObservationTabularMixin
    ├── MultiSiteCompositionMixin
    ├── TabularApiMixin
    └── optimizer.py              # tabular core adapter

builders.py                       # config boundary mapping
converter.py                      # DataFrame / numpy <-> tensor
prediction.py                     # tabular prediction formatting
multi_output_categories.py        # target category metadata / label resolution
composition/*                     # composition transforms / search space
composition_* components          # bounds / total / element constraints
```

最終的な public class 以外で compatibility のためだけに `TabularBayesianOptimizer` を重ねる設計は採用しません。

## 開発時の確認

```bash
pytest -q tests/test_tabular_architecture.py
pytest -q tests/test_tabular_composition_sites.py
pytest -q tests/test_tabular_multi_output_categories.py
pytest -q tests/test_tabular_prediction_labels.py
ruff check src/bochan/tabular tests/test_tabular_architecture.py
```

変更時は public entry point、MRO、カテゴリ metadata、組成変換、candidate repair の契約を同時に確認してください。
