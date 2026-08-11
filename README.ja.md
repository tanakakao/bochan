# bochan

`bochan` は、複数のサロゲートモデル family にわたる Bayesian optimization、active learning、level-set estimation のための BoTorch 指向の実験的ライブラリです。

このプロジェクトは、Gaussian regression、binary / multiclass classification、ordinal regression、hybrid multi-output models、tabular data workflows、および ask/tell optimization loops の周辺で一貫したインターフェースを提供することに重点を置いています。Acquisition functions、objectives、candidate repair、および optimization backends は、これらの model families 全体で再利用できるように設計されています。

この codebase は現在も活発に開発中です。Backward support はまだ主な優先事項ではありません。API design、tensor shapes、および BoTorch-like behavior の一貫性が優先されます。

---

## このライブラリの用途

`bochan` は、次のような workflow を想定しています。

- continuous、categorical、および mixed variables を用いた Bayesian optimization
- regression、binary classification、multiclass classification、ordinal regression、および non-Gaussian response models のための active learning
- level-set estimation と boundary exploration
- constrained および multi-objective optimization
- input perturbation と risk aggregation を用いた robust optimization
- PCA、REMBO、SAAS、または関連する wrapper を用いた high-dimensional optimization
- GP、DeepGP、Deep Kernel GP、heteroscedastic GP、および robust relevance pursuit variants をめぐる model experimentation
- regression / binary / multiclass / ordinal outputs にわたる hybrid multi-output modeling
- `BochanStudy` による Optuna / Ax 風の optimization loops
- `ask()` / `tell()` を使用する human-in-the-loop experiments
- automatic evaluation を伴う simulation または Python-function optimization
- `bochan.tabular` による pandas / numpy / CSV ベースの optimization
- FastAPI による HTTP / JSON model serving

実装は、可能なところでは BoTorch の概念に近い状態を保つように設計されています。

- model wrappers は `posterior(X)` を公開します。
- latent-response models は、必要な場合に `latent_posterior(X)` を公開します。
- acquisition functions は q-batch tensors 上で動作します。
- objective classes は scalarization、probability / utility conversion、input-perturbation aggregation、および risk aggregation を処理します。
- BoTorch standard acquisition functions は、必要な挙動をすでにカバーしている場合に再利用されます。

---

## インストール

Core package:

```bash
pip install -e .
```

一般的な development setup:

```bash
pip install -e ".[dev,api,tabular,visualization,evo]"
```

FastAPI serving のみ:

```bash
pip install -e ".[api]"
```

Tabular DataFrame / numpy / CSV workflows:

```bash
pip install -e ".[tabular]"
```

Notebook examples:

```bash
pip install -e ".[notebook]"
```

Documentation tooling:

```bash
pip install -e ".[docs]"
```

`pyproject.toml` で定義されているすべての optional extras:

```bash
pip install -e ".[all]"
```

---

## どの API を使うべきですか？

| API | 使用する場面 |
|---|---|
| `bochan.api.BayesianOptimizer` | 直接 tensor-based の model fitting、prediction、および candidate generation を行いたい場合。 |
| `bochan.api.BochanStudy` | Optuna / Ax 風の optimization loop、`ask()` / `tell()`、save / load、early stopping、または generation scheduling が必要な場合。 |
| `bochan.tabular.TabularBayesianOptimizer` | pandas DataFrames、numpy arrays、または CSV files から column names を使って作業したい場合。 |
| `bochan.serving.fastapi` | applications または external systems 向けに HTTP / JSON model serving が必要な場合。 |

低レベルの 4-step API は、現在も内部設計単位です。

```python
bundle = build_model(train_X, train_Y, model_config)
bundle = fit_model(bundle, fit_config)
acqf = build_acquisition(bundle, acq_config, data_context)
candidates, acq_value = optimize_candidates(acqf, bounds, opt_config)
```

`BayesianOptimizer`、`BochanStudy`、`TabularBayesianOptimizer`、および FastAPI は、その設計を取り巻く higher-level wrappers です。

---

## Quick start: tensor-based optimizer

```python
import torch

from bochan.api import AcquisitionConfig, BayesianOptimizer, FitConfig, ModelConfig, OptimizeConfig

train_X = torch.rand(40, 2, dtype=torch.double)
train_Y = torch.sin(train_X[:, :1] * 6.28)
bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

bo = BayesianOptimizer(
    model_config=ModelConfig(task_type="regression", model_type="base"),
    fit_config=FitConfig(maxiter=128),
    bounds=bounds,
)
bo.fit(train_X, train_Y)

candidates, acq_value = bo.candidate(
    acq_config=AcquisitionConfig(name="EI", acqf_kwargs={"best_f": train_Y.max()}),
    opt_config=OptimizeConfig(q=3, num_restarts=10, raw_samples=128, sequential=True),
)
```

### FitConfig.beta

`FitConfig.beta` は `mll_kwargs["beta"]` の convenience alias です。DeepGP や DeepKernel classifiers などの variational models に有用です。

```python
fit_config = FitConfig(
    num_epochs=300,
    lr=0.03,
    beta=0.5,
)
```

`beta` と `mll_kwargs["beta"]` の両方が指定された場合、`mll_kwargs` 内の明示的な値が優先されます。

### Optimizer backend and evo_method

`OptimizeConfig` は backend-family names を使用します。Mixed / non-mixed dispatch は、可能な場合 categorical settings から解決されます。

```python
OptimizeConfig(optimizer="optimize_acqf")
OptimizeConfig(optimizer="evo", evo_method="ga")
OptimizeConfig(optimizer="torch")
OptimizeConfig(optimizer="nsgaii")
OptimizeConfig(optimizer="thompson_sampling")
```

Evolutionary methods は直接選択することもできます。

```python
OptimizeConfig(optimizer="ga")
OptimizeConfig(optimizer="pso")
OptimizeConfig(optimizer="sa")
OptimizeConfig(optimizer="cmaes")
```

これらの直接名は `optimizer="evo"` に正規化され、`evo_method` に保存されます。

---

## Python API: classification and level-set examples

### Single-output multiclass active learning

```python
import torch

from bochan.api import AcquisitionConfig, BayesianOptimizer, FitConfig, ModelConfig, OptimizeConfig

train_X = torch.rand(40, 2, dtype=torch.double)
train_Y = torch.randint(0, 3, (40,), dtype=torch.long)
bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

bo = BayesianOptimizer(
    model_config=ModelConfig(
        task_type="multiclass",
        model_type="base",
        model_kwargs={
            "num_classes": 3,
            "num_inducing_points": 32,
        },
    ),
    fit_config=FitConfig(num_epochs=250, lr=0.03),
    bounds=bounds,
)
bo.fit(train_X, train_Y)

candidates, acq_value = bo.candidate(
    acq_config=AcquisitionConfig(name="entropy"),
    opt_config=OptimizeConfig(q=3, num_restarts=10, raw_samples=128, sequential=True),
)
```

active learning では、これらの contextual aliases は `task_type`、`model_type`、および model が multi-output かどうかに応じて解決されます。

```python
AcquisitionConfig(name="entropy")
AcquisitionConfig(name="BALD")
AcquisitionConfig(name="JointBALD")
AcquisitionConfig(name="GreedyJointBALD")
AcquisitionConfig(name="variance")
AcquisitionConfig(name="margin")
AcquisitionConfig(name="NIPV")
```

### Multiclass level-set estimation

Multiclass level-set estimation は target-class probability に基づきます。

```text
p(target_class | x)
```

例: class 2 probability が 0.5 付近である boundary を探索します。

```python
candidates, acq_value = bo.candidate(
    acq_config=AcquisitionConfig(
        name="straddle",
        acqf_kwargs={
            "target_class": 2,
            "threshold": 0.5,
        },
    ),
    opt_config=OptimizeConfig(q=3, num_restarts=10, raw_samples=128),
)
```

Useful aliases:

```python
AcquisitionConfig(name="straddle", acqf_kwargs={"target_class": 1, "threshold": 0.5})
AcquisitionConfig(name="ICU", acqf_kwargs={"target_class": 1, "threshold": 0.5})
AcquisitionConfig(name="boundaryvariance", acqf_kwargs={"target_class": 1, "threshold": 0.5})
AcquisitionConfig(name="classentropy")
AcquisitionConfig(name="poe", acqf_kwargs={"target_class": 1, "threshold": 0.5})
AcquisitionConfig(name="levelset", acqf_kwargs={"target_class": 1, "threshold": 0.5})
```

### Multiclass Bayesian optimization

multiclass Bayesian optimization では、`target_class` が必須です。objective は次のとおりです。

```text
maximize p(target_class | x)
```

```python
candidates, acq_value = bo.candidate(
    acq_config=AcquisitionConfig(
        name="EI",
        acqf_kwargs={
            "target_class": 2,
            "best_f": 0.70,
            "num_samples": 128,
        },
    ),
    opt_config=OptimizeConfig(q=1, num_restarts=10, raw_samples=128),
)
```

Supported contextual aliases:

```python
AcquisitionConfig(name="EI", acqf_kwargs={"target_class": 2, "best_f": 0.7})
AcquisitionConfig(name="PI", acqf_kwargs={"target_class": 2, "best_f": 0.7})
AcquisitionConfig(name="UCB", acqf_kwargs={"target_class": 2, "beta": 2.0})
AcquisitionConfig(name="PoF", acqf_kwargs={"target_class": 2})
```

`ObjectiveConfig` は現在、multiclass objectives を自動構築しません。multiclass acquisition 自身の `target_class`、`threshold`、`best_f`、および関連する keyword arguments を使用するか、`objective` / `objective_factory` を明示的に渡してください。

---

## Multi-output and heteroscedastic examples

### Mixed-input multiclass model

```python
bo = BayesianOptimizer(
    model_config=ModelConfig(
        task_type="multiclass",
        model_type="base",
        cat_dims=[2],
        model_kwargs={"num_classes": 3},
    ),
    fit_config=FitConfig(num_epochs=250, lr=0.03),
    bounds=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]], dtype=torch.double),
)
```

`cat_dims` が設定され、`input_type` が省略されている場合、Python API は `input_type="mixed"` を自動的に推論します。

### Multi-output multiclass model

Multi-output multiclass は independent submodels から構築され、hybrid multi-output wrapper を通じて wrap されます。fitted bundle が `multi_output=True` を記録するため、acquisition classes は引き続き `qMultiOutputMulticlass...` variants に解決されます。

```python
from bochan.api import MultiOutputConfig

bo = BayesianOptimizer(
    model_config=ModelConfig(
        task_type="multiclass",
        model_type="base",
        model_kwargs={"num_classes": 3},
        multi_output_config=MultiOutputConfig(
            output_task_types=["multiclass", "multiclass"],
            output_names=["defect_type_a", "defect_type_b"],
            use_hybrid=True,
        ),
    ),
    fit_config=FitConfig(num_epochs=200, lr=0.03),
    bounds=bounds,
)
bo.fit(train_X, train_Y_multi)  # train_Y_multi shape: n x m

candidates, acq_value = bo.candidate(
    acq_config=AcquisitionConfig(name="entropy", acqf_kwargs={"output_reduction": "mean"}),
    opt_config=OptimizeConfig(q=3, num_restarts=10, raw_samples=128),
)
```

output aggregation では、multi-output multiclass acquisitions は次のような options を受け取ります。

```python
output_reduction="mean"
output_reduction="sum"
output_reduction="max"
output_reduction="min"
output_reduction="weighted_mean"
```

### Heteroscedastic multiclass model

```python
bo = BayesianOptimizer(
    model_config=ModelConfig(
        task_type="multiclass",
        model_type="hetero",
        model_kwargs={"num_classes": 3},
    ),
    fit_config=FitConfig(num_epochs=250, lr=0.03),
    bounds=bounds,
)
```

`model_type="hetero"` では、contextual acquisition aliases は `qHeteroMulticlass...` または `qHeteroMultiOutputMulticlass...` variants に解決されます。

Noise-aware acquisitions は次のような options を受け取る場合があります。

```python
noise_mode="inverse_linear"
noise_mode="exp"
noise_mode="custom"
noise_mode="none"

noise_combine="multiply"
noise_combine="subtract"
```

---

## Input transform and robust objective examples

`InputTransformConfig` は API settings から Normalize と input perturbation transforms を構築できます。

```python
from bochan.api import InputTransformConfig, ObjectiveConfig

model_config = ModelConfig(
    task_type="regression",
    model_type="base",
    input_transform_config=InputTransformConfig(
        normalize=True,
        perturbation=True,
        n_w=8,
        std=0.1,
        bounds=bounds,
    ),
)

acq_config = AcquisitionConfig(
    name="EI",
    objective_config=ObjectiveConfig(
        mode="scalar",
        output=0,
        n_w=8,
        risk_type="cvar",
        alpha=0.8,
    ),
    acqf_kwargs={"best_f": train_Y.max()},
)
```

acquisition が expanded `q * n_w` samples を `q` に戻して aggregate する必要がある場合、`InputTransformConfig(n_w=...)` と `ObjectiveConfig(n_w=...)` は一致している必要があります。

---

## Outcome constraints

user-facing feasibility constraints には `OutcomeConstraintConfig` を使用してください。

```python
from bochan.api import OutcomeConstraintConfig

acq_config = AcquisitionConfig(
    name="NEHVI",
    outcome_constraint_config=OutcomeConstraintConfig(
        output_indices=[0, 1],
        operators=["ge", "le"],
        thresholds=[0.5, 1.2],
    ),
)
```

model-dependent feasibility constraints の場合は、constraint specs を `outcome_constraint_config.constraints` 経由で渡してください。

```python
acq_config = AcquisitionConfig(
    name="EI",
    outcome_constraint_config=OutcomeConstraintConfig(
        constraints=[
            {
                "kind": "feasibility",
                "output": "defect",
                "operator": "le",
                "threshold": 0.2,
            }
        ],
        eta=1e-3,
        reduce_constraints="prod",
        reduce_q="mean",
    ),
    acqf_kwargs={"best_f": train_Y.max()},
)
```

low-level `constraints` と `outcome_constraint_config` を同時に指定しないでください。

---

## Candidate optimization and repair

`OptimizeConfig` は candidate optimization backend を制御します。

```python
OptimizeConfig(optimizer="optimize_acqf")
OptimizeConfig(optimizer="evo", evo_method="ga")
OptimizeConfig(optimizer="torch")
OptimizeConfig(optimizer="nsgaii")
OptimizeConfig(optimizer="thompson_sampling")
```

Mixed optimization では `fixed_features_list` を使用できます。

```python
opt_config = OptimizeConfig(
    optimizer="optimize_acqf_mixed",
    q=1,
    fixed_features_list=[
        {2: 0.0},
        {2: 1.0},
    ],
)
```

Candidate repair は `CandidateRepairConfig` を通じて設定されます。

```python
from bochan.api import CandidateRepairConfig

opt_config = OptimizeConfig(
    q=3,
    repair_config=CandidateRepairConfig(
        bounds=bounds,
        numeric_indices=[0, 1, 2, 3],
        steps=[0.1, 0.1, 0.1, 0.1],
        comp_idx=[0, 1, 2, 3],
        k=2,
        inequality_constraints=ineq_constraints,
        inequality_sense="le",
        final_priority="constraints",
    ),
)
```

Notes:

- `steps=None` は grid rounding を無効にします。
- `comp_idx=None` または `comp_idx=[]` は、repair function に k-sparse support selection なしで rounding / constraints を実行させます。
- `support_selection="topk"` は最大の score entries を選択します。
- `support_selection="sample"` は `sample_tau` と `sample_eps` を使用して support entries をサンプリングします。
- `final_priority="grid"` は final grid alignment を優先します。
- `final_priority="constraints"` は final constraint satisfaction を優先します。

---

## Optimization loop API: `BochanStudy`

`BochanStudy` は `BayesianOptimizer` を取り巻く Optuna / Ax 風の loop wrapper です。candidate generation と evaluation が別々の operations である場合に有用です。

これは 2 つの主要な patterns をサポートします。

1. Python objective function の automatic evaluation;
2. `ask()` / `tell()` による human-in-the-loop または external evaluation。

### Automatic Python-function optimization

```python
import torch

from bochan.api import BochanStudy

bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

study = BochanStudy(
    bounds=bounds,
    n_initial_random=10,
)

study.optimize(
    objective_func=lambda X: X.sum(dim=-1),
    n_trials=20,
    q=2,
    save_path="study.json",
)

train_X, train_Y = study.completed_data()
```

### Human-in-the-loop / simulation workflow

```python
batch = study.ask(q=3, mark_running=True, return_batch=True)

# Send batch.candidates to an experiment, Web UI, or external simulator.
# Register measured values when they become available.

study.tell(batch, measured_values)
study.save("study.json")
```

trial history を読み込み、runtime configs を再注入することで、後で再開できます。

```python
study = BochanStudy.load(
    "study.json",
    model_config=model_config,
    fit_config=fit_config,
    acq_config=acq_config,
    opt_config=opt_config,
    data_context=data_context,
    bounds=bounds,
)

next_batch = study.ask(q=3, return_batch=True)
```

`BochanStudy` は次もサポートします。

- target-based または no-improvement stopping のための `EarlyStoppingConfig`;
- 実行中に `q`、acquisition、optimization settings、または data context を切り替えるための `GenerationSchedule` と `GenerationStep`;
- failed experiments または simulations のための `mark_failed(...)`;
- run history inspection のための `trials_dataframe()`。

詳細な examples については `src/bochan/api/STUDY_README.md` を参照してください。

---

## Tabular API: pandas / numpy / CSV

`bochan.tabular` は tensor-based API 上の薄い wrapper です。内部では DataFrame / numpy / CSV inputs を tensors に変換しながら、users が column names と DataFrame outputs を使って作業できるようにします。

### DataFrame example

```python
import pandas as pd

from bochan.tabular import TabularBayesianOptimizer


df = pd.read_csv("data.csv")

bo = TabularBayesianOptimizer(
    task_type="regression",
    model_type="base",
    input_cols=["x1", "x2", "x3", "machine"],
    target_cols="y",
    categorical_cols=["machine"],
    bounds={
        "x1": [0.0, 1.0],
        "x2": [0.0, 1.0],
        "x3": [0.0, 1.0],
        "machine": [0, 2],
    },
)

bo.fit(df)

candidates_df, acq_value = bo.candidate(
    acq_name="NIPV",
    q=10,
    optimizer="evo",
    evo_method="ga",
    numeric_indices=["x1", "x2", "x3"],
    steps={"x1": 0.1, "x2": 0.1, "x3": 0.1},
    comp_idx=["x1", "x2", "x3"],
    k=2,
)
```

tabular API は次をサポートします。

- explicit config objects の代わりとなる direct keyword arguments;
- 既存の `ModelConfig`、`FitConfig`、`AcquisitionConfig`、`OptimizeConfig`、および `CandidateRepairConfig` objects;
- `FitConfig.beta` のための `fit_beta` / `beta`;
- evolutionary backend selection のための `evo_method`;
- user-facing outcome constraints のための `outcome_constraint_config`;
- label encoding と candidate decoding を通じた string categorical input columns;
- binary / classification workflows のための string categorical target columns;
- missing-value deletion または imputation;
- column-name based の `bounds`、`steps`、`comp_idx`、`fixed_features`、および `fixed_features_list`。

詳細な examples については `src/bochan/tabular/README.md` を参照してください。

---

## FastAPI serving

FastAPI serving は次の下にあります。

```text
bochan.serving.fastapi
```

Install and start:

```bash
pip install -e ".[api]"
uvicorn bochan.serving.fastapi.app:app --reload
```

FastAPI layer は Python API を反映します。これは `ModelConfig`、`FitConfig`、`AcquisitionConfig`、`OutcomeConstraintConfig`、`OptimizeConfig`、および `DataContext` の JSON versions を受け取ります。

Important endpoints:

| method | path | Purpose |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/models` | model を fit し、memory に保存する |
| `GET` | `/models` | 保存された model ids を一覧表示する |
| `POST` | `/models/{model_id}/predict` | Predict |
| `POST` | `/models/{model_id}/candidates` | candidates を生成する |
| `POST` | `/models/{model_id}/ask` | candidate generation の alias |
| `POST` | `/models/{model_id}/tell` | observations を追加し、任意で refit する |
| `POST` | `/models/{model_id}/refit` | 既存の optimizer を refit する |
| `POST` | `/models/{model_id}/candidates/compare` | 複数の acquisitions を比較する |
| `GET` | `/acquisitions/names` | acquisition aliases を一覧表示する |

Example candidate payload:

```json
{
  "acquisition_config": {"name": "EI", "acqf_kwargs": {"best_f": 1.0}},
  "optimize_config": {"q": 1, "num_restarts": 10, "raw_samples": 256}
}
```

serving layer は、JSON-to-tensor conversion のための `tensor_options` も受け取ります。

```json
{
  "tensor_options": {"dtype": "float64", "device": "cpu"}
}
```

HTTP examples については `src/bochan/serving/fastapi/README.md` を参照してください。これには tensor options、candidate repair、`evo_method`、outcome constraints、multiclass model fitting、および target-class BO が含まれます。

---

## Package layout

```text
src/bochan/
├── api/
├── acquisition/
│   ├── objective/
│   ├── feasible/
│   ├── regression/
│   ├── binary/
│   ├── multiclass/
│   ├── ordinal/
│   └── non_gaussian/
├── fit/
├── models/
│   ├── components/
│   ├── transforms/
│   ├── regression/
│   │   ├── gaussian/
│   │   └── non_gaussian/
│   ├── classification/
│   │   ├── binary/
│   │   └── multiclass/
│   ├── ordinal/
│   └── hybrid/
├── tabular/
├── visualization/
└── serving/
    └── fastapi/
```

### Model layout

Model families は次の broad structure を使用します。

```text
models/
├── regression/
│   ├── gaussian/
│   │   ├── base/
│   │   ├── deep/
│   │   ├── high_dim/
│   │   └── robust/
│   └── non_gaussian/
│       ├── poisson/
│       ├── beta/
│       ├── gamma/
│       └── negative_binomial/
├── classification/
│   ├── binary/
│   └── multiclass/
├── ordinal/
└── hybrid/
```

Major model families:

| Family | Purpose |
|---|---|
| `regression/gaussian` | Standard continuous-output Gaussian regression models. |
| `regression/beta`, `regression/gamma`, `regression/count` | Beta、Gamma、Poisson、および Negative Binomial response models. |
| `classification/binary` | Binary GP classification および related wrappers. |
| `classification/multiclass` | Multiclass GP classification および related wrappers. |
| `ordinal` | Ordered-label / ordinal-regression GP wrappers. |
| `hybrid` | heterogeneous task families のための Multi-output wrapper. |
| `components` | Shared likelihoods、posterior wrappers、transforms、decomposition utilities、および helper functions. |
| `transforms` | Normalize と input perturbation のための Input transform builders. |

### High-level model registry

default API registry は、これらの `task_type` values を公開します。

```python
"regression"
"multi_objective"
"binary"
"multiclass"
"ordinal"
"hybrid"
```

登録済みの `model_type` values は次のとおりです。

```python
"base"
"deepgp"
"deepkernel"
"deepgpdeepkernel"
"saas"
"pca"
"rembo"
"rrp"
"hetero"
```

`multiclass` では、`deepgpdeepkernel` は現在、別個の model type として登録されていません。分布固有の回帰モデルは `models/regression/beta/`、`models/regression/gamma/`、`models/regression/count/` に整理され、canonical な model registry path から解決されます。

`cat_dims` が指定され、`input_type` が省略されている場合、API は `input_type="mixed"` を推論します。それ以外の場合は `input_type="normal"` を使用します。

---

## Core wrapper conventions

### `posterior(X)`

Public prediction API です。これは acquisition functions が期待する prediction object を返す必要があります。

Examples:

- Gaussian regression: continuous response posterior
- Binary classification: probability-scale posterior
- Multiclass classification: class-probability posterior
- Ordinal regression: ordinal class-probability / utility-supported posterior
- Non-Gaussian regression: rate や mean などの response-scale posterior
- Hybrid multi-output: task-aware output collection または objective-space posterior

### `latent_posterior(X)`

model が latent GP を持っているが、public posterior が likelihood または link function を通じて transformed される場合にこれを使用します。

Typical examples:

- binary classification: latent `f` -> sigmoid probability
- multiclass classification: class-wise latent GP -> class probabilities
- ordinal regression: latent `f` -> cutpoint probabilities
- Poisson regression: latent `f` -> positive rate
- Beta regression: latent `f` -> response mean in `(0, 1)`

### `forward(X)`

GPyTorch-trained wrappers では、`forward(X)` は fitting 中に likelihood によって使用される latent GP distribution を返す必要があります。

### `make_mll()`

Wrappers は、`ExactMarginalLogLikelihood` や `VariationalELBO` などの recommended training objective がある場合、`make_mll()` を公開する必要があります。

### `train_inputs` and `train_inputs_raw`

次の区別を使用してください。

```text
train_inputs      = inputs actually used by the internal latent / BoTorch model
train_inputs_raw  = original raw search-space inputs
```

この区別は、input transforms、high-dimensional wrappers、mixed variables、および candidate-update logic にとって重要です。

### `condition_on_observations`

サポートされている場合、この method は raw `X` を受け取り、`Y` を適切に準備し、model-family settings を保持し、新しい wrapper instance を返す必要があります。

non-Gaussian likelihoods に対する Gaussian-style `noise=` のような unsupported options は、無視されるのではなく、明示的な `NotImplementedError` を発生させる必要があります。

---

## Documentation map

| File | Contents |
|---|---|
| `docs/theory/README.md` | GP models、Bayesian optimization、acquisition functions、active learning、level-set estimation、classification / ordinal BO、multi-objective constraints、input perturbation、risk、および tensor shape conventions の theoretical background. |
| `src/bochan/models/README.md` | Model family overview、default model registry、wrapper API conventions、および model implementation checklist. |
| `src/bochan/acquisition/README.md` | Acquisition family overview、objectives、feasibility、active learning、level-set estimation、multiclass acquisitions、および non-Gaussian acquisitions. |
| `src/bochan/acquisition/feasible/README.md` | Feasibility constraints と feasibility wrapper usage. |
| `src/bochan/api/README.md` | Tensor-based Python API usage、config objects、registries、objectives、candidate optimization、および repair. |
| `src/bochan/api/STUDY_README.md` | `BochanStudy` optimization loop、`ask()` / `tell()`、`optimize()`、save / load、early stopping、および generation schedules. |
| `src/bochan/tabular/README.md` | pandas / numpy / CSV wrapper、column-name based settings、categorical encoding、imputation、candidate repair、`fit_beta`、`evo_method`、および constraints. |
| `src/bochan/serving/fastapi/README.md` | HTTP / JSON serving examples、tensor conversion、optimizer settings、candidate repair、constraints、および multiclass workflows. |

---

## Development status

この repository は active development 中です。

Current priorities:

- model wrappers を BoTorch-supported に保つ;
- regression / binary / multiclass / ordinal / non-Gaussian families 全体で naming と arguments を揃える;
- tensor shapes を q-batch safe に保つ;
- optimization-loop APIs を Python functions と human-in-the-loop experiments の両方から使えるようにする;
- DataFrame / CSV wrappers を薄く保ち、tensor API と一貫させる;
- HTTP / JSON payloads を public `bochan.api` config objects と揃える;
- distribution-specific duplication よりも shared implementation を優先する;
- 可能な限り BoTorch standard functionality を再利用する。
