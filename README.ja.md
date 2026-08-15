# bochan

`bochan` は、複数のサロゲートモデル family にわたる Bayesian optimization、active learning、level-set estimation のための BoTorch 指向の実験的ライブラリです。

このプロジェクトは、Gaussian / non-Gaussian regression、binary / multiclass classification、ordinal regression、multi-task / multi-output models、tabular workflows、および ask/tell optimization loops の周辺で一貫したインターフェースを提供することに重点を置いています。Acquisition functions、objectives、candidate repair、および optimization backends は、これらの model families 全体で再利用できるように設計されています。

この codebase は現在も活発に開発中です。Backward support はまだ主な優先事項ではなく、API design、tensor shapes、および BoTorch-like behavior の一貫性を優先します。

---

## このライブラリの用途

`bochan` は、次のような workflow を想定しています。

- continuous、categorical、mixed、および composition variables を用いた Bayesian optimization
- regression、binary classification、multiclass classification、ordinal regression、および non-Gaussian response models のための active learning
- level-set estimation と boundary exploration
- constrained および multi-objective optimization
- input perturbation と risk aggregation を用いた robust optimization
- PCA、REMBO、SAAS、VAE などを用いた high-dimensional optimization
- multi-task、multi-fidelity、および independent multi-output modeling
- 対応する task での LightGBM、NGBoost、Random Forest、PFN、TabPFN などの external / foundation estimators
- `BochanStudy` による Optuna / Ax 風の optimization loops
- `ask()` / `tell()` を使用する human-in-the-loop experiments
- automatic evaluation を伴う simulation または Python-function optimization
- `bochan.tabular` による pandas / numpy / CSV ベースの optimization
- FastAPI による HTTP / JSON model serving
- `bochan.serving.webapp` による React Web workbench の backend

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

Web workbench dependencies:

```bash
pip install -e ".[web]"
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
| `bochan.api.BochanStudy` | Optuna / Ax 風の optimization loop、`ask()` / `tell()`、save / load、early stopping、generation scheduling が必要な場合。 |
| `bochan.tabular.TabularBayesianOptimizer` | pandas DataFrames、numpy arrays、CSV files を column names で扱いたい場合。 |
| `bochan.serving.fastapi` | core Python API を HTTP / JSON 経由で利用したい場合。 |
| `bochan.serving.webapp` | interactive Web workbench 用 backend を利用したい場合。 |

低レベルの 4-step API は、現在も内部設計単位です。

```python
bundle = build_model(train_X, train_Y, model_config)
bundle = fit_model(bundle, fit_config)
acqf = build_acquisition(bundle, acq_config, data_context)
candidates, acq_value = optimize_candidates(acqf, bounds, opt_config)
```

`BayesianOptimizer`、`BochanStudy`、`TabularBayesianOptimizer`、および serving adapters は、その設計を取り巻く higher-level wrappers です。

---

## Quick start: tensor-based optimizer

```python
import torch

from bochan.api import (
    AcquisitionConfig,
    BayesianOptimizer,
    FitConfig,
    ModelConfig,
    OptimizeConfig,
)

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
    acq_config=AcquisitionConfig(
        name="EI",
        acqf_kwargs={"best_f": train_Y.max()},
    ),
    opt_config=OptimizeConfig(
        q=3,
        num_restarts=10,
        raw_samples=128,
        sequential=True,
    ),
)
```

### `FitConfig.beta`

`FitConfig.beta` は `mll_kwargs["beta"]` の convenience alias です。DeepGP や DeepKernel classifiers などの variational models に利用できます。

```python
fit_config = FitConfig(
    num_epochs=300,
    lr=0.03,
    beta=0.5,
)
```

`beta` と `mll_kwargs["beta"]` の両方を指定した場合、`mll_kwargs` 内の明示値が優先されます。

### Optimizer backend と `evo_method`

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

## Classification、active learning、level-set estimation

### Multiclass active learning

```python
train_X = torch.rand(40, 2, dtype=torch.double)
train_Y = torch.randint(0, 3, (40,), dtype=torch.long)

bo = BayesianOptimizer(
    model_config=ModelConfig(
        task_type="multiclass",
        model_type="base",
        model_kwargs={"num_classes": 3, "num_inducing_points": 32},
    ),
    fit_config=FitConfig(num_epochs=250, lr=0.03),
    bounds=bounds,
)
bo.fit(train_X, train_Y)

candidates, acq_value = bo.candidate(
    acq_config=AcquisitionConfig(name="entropy"),
    opt_config=OptimizeConfig(q=3, num_restarts=10, raw_samples=128),
)
```

Contextual active-learning aliases には、対応する model family で `entropy`、`BALD`、`JointBALD`、`GreedyJointBALD`、`variance`、`margin`、`NIPV` などがあります。

### Multiclass level-set estimation

Multiclass level-set estimation は target-class probability を基準にします。例えば class 2 probability が 0.5 付近の boundary を探索する場合:

```python
candidates, acq_value = bo.candidate(
    acq_config=AcquisitionConfig(
        name="straddle",
        acqf_kwargs={"target_class": 2, "threshold": 0.5},
    ),
    opt_config=OptimizeConfig(q=3, num_restarts=10, raw_samples=128),
)
```

### Multiclass Bayesian optimization

選択した class probability に対して acquisition を適用する multiclass Bayesian optimization では `target_class` が必要です。

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

`ObjectiveConfig` は現在 multiclass objective を自動構築しません。multiclass acquisition 自身の `target_class`、`threshold`、`best_f` などを利用するか、`objective` / `objective_factory` を明示してください。

---

## Input transforms、risk、constraints、repair

`InputTransformConfig` では normalization と input perturbation transforms を構築できます。Acquisition が展開された `q * n_w` samples を `q` に集約する場合、input transform と objective の `n_w` を一致させます。

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

User-facing feasibility constraints には `OutcomeConstraintConfig` を使用します。候補の rounding、k-sparse support、最終 constraint repair は `CandidateRepairConfig` / `OptimizeConfig` で設定します。

詳細な config contract と examples は `src/bochan/api/README.md` を参照してください。

---

## Optimization loop API: `BochanStudy`

`BochanStudy` は `BayesianOptimizer` を包む Optuna / Ax 風の loop wrapper です。Python function の自動評価と、`ask()` / `tell()` による human-in-the-loop / external evaluation の両方に対応します。

```python
from bochan.api import BochanStudy

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
```

外部実験では次のように利用できます。

```python
batch = study.ask(q=3, mark_running=True, return_batch=True)
# batch.candidates で実験を実行する。
study.tell(batch, measured_values)
study.save("study.json")
```

Save / load、early stopping、generation schedules、failed trials、trial history については `src/bochan/api/STUDY_README.md` を参照してください。

---

## Tabular API: pandas / numpy / CSV

`bochan.tabular` は tensor-oriented な `bochan.api` の DataFrame / numpy boundary です。

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
```

tabular API は次をサポートします。

- direct keyword arguments または canonical config objects;
- `FitConfig.beta` に対応する direct field `beta`（`fit_beta` compatibility alias はありません）;
- evolutionary backend selection のための `evo_method`;
- user-facing outcome constraints;
- string categorical input / target columns;
- missing-value deletion または imputation;
- column-name based bounds、steps、composition indices、fixed features;
- `bochan.composition` と `bochan.tabular.composition` による composition-domain integration。

Canonical な tabular contract は `src/bochan/tabular/README.md` と `src/bochan/tabular/ARCHITECTURE.md` を参照してください。

---

## FastAPI serving

FastAPI serving は `bochan.serving.fastapi` にあります。

```bash
pip install -e ".[api]"
uvicorn bochan.serving.fastapi.app:app --reload
```

既定の API prefix は `/api/v1` です。

| method | path | 内容 |
|---|---|---|
| `GET` | `/api/v1/health` | Health check |
| `POST` | `/api/v1/models` | model を fit して memory に保存 |
| `GET` | `/api/v1/models` | 保存された model ids を一覧表示 |
| `POST` | `/api/v1/models/{model_id}/predict` | Predict |
| `POST` | `/api/v1/models/{model_id}/candidates` | candidates を生成 |
| `POST` | `/api/v1/models/{model_id}/ask` | candidates を要求 |
| `POST` | `/api/v1/models/{model_id}/tell` | observations を追加し任意で refit |
| `POST` | `/api/v1/models/{model_id}/refit` | 既存 optimizer を refit |
| `POST` | `/api/v1/models/{model_id}/candidates/compare` | acquisitions を比較 |
| `GET` | `/api/v1/acquisitions/names` | acquisition aliases を一覧表示 |

完全な endpoint / payload reference は `src/bochan/serving/fastapi/README.md` を参照してください。

Web backend は `bochan.serving.webapp` に分離されています。Web app は core FastAPI routes を composition でき、`bochan.serving.fastapi` から `bochan.serving.webapp` への逆依存を作らずに Web-workbench 固有 routes を追加します。

---

## Package layout

現在の top-level ownership は次のとおりです。

```text
src/bochan/
├── acquisition/       # acquisition functions / objectives
├── api/               # high-level tensor API
├── composition/       # pandas-independent composition domain logic
├── constraints/       # reusable constraint utilities
├── fit/               # fitting helpers
├── inspection/        # feature importance / fitted-model diagnostics
├── llm/               # LLM planning / explanation support
├── models/            # surrogate models
├── optim/             # acquisition optimization backends
├── serving/
│   ├── fastapi/       # core HTTP / JSON adapter
│   ├── webapp/        # Web-workbench backend adapter
│   └── workbench/     # shared workbench application state / services
├── tabular/           # DataFrame / numpy adapter
└── visualization/     # visualization utilities
```

`model_artifact.py`、`tabpfn_assets.py`、`tabpfn_preload.py` などの cross-cutting root modules は、複数 surface で共有する処理や deployment-time tooling のため、意図的に Web adapter の外にあります。

### Model layout

Model code は model family と cross-cutting strategy packages の2軸で整理されています。

```text
models/
├── regression/
│   ├── gaussian/
│   ├── beta/
│   ├── gamma/
│   ├── count/
│   ├── external/
│   ├── foundation/
│   └── neural/
├── classification/
│   ├── binary/
│   ├── multiclass/
│   └── common/
├── ordinal/
├── hybrid/
├── multitask/
├── multioutput/
├── multifidelity/
├── external/
├── components/
└── transforms/
```

`multitask` は相関した task/output mechanics、`multioutput` は独立に fitted された outputs を束ねる wrappers、`multifidelity` は shared fidelity-axis abstractions を担当します。Likelihood-specific な具体モデルは owning task/model family 側に残します。

Ownership rule は `src/bochan/models/ARCHITECTURE.md`、model convention は `src/bochan/models/README.md` を参照してください。

### High-level model registry

正確な `model_type` key は task-dependent であり、registry の拡張に伴って変化します。Source of truth は `bochan.api.registry.model` です。Root README の単一 flat list から availability を判断しないでください。

代表的な group には次があります。

- 対応する task での Gaussian GP strategies: `base`、`kronecker`、`multitask`、`multifidelity`、`deepgp`、`deepkernel`、`deepgpdeepkernel`、`saas`、`pca`、`rembo`、`vae`、`rrp`、`hetero`;
- external / neural / foundation estimators: `lightgbm`、`ngboost`、`random_forest`、`deep_ensemble`、`pfn`、`tabpfn`;
- distribution-specific regression: `beta_`、`gamma_`、`poisson_`、`negative_binomial_` prefixes。

`cat_dims` を指定し `input_type` を省略した場合、requested model family が対応していれば mixed input handling を推論できます。

---

## Core wrapper conventions

Model family によって uncertainty の意味は同一ではありません。

- Gaussian regression: continuous-response posterior
- binary classification: probability-scale prediction と latent GP
- multiclass classification: class probabilities と latent class GPs
- ordinal: latent score / cutpoint semantics と class probabilities を区別
- non-Gaussian regression: rate / mean などの response-scale quantities
- hybrid: task-specific semantics を保持

`train_inputs` は internal model が実際に使用する inputs、`train_inputs_raw` は元の search-space inputs を表します。詳細な posterior / shape contract は `src/bochan/models/README.md` と theory reference を参照してください。

---

## Documentation map

| File | 内容 |
|---|---|
| `docs/theory/README.md` | 英語 / 日本語 theory reference の入口。 |
| `docs/theory/ja/README.md` | 日本語 theory chapters と推奨読書順。 |
| `src/bochan/models/README.md` | Model families、registry guidance、wrapper conventions。 |
| `src/bochan/models/ARCHITECTURE.md` | Model family / cross-cutting ownership rules。 |
| `src/bochan/acquisition/README.md` | Acquisition families、objectives、active learning、LSE、non-Gaussian acquisitions。 |
| `src/bochan/api/README.md` | Tensor Python API、configs、objectives、candidate optimization、repair。 |
| `src/bochan/api/STUDY_README.md` | `BochanStudy`、`ask()` / `tell()`、save / load、scheduling。 |
| `src/bochan/tabular/README.md` | DataFrame / numpy / CSV adapter、categorical data、`beta`、repair、constraints。 |
| `src/bochan/tabular/ARCHITECTURE.md` | Canonical tabular ownership と dependency direction。 |
| `src/bochan/serving/fastapi/README.md` | HTTP / JSON endpoints、conversion、payloads、serving examples。 |
| `src/bochan/serving/fastapi/ARCHITECTURE.md` | Transport-layer ownership rules。 |

---

## Development status

この repository は active development 中です。Model wrappers の BoTorch compatibility、tensor / response-space contracts の明示、tabular / serving adapters と canonical API の整合、compatibility shim や duplicate domain logic を増やさないことを重視しています。
