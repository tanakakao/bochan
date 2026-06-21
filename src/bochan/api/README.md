# bochan API

`bochan.api` は、モデル構築・学習・予測・獲得関数生成・候補点最適化をまとめる高レベル API です。

```python
bundle = build_model(train_X, train_Y, model_config)
bundle = fit_model(bundle, fit_config)
acqf = build_acquisition(bundle, acq_config, data_context)
candidates, value = optimize_candidates(acqf, bounds, opt_config)
```

通常は `BayesianOptimizer` から利用します。

```python
from bochan.api import BayesianOptimizer, FitConfig, ModelConfig

bo = BayesianOptimizer(
    model_config=ModelConfig(task_type="regression", model_type="base"),
    fit_config=FitConfig(maxiter=128),
    bounds=bounds,
)
bo.fit(train_X, train_Y)
posterior = bo.predict(test_X)
```

## 1. 主な設定

| 設定 | 用途 |
|---|---|
| `ModelConfig` | task、model、mixed列、transform、model固有引数 |
| `FitConfig` | epoch、learning rate、maxiter、fit override |
| `InputTransformConfig` | Normalize、input perturbation |
| `AcquisitionConfig` | acquisition、objective、固有引数 |
| `ObjectiveConfig` | scalar / multi-output objective、risk aggregation |
| `DataContext` | baseline、best_f、ref_point、pending |
| `OptimizeConfig` | q、optimizer、fixed feature、制約 |
| `CandidateRepairConfig` | grid rounding、k-sparse、制約補修 |

## 2. model registry

標準 `task_type`:

```text
regression, multi_objective, binary, multiclass, ordinal, hybrid
```

主要な `model_type`:

```text
base, deepgp, deepkernel, deepgpdeepkernel,
saas, pca, rembo, rrp, hetero,
kronecker, multitask
```

`input_type` を省略した場合、`cat_dims` があれば `mixed`、なければ `normal` として解決します。

### mixed registry

| `task_type` | `kronecker` | `multitask` |
|---|---|---|
| `regression` | `MixedKroneckerMultiTaskGP` | `MixedMultiTaskGP` |
| `multi_objective` | `MixedKroneckerMultiTaskGP` | `MixedMultiTaskGP` |
| `binary` | `KroneckerMultiTaskBinaryClassificationMixedGPModel` | `MultiTaskBinaryClassificationMixedGPModel` |
| `multiclass` | `KroneckerMultiTaskMulticlassClassificationMixedGPModel` | `MultiTaskMulticlassClassificationMixedGPModel` |
| `ordinal` | `KroneckerMultiTaskOrdinalMixedGPModel` | `MultiTaskOrdinalMixedGPModel` |

通常の `base`、`deepgp`、`deepkernel`、`saas`、`pca`、`rembo`、`rrp`、`hetero` も従来どおり利用できます。

## 3. multi-output と multi-task の違い

### independent multi-output

同じ入力から複数出力を得ますが、submodel 間の相関は学習しません。

```text
train_X: [n, d]
train_Y: [n, m]
```

### task-feature multi-task

入力に task-id 列を追加します。タスクごとに入力位置・観測数が異なる場合や、欠測タスクがある場合に使います。

```text
train_X: [N, d + 1]
train_Y: [N] or [N, 1]
```

```text
K((x,t),(x',t')) = K_mixed(x,x') * K_task(t,t')
```

### Kronecker multi-task

全タスクが同じ入力点で観測される block design 専用です。

```text
train_X: [n, d]
train_Y: [n, m]
```

| 条件 | 推奨 |
|---|---|
| タスクごとに入力位置や観測数が異なる | `model_type="multitask"` |
| 一部タスクが欠測 | `model_type="multitask"` |
| 全タスクが同じ入力点で観測 | `model_type="kronecker"` |
| 出力間相関を使わない | independent multi-output |

## 4. mixed task-feature model

例の列構成:

```text
continuous_0 | task_id | category_0
```

### binary

```python
model_config = ModelConfig(
    task_type="binary",
    input_type="mixed",
    model_type="multitask",
    cat_dims=[2],
    model_kwargs={
        "num_tasks": 3,
        "task_feature": 1,
        "rank": 2,
        "num_inducing_points": 64,
    },
)
```

### multiclass

```python
model_config = ModelConfig(
    task_type="multiclass",
    input_type="mixed",
    model_type="multitask",
    cat_dims=[2],
    model_kwargs={
        "num_classes": 3,
        "num_tasks": 3,
        "task_feature": 1,
        "rank": 2,
    },
)
```

multiclass ではクラス logit ごとに task covariance を持つため、`task_covar_matrix.shape == [C, m, m]` です。

### ordinal

```python
model_config = ModelConfig(
    task_type="ordinal",
    input_type="mixed",
    model_type="multitask",
    cat_dims=[2],
    model_kwargs={
        "num_classes": 4,
        "num_tasks": 3,
        "task_feature": 1,
        "rank": 2,
    },
)
```

ordinal multi-task は全タスクでクラス定義と cutpoint を共有します。

### Gaussian regression

```python
model_config = ModelConfig(
    task_type="regression",
    input_type="mixed",
    model_type="multitask",
    cat_dims=[2],
    model_kwargs={
        "task_feature": 1,
        "rank": 2,
    },
)
```

Gaussian 版は exact `MultiTaskGP` wrapper です。

## 5. mixed Kronecker model

```python
model_config = ModelConfig(
    task_type="binary",
    input_type="mixed",
    model_type="kronecker",
    cat_dims=[2],
    model_kwargs={
        "rank": 2,
        "num_inducing_points": 64,
    },
)
```

classification / ordinal Kronecker model は `make_mll()` を持ち、API の fit 解決ではこれを優先します。

## 6. InputTransform

カテゴリ列と task-id 列は Normalize / perturbation から除外します。

```python
from bochan.api import InputTransformConfig

model_config = ModelConfig(
    task_type="binary",
    input_type="mixed",
    model_type="multitask",
    cat_dims=[2],
    model_kwargs={
        "num_tasks": 2,
        "task_feature": 1,
    },
    input_transform_config=InputTransformConfig(
        normalize=True,
        categorical_idx=[1, 2],
    ),
)
```

`categorical_idx` は通常カテゴリだけでなく task-id 列の保護にも使います。

直接 transform を渡す場合:

```python
from botorch.models.transforms.input import Normalize

Normalize(d=3, indices=[0])  # continuous列だけ
```

次は無効です。

```python
Normalize(d=3)  # categoryとtask-idを変換するためValueError
```

`task_feature` を `cat_dims` に重複指定することもできません。

## 7. FitConfig

```python
FitConfig(maxiter=128)               # exact GP
FitConfig(num_epochs=300, lr=0.01)  # variational model
FitConfig(skip_fit=True)            # already fitted
```

自動解決:

| model | fit |
|---|---|
| model に `make_mll()` がある | `model.make_mll()` を優先 |
| regression exact GP | `ExactMarginalLogLikelihood` |
| binary | `VariationalELBO` + binary fit helper |
| multiclass | `VariationalELBO` + multiclass fit helper |
| ordinal | ordinal MLL + ordinal fit helper |

## 8. 予測契約

```python
posterior = bo.predict(X)
mean = bo.predict(X, return_type="mean")
mean, variance = bo.predict(X, return_type="mean_variance")
result = bo.predict(X, return_result=True)
```

### binary

利用可能なら `probability_posterior()` を優先します。

- `mean`: `P(y=1 | x)`
- `variance`: 通常 `p * (1-p)`
- `prediction_space="probability"`

この variance は epistemic uncertainty そのものではありません。

### Gaussian task-feature

training `train_X` は task-id 列を含みますが、予測 `X` は task-id 列を含まない data feature です。タスクは `output_indices` で選びます。

```python
result = bo.predict(
    X_without_task_column,
    return_result=True,
    posterior_kwargs={"output_indices": [0, 1]},
)
```

Gaussian task-feature model の `bounds` は candidate 側の data feature 次元に合わせて明示してください。

### binary / multiclass / ordinal task-feature

予測 `X` 自体に task-id 列を含めます。

```python
X = torch.tensor(
    [
        [0.25, 0.0, 1.0],
        [0.25, 1.0, 1.0],
    ],
    dtype=torch.double,
)
result = bo.predict(X, return_result=True)
```

## 9. acquisition と objective

```python
AcquisitionConfig(name="EI")
AcquisitionConfig(name="UCB")
AcquisitionConfig(name="BALD")
AcquisitionConfig(name="Entropy")
AcquisitionConfig(name="Straddle")
AcquisitionConfig(name="NEHVI")
```

contextual alias は `task_type`、`model_type`、multi-output 状態に応じて解決されます。

multiclass BO では acquisition 固有引数を渡します。

```python
AcquisitionConfig(
    name="EI",
    acqf_kwargs={
        "target_class": 2,
        "best_f": 0.7,
    },
)
```

ordinal expected utility:

```python
ObjectiveConfig(
    mode="scalar",
    utility_values=[0.0, 1.0, 3.0, 6.0],
)
```

input perturbation と risk aggregation:

```python
ObjectiveConfig(
    mode="scalar",
    n_w=8,
    risk_type="cvar",
    alpha=0.8,
)
```

## 10. taskを固定した候補最適化

binary / multiclass / ordinal task-feature model では candidate tensor に task-id 列を含めます。

```python
from bochan.api import OptimizeConfig

# columns: continuous, task_id, category
opt_config = OptimizeConfig(
    optimizer="optimize_acqf_mixed",
    q=1,
    fixed_features={1: 1.0},
    fixed_features_list=[
        {2: 0.0},
        {2: 1.0},
    ],
)
```

概念的には次を列挙します。

```text
{task_id: 1, category: 0}
{task_id: 1, category: 1}
```

Gaussian `MultiTaskGP` 系では candidate tensor に task-id 列を含めず、posterior output を選択します。

## 11. binary end-to-end

```python
import torch

from bochan.api import (
    AcquisitionConfig,
    BayesianOptimizer,
    FitConfig,
    InputTransformConfig,
    ModelConfig,
    OptimizeConfig,
)

train_X = torch.tensor(
    [
        [0.05, 0.0, 0.0],
        [0.20, 0.0, 1.0],
        [0.60, 0.0, 0.0],
        [0.10, 1.0, 1.0],
        [0.45, 1.0, 0.0],
        [0.90, 1.0, 1.0],
    ],
    dtype=torch.double,
)
train_Y = torch.tensor([0, 0, 1, 0, 1, 1], dtype=torch.double)
bounds = torch.tensor(
    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    dtype=torch.double,
)

bo = BayesianOptimizer(
    model_config=ModelConfig(
        task_type="binary",
        input_type="mixed",
        model_type="multitask",
        cat_dims=[2],
        model_kwargs={
            "num_tasks": 2,
            "task_feature": 1,
            "rank": 2,
            "num_inducing_points": 32,
        },
        input_transform_config=InputTransformConfig(
            normalize=True,
            categorical_idx=[1, 2],
        ),
    ),
    fit_config=FitConfig(num_epochs=300, lr=0.01),
    bounds=bounds,
)
bo.fit(train_X, train_Y)

candidates, value = bo.candidate(
    acq_config=AcquisitionConfig(name="Entropy"),
    opt_config=OptimizeConfig(
        optimizer="optimize_acqf_mixed",
        q=1,
        fixed_features={1: 1.0},
        fixed_features_list=[{2: 0.0}, {2: 1.0}],
    ),
)
```

## 12. multi-output / hybrid

```python
ModelConfig(
    task_type="regression",
    model_type="base",
    multi_output_config=MultiOutputConfig(),
)
```

hybrid:

```python
ModelConfig(
    task_type="hybrid",
    multi_output_config=MultiOutputConfig(
        output_configs=[
            OutputConfig(task_type="regression", name="strength"),
            OutputConfig(task_type="binary", name="defect"),
            OutputConfig(
                task_type="multiclass",
                name="defect_type",
                model_kwargs={"num_classes": 3},
            ),
        ],
        use_hybrid=True,
    ),
)
```

## 13. candidate repair

```python
CandidateRepairConfig(
    bounds=bounds,
    numeric_indices=[0, 1, 2, 3],
    steps=[0.1, 0.1, 0.1, 0.1],
    comp_idx=[0, 1, 2, 3],
    k=2,
    inequality_constraints=ineq_constraints,
    inequality_sense="le",
)
```

- `steps=None`: grid rounding なし
- `comp_idx=None` / `[]`: k-sparse なし
- `support_selection="topk"` / `"sample"`

## 14. BochanStudy / serving / tabular

- ask / tell loop: `src/bochan/api/STUDY_README.md`
- FastAPI: `src/bochan/serving/fastapi/README.md`
- pandas / numpy / CSV: `src/bochan/tabular/README.md`
- mixed task-feature 詳細: `docs/mixed_task_feature_multitask_models.md`

FastAPI でも `model_type="multitask"` / `"kronecker"`、`model_kwargs.task_feature`、`cat_dims`、`fixed_features` を同じ意味で指定します。

## 15. 注意点

- task-id を `cat_dims` に含めない
- category と task-id を Normalize / perturbation しない
- classification / ordinal candidate では task-id を固定する
- task covariance は観測値の単純相関ではなく latent GP 共分散
- multiclass task covariance は `[C, m, m]`
- ordinal multi-task は全タスクで cutpoint を共有
- Kronecker model は block design 専用

## 16. 関連テスト

```text
tests/test_mixed_task_feature_multitask_models.py
tests/test_mixed_task_feature_multitask_registry.py
tests/test_kronecker_multitask_classification_ordinal_models.py
tests/test_kronecker_multitask_multiclass_model.py
tests/test_binary_api_prediction.py
```
