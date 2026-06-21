# Gaussian regression models

`bochan.models.regression.gaussian` は、連続値を Gaussian likelihood で扱う回帰モデル群です。この README では、標準回帰、mixed input、独立 multi-output、task-id 形式の multi-task、block-design の Kronecker multi-task を説明します。

## 1. データ形式

### single-output / independent multi-output

```python
import torch

train_X = torch.rand(40, 5, dtype=torch.double)  # [n, d]
train_Y = torch.rand(40, 1, dtype=torch.double)  # [n, 1]
```

複数出力を同じ入力点で観測し、出力間を独立に扱う場合は次の形です。

```python
train_Y = torch.rand(40, 3, dtype=torch.double)  # [n, m]
```

### task-feature long format

タスクごとに入力点が異なる場合や、一部タスクだけが観測されている場合は、task-id 列を含む long format を使います。

```text
train_X: [N, d + 1]
train_Y: [N, 1]
```

例:

```text
continuous_0 | task_id | category_0
0.10         | 0       | 1
0.35         | 1       | 0
0.80         | 1       | 2
```

### Kronecker block design

すべてのタスクが同じ入力点で観測されている場合は次の形です。

```text
train_X: [n, d]
train_Y: [n, m]
```

モデル、学習データ、bounds は同じ dtype / device に揃えてください。連続説明変数は通常 `[0, 1]` へ正規化し、目的変数は出力ごとに標準化します。

## 2. モデル選択

| 用途 | 通常入力 | mixed input |
|---|---|---|
| 標準 exact GP | `SingleTaskGP` | `MixedSingleTaskGP` |
| 独立 multi-output | `SingleTaskGP(train_Y=[n, m])` / `ModelListGP` | 各 submodel に mixed model を使用 |
| task-id long-format multi-task | `MultiTaskGP` | `MixedMultiTaskGP` |
| block-design Kronecker multi-task | `KroneckerMultiTaskGP` | `MixedKroneckerMultiTaskGP` |
| DeepGP | `DeepGPModel` | `DeepMixedGPModel` |
| DeepKernel | `DeepKernelGPModel` | `DeepKernelMixedGPModel` |
| 高次元 SAAS | `SaasSingleTaskGP` | `SaasMixedSingleTaskGP` |
| PCA / REMBO | `PCASingleTaskGP` / `REMBOSingleTaskGP` | 対応 mixed model |
| robust / heteroscedastic | `SafeRobustRelevancePursuitSingleTaskGP` / `HeteroscedasticSingleTaskGP` | 対応 mixed model |

使い分けの基準:

- 各出力を独立に扱う: independent multi-output
- タスクごとに入力位置や観測数が異なる: task-feature multi-task
- 全タスクが同じ入力点で観測される: Kronecker multi-task
- 連続列とカテゴリ列が混在する: mixed 版

## 3. 標準回帰の最小例

```python
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood


torch.manual_seed(0)
dtype = torch.double

train_X = torch.rand(40, 3, dtype=dtype)
train_Y = (
    torch.sin(2.0 * torch.pi * train_X[:, 0])
    + 0.5 * train_X[:, 1]
    - 0.2 * train_X[:, 2]
).unsqueeze(-1)

bounds = torch.tensor(
    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    dtype=dtype,
)

model = SingleTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    input_transform=Normalize(d=train_X.shape[-1], bounds=bounds),
    outcome_transform=Standardize(m=1),
)

mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

X_test = torch.rand(10, 3, dtype=dtype)
posterior = model.posterior(X_test)

print(posterior.mean.shape)      # [10, 1]
print(posterior.variance.shape)  # [10, 1]
```

`posterior()` は `outcome_transform` を自動的に逆変換するため、予測値は元の目的変数 scale で返ります。

## 4. mixed input

カテゴリ列を同じ Tensor へ格納し、`cat_dims` で列番号を指定します。

```python
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

continuous_X = torch.rand(50, 3, dtype=torch.double)
category = torch.randint(0, 4, (50, 1)).to(torch.double)
train_X = torch.cat([continuous_X, category], dim=-1)

train_Y = (
    continuous_X[:, 0]
    + 0.2 * continuous_X[:, 1]
    + 0.3 * category.squeeze(-1)
).unsqueeze(-1)

model = MixedSingleTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    cat_dims=[3],
)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)
```

カテゴリ列を通常の連続変数として正規化しないでください。

## 5. 独立 multi-output

```python
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize

train_Y = torch.stack(
    [
        torch.sin(2.0 * torch.pi * train_X[:, 0]),
        train_X[:, 0] + train_X[:, 1],
        train_X[:, 2] ** 2,
    ],
    dim=-1,
)

model = SingleTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    outcome_transform=Standardize(m=train_Y.shape[-1]),
)

mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

posterior = model.posterior(X_test)
print(posterior.mean.shape)  # [q, m]
```

この形式は各出力を独立に学習します。出力間相関をモデル化する場合は multi-task model を使用します。

## 6. task-id 列を使う multi-task

### 6.1 continuous input: BoTorch `MultiTaskGP`

```python
from botorch.models.multitask import MultiTaskGP

X_task0 = torch.rand(30, 2, dtype=torch.double)
X_task1 = torch.rand(20, 2, dtype=torch.double)

train_X = torch.cat(
    [
        torch.cat([X_task0, torch.zeros(30, 1, dtype=torch.double)], dim=-1),
        torch.cat([X_task1, torch.ones(20, 1, dtype=torch.double)], dim=-1),
    ],
    dim=0,
)
train_Y = torch.cat(
    [
        torch.sin(2.0 * torch.pi * X_task0[:, 0]),
        0.7 * torch.sin(2.0 * torch.pi * X_task1[:, 0]) + 0.2,
    ],
    dim=0,
).unsqueeze(-1)

model = MultiTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    task_feature=-1,
    rank=2,
)

mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)
```

### 6.2 mixed input: `MixedMultiTaskGP`

`MixedMultiTaskGP` は、連続列・カテゴリ列・task-id 列を持つ long-format exact GP です。タスク列は `IndexKernel` のみに渡され、mixed data kernel には含まれません。

```python
from botorch.models.transforms.input import Normalize
from bochan.models.regression.gaussian import MixedMultiTaskGP

# columns: continuous, task_id, category
train_X = torch.tensor(
    [
        [0.05, 0.0, 0.0],
        [0.25, 0.0, 1.0],
        [0.55, 0.0, 0.0],
        [0.10, 1.0, 1.0],
        [0.40, 1.0, 0.0],
        [0.85, 1.0, 1.0],
    ],
    dtype=torch.double,
)
train_Y = (
    torch.sin(2.0 * torch.pi * train_X[:, 0])
    + 0.25 * train_X[:, 1]
    + 0.15 * train_X[:, 2]
).unsqueeze(-1)

model = MixedMultiTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    task_feature=1,
    cat_dims=[2],
    rank=2,
    input_transform=Normalize(
        d=3,
        indices=[0],
    ),
)

mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)
```

BoTorch `MultiTaskGP.posterior()` は予測時に task 列を含まない入力を受け取り、`output_indices` でタスクを選択します。

```python
X_test = torch.tensor(
    [[0.20, 0.0], [0.80, 1.0]],
    dtype=torch.double,
)

posterior = model.posterior(X_test, output_indices=[0, 1])
print(posterior.mean.shape)  # [2, 2]
```

### 6.3 InputTransform の注意

カテゴリ列と task-id 列は正規化・摂動しないでください。

```python
Normalize(d=3, indices=[0])  # OK
Normalize(d=3)               # NG
```

`task_feature` を `cat_dims` に重複して指定することもできません。

## 7. Kronecker multi-task を使う block design

### 7.1 continuous input

```python
from botorch.models.multitask import KroneckerMultiTaskGP
from botorch.models.transforms.outcome import Standardize

n = 50
num_tasks = 3
train_X = torch.rand(n, 3, dtype=torch.double)
base = torch.sin(2.0 * torch.pi * train_X[:, 0])
train_Y = torch.stack(
    [
        base,
        0.8 * base + 0.4 * train_X[:, 1],
        -0.5 * base + train_X[:, 2],
    ],
    dim=-1,
)

model = KroneckerMultiTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    rank=2,
    outcome_transform=Standardize(m=num_tasks),
)

mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)
```

### 7.2 mixed input

`MixedKroneckerMultiTaskGP` は、同じ入力点で全タスクが観測された mixed block design を扱います。

```python
from bochan.models.regression.gaussian import MixedKroneckerMultiTaskGP

model = MixedKroneckerMultiTaskGP(
    train_X=train_X_mixed,
    train_Y=train_Y,
    cat_dims=[2],
    rank=2,
)
```

### 7.3 タスク共分散と相関

```python
task_covar = model.covar_module.task_covar_module.covar_matrix.to_dense()

task_std = task_covar.diag().clamp_min(1e-12).sqrt()
task_corr = task_covar / task_std[:, None] / task_std[None, :]
```

`task_covar` の非対角成分は latent function の共変動です。目的変数から直接計算した Pearson 相関とは異なります。

Kronecker 版は block design 専用です。タスクごとに入力点が異なる場合や欠測タスクがある場合は `MultiTaskGP` または `MixedMultiTaskGP` を使用してください。

## 8. Bayesian optimization

Gaussian exact GP は BoTorch 標準 posterior を返すため、EI、UCB、KG、EHVI、NEHVI などへ接続できます。

- Gaussian `MultiTaskGP` 系: candidate tensor は task 列を含めず、posterior の `output_indices` でタスクを選択
- binary / multiclass / ordinal task-feature model: candidate tensor 自体に task-id 列を含める

## 9. high-level API

mixed task-feature model は `model_type="multitask"` で解決できます。

```python
from bochan.api import (
    BayesianOptimizer,
    FitConfig,
    InputTransformConfig,
    ModelConfig,
)

bo = BayesianOptimizer(
    model_config=ModelConfig(
        task_type="regression",
        input_type="mixed",
        model_type="multitask",
        cat_dims=[2],
        model_kwargs={
            "task_feature": 1,
            "rank": 2,
        },
        input_transform_config=InputTransformConfig(
            normalize=True,
            categorical_idx=[1, 2],
        ),
    ),
    fit_config=FitConfig(maxiter=128),
    bounds=torch.tensor(
        [[0.0, 0.0], [1.0, 1.0]],
        dtype=torch.double,
    ),
)
bo.fit(train_X, train_Y)

X_test_without_task = torch.tensor(
    [[0.20, 0.0], [0.80, 1.0]],
    dtype=torch.double,
)
result = bo.predict(
    X_test_without_task,
    return_result=True,
    posterior_kwargs={"output_indices": [0, 1]},
)
```

mixed block-design Kronecker は `model_type="kronecker"` です。

```python
ModelConfig(
    task_type="regression",
    input_type="mixed",
    model_type="kronecker",
    cat_dims=[2],
    model_kwargs={"rank": 2},
)
```

## 10. よくある問題

### 全行の予測平均が同じになる

1. 学習が完了しているか
2. `input_transform` を二重適用していないか
3. `train_X` に重複や極端な距離集中がないか
4. lengthscale が過度に大きくなっていないか
5. outcome standardization と保存・復元が整合しているか

### task covariance とデータ相関が一致しない

正常です。task covariance は latent GP の共分散であり、単純な観測値相関ではありません。

### category / task-id が変化する

`Normalize(indices=...)` にカテゴリ列または task-id 列を含めないでください。API の `InputTransformConfig` では両方を `categorical_idx` に含めます。

## 11. 関連実装・テスト

```text
botorch.models.SingleTaskGP
botorch.models.multitask.MultiTaskGP
botorch.models.multitask.KroneckerMultiTaskGP
src/bochan/models/regression/gaussian/multitask.py
src/bochan/models/regression/gaussian/kronecker_multitask.py
src/bochan/models/components/mixed_multitask.py
tests/test_mixed_task_feature_multitask_models.py
tests/test_mixed_task_feature_multitask_registry.py
```
