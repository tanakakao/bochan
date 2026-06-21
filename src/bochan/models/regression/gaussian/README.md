# Gaussian regression models

`bochan.models.regression.gaussian` は、連続値をGaussian likelihoodで扱う回帰モデル群です。このREADMEでは、標準回帰、mixed input、独立multi-output、task-id形式のmulti-task、block-designのKronecker multi-taskを説明します。

## 1. データ形式

標準回帰では、入力と目的変数を次のshapeで用意します。

```python
import torch

train_X = torch.rand(40, 5, dtype=torch.double)  # [n, d]
train_Y = torch.rand(40, 1, dtype=torch.double)  # [n, 1]
```

複数出力を同じ入力点で観測している場合は、`train_Y.shape == [n, m]`です。

```python
train_Y = torch.rand(40, 3, dtype=torch.double)  # [n, m]
```

モデル、学習データ、boundsは同じdtype / deviceに揃えてください。連続説明変数は通常`[0, 1]`へ正規化し、目的変数は出力ごとに標準化します。

## 2. モデル選択

| 用途 | モデル |
|---|---|
| 標準exact GP | `SingleTaskGP` |
| continuous + categorical | `MixedSingleTaskGP` |
| 独立multi-output | `SingleTaskGP(train_Y=[n, m])` または `ModelListGP` |
| task-id列を使うmulti-task | `MultiTaskGP` |
| block-designの相関multi-task | `KroneckerMultiTaskGP` |
| DeepGP | `DeepGPModel` / `DeepMixedGPModel` |
| DeepKernel | `DeepKernelGPModel` / `DeepKernelMixedGPModel` |
| 高次元SAAS | `SaasSingleTaskGP` / `SaasMixedSingleTaskGP` |
| PCA / REMBO | `PCASingleTaskGP` / `REMBOSingleTaskGP` |
| robust / heteroscedastic | `SafeRobustRelevancePursuitSingleTaskGP` / `HeteroscedasticSingleTaskGP` |

最初は`SingleTaskGP`を基準にし、出力間相関を利用したい場合だけ`MultiTaskGP`または`KroneckerMultiTaskGP`を検討します。

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

`posterior()`は`outcome_transform`を自動的に逆変換するため、予測値は元の目的変数scaleで返ります。

## 4. mixed input

カテゴリ列を同じTensorへ格納し、`cat_dims`で列番号を指定します。

```python
import torch
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

## 5. 独立multi-output

同じ入力に対する複数の連続出力を独立に扱う場合、`SingleTaskGP`へ`train_Y: [n, m]`を渡せます。

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

この形式は各出力を独立に学習します。出力間相関をモデル化する場合はmulti-task modelを使用します。

## 6. task-id列を使うmulti-task

各タスクの入力点が異なる場合や、一部タスクだけが観測されている場合は`MultiTaskGP`を使います。入力にはtask-id列を追加します。

```python
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.multitask import MultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

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

task-id列は連続説明変数として正規化しないでください。

## 7. Kronecker multi-taskを使うblock design

すべてのタスクが同じ入力点で観測されている場合は、BoTorchの`KroneckerMultiTaskGP`を使用できます。

```text
train_X: [n, d]
train_Y: [n, m]
```

Gaussian likelihoodによるexact GPであり、共分散は概念的に`K_X ⊗ K_task`となります。

```python
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.multitask import KroneckerMultiTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood


torch.manual_seed(0)
dtype = torch.double

n = 50
num_tasks = 3
train_X = torch.rand(n, 3, dtype=dtype)

base = torch.sin(2.0 * torch.pi * train_X[:, 0])
train_Y = torch.stack(
    [
        base,
        0.8 * base + 0.4 * train_X[:, 1],
        -0.5 * base + train_X[:, 2],
    ],
    dim=-1,
)  # [n, 3]

bounds = torch.tensor(
    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    dtype=dtype,
)

model = KroneckerMultiTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    rank=2,
    input_transform=Normalize(d=train_X.shape[-1], bounds=bounds),
    outcome_transform=Standardize(m=num_tasks),
)

mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

X_test = torch.rand(10, 3, dtype=dtype)
posterior = model.posterior(X_test)

print(posterior.mean.shape)      # [10, 3]
print(posterior.variance.shape)  # [10, 3]
```

観測ノイズを含む予測分布が必要な場合は、次のように指定します。

```python
posterior_with_noise = model.posterior(X_test, observation_noise=True)
```

### タスク共分散と相関

```python
task_covar = model.covar_module.task_covar_module.covar_matrix.to_dense()

task_std = task_covar.diag().clamp_min(1e-12).sqrt()
task_corr = task_covar / task_std[:, None] / task_std[None, :]

print(task_covar.shape)  # [m, m]
print(task_corr.shape)   # [m, m]
```

`task_covar`の非対角成分はlatent functionの共変動です。目的変数から直接計算したPearson相関とは異なり、入力依存構造とノイズを考慮したモデル上の関係です。

### rank

- `rank=None`: full rank。既定ではタスク数と同じrank
- 小さい`rank`: タスク関係を低rankで表現し、パラメータ数を抑制
- タスク数が少ない場合は、まずfull rankまたは`rank=2`程度から比較

`KroneckerMultiTaskGP`はblock design専用です。タスクごとに入力点が異なる場合や欠測タスクがある場合は`MultiTaskGP`を使用してください。

## 8. 新しい観測とBayesian optimization

Gaussian exact GPはBoTorch標準posteriorを返すため、EI、UCB、KG、EHVI、NEHVIなどの獲得関数へ直接接続できます。

追加観測による更新では、BoTorchの`condition_on_observations`または再構築・再学習を利用します。Kronecker modelへ追加する`Y`は引き続き`[n_new, m]`のblock-design形式である必要があります。

## 9. high-level API

標準のsingle-output / independent multi-output regressionはhigh-level APIから構築できます。

```python
from bochan.api import BayesianOptimizer, FitConfig, ModelConfig

bo = BayesianOptimizer(
    model_config=ModelConfig(
        task_type="regression",
        model_type="base",
    ),
    fit_config=FitConfig(),
    bounds=bounds,
)
bo.fit(train_X, train_Y[:, [0]])
```

`MultiTaskGP`と`KroneckerMultiTaskGP`は現在の標準registry専用keyには含まれていないため、直接構築するかcustom modelとして接続してください。

## 10. よくある問題

### 全行の予測平均が同じになる

次を確認してください。

1. 学習が完了しているか
2. `input_transform`を二重適用していないか
3. `train_X`に重複や極端な高次元距離集中がないか
4. lengthscaleが過度に大きくなっていないか
5. outcome standardizationと保存・復元が整合しているか

### task covarianceとデータ相関が一致しない

正常です。task covarianceはlatent GPの共分散であり、単純な観測値相関ではありません。符号、相対的な強さ、学習データ数に対する安定性を確認してください。

### 高次元で学習が難しい

まずARDのlengthscaleを確認し、必要に応じてSAAS、PCA、REMBO、DeepKernelを比較してください。

## 11. 関連実装

```text
botorch.models.SingleTaskGP
botorch.models.multitask.MultiTaskGP
botorch.models.multitask.KroneckerMultiTaskGP
src/bochan/models/regression/gaussian/deep/
src/bochan/models/regression/gaussian/high_dim/
src/bochan/models/regression/gaussian/robust/
```
