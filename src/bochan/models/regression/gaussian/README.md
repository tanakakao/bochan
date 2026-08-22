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
| DeepGP | `DeepGaussianGPModel` / `DeepGaussianMixedGPModel` |
| DeepKernel | `DeepKernelGaussianGPModel` / `DeepKernelGaussianMixedGPModel` |
| frozen CrabNet + exact GP | `CrabNetGPModel` |
| 高次元SAAS | `SaasGaussianGPModel` / `SaasGaussianMixedGPModel` |
| PCA / REMBO | `PCAGaussianGPModel` / `REMBOGaussianGPModel` |
| robust / heteroscedastic | `RobustRelevancePursuitGaussianGPModel` / `HeteroscedasticGaussianGPModel` |

最初は`SingleTaskGP`を基準にし、出力間相関を利用したい場合だけ`MultiTaskGP`または`KroneckerMultiTaskGP`を検討します。

### DeepKernelの任意feature extractor

Gaussian DeepKernelは、入力次元とlatent次元を分離できます。`feature_extractor=None`
では従来どおり入力次元と同じ幅のMLPを使います。任意の`nn.Module`を渡す場合は、
最後の次元を`latent_dim`に合わせてください。GP kernelのARD次元にも同じ値が使われます。

```python
import torch
from torch import nn

from bochan.models.regression.gaussian.deep import DeepKernelGaussianGPModel


feature_extractor = nn.Sequential(
    nn.Linear(train_X.shape[-1], 64),
    nn.SiLU(),
    nn.Linear(64, 16),
)
model = DeepKernelGaussianGPModel(
    train_X=train_X,
    train_Y=train_Y,
    feature_extractor=feature_extractor,
    latent_dim=16,
)
```

`latent_dim`を省略した場合は、feature extractorの`output_dim`属性、またはsample
forwardから出力幅を解決します。明示した`latent_dim`と実際の出力幅が異なる場合は、
kernel評価前に具体的なshape errorを返します。

### CrabNet-GPの低レベルAPI

`CrabNetGPModel`は、固定したCrabNet material encoderの表現と連続process特徴を
concatし、trainableな線形projectionを通してexact GPへ渡します。最初の組成列は
`element_ids`と同じ順序のfractionで、残りの列がprocess特徴です。これは
`composition_sites`による前処理後の低レベルTensor契約であり、組成式用の別APIでは
ありません。

```python
import torch

from bochan.composition import CrabNetEncoder
from bochan.models.regression.gaussian.deep import CrabNetGPModel


element_ids = torch.tensor([26, 27, 28], dtype=torch.long)  # Fe, Co, Ni
fractions = torch.tensor(
    [[0.5, 0.3, 0.2], [0.6, 0.2, 0.2], [0.4, 0.4, 0.2]],
    dtype=torch.double,
)
process_X = torch.tensor([[900.0, 2.0], [950.0, 4.0], [1000.0, 3.0]], dtype=torch.double)
train_X = torch.cat([fractions, process_X], dim=-1)
train_Y = torch.tensor([[1.1], [1.4], [1.2]], dtype=torch.double)

model = CrabNetGPModel(
    train_X=train_X,
    train_Y=train_Y,
    element_ids=element_ids,
    encoder=CrabNetEncoder(checkpoint="crabnet.pth"),
    latent_dim=32,
)
posterior = model.posterior(train_X[:1])
```

既定のinput transformはprocess列だけを正規化し、fractionは変更しません。encoder
parameterはfreezeされますが、composition/process入力からposteriorまでのgradientは
保持されます。初期版ではcategorical processと`train_Yvar`は未対応です。

#### 組成とprocess条件の同時最適化

fractionを探索座標として直接使う場合は、各候補のfraction和を1に保つため、BoTorchの
intra-point equality constraintを渡します。1次元のindex tensorを使うと、この制約は
q-batch内の各候補へ個別に適用されます。process列は同じ`bounds`内で同時に最適化され
ます。

```python
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf


acqf = qLogExpectedImprovement(
    model=model,
    best_f=train_Y.max(),
)
bounds = torch.tensor(
    [
        [0.05, 0.05, 0.05, 850.0, 0.5],
        [0.80, 0.80, 0.80, 1300.0, 6.0],
    ],
    dtype=train_X.dtype,
)
composition_constraint = (
    torch.arange(model.composition_dim, device=bounds.device, dtype=torch.long),
    torch.ones(model.composition_dim, device=bounds.device, dtype=bounds.dtype),
    1.0,
)

candidate, acq_value = optimize_acqf(
    acq_function=acqf,
    bounds=bounds,
    q=3,
    num_restarts=10,
    raw_samples=256,
    equality_constraints=[composition_constraint],
)
```

この直接fraction最適化はPhase 5の低レベル契約です。ILR等のcomposition coordinateから
fractionへautogradを維持して戻す場合は、canonical domain API の
`bochan.composition.TorchSimplexTransform`を使います。

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
