# Binary classification models

`bochan.models.classification.binary` は、BoTorch / GPyTorch ベースの2値分類モデル群です。この README では、標準分類、mixed input、独立 multi-output、task-id long-format multi-task、block-design Kronecker multi-task、不確かさの扱いを説明します。

## 1. データ形式

### single-output

```python
import torch

train_X = torch.rand(40, 5, dtype=torch.double)
train_Y = torch.randint(0, 2, (40, 1)).to(torch.double)
```

single-output の `train_Y` は `[n]` と `[n, 1]` の両方を受け付けます。モデル、学習データ、bounds は同じ dtype / device に揃えてください。

### mixed input

```python
continuous_X = torch.rand(40, 4, dtype=torch.double)
category = torch.randint(0, 3, (40, 1)).to(torch.double)
train_X = torch.cat([continuous_X, category], dim=-1)
cat_dims = [4]
```

### task-feature long format

タスクごとに入力位置や観測数が異なる場合は、task-id 列を入力に含めます。

```text
train_X: [N, d + 1]
train_Y: [N] or [N, 1]
```

### Kronecker block design

同じ入力点で全タスクが観測される場合は次の形です。

```text
train_X: [n, d]
train_Y: [n, m]
```

## 2. モデル選択

| 用途 | 通常入力 | mixed input |
|---|---|---|
| 標準 SVGP | `BinaryClassificationGPModel` | `BinaryClassificationMixedGPModel` |
| 独立 multi-output | `MultiOutputBinaryClassificationModel` | 各 submodel に mixed model を使用 |
| task-id long-format multi-task | `MultiTaskBinaryClassificationGPModel` | `MultiTaskBinaryClassificationMixedGPModel` |
| block-design Kronecker multi-task | `KroneckerMultiTaskBinaryClassificationGPModel` | `KroneckerMultiTaskBinaryClassificationMixedGPModel` |
| DeepGP | `BinaryClassificationDeepGPModel` | `BinaryClassificationMixedDeepGPModel` |
| DeepKernel | `DeepKernelBinaryClassificationGPModel` | `DeepKernelBinaryClassificationMixedGPModel` |
| DeepKernel + DeepGP | `DeepKernelBinaryClassificationDeepGPModel` | `DeepKernelBinaryClassificationMixedDeepGPModel` |
| 高次元 SAAS | `SaasBinaryClassificationGPModel` | `SaasBinaryClassificationMixedGPModel` |
| PCA / REMBO | `PCABinaryClassificationGPModel` / `REMBOBinaryClassificationGPModel` | 対応 mixed model |
| 外れラベル RRP | `OutlierRelevancePursuitBinaryClassificationGPModel` | 対応 mixed model |
| 不均一ノイズ | `HeteroscedasticBinaryClassificationGPModel` | 対応 mixed model |

使い分け:

- 出力を独立に扱う: multi-output wrapper
- タスク相関を使い、入力位置・観測数がタスクごとに異なる: task-feature multi-task
- タスク相関を使い、全タスクが同じ入力点で観測される: Kronecker multi-task

## 3. 標準 binary の最小例

```python
import torch
from botorch.models.transforms.input import Normalize
from gpytorch.mlls import VariationalELBO

from bochan.fit import fit_binary_classifier_mll
from bochan.models.classification.binary.base import BinaryClassificationGPModel


torch.manual_seed(0)
dtype = torch.double

train_X = torch.rand(40, 3, dtype=dtype)
score = 1.2 * train_X[:, 0] - 0.8 * train_X[:, 1] + 0.4 * train_X[:, 2]
train_Y = (score > score.median()).to(dtype).unsqueeze(-1)

model = BinaryClassificationGPModel(
    train_X=train_X,
    train_Y=train_Y,
    input_transform=Normalize(d=train_X.shape[-1]),
    num_inducing_points=32,
)

mll = VariationalELBO(
    likelihood=model.likelihood,
    model=model.model,
    num_data=train_X.shape[-2],
)
fit_binary_classifier_mll(mll, num_epochs=300, lr=0.01)

X_test = torch.rand(10, 3, dtype=dtype)
posterior = model.posterior(X_test)
probability = posterior.mean
prediction = (probability >= 0.5).to(torch.long)

print(probability.shape)  # [10, 1]
```

主な引数:

- `num_inducing_points`: inducing point 数
- `inducing_points`: inducing point を明示する場合に指定
- `learn_inducing_locations`: inducing point 位置を学習するか
- `input_transform`: `Normalize` などの入力変換
- `mean_module`, `covar_module`: latent GP のカスタマイズ

## 4. mixed input

カテゴリ列を正規化対象から除外します。

```python
from bochan.models.classification.binary.base import BinaryClassificationMixedGPModel

model = BinaryClassificationMixedGPModel(
    train_X=train_X,
    train_Y=train_Y,
    cat_dims=[3],
    input_transform=Normalize(
        d=train_X.shape[-1],
        indices=[0, 1, 2],
    ),
    num_inducing_points=32,
)

mll = VariationalELBO(
    likelihood=model.likelihood,
    model=model.model,
    num_data=train_X.shape[-2],
)
fit_binary_classifier_mll(mll, num_epochs=300, lr=0.01)
```

カテゴリ値は候補点生成時も学習時と同じ値集合を使ってください。

## 5. posterior と不確かさ

### probability posterior

```python
posterior = model.posterior(X_test)
p = posterior.mean
variance = posterior.variance
```

`mean` はクラス1確率 `P(y=1 | x, D)` です。通常の `variance` は Bernoulli observation variance `p * (1-p)` であり、epistemic uncertainty そのものではありません。

### latent posterior

```python
latent_posterior = model.latent_posterior(X_test)
latent_mean = latent_posterior.mean
latent_variance = latent_posterior.variance
```

### probability-scale epistemic uncertainty

```python
from bochan.acquisition.binary.epistemic import binary_probability_moments

mean_probability, epistemic_variance, aleatoric_variance, total_variance = (
    binary_probability_moments(model, X_test, num_samples=256)
)
```

- `epistemic_variance`: posterior sample 間の確率分散
- `aleatoric_variance`: `E[p(1-p)]`
- `total_variance`: 0/1 label の全分散

active learning や variance 系獲得関数では、原則として epistemic variance を使います。

## 6. 独立 multi-output

```python
from bochan.models.classification.binary.base import (
    BinaryClassificationGPModel,
    MultiOutputBinaryClassificationModel,
)

submodels = []
for j in range(train_Y_multi.shape[-1]):
    submodel = BinaryClassificationGPModel(
        train_X=train_X,
        train_Y=train_Y_multi[:, [j]],
        num_inducing_points=32,
    )
    mll = VariationalELBO(
        likelihood=submodel.likelihood,
        model=submodel.model,
        num_data=train_X.shape[-2],
    )
    fit_binary_classifier_mll(mll, num_epochs=300, lr=0.01)
    submodels.append(submodel)

model = MultiOutputBinaryClassificationModel(*submodels)
posterior = model.posterior(X_test)
print(posterior.mean.shape)  # [q, m]
```

この wrapper は出力間を独立として扱います。出力間相関を学習する場合は multi-task model を使用します。

## 7. task-id 列を使う multi-task

### 7.1 continuous input

```python
from bochan.models.classification.binary.base import (
    MultiTaskBinaryClassificationGPModel,
)

X_data = torch.rand(60, 3, dtype=torch.double)
task_id = torch.randint(0, 3, (60, 1)).to(torch.double)
train_X = torch.cat([X_data, task_id], dim=-1)
train_Y = torch.randint(0, 2, (60, 1)).to(torch.double)

model = MultiTaskBinaryClassificationGPModel(
    train_X=train_X,
    train_Y=train_Y,
    num_tasks=3,
    task_feature=-1,
    rank=2,
    num_inducing_points=32,
)

mll = VariationalELBO(
    likelihood=model.likelihood,
    model=model.model,
    num_data=train_X.shape[-2],
)
fit_binary_classifier_mll(mll, num_epochs=300, lr=0.01)
```

### 7.2 mixed input

`MultiTaskBinaryClassificationMixedGPModel` は、continuous・category・task-id を含む long-format variational GP です。タスクごとに入力位置や観測数が異なるデータを扱えます。

```python
from bochan.models.classification.binary.base import (
    MultiTaskBinaryClassificationMixedGPModel,
)

# columns: continuous, task_id, category
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

model = MultiTaskBinaryClassificationMixedGPModel(
    train_X=train_X,
    train_Y=train_Y,
    cat_dims=[2],
    num_tasks=2,
    task_feature=1,
    rank=2,
    input_transform=Normalize(d=3, indices=[0]),
    num_inducing_points=32,
)

mll = VariationalELBO(
    likelihood=model.likelihood,
    model=model.model,
    num_data=train_X.shape[-2],
)
fit_binary_classifier_mll(mll, num_epochs=300, lr=0.01)
```

予測候補にも task-id 列を含めます。

```python
X_test = torch.tensor(
    [
        [0.25, 0.0, 1.0],
        [0.25, 1.0, 1.0],
    ],
    dtype=torch.double,
)

probability = model.posterior(X_test).mean
print(probability.shape)  # [2, 1]
```

### 7.3 task covariance

```python
task_covar = model.task_covar_matrix  # [m, m]

task_std = task_covar.diag().clamp_min(1e-12).sqrt()
task_corr = task_covar / task_std[:, None] / task_std[None, :]
```

`task_covar_matrix` は latent function のタスク共分散です。観測ラベルから直接計算した相関とは異なります。

### 7.4 InputTransform の注意

カテゴリ列と task-id 列を正規化・摂動しないでください。

```python
Normalize(d=3, indices=[0])  # OK
Normalize(d=3)               # NG
```

`task_feature` を `cat_dims` に重複指定することもできません。

## 8. Kronecker multi-task を使う block design

### 8.1 continuous input

```python
from bochan.models.classification.binary.base import (
    KroneckerMultiTaskBinaryClassificationGPModel,
)

model = KroneckerMultiTaskBinaryClassificationGPModel(
    train_X=train_X_block,  # [n, d]
    train_Y=train_Y_block,  # [n, m]
    rank=2,
    num_inducing_points=32,
)

mll = model.make_mll()
fit_binary_classifier_mll(
    mll,
    num_epochs=300,
    lr=0.01,
    batch_size=None,
)
```

### 8.2 mixed input

```python
from bochan.models.classification.binary.base import (
    KroneckerMultiTaskBinaryClassificationMixedGPModel,
)

model = KroneckerMultiTaskBinaryClassificationMixedGPModel(
    train_X=train_X_mixed,  # [n, d]
    train_Y=train_Y_block,  # [n, m]
    cat_dims=[2],
    rank=2,
    input_transform=Normalize(d=train_X_mixed.shape[-1], indices=[0, 1]),
    num_inducing_points=32,
)

mll = model.make_mll()
fit_binary_classifier_mll(mll, num_epochs=300, lr=0.01, batch_size=None)
```

Kronecker 版は block design 専用です。入力点ごとに観測されるタスクが異なる場合は task-feature 版を使用してください。

## 9. 新しい観測の追加

```python
updated_model = model.condition_on_observations(
    X=X_new,
    Y=Y_new,
)
```

これは厳密な closed-form conditioning ではなく、既存データと新規データから model を再構築する近似処理です。必要に応じて再学習してください。

- task-feature model: `X_new` に task-id 列を含める
- Kronecker model: `Y_new` は `[n_new, m]` で全タスクを含める

## 10. 候補点最適化

binary task-feature model では candidate tensor に task-id 列を含め、探索対象タスクを固定します。

```python
from botorch.optim import optimize_acqf_mixed
from bochan.acquisition.binary.active_learning import qBinaryPredictiveEntropy

acq_function = qBinaryPredictiveEntropy(model)

fixed_features_list = [
    {1: 1.0, 2: 0.0},  # task 1, category 0
    {1: 1.0, 2: 1.0},  # task 1, category 1
]

candidates, acq_value = optimize_acqf_mixed(
    acq_function=acq_function,
    bounds=bounds,
    q=1,
    num_restarts=10,
    raw_samples=256,
    fixed_features_list=fixed_features_list,
)
```

## 11. advanced model の学習

DeepKernel は専用 fit helper を使います。

```python
from bochan.fit import fit_deepkernel_mll
from bochan.models.classification.binary.deep import (
    DeepKernelBinaryClassificationGPModel,
)

model = DeepKernelBinaryClassificationGPModel(
    train_X=train_X,
    train_Y=train_Y,
    num_inducing_points=32,
)
fit_deepkernel_mll(model.make_mll(), num_epochs=300, lr=0.01)
```

RRP は外れラベル候補を学習点単位で疎に推定します。説明変数選択モデルではありません。

## 12. high-level API

mixed task-feature binary は `model_type="multitask"` で構築できます。

```python
from bochan.api import (
    BayesianOptimizer,
    FitConfig,
    InputTransformConfig,
    ModelConfig,
    OptimizeConfig,
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
            categorical_idx=[1, 2],  # task-idとcategoryを保護
        ),
    ),
    fit_config=FitConfig(num_epochs=300, lr=0.01),
    bounds=torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        dtype=torch.double,
    ),
)
bo.fit(train_X, train_Y)
```

候補生成では task-id を固定します。

```python
opt_config = OptimizeConfig(
    optimizer="optimize_acqf_mixed",
    q=1,
    fixed_features={1: 1.0},
    fixed_features_list=[{2: 0.0}, {2: 1.0}],
)
```

mixed Kronecker は `model_type="kronecker"` です。

```python
ModelConfig(
    task_type="binary",
    input_type="mixed",
    model_type="kronecker",
    cat_dims=[2],
    model_kwargs={
        "rank": 2,
        "num_inducing_points": 32,
    },
)
```

## 13. よくある問題

### `posterior.variance` が 0.25 付近になる

確率 0.5 付近では Bernoulli observation variance が 0.25 になります。epistemic uncertainty は `binary_probability_moments` で確認してください。

### mixed model で category / task-id が変化する

`Normalize(indices=...)` にカテゴリ列や task-id 列を含めないでください。API の `InputTransformConfig` では両方を `categorical_idx` に指定します。

### EI / PI で `best_f=1` になる

0/1 label の最大値をそのまま使うと改善余地がなくなります。binary probability scale の有限 scalar を使うか、`compute_binary_best_f` を利用してください。

### 学習が不安定

1. `train_Y` に 0 と 1 の両方が含まれる
2. dtype / device が一致している
3. continuous 列だけを Normalize している
4. learning rate を下げる
5. `num_inducing_points` を調整する
6. model を複雑にしすぎていない

## 14. 関連実装・テスト

```text
src/bochan/models/classification/binary/base/multitask.py
src/bochan/models/classification/binary/base/multitask_mixed.py
src/bochan/models/classification/binary/base/kronecker_multitask.py
src/bochan/models/classification/binary/base/kronecker_multitask_mixed.py
src/bochan/models/components/mixed_multitask.py
tests/test_mixed_task_feature_multitask_models.py
tests/test_mixed_task_feature_multitask_registry.py
tests/test_kronecker_multitask_classification_ordinal_models.py
```
