# Ordinal models

`bochan.models.ordinal` は、順序を持つカテゴリラベルを latent Gaussian process と ordered-logit likelihood で扱うモデル群です。この README では、標準 ordinal、mixed input、独立 multi-output、task-id long-format multi-task、block-design Kronecker multi-task を説明します。

## 1. データ形式

ordinal label は、順序を保った連続整数 `0, 1, ..., K - 1` で表します。

```python
import torch

train_X = torch.rand(50, 4, dtype=torch.double)         # [n, d]
train_Y = torch.randint(0, 4, (50,), dtype=torch.long) # [n]
```

### task-feature long format

タスクごとに入力点や観測数が異なる場合は、task-id 列を入力に含めます。

```text
train_X: [N, d + 1]
train_Y: [N]
```

### Kronecker block design

同じ入力点で全タスクが観測される場合は次の形です。

```text
train_X: [n, d]
train_Y: [n, m]
```

- `K`: ordinal class 数。現在の ordinal model は 3 クラス以上を前提
- `m`: ordinal task 数
- label dtype: `torch.long`
- label は 0 から始まる連続整数

クラス番号の差を連続値として直接回帰するのではなく、latent score と cutpoint からクラス確率を計算します。

## 2. モデル選択

| 用途 | 通常入力 | mixed input |
|---|---|---|
| 標準 ordinal SVGP | `OrdinalGPModel` | `OrdinalMixedGPModel` |
| 独立 multi-output | `MultiOutputOrdinalModel` | 各 submodel に mixed model を使用 |
| task-id long-format multi-task | `MultiTaskOrdinalGPModel` | `MultiTaskOrdinalMixedGPModel` |
| block-design Kronecker multi-task | `KroneckerMultiTaskOrdinalGPModel` | `KroneckerMultiTaskOrdinalMixedGPModel` |
| DeepGP | `OrdinalDeepGPModel` | `OrdinalMixedDeepGPModel` |
| DeepKernel | `DeepKernelOrdinalGPModel` | `DeepKernelOrdinalMixedGPModel` |
| 高次元 SAAS | `SaasOrdinalGPModel` | `SaasOrdinalMixedGPModel` |
| PCA / REMBO | `PCAOrdinalGPModel` / `REMBOOrdinalGPModel` | 対応 mixed model |
| 外れラベル RRP | `OutlierRelevancePursuitOrdinalGPModel` | 対応 mixed model |
| 不均一ノイズ | `HeteroscedasticOrdinalGPModel` | 対応 mixed model |

使い分け:

- 出力ごとにクラス定義や cutpoint が異なる: independent multi-output
- 共通尺度で、タスクごとに入力位置や観測数が異なる: task-feature multi-task
- 共通尺度で、全タスクが同じ入力点にある: Kronecker multi-task

## 3. 標準 ordinal の最小例

```python
import torch
from botorch.models.transforms.input import Normalize

from bochan.fit import fit_ordinal_gp
from bochan.models.ordinal.base import OrdinalGPModel


torch.manual_seed(0)
dtype = torch.double
num_classes = 4

train_X = torch.rand(60, 3, dtype=dtype)
latent_score = (
    1.5 * train_X[:, 0]
    - 0.8 * train_X[:, 1]
    + 0.5 * train_X[:, 2]
)
train_Y = torch.bucketize(
    latent_score,
    boundaries=torch.tensor([-0.2, 0.3, 0.8], dtype=dtype),
).long()

model = OrdinalGPModel(
    train_X=train_X,
    train_Y=train_Y,
    num_classes=num_classes,
    input_transform=Normalize(d=train_X.shape[-1]),
    inducing_points_num=32,
)
fit_ordinal_gp(model, num_epochs=300, lr=0.03)

X_test = torch.rand(10, 3, dtype=dtype)
latent_posterior = model.posterior(X_test)
class_probability = model.class_probs(X_test)
prediction = model.predict_class(X_test)

print(latent_posterior.mean.shape)  # [10]
print(class_probability.shape)     # [10, 4]
print(prediction.shape)            # [10]
```

`posterior()` はクラス確率ではなく latent score `f(x)` の Gaussian posterior を返します。クラス確率には `class_probs()` を使用してください。

## 4. cutpoint・クラス確率・expected utility

```python
cutpoints = model.ordinal_likelihood.cutpoints
print(cutpoints.shape)  # [K - 1]

probability = model.class_probs(X_test)
print(probability.shape)        # [q, K]
print(probability.sum(dim=-1))  # approximately ones
```

ordinal class に実用上の価値を割り当てる場合は expected utility を計算できます。

```python
utilities = torch.tensor([0.0, 1.0, 3.0, 6.0], dtype=dtype)
expected_utility = model.expected_utility(X_test, utilities)
print(expected_utility.shape)  # [q]
```

クラス番号そのものを期待値化する場合は `utilities=torch.arange(K)` を使用します。ただし ordinal class 間隔が等しいとは限らないため、用途に応じた utility を指定してください。

## 5. mixed input

```python
from botorch.models.transforms.input import Normalize
from bochan.models.ordinal.base import OrdinalMixedGPModel

continuous_X = torch.rand(60, 3, dtype=torch.double)
category = torch.randint(0, 4, (60, 1)).to(torch.double)
train_X = torch.cat([continuous_X, category], dim=-1)

model = OrdinalMixedGPModel(
    train_X=train_X,
    train_Y=train_Y,
    cat_dims=[3],
    num_classes=4,
    input_transform=Normalize(
        d=train_X.shape[-1],
        indices=[0, 1, 2],
    ),
    inducing_points_num=32,
)
fit_ordinal_gp(model, num_epochs=300, lr=0.03)
```

カテゴリ列を `Normalize(indices=...)` に含めないでください。

## 6. 独立 multi-output ordinal

同じ入力に対する複数の ordinal 出力を独立に学習する場合は、出力ごとに submodel を作り、`MultiOutputOrdinalModel` で包みます。

```python
from bochan.models.ordinal.base import MultiOutputOrdinalModel, OrdinalGPModel

submodels = []
for task_index in range(train_Y_multi.shape[-1]):
    submodel = OrdinalGPModel(
        train_X=train_X,
        train_Y=train_Y_multi[:, task_index],
        num_classes=4,
        inducing_points_num=32,
    )
    fit_ordinal_gp(submodel, num_epochs=300, lr=0.03)
    submodels.append(submodel)

model = MultiOutputOrdinalModel(*submodels)
probability = model.class_probs(X_test)
print(probability.shape)  # [q, m, K] when class counts are common
```

この wrapper はタスク間共分散を学習しません。出力ごとにクラス数や cutpoint を変えられます。

## 7. task-id 列を使う multi-task

### 7.1 continuous input

```python
from bochan.models.ordinal.base import MultiTaskOrdinalGPModel

X_data = torch.rand(70, 3, dtype=torch.double)
task_id = torch.randint(0, 3, (70, 1)).to(torch.double)
train_X = torch.cat([X_data, task_id], dim=-1)
train_Y = torch.randint(0, 4, (70,), dtype=torch.long)

model = MultiTaskOrdinalGPModel(
    train_X=train_X,
    train_Y=train_Y,
    num_classes=4,
    num_tasks=3,
    task_feature=-1,
    rank=2,
    inducing_points_num=32,
)
fit_ordinal_gp(model, num_epochs=300, lr=0.03)
```

### 7.2 mixed input

`MultiTaskOrdinalMixedGPModel` は、連続列・カテゴリ列・task-id 列を含む long format を扱います。タスクごとに異なる入力位置と欠測を許容します。

```python
from botorch.models.transforms.input import Normalize
from bochan.models.ordinal.base import MultiTaskOrdinalMixedGPModel

# columns: continuous, task_id, category
train_X = torch.tensor(
    [
        [0.05, 0.0, 0.0],
        [0.20, 0.0, 1.0],
        [0.65, 0.0, 0.0],
        [0.10, 1.0, 1.0],
        [0.45, 1.0, 0.0],
        [0.90, 1.0, 1.0],
    ],
    dtype=torch.double,
)
train_Y = torch.tensor([0, 1, 3, 0, 2, 3], dtype=torch.long)

model = MultiTaskOrdinalMixedGPModel(
    train_X=train_X,
    train_Y=train_Y,
    cat_dims=[2],
    num_classes=4,
    num_tasks=2,
    task_feature=1,
    rank=2,
    input_transform=Normalize(d=3, indices=[0]),
    inducing_points_num=32,
)
fit_ordinal_gp(model, num_epochs=300, lr=0.03)
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

probability = model.class_probs(X_test)
expected_utility = model.expected_utility(X_test, utilities)

print(probability.shape)       # [2, K]
print(expected_utility.shape)  # [2]
```

### 7.3 共通尺度の前提

`MultiTaskOrdinalGPModel` と `MultiTaskOrdinalMixedGPModel` は全タスクで次を共有します。

- `num_classes`
- ordinal class の意味
- ordered-logit cutpoint
- utility の解釈

タスクごとに尺度が異なる場合は independent multi-output を使用してください。

### 7.4 task covariance

```python
task_covar = model.task_covar_matrix  # [m, m]

task_std = task_covar.diag().clamp_min(1e-12).sqrt()
task_corr = task_covar / task_std[:, None] / task_std[None, :]
```

これは latent score のタスク共分散です。観測ラベルから直接計算した相関とは異なります。

## 8. Kronecker multi-task を使う block design

### 8.1 continuous input

```python
from bochan.fit import fit_ordinal_mll
from bochan.models.ordinal.base import KroneckerMultiTaskOrdinalGPModel

model = KroneckerMultiTaskOrdinalGPModel(
    train_X=train_X_block,  # [n, d]
    train_Y=train_Y_block,  # [n, m]
    num_classes=4,
    rank=2,
    inducing_points_num=32,
)

mll = model.make_mll()
fit_ordinal_mll(
    mll,
    fit_model=model,
    num_epochs=300,
    lr=0.03,
    batch_size=None,
)
```

### 8.2 mixed input

```python
from bochan.models.ordinal.base import KroneckerMultiTaskOrdinalMixedGPModel

model = KroneckerMultiTaskOrdinalMixedGPModel(
    train_X=train_X_mixed,  # [n, d]
    train_Y=train_Y_block,  # [n, m]
    cat_dims=[2],
    num_classes=4,
    rank=2,
)
```

Kronecker 版も全タスクで共通の cutpoint を使用します。block design 専用であり、タスクごとに入力点が異なる場合は task-feature 版を使用してください。

## 9. 新しい観測の追加

```python
updated_model = model.condition_on_observations(
    X=X_new,
    Y=Y_new,
    refit=True,
    num_steps=50,
    lr=0.01,
)
```

非 Gaussian likelihood のため closed-form conditioning ではなく、モデルを再構築して必要に応じて再学習します。

- task-feature model: `X_new` に task-id 列を含める
- Kronecker model: `Y_new` は `[n_new, m]` で全タスクを含める

## 10. 候補点最適化

classification / ordinal の task-feature model では、candidate tensor 自体に task-id 列を含めます。探索対象タスクを固定し、通常カテゴリを列挙します。

```python
from botorch.optim import optimize_acqf_mixed

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

## 11. high-level API

mixed task-feature ordinal は `model_type="multitask"` で構築できます。

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
        task_type="ordinal",
        input_type="mixed",
        model_type="multitask",
        cat_dims=[2],
        model_kwargs={
            "num_classes": 4,
            "num_tasks": 2,
            "task_feature": 1,
            "rank": 2,
        },
        input_transform_config=InputTransformConfig(
            normalize=True,
            categorical_idx=[1, 2],  # task-idとcategoryを保護
        ),
    ),
    fit_config=FitConfig(num_epochs=300, lr=0.03),
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

mixed block-design Kronecker は `model_type="kronecker"` です。

```python
ModelConfig(
    task_type="ordinal",
    input_type="mixed",
    model_type="kronecker",
    cat_dims=[2],
    model_kwargs={
        "num_classes": 4,
        "rank": 2,
    },
)
```

## 12. よくある問題

### `posterior.mean` がクラス番号ではない

`posterior()` は latent score を返します。クラス確率は `class_probs()`、予測クラスは `predict_class()` を使用してください。

### cutpoint が狭い、または一部クラスが予測されない

クラス不均衡、データ数、learning rate、`init_gap`、学習回数を確認してください。

### task-feature mixed model で category / task-id が変化する

`Normalize(indices=...)` にカテゴリ列や task-id 列を含めないでください。`InputTransformConfig` では両方を `categorical_idx` に指定します。

### task covariance が不自然

タスク間で ordinal 尺度やクラス定義が本当に共通か確認してください。尺度が異なる場合、cutpoint 共有の前提が成立しません。

### label が 1 から始まる

学習前に `0, ..., K - 1` へ変換してください。

## 13. 関連実装・テスト

```text
src/bochan/models/ordinal/base/multitask.py
src/bochan/models/ordinal/base/multitask_mixed.py
src/bochan/models/ordinal/base/kronecker_multitask.py
src/bochan/models/ordinal/base/kronecker_multitask_mixed.py
src/bochan/models/components/mixed_multitask.py
tests/test_mixed_task_feature_multitask_models.py
tests/test_mixed_task_feature_multitask_registry.py
tests/test_kronecker_multitask_classification_ordinal_models.py
```
