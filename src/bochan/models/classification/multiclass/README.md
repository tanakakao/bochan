# Multiclass classification models

`bochan.models.classification.multiclass` は、3クラス以上の分類を扱う BoTorch / GPyTorch ベースのモデル群です。この README では、標準 multiclass、mixed input、独立 multi-output、task-id long-format multi-task、block-design Kronecker multi-task を説明します。

## 1. データ形式

single-output multiclass では、label を `0, 1, ..., C - 1` の整数で表します。

```python
import torch

train_X = torch.rand(50, 4, dtype=torch.double)             # [n, d]
train_Y = torch.randint(0, 3, (50,), dtype=torch.long)      # [n]
```

### task-feature long format

タスクごとに入力位置や観測数が異なる場合は task-id 列を入力に含めます。

```text
train_X: [N, d + 1]
train_Y: [N]
class labels: 0, ..., C - 1
```

### Kronecker block design

同じ入力点で全タスクが観測される場合は次の形です。

```text
train_X: [n, d]
train_Y: [n, m]
class labels: 0, ..., C - 1
```

- `C`: クラス数
- `m`: multiclass task 数
- label dtype: `torch.long`
- model、`train_X`、bounds は同じ floating dtype / device

## 2. モデル選択

| 用途 | 通常入力 | mixed input |
|---|---|---|
| 標準 SVGP | `MulticlassClassificationGPModel` | `MulticlassClassificationMixedGPModel` |
| 独立 multi-output | `MultiOutputMulticlassClassificationModel` | 各 submodel に mixed model を使用 |
| task-id long-format multi-task | `MultiTaskMulticlassClassificationGPModel` | `MultiTaskMulticlassClassificationMixedGPModel` |
| block-design Kronecker multi-task | `KroneckerMultiTaskMulticlassClassificationGPModel` | `KroneckerMultiTaskMulticlassClassificationMixedGPModel` |
| DeepGP | `MulticlassDeepGPModel` | `MulticlassMixedDeepGPModel` |
| DeepKernel | `DeepKernelMulticlassClassificationGPModel` | `DeepKernelMulticlassClassificationMixedGPModel` |
| 高次元 SAAS | `SaasMulticlassClassificationGPModel` | `SaasMulticlassClassificationMixedGPModel` |
| PCA / REMBO | `PCAMulticlassClassificationGPModel` / `REMBOMulticlassClassificationGPModel` | 対応 mixed model |
| 外れラベル RRP | `OutlierRelevancePursuitMulticlassClassificationGPModel` | 対応 mixed model |
| 不均一ノイズ | `HeteroscedasticMulticlassClassificationGPModel` | 対応 mixed model |

使い分け:

- タスクを独立に扱う: independent multi-output
- タスク相関を使い、入力位置・観測数がタスクごとに異なる: task-feature multi-task
- タスク相関を使い、全タスクが同じ入力点にある: Kronecker multi-task

## 3. 標準 multiclass の最小例

```python
import torch
from botorch.models.transforms.input import Normalize

from bochan.fit import fit_multiclass_gp
from bochan.models.classification.multiclass.base import (
    MulticlassClassificationGPModel,
)


torch.manual_seed(0)
dtype = torch.double
num_classes = 3

train_X = torch.rand(60, 3, dtype=dtype)
score = torch.stack(
    [
        1.2 * train_X[:, 0] - train_X[:, 1],
        train_X[:, 1] + 0.5 * train_X[:, 2],
        -train_X[:, 0] + train_X[:, 2],
    ],
    dim=-1,
)
train_Y = score.argmax(dim=-1).long()

model = MulticlassClassificationGPModel(
    train_X=train_X,
    train_Y=train_Y,
    num_classes=num_classes,
    input_transform=Normalize(d=train_X.shape[-1]),
    num_inducing_points=32,
)
fit_multiclass_gp(model, num_epochs=300, lr=0.01)

X_test = torch.rand(10, 3, dtype=dtype)
posterior = model.posterior(X_test)
probability = posterior.mean
prediction = model.predict_class(X_test)

print(probability.shape)  # [10, 3]
print(prediction.shape)   # [10]
```

`posterior.mean[..., c]` はクラス `c` の予測確率です。最後の次元の和は1になります。

## 4. mixed input

```python
from bochan.models.classification.multiclass.base import (
    MulticlassClassificationMixedGPModel,
)

continuous_X = torch.rand(60, 3, dtype=torch.double)
category = torch.randint(0, 4, (60, 1)).to(torch.double)
train_X = torch.cat([continuous_X, category], dim=-1)

model = MulticlassClassificationMixedGPModel(
    train_X=train_X,
    train_Y=train_Y,
    cat_dims=[3],
    num_classes=3,
    input_transform=Normalize(
        d=train_X.shape[-1],
        indices=[0, 1, 2],
    ),
    num_inducing_points=32,
)
fit_multiclass_gp(model, num_epochs=300, lr=0.01)
```

`Normalize(indices=...)` にカテゴリ列を含めないでください。

## 5. posterior と不確かさ

### probability posterior

```python
posterior = model.posterior(X_test)
probability = posterior.mean
bernoulli_like_variance = posterior.variance
```

`posterior.variance` は各クラス確率について `p * (1 - p)` を返します。クラス間共分散や epistemic uncertainty そのものではありません。

### latent posterior

```python
latent = model.latent_posterior(X_test)
latent_mean = latent.mean
latent_variance = latent.variance
```

latent 値は softmax 適用前の logit です。

### posterior sampling

```python
samples = posterior.rsample(torch.Size([128]))
print(samples.shape)  # [128, q, C]
```

active learning で epistemic uncertainty を評価する場合は、latent posterior または posterior sample 間のばらつきを利用します。

## 6. 独立 multi-output multiclass

```python
from bochan.models.classification.multiclass.base import (
    MultiOutputMulticlassClassificationModel,
    MulticlassClassificationGPModel,
)

submodels = []
for task_index in range(train_Y_multi.shape[-1]):
    submodel = MulticlassClassificationGPModel(
        train_X=train_X,
        train_Y=train_Y_multi[:, task_index],
        num_classes=3,
        num_inducing_points=32,
    )
    fit_multiclass_gp(submodel, num_epochs=300, lr=0.01)
    submodels.append(submodel)

model = MultiOutputMulticlassClassificationModel(*submodels)
probability = model.class_probs(X_test)
print(probability.shape)  # [q, m, C]
```

出力ごとにクラス数が異なる場合は `class_probs_list()` または `padded_class_probs()` を使用します。この wrapper は出力間相関を学習しません。

## 7. task-id 列を使う multi-task

### 7.1 continuous input

`MultiTaskMulticlassClassificationGPModel` は、今回追加された通常入力の task-feature multiclass model です。

```python
from bochan.models.classification.multiclass.base import (
    MultiTaskMulticlassClassificationGPModel,
)

X_data = torch.rand(80, 3, dtype=torch.double)
task_id = torch.randint(0, 2, (80, 1)).to(torch.double)
train_X = torch.cat([X_data, task_id], dim=-1)
train_Y = torch.randint(0, 3, (80,), dtype=torch.long)

model = MultiTaskMulticlassClassificationGPModel(
    train_X=train_X,
    train_Y=train_Y,
    num_classes=3,
    num_tasks=2,
    task_feature=-1,
    rank=2,
    num_inducing_points=32,
)
fit_multiclass_gp(model, num_epochs=300, lr=0.01)
```

### 7.2 mixed input

`MultiTaskMulticlassClassificationMixedGPModel` は、continuous・category・task-id を含む long-format multiclass model です。

```python
from bochan.models.classification.multiclass.base import (
    MultiTaskMulticlassClassificationMixedGPModel,
)

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
train_Y = torch.tensor([0, 1, 2, 0, 2, 1], dtype=torch.long)

model = MultiTaskMulticlassClassificationMixedGPModel(
    train_X=train_X,
    train_Y=train_Y,
    cat_dims=[2],
    num_classes=3,
    num_tasks=2,
    task_feature=1,
    rank=2,
    input_transform=Normalize(d=3, indices=[0]),
    num_inducing_points=32,
)
fit_multiclass_gp(model, num_epochs=300, lr=0.01)
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
prediction = model.predict_class(X_test)

print(probability.shape)  # [2, C]
print(prediction.shape)   # [2]
```

### 7.3 クラスごとの task covariance

multiclass task-feature model は、クラス logit ごとに task covariance を持ちます。

```python
task_covar = model.task_covar_matrix  # [C, m, m]

task_var = task_covar.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12)
task_std = task_var.sqrt()
task_corr = (
    task_covar
    / task_std.unsqueeze(-1)
    / task_std.unsqueeze(-2)
)

print(task_covar.shape)  # [3, 2, 2]
```

`task_corr[c]` はクラス `c` の latent logit に関するタスク相関です。観測 label の相関や混同行列とは異なります。

### 7.4 共通クラス集合

全タスクが同じ `num_classes` とクラス定義を使う必要があります。タスクごとにクラス数が異なる場合は independent multi-output wrapper を使用してください。

### 7.5 InputTransform の注意

カテゴリ列と task-id 列を正規化・摂動しないでください。

```python
Normalize(d=3, indices=[0])  # OK
Normalize(d=3)               # NG
```

## 8. Kronecker multi-task を使う block design

### 8.1 continuous input

```python
from bochan.models.classification.multiclass.base import (
    KroneckerMultiTaskMulticlassClassificationGPModel,
)

model = KroneckerMultiTaskMulticlassClassificationGPModel(
    train_X=train_X_block,  # [n, d]
    train_Y=train_Y_block,  # [n, m]
    num_classes=3,
    rank=2,
    num_inducing_points=32,
)
fit_multiclass_gp(model, num_epochs=300, lr=0.01, batch_size=None)
```

### 8.2 mixed input

```python
from bochan.models.classification.multiclass.base import (
    KroneckerMultiTaskMulticlassClassificationMixedGPModel,
)

model = KroneckerMultiTaskMulticlassClassificationMixedGPModel(
    train_X=train_X_mixed,  # [n, d]
    train_Y=train_Y_block,  # [n, m]
    cat_dims=[2],
    num_classes=3,
    rank=2,
    input_transform=Normalize(d=train_X_mixed.shape[-1], indices=[0, 1]),
    num_inducing_points=32,
)
fit_multiclass_gp(model, num_epochs=300, lr=0.01, batch_size=None)
```

Kronecker 版もクラスごとに `[C, m, m]` の task covariance を持ちます。block design 専用です。

## 9. 新しい観測の追加

```python
updated_model = model.condition_on_observations(
    X=X_new,
    Y=Y_new,
)
```

closed-form conditioning ではなく、学習データを追加して variational model を再構築します。

- task-feature model: `X_new` に task-id 列を含める
- Kronecker model: `Y_new` は `[n_new, m]` で全タスクを含める

## 10. 候補点最適化

multiclass task-feature model では candidate tensor に task-id 列を含め、探索対象タスクを固定します。

```python
from botorch.optim import optimize_acqf_mixed

fixed_features_list = [
    {1: 1.0, 2: 0.0},
    {1: 1.0, 2: 1.0},
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

multiclass acquisition では `target_class`、`threshold`、`best_f` などを acquisition class に応じて指定してください。

## 11. high-level API

mixed task-feature multiclass は `model_type="multitask"` で構築できます。

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
        task_type="multiclass",
        input_type="mixed",
        model_type="multitask",
        cat_dims=[2],
        model_kwargs={
            "num_classes": 3,
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
    task_type="multiclass",
    input_type="mixed",
    model_type="kronecker",
    cat_dims=[2],
    model_kwargs={
        "num_classes": 3,
        "rank": 2,
    },
)
```

## 12. よくある問題

### 確率の shape が想定と異なる

- single-output: `[q, C]`
- task-feature model: candidate ごとに1つのtask-idを持つため `[q, C]`
- independent / Kronecker multi-output: `[q, m, C]`
- Kronecker latent mean: `[C, q, m]`

### すべての確率がほぼ一様になる

学習初期は `1 / C` 付近になりやすいため、loss、クラス数、label dtype、learning rate、inducing point 数を確認してください。

### mixed model で category / task-id が変化する

`Normalize(indices=...)` にカテゴリ列や task-id 列を含めないでください。API の `InputTransformConfig` では両方を `categorical_idx` に指定します。

### task covariance が不安定

データ数に対して `rank` が高すぎる可能性があります。`rank=1`、`rank=2`、full rank を比較してください。

## 13. 関連実装・テスト

```text
src/bochan/models/classification/multiclass/base/multitask.py
src/bochan/models/classification/multiclass/base/kronecker_multitask.py
src/bochan/models/classification/multiclass/base/kronecker_multitask_mixed.py
src/bochan/models/components/mixed_multitask.py
tests/test_mixed_task_feature_multitask_models.py
tests/test_mixed_task_feature_multitask_registry.py
tests/test_kronecker_multitask_multiclass_model.py
```
