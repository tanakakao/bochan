# Multiclass classification models

`bochan.models.classification.multiclass` は、3クラス以上の分類を扱うBoTorch / GPyTorchベースのモデル群です。このREADMEでは、標準multiclass、mixed input、独立multi-output、block-designのKronecker multi-taskを説明します。

## 1. データ形式

single-output multiclassでは、ラベルを`0, 1, ..., C - 1`の整数で表します。

```python
import torch

train_X = torch.rand(50, 4, dtype=torch.double)  # [n, d]
train_Y = torch.randint(0, 3, (50,), dtype=torch.long)  # [n]
```

複数のmulticlassタスクを同じ入力点で観測しているblock designでは、`train_Y.shape == [n, m]`です。

```python
train_Y = torch.randint(0, 3, (50, 2), dtype=torch.long)  # [n, m]
```

- `C`: クラス数
- `m`: multiclassタスク数
- label dtype: `torch.long`
- モデルと`train_X`、boundsは同じfloating dtype / device

## 2. モデル選択

| 用途 | 通常入力 | mixed input |
|---|---|---|
| 標準SVGP | `MulticlassClassificationGPModel` | `MulticlassClassificationMixedGPModel` |
| 独立multi-output | `MultiOutputMulticlassClassificationModel` | 各submodelにmixed modelを使用 |
| 相関ありblock-design multi-task | `KroneckerMultiTaskMulticlassClassificationGPModel` | - |
| DeepGP | `MulticlassDeepGPModel` | `MulticlassMixedDeepGPModel` |
| DeepKernel | `DeepKernelMulticlassClassificationGPModel` | `DeepKernelMulticlassClassificationMixedGPModel` |
| 高次元SAAS | `SaasMulticlassClassificationGPModel` | `SaasMulticlassClassificationMixedGPModel` |
| PCA / REMBO | `PCAMulticlassClassificationGPModel` / `REMBOMulticlassClassificationGPModel` | 対応mixed model |
| 外れラベルRRP | `OutlierRelevancePursuitMulticlassClassificationGPModel` | 対応mixed model |
| 不均一ノイズ | `HeteroscedasticMulticlassClassificationGPModel` | 対応mixed model |

出力間を独立に扱うならmulti-output wrapper、出力間相関も学習するならKronecker multi-taskを使用します。

## 3. 標準multiclassの最小例

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

bounds = torch.tensor(
    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    dtype=dtype,
)

model = MulticlassClassificationGPModel(
    train_X=train_X,
    train_Y=train_Y,
    num_classes=num_classes,
    input_transform=Normalize(d=train_X.shape[-1], bounds=bounds),
    num_inducing_points=32,
)

fit_result = fit_multiclass_gp(
    model,
    num_epochs=300,
    lr=0.01,
)

X_test = torch.rand(10, 3, dtype=dtype)
posterior = model.posterior(X_test)
probability = posterior.mean
prediction = model.predict_class(X_test)

print(probability.shape)  # [10, 3]
print(prediction.shape)   # [10]
```

`posterior.mean[..., c]`はクラス`c`の予測確率です。最後の次元の和は1になります。

## 4. mixed input

カテゴリ列を同じTensorへ格納し、`cat_dims`で列番号を指定します。

```python
from botorch.models.transforms.input import Normalize

from bochan.fit import fit_multiclass_gp
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

`Normalize(indices=...)`にカテゴリ列を含めないでください。

## 5. posteriorと不確かさ

### probability posterior

```python
posterior = model.posterior(X_test)
probability = posterior.mean
bernoulli_like_variance = posterior.variance
```

`posterior.variance`は各クラス確率について`p * (1 - p)`を返します。クラス間共分散やepistemic uncertaintyそのものではありません。

### latent posterior

```python
latent = model.latent_posterior(X_test)
latent_mean = latent.mean
latent_variance = latent.variance
```

latent値はsoftmax適用前のlogitです。確率ではないため、0から1の範囲には限定されません。

### posterior sampling

```python
samples = posterior.rsample(torch.Size([128]))
print(samples.shape)  # [128, q, C]
```

active learningでepistemic uncertaintyを評価する場合は、latent posteriorまたはposterior sample間のばらつきを利用します。

## 6. 独立multi-output multiclass

同じ入力に対して複数の独立したmulticlass出力を予測する場合は、出力ごとにsubmodelを学習してwrapperで包みます。

```python
from bochan.fit import fit_multiclass_gp
from bochan.models.classification.multiclass.base import (
    MultiOutputMulticlassClassificationModel,
    MulticlassClassificationGPModel,
)

submodels = []
for task_index in range(train_Y.shape[-1]):
    submodel = MulticlassClassificationGPModel(
        train_X=train_X,
        train_Y=train_Y[:, task_index],
        num_classes=3,
        num_inducing_points=32,
    )
    fit_multiclass_gp(submodel, num_epochs=300, lr=0.01)
    submodels.append(submodel)

model = MultiOutputMulticlassClassificationModel(*submodels)
probability = model.class_probs(X_test)

print(probability.shape)  # [q, m, C]
```

出力ごとにクラス数が異なる場合は、`class_probs_list()`または`padded_class_probs()`を使用します。このwrapperでは出力間相関を学習しません。

## 7. Kronecker multi-taskを使うblock design

すべてのmulticlassタスクが同じ入力点で観測され、同じクラス集合を使う場合は、`KroneckerMultiTaskMulticlassClassificationGPModel`を使用できます。

```text
train_X: [n, d]
train_Y: [n, m]
class labels: 0, ..., C - 1
```

各クラスlogitに対してタスク間ICM共分散を持ち、概念的にはクラス`c`ごとに`K_X,c ⊗ K_task,c`を学習します。

```python
import torch
from botorch.models.transforms.input import Normalize

from bochan.fit import fit_multiclass_gp
from bochan.models.classification.multiclass.base import (
    KroneckerMultiTaskMulticlassClassificationGPModel,
)


torch.manual_seed(0)
dtype = torch.double
num_classes = 3
num_tasks = 2
n = 70

train_X = torch.rand(n, 3, dtype=dtype)

logits_task0 = torch.stack(
    [
        1.2 * train_X[:, 0] - train_X[:, 1],
        train_X[:, 1] + 0.4 * train_X[:, 2],
        -train_X[:, 0] + train_X[:, 2],
    ],
    dim=-1,
)
logits_task1 = logits_task0 + torch.stack(
    [
        0.2 * train_X[:, 2],
        -0.1 * train_X[:, 0],
        0.3 * train_X[:, 1],
    ],
    dim=-1,
)

train_Y = torch.stack(
    [
        logits_task0.argmax(dim=-1),
        logits_task1.argmax(dim=-1),
    ],
    dim=-1,
).long()  # [n, 2]

bounds = torch.tensor(
    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    dtype=dtype,
)

model = KroneckerMultiTaskMulticlassClassificationGPModel(
    train_X=train_X,
    train_Y=train_Y,
    num_classes=num_classes,
    rank=2,
    input_transform=Normalize(d=train_X.shape[-1], bounds=bounds),
    num_inducing_points=32,
)

# model.make_mll()がblock-design用ELBOを返す
fit_result = fit_multiclass_gp(
    model,
    num_epochs=300,
    lr=0.01,
    batch_size=None,
)

X_test = torch.rand(10, 3, dtype=dtype)
posterior = model.posterior(X_test)
probability = posterior.mean
prediction = model.predict_class(X_test)

print(probability.shape)  # [10, 2, 3] = [q, m, C]
print(prediction.shape)   # [10, 2]
```

### latent posterior

```python
latent = model.latent_posterior(X_test)
print(latent.mean.shape)  # [C, q, m] = [3, 10, 2]
```

クラスごとにタスク相関を持つため、latent posteriorの先頭batch次元がクラスです。

### クラスごとのタスク共分散

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
print(task_corr.shape)   # [3, 2, 2]
```

`task_corr[c]`はクラス`c`のlatent logitについてのタスク相関です。観測ラベルから直接計算した相関や混同行列とは異なります。

### 出力タスクの選択

```python
selected = model.posterior(X_test, output_indices=[1])
print(selected.mean.shape)  # [10, 1, 3]
```

`latent_posterior()`は相関構造を保つため全タスクをまとめて返します。タスク選択は`posterior()`または`class_probs()`で行います。

### 共通クラス集合の制約

Kronecker multiclassでは全タスクが同じ`num_classes`とクラス定義を使う必要があります。タスクごとにクラス数が異なる場合は、独立multi-output wrapperを使用してください。

## 8. 新しい観測の追加

```python
updated_model = model.condition_on_observations(
    X=X_new,
    Y=Y_new,  # [n_new, m]
)
```

これはclosed-form conditioningではなく、学習データを追加してvariational modelを再構築します。追加後は必要に応じて再学習してください。

## 9. high-level API

標準single-output multiclassはhigh-level APIから構築できます。

```python
from bochan.api import BayesianOptimizer, FitConfig, ModelConfig

bo = BayesianOptimizer(
    model_config=ModelConfig(
        task_type="multiclass",
        model_type="base",
        model_kwargs={"num_classes": 3},
    ),
    fit_config=FitConfig(),
    bounds=bounds,
)
bo.fit(train_X, train_Y_single)
```

Kronecker multi-task専用registry keyは現在ないため、モデルを直接構築してください。

## 10. よくある問題

### 確率のshapeが想定と異なる

- single-output: `[q, C]`
- independent / Kronecker multi-output: `[q, m, C]`
- Kronecker latent mean: `[C, q, m]`

### すべての確率がほぼ一様になる

学習初期は`1 / C`付近になりやすいため、lossの低下、クラス数、label dtype、learning rate、inducing point数を確認してください。

### タスク共分散が不安定

データ数に対して`rank`が高すぎる可能性があります。`rank=1`または`rank=2`とfull rankを比較してください。

## 11. 関連テスト

```text
tests/test_kronecker_multitask_multiclass_model.py
src/bochan/models/classification/multiclass/base/
src/bochan/models/classification/multiclass/deep/
src/bochan/models/classification/multiclass/high_dim/
src/bochan/models/classification/multiclass/robust/
```
