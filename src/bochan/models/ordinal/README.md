# Ordinal models

`bochan.models.ordinal` は、順序を持つカテゴリラベルをlatent Gaussian processとordered-logit likelihoodで扱うモデル群です。このREADMEでは、標準ordinal、mixed input、独立multi-output、task-id形式のmulti-task、block-designのKronecker multi-taskを説明します。

## 1. データ形式

ordinal labelは、順序を保った連続整数`0, 1, ..., K - 1`で表します。

```python
import torch

train_X = torch.rand(50, 4, dtype=torch.double)  # [n, d]
train_Y = torch.randint(0, 4, (50,), dtype=torch.long)  # [n]
```

block-design multi-taskでは、同じ入力点に対して複数のordinal labelを持ちます。

```python
train_Y = torch.randint(0, 4, (50, 3), dtype=torch.long)  # [n, m]
```

- `K`: ordinal class数。現在のordinalモデルは3クラス以上を前提
- `m`: ordinalタスク数
- label dtype: `torch.long`
- labelは0から始まる連続整数

クラス番号の差を通常の連続値として直接回帰するのではなく、latent scoreとcutpointからクラス確率を計算します。

## 2. モデル選択

| 用途 | 通常入力 | mixed input |
|---|---|---|
| 標準ordinal SVGP | `OrdinalGPModel` | `OrdinalMixedGPModel` |
| 独立multi-output | `MultiOutputOrdinalModel` | 各submodelにmixed modelを使用 |
| task-id列を使うmulti-task | `MultiTaskOrdinalGPModel` | - |
| 相関ありblock-design multi-task | `KroneckerMultiTaskOrdinalGPModel` | - |
| DeepGP | `DeepOrdinalGPModel` | `DeepOrdinalMixedGPModel` |
| DeepKernel | `DeepKernelOrdinalGPModel` | `DeepKernelOrdinalMixedGPModel` |
| 高次元SAAS | `SaasOrdinalGPModel` | `SaasOrdinalMixedGPModel` |
| PCA / REMBO | `PCAOrdinalGPModel` / `REMBOOrdinalGPModel` | 対応mixed model |
| 外れラベルRRP | `RobustRelevancePursuitOrdinalGPModel` | 対応mixed model |
| 不均一ノイズ | `HeteroscedasticOrdinalGPModel` | 対応mixed model |

各出力を独立に扱うなら`MultiOutputOrdinalModel`、同一尺度のタスク間相関を利用するならKronecker multi-taskを使用します。

## 3. 標準ordinalの最小例

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

bounds = torch.tensor(
    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    dtype=dtype,
)

model = OrdinalGPModel(
    train_X=train_X,
    train_Y=train_Y,
    num_classes=num_classes,
    input_transform=Normalize(d=train_X.shape[-1], bounds=bounds),
    num_inducing=32,
)

fit_ordinal_gp(
    model,
    num_epochs=300,
    lr=0.03,
)

X_test = torch.rand(10, 3, dtype=dtype)
latent_posterior = model.posterior(X_test)
class_probability = model.class_probs(X_test)
prediction = model.predict_class(X_test)

print(latent_posterior.mean.shape)  # [10]
print(class_probability.shape)     # [10, 4]
print(prediction.shape)            # [10]
```

`posterior()`はクラス確率ではなくlatent score `f(x)`のGaussian posteriorを返します。クラス確率には`class_probs()`を使用してください。

## 4. cutpointとクラス確率

ordered-logitでは、学習されたcutpointでlatent scoreを順序クラスへ分割します。

```python
cutpoints = model.ordinal_likelihood.cutpoints
print(cutpoints.shape)  # [K - 1]
```

クラス確率は次のshapeです。

```python
probability = model.class_probs(X_test)
print(probability.shape)          # [q, K]
print(probability.sum(dim=-1))    # approximately ones
```

予測クラスは最大確率クラスです。

```python
prediction = probability.argmax(dim=-1)
# model.predict_class(X_test) と同じ
```

## 5. expected utility

ordinal classに実用上の価値を割り当てる場合は、expected utilityを計算できます。

```python
utilities = torch.tensor([0.0, 1.0, 3.0, 6.0], dtype=dtype)
expected_utility = model.expected_utility(X_test, utilities)

print(expected_utility.shape)  # [q]
```

クラス番号そのものを期待値化する場合は、`utilities=torch.arange(K)`を使用します。ただし、ordinal class間隔が等しいとは限らないため、用途に応じたutilityを指定してください。

## 6. mixed input

カテゴリ列を同じTensorに格納し、`cat_dims`を指定します。

```python
from botorch.models.transforms.input import Normalize

from bochan.fit import fit_ordinal_gp
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
    num_inducing=32,
)
fit_ordinal_gp(model, num_epochs=300, lr=0.03)
```

カテゴリ列を`Normalize(indices=...)`に含めないでください。

## 7. 独立multi-output ordinal

同じ入力に対する複数のordinal出力を独立に学習する場合は、出力ごとにsubmodelを作り、`MultiOutputOrdinalModel`で包みます。

```python
from bochan.fit import fit_ordinal_gp
from bochan.models.ordinal.base import MultiOutputOrdinalModel, OrdinalGPModel

submodels = []
for task_index in range(train_Y.shape[-1]):
    submodel = OrdinalGPModel(
        train_X=train_X,
        train_Y=train_Y[:, task_index],
        num_classes=4,
        num_inducing=32,
    )
    fit_ordinal_gp(submodel, num_epochs=300, lr=0.03)
    submodels.append(submodel)

model = MultiOutputOrdinalModel(*submodels)
class_probability = model.class_probs(X_test)

print(class_probability.shape)  # [q, m, K] when class counts are common
```

このwrapperは各出力を独立に扱います。出力ごとにcutpointやクラス数を変えられますが、タスク間共分散は学習しません。

## 8. task-id列を使うmulti-task

各タスクの入力点が異なる場合や欠測タスクがある場合は、`MultiTaskOrdinalGPModel`を使います。task-id列を含むlong formatです。

```python
import torch

from bochan.fit import fit_ordinal_gp
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
    num_inducing=32,
)
fit_ordinal_gp(model, num_epochs=300, lr=0.03)
```

task-idは`0, ..., num_tasks - 1`の整数値にし、連続説明変数として正規化しないでください。

## 9. Kronecker multi-taskを使うblock design

すべてのordinalタスクが同じ入力点で観測され、同じordinal尺度を使う場合は、`KroneckerMultiTaskOrdinalGPModel`を使用できます。

```text
train_X: [n, d]
train_Y: [n, m]
labels: 0, ..., K - 1
```

latent scoreのタスク共分散は概念的に`K_X ⊗ K_task`です。1組のordered-logit cutpointを全タスクで共有します。

```python
import torch
from botorch.models.transforms.input import Normalize

from bochan.fit import fit_ordinal_mll
from bochan.models.ordinal.base import KroneckerMultiTaskOrdinalGPModel


torch.manual_seed(0)
dtype = torch.double
num_classes = 4
num_tasks = 3
n = 70

train_X = torch.rand(n, 3, dtype=dtype)
shared_score = 1.4 * train_X[:, 0] - 0.8 * train_X[:, 1]

task_scores = torch.stack(
    [
        shared_score,
        shared_score + 0.4 * train_X[:, 2],
        0.6 * shared_score - 0.3 * train_X[:, 2],
    ],
    dim=-1,
)

boundaries = torch.tensor([-0.2, 0.3, 0.8], dtype=dtype)
train_Y = torch.bucketize(task_scores, boundaries=boundaries).long()  # [n, 3]

bounds = torch.tensor(
    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    dtype=dtype,
)

model = KroneckerMultiTaskOrdinalGPModel(
    train_X=train_X,
    train_Y=train_Y,
    num_classes=num_classes,
    rank=2,
    input_transform=Normalize(d=train_X.shape[-1], bounds=bounds),
    num_inducing=32,
)

# block-design用ELBOを使うため、model.make_mll()を使用
mll = model.make_mll()
fit_ordinal_mll(
    mll,
    fit_model=model,
    num_epochs=300,
    lr=0.03,
    batch_size=None,
)

X_test = torch.rand(10, 3, dtype=dtype)
latent_posterior = model.posterior(X_test)
class_probability = model.class_probs(X_test)
prediction = model.predict_class(X_test)

print(latent_posterior.mean.shape)  # [10, 3] = [q, m]
print(class_probability.shape)     # [10, 3, 4] = [q, m, K]
print(prediction.shape)            # [10, 3]
```

### 共有cutpoint

```python
cutpoints = model.ordinal_likelihood.cutpoints
print(cutpoints.shape)  # [K - 1]
```

全タスクで同じ評価尺度を使う設計です。タスクごとにクラス定義やcutpointが異なる場合は、独立multi-output modelを使用してください。

### タスク共分散と相関

```python
task_covar = model.task_covar_matrix  # [m, m]

task_std = task_covar.diag().clamp_min(1e-12).sqrt()
task_corr = task_covar / task_std[:, None] / task_std[None, :]

print(task_covar.shape)  # [3, 3]
print(task_corr.shape)   # [3, 3]
```

この相関はlatent ordinal score間のモデル相関です。クラス番号のPearson相関や順位相関と同じ値になる必要はありません。

### expected utility

```python
utilities = torch.tensor([0.0, 1.0, 3.0, 6.0], dtype=dtype)
expected_utility = model.expected_utility(X_test, utilities)

print(expected_utility.shape)  # [q, m]
```

### 出力タスクの選択

```python
selected_probs = model.class_probs(X_test, output_indices=[0, 2])
selected_utility = model.expected_utility(
    X_test,
    utilities,
    output_indices=[0, 2],
)

print(selected_probs.shape)    # [q, 2, K]
print(selected_utility.shape)  # [q, 2]
```

`posterior()`は相関を保ったlatent distributionを全タスクまとめて返すため、`output_indices`には対応していません。タスク選択は`class_probs()`または`expected_utility()`で行います。

`KroneckerMultiTaskOrdinalGPModel`はblock design専用です。タスクごとに入力点が異なる場合は`MultiTaskOrdinalGPModel`を使用してください。

## 10. 新しい観測の追加

Kronecker modelでは、追加する`Y`も`[n_new, m]`で全タスクを含めます。

```python
updated_model = model.condition_on_observations(
    X=X_new,
    Y=Y_new,
    refit=True,
    num_steps=50,
    lr=0.01,
)
```

非Gaussian likelihoodのためclosed-form conditioningではなく、モデルを再構築して必要に応じて再学習します。

## 11. high-level API

標準single-output ordinalはhigh-level APIから構築できます。

```python
from bochan.api import BayesianOptimizer, FitConfig, ModelConfig

bo = BayesianOptimizer(
    model_config=ModelConfig(
        task_type="ordinal",
        model_type="base",
        model_kwargs={"num_classes": 4},
    ),
    fit_config=FitConfig(),
    bounds=bounds,
)
bo.fit(train_X, train_Y_single)
```

Kronecker multi-task専用registry keyは現在ないため、モデルを直接構築してください。

## 12. よくある問題

### `posterior.mean`がクラス番号ではない

`posterior()`はlatent scoreを返します。クラス確率は`class_probs()`、予測クラスは`predict_class()`を使用してください。

### cutpointが狭い、または一部クラスが予測されない

クラス不均衡、データ数、learning rate、`init_gap`、学習回数を確認してください。各クラスの観測が極端に少ない場合、cutpoint推定は不安定になります。

### Kronecker modelのタスク間比較が不自然

タスク間でordinal尺度やクラス定義が本当に共通か確認してください。尺度が異なる場合、cutpoint共有の前提が成立しません。

### labelが1から始まる

学習前に`0, ..., K - 1`へ変換してください。

## 13. 関連テスト

```text
tests/test_kronecker_multitask_classification_ordinal_models.py
src/bochan/models/ordinal/base/
src/bochan/models/ordinal/deep/
src/bochan/models/ordinal/high_dim/
src/bochan/models/ordinal/robust/
```
