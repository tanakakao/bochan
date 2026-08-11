# Binary classification models

`bochan.models.classification.binary` は、BoTorch / GPyTorch ベースの2値分類モデル群です。このREADMEでは、モデルの選択、学習、予測、mixed input、multi-output、multi-task、不確かさの扱いを説明します。

## 1. データ形式

通常入力は `train_X.shape == [n, d]`、binary labelは0/1です。

```python
import torch

train_X = torch.rand(40, 5, dtype=torch.double)
train_Y = torch.randint(0, 2, (40, 1)).to(torch.double)
```

single-outputの`train_Y`は`[n]`と`[n, 1]`の両方を受け付けますが、例では`[n, 1]`を推奨します。モデル、学習データ、boundsは同じdtype / deviceに揃えてください。

mixed modelではカテゴリ列を同じTensorに格納し、`cat_dims`で列番号を指定します。

```python
continuous_X = torch.rand(40, 4, dtype=torch.double)
category = torch.randint(0, 3, (40, 1)).to(torch.double)
train_X = torch.cat([continuous_X, category], dim=-1)
cat_dims = [4]
```

## 2. モデル選択

| 用途 | 通常入力 | mixed input |
|---|---|---|
| 標準SVGP | `BinaryClassificationGPModel` | `BinaryClassificationMixedGPModel` |
| 相関ありblock-design multi-task | `KroneckerMultiTaskBinaryClassificationGPModel` | - |
| DeepGP | `DeepBinaryClassificationGPModel` | `DeepBinaryClassificationMixedGPModel` |
| DeepKernel | `DeepKernelBinaryClassificationGPModel` | `DeepKernelBinaryClassificationMixedGPModel` |
| DeepKernel + DeepGP | `DeepKernelDeepBinaryClassificationGPModel` | `DeepKernelDeepBinaryClassificationMixedGPModel` |
| 高次元SAAS | `SaasBinaryClassificationGPModel` | `SaasBinaryClassificationMixedGPModel` |
| PCA | `PCABinaryClassificationGPModel` | `PCABinaryClassificationMixedGPModel` |
| REMBO | `REMBOBinaryClassificationGPModel` | `REMBOBinaryClassificationMixedGPModel` |
| 外れラベルRRP | `RobustRelevancePursuitBinaryClassificationGPModel` | `RobustRelevancePursuitBinaryClassificationMixedGPModel` |
| 不均一ノイズ | `HeteroscedasticBinaryClassificationGPModel` | `HeteroscedasticBinaryClassificationMixedGPModel` |

基本方針は、まずbase modelを試し、不要変数が多い高次元問題ではSAAS、固定kernelでは表現力が不足する場合はDeepKernel / DeepGP、外れラベルが疑われる場合はRRPを検討します。

## 3. 最小例

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

bounds = torch.tensor(
    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    dtype=dtype,
)

model = BinaryClassificationGPModel(
    train_X=train_X,
    train_Y=train_Y,
    input_transform=Normalize(d=train_X.shape[-1], bounds=bounds),
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

- `num_inducing_points`: inducing point数。公開base modelの既定値は最大128
- `inducing_points`: inducing pointを明示する場合に指定
- `learn_inducing_locations`: inducing point位置を学習するか
- `input_transform`: `Normalize`などの入力変換
- `mean_module`, `covar_module`: latent GPをカスタマイズする場合に指定

## 4. mixed input

カテゴリ列を正規化対象から除外します。

```python
from botorch.models.transforms.input import Normalize
from gpytorch.mlls import VariationalELBO

from bochan.fit import fit_binary_classifier_mll
from bochan.models.classification.binary.base import BinaryClassificationMixedGPModel

cat_dims = [3]
continuous_dims = [0, 1, 2]

input_transform = Normalize(
    d=train_X.shape[-1],
    bounds=bounds,
    indices=continuous_dims,
)

model = BinaryClassificationMixedGPModel(
    train_X=train_X,
    train_Y=train_Y,
    cat_dims=cat_dims,
    input_transform=input_transform,
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

## 5. posteriorと不確かさ

### 確率posterior

```python
posterior = model.posterior(X_test)
p = posterior.mean
variance = posterior.variance
```

`mean`はクラス1確率`P(y=1 | x, D)`です。通常の`variance`はBernoulli observation variance `p * (1-p)`であり、モデルのepistemic uncertaintyそのものではありません。

### latent posterior

```python
latent_posterior = model.latent_posterior(X_test)
latent_mean = latent_posterior.mean
latent_variance = latent_posterior.variance
```

こちらはlikelihood適用前のlatent function `f(x)`です。0から1の範囲には限定されません。

### probability-scale epistemic uncertainty

```python
from bochan.acquisition.binary.epistemic import binary_probability_moments

mean_probability, epistemic_variance, aleatoric_variance, total_variance = (
    binary_probability_moments(model, X_test, num_samples=256)
)
```

- `epistemic_variance`: latent posterior sampleを確率へ写像したsample間分散
- `aleatoric_variance`: `E[p(1-p)]`
- `total_variance`: 0/1ラベルの全分散

active learningやvariance系獲得関数では、原則としてepistemic varianceを使います。

## 6. multi-output

同じ入力に対して複数の独立したbinary labelを予測する場合は、出力ごとにsingle-output modelを学習し、wrapperで包みます。

```python
from gpytorch.mlls import VariationalELBO

from bochan.fit import fit_binary_classifier_mll
from bochan.models.classification.binary.base import (
    BinaryClassificationGPModel,
    MultiOutputBinaryClassificationModel,
)

submodels = []
for j in range(train_Y.shape[-1]):
    submodel = BinaryClassificationGPModel(
        train_X=train_X,
        train_Y=train_Y[:, [j]],
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
posterior = model.posterior(train_X)
print(posterior.mean.shape)  # [n, m]
```

出力の一部だけを選択できます。

```python
subset_probability = model.posterior(train_X, output_indices=[0, 2])
subset_latent = model.latent_posterior(train_X, output_indices=[0])

print(subset_probability.mean.shape)  # [n, 2]
print(subset_latent.mean.shape)       # [n, 1]
```

このwrapperは出力間を独立として扱います。出力間相関を学習する場合はmulti-task modelを検討します。

## 7. multi-task

### task-id列を使うlong format

`MultiTaskBinaryClassificationGPModel`はtask-id列を含むlong formatを使います。

```python
from gpytorch.mlls import VariationalELBO

from bochan.fit import fit_binary_classifier_mll
from bochan.models.classification.binary.base import MultiTaskBinaryClassificationGPModel

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

task-id列は連続変数として正規化しないでください。

### Kronecker multi-taskを使うblock design

すべてのタスクが同じ入力点で観測されている場合は、`KroneckerMultiTaskBinaryClassificationGPModel`を使用できます。`train_X`は`[n, d]`、`train_Y`は`[n, m]`です。`m`は相関を学習するbinaryタスク数です。

```python
import torch
from botorch.models.transforms.input import Normalize

from bochan.fit import fit_binary_classifier_mll
from bochan.models.classification.binary.base import (
    KroneckerMultiTaskBinaryClassificationGPModel,
)


torch.manual_seed(0)
dtype = torch.double

n = 60
num_tasks = 3
train_X = torch.rand(n, 3, dtype=dtype)

shared_score = 1.5 * train_X[:, 0] - train_X[:, 1]
task_scores = torch.stack(
    [
        shared_score,
        shared_score + 0.5 * train_X[:, 2] - 0.1,
        -0.6 * shared_score + train_X[:, 2],
    ],
    dim=-1,
)
train_Y = (task_scores > 0.0).to(dtype)  # [n, 3]

bounds = torch.tensor(
    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    dtype=dtype,
)

model = KroneckerMultiTaskBinaryClassificationGPModel(
    train_X=train_X,
    train_Y=train_Y,
    rank=2,
    input_transform=Normalize(d=train_X.shape[-1], bounds=bounds),
    num_inducing_points=32,
)

# block-design用のELBOを使うため、model.make_mll()を推奨
mll = model.make_mll()
fit_binary_classifier_mll(
    mll,
    num_epochs=300,
    lr=0.01,
    batch_size=None,
)

X_test = torch.rand(10, 3, dtype=dtype)
posterior = model.posterior(X_test)
probability = posterior.mean
prediction = model.predict_class(X_test)
class_probability = model.class_probs(X_test)

print(probability.shape)       # [10, 3]: P(y_task=1)
print(prediction.shape)        # [10, 3]
print(class_probability.shape) # [10, 3, 2]: [P(y=0), P(y=1)]
```

latent GPの相関構造は次のように確認できます。

```python
latent_posterior = model.latent_posterior(X_test)
task_covar = model.task_covar_matrix

print(latent_posterior.mean.shape)  # [10, 3]
print(task_covar.shape)             # [3, 3]
```

`task_covar_matrix`はLMC係数から得られる正半定値行列です。対角成分は各タスクのlatent scale、非対角成分はタスク間の共変動を表します。相関係数として比較する場合は、対角成分で標準化します。

```python
task_std = task_covar.diag().clamp_min(1e-12).sqrt()
task_corr = task_covar / task_std[:, None] / task_std[None, :]
```

出力の一部だけを確率scaleで選択できます。

```python
selected = model.posterior(X_test, output_indices=[0, 2])
print(selected.mean.shape)  # [10, 2]
```

`KroneckerMultiTaskBinaryClassificationGPModel`はblock design専用です。入力点ごとに観測されるタスクが異なる欠測・long formatデータには、`MultiTaskBinaryClassificationGPModel`を使用してください。

## 8. advanced modelの学習

DeepKernelは専用fit helperを使います。

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

RRPは外れラベル候補を学習点単位で疎に推定します。説明変数選択モデルではありません。

```python
from bochan.fit import fit_rrp_binary_classifier_mll
from bochan.models.classification.binary.robust import (
    RobustRelevancePursuitBinaryClassificationGPModel,
)

model = RobustRelevancePursuitBinaryClassificationGPModel(
    train_X=train_X,
    train_Y=train_Y,
)
fit_rrp_binary_classifier_mll(
    model.make_mll(),
    method="forward",
    sparsity_levels=[0, 1, 2, 4],
    optimizer_kwargs={"num_epochs": 100, "lr": 0.01},
)
```

SAAS / PCA / REMBO / heteroscedastic modelの詳細な引数は、各model fileと対応する`tests/test_binary_classification_*`を参照してください。

## 9. 新しい観測の追加

対応modelでは`condition_on_observations`を利用できます。

```python
updated_model = model.condition_on_observations(
    X=X_new,
    Y=Y_new,
)
```

これは厳密なclosed-form conditioningではなく、既存データと新規データからmodelを再構成する近似処理です。必要に応じて再学習してください。

## 10. high-level API

文字列設定でmodelを構築することもできます。

```python
from bochan.api import BayesianOptimizer, FitConfig, ModelConfig

bo = BayesianOptimizer(
    model_config=ModelConfig(
        task_type="binary",
        model_type="base",
        model_kwargs={"num_inducing_points": 64},
    ),
    fit_config=FitConfig(),
    bounds=bounds,
)
bo.fit(train_X, train_Y)
result = bo.predict(X_test, return_result=True)
```

標準registryのbinary `model_type`:

```text
base, deepgp, deepkernel, deepgpdeepkernel,
saas, pca, rembo, rrp, hetero
```

API全体は`src/bochan/api/README.md`を参照してください。

## 11. よくある問題

### `posterior.variance`が0.25付近になる

確率0.5付近ではBernoulli observation varianceが0.25になります。epistemic uncertaintyは`binary_probability_moments`で確認してください。

### mixed modelでカテゴリ値が変化する

`Normalize(indices=...)`にカテゴリ列を含めないでください。

### EI / PIで`best_f=1`になる

0/1ラベルの最大値をそのまま使うと改善余地がなくなります。binary probability scaleの有限scalarを使うか、`compute_binary_best_f`を利用してください。

### RRPで`transform_inputs`エラーが出る

現行のbinary latent RRP modelは内部にidentity `transform_inputs`を持ちます。古い実装を使用している場合は最新版へ更新してください。

### 学習が不安定

次を確認してください。

1. `train_Y`に0と1の両方が含まれる
2. dtype / deviceが一致している
3. continuous列だけをNormalizeしている
4. learning rateを下げる
5. `num_inducing_points`を調整する
6. modelを複雑にしすぎていない

## 12. 関連テスト

```text
tests/test_binary_classification_base_single_output.py
tests/test_binary_classification_base_multi_output.py
tests/test_binary_classification_deepgp_single_output.py
tests/test_binary_classification_deepkernel_single_output.py
tests/test_binary_classification_saas_single_output.py
tests/test_binary_classification_pca_rembo_single_output.py
tests/test_binary_classification_rrp_single_output.py
tests/test_binary_classification_hetero_single_output.py
tests/test_binary_probability_bo_pointwise.py
tests/test_kronecker_multitask_classification_ordinal_models.py
```
