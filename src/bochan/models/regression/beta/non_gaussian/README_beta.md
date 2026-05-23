# Beta regression models

このフォルダは、割合・率など `0 < y < 1` の連続値 target 用の Beta GP 回帰モデル群です。

## 基本方針

Beta 回帰は非ガウス尤度なので、Poisson / Gamma と同様に SVGP / VariationalELBO を基本にします。

```text
latent f(x)
  -> sigmoid / probit
  -> mean μ(x)

φ = learnable scalar concentration
α(x) = μ(x) φ
β(x) = (1 - μ(x)) φ

y ~ Beta(α(x), β(x))
```

## モデル一覧

```text
models/components/beta.py
  BetaLogLikelihood
  BetaPosterior

models/regression/non_gaussian/beta.py
  BetaGPModel
  BetaMixedGPModel

models/regression/high_dim/beta_decomposition.py
  PCABetaGPModel
  REMBOBetaGPModel
  PCABetaMixedGPModel
  REMBOBetaMixedGPModel

models/regression/high_dim/beta_saas.py
  SaasBetaGPModel
  SaasBetaMixedGPModel

models/regression/deep/beta_deepkernel.py
  DeepKernelBetaGPModel
  DeepKernelBetaMixedGPModel

models/regression/deep/beta_deepgp.py
  BetaDeepGPModel
  BetaMixedDeepGPModel

models/regression/robust/beta_relevance_pursuit.py
  OutlierRelevancePursuitBetaGPModel
  OutlierRelevancePursuitBetaMixedGPModel

models/regression/robust/beta_heteroscedastic.py
  HeteroscedasticBetaGPModel
  HeteroscedasticBetaMixedGPModel
```

## posterior の意味

```python
post = model.posterior(X)
```

`post.mean` は Beta mean `μ(x)` の近似平均です。

```python
mu = model.predict_mean(X)
phi = model.predict_concentration()
alpha, beta = model.predict_beta_params(X)
```

latent GP posterior が必要な場合は以下を使います。

```python
latent_post = model.latent_posterior(X)
```

## fit 例

```python
import torch
from bochan.models.regression.non_gaussian.beta import BetaGPModel
from bochan.models.regression.non_gaussian.fit import fit_non_gaussian_gp

train_X = torch.rand(40, 2, dtype=torch.double)
true_mu = torch.sigmoid(4.0 * (train_X[:, 0] - 0.5))
phi = torch.tensor(20.0, dtype=torch.double)
train_Y = torch.distributions.Beta(true_mu * phi, (1.0 - true_mu) * phi).sample()

model = BetaGPModel(
    train_X=train_X,
    train_Y=train_Y,
    link="sigmoid",
    init_concentration=10.0,
)

result = fit_non_gaussian_gp(model, lr=0.01, num_epochs=300)
post = result.model.posterior(train_X)
```

MLL を明示する場合は以下です。

```python
from gpytorch.mlls import VariationalELBO

mll = VariationalELBO(
    likelihood=model.likelihood,
    model=model.model,
    num_data=model.train_inputs_raw[0].shape[-2],
)
```

DeepGP の場合は以下です。

```python
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO

base_mll = VariationalELBO(
    likelihood=model.likelihood,
    model=model,
    num_data=model.train_inputs_raw[0].shape[-2],
)
mll = DeepApproximateMLL(base_mll)
```

## 注意点

- Beta 分布は厳密には `0 < y < 1` のみ扱います。
- 実装では `clip_targets=True` の場合、`[eps, 1-eps]` に自動 clipping します。
- `0` や `1` が頻繁に出る場合は、zero-one inflated beta を別途検討してください。
- `heteroscedastic` 版は pragmatic な追加分散補正です。本来の heteroscedastic Beta は `φ(x)` をモデル化する方が自然です。
