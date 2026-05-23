# Negative Binomial regression models

このフォルダは、過分散を持つ count target 用の Negative Binomial GP 回帰モデル群です。

## 基本方針

Negative Binomial 回帰は非ガウス尤度なので、Poisson / Gamma と同様に SVGP / VariationalELBO を基本にします。

```text
latent f(x)
  -> link function
  -> mean μ(x)

r = learnable scalar total_count / dispersion

y ~ NegativeBinomial(total_count=r, logits=log(μ/r))
E[y|x] = μ
Var[y|x] = μ + μ²/r
```

`r` が大きいほど Poisson に近づき、`r` が小さいほど過分散が強くなります。

## モデル一覧

```text
models/components/negative_binomial.py
  NegativeBinomialLogLikelihood
  NegativeBinomialPosterior

models/regression/non_gaussian/negative_binomial.py
  NegativeBinomialGPModel
  NegativeBinomialMixedGPModel

models/regression/high_dim/negative_binomial_decomposition.py
  PCANegativeBinomialGPModel
  REMBONegativeBinomialGPModel
  PCANegativeBinomialMixedGPModel
  REMBONegativeBinomialMixedGPModel

models/regression/high_dim/negative_binomial_saas.py
  SaasNegativeBinomialGPModel
  SaasNegativeBinomialMixedGPModel

models/regression/deep/negative_binomial_deepkernel.py
  DeepKernelNegativeBinomialGPModel
  DeepKernelNegativeBinomialMixedGPModel

models/regression/deep/negative_binomial_deepgp.py
  NegativeBinomialDeepGPModel
  NegativeBinomialMixedDeepGPModel

models/regression/robust/negative_binomial_relevance_pursuit.py
  SparseOutlierNegativeBinomialLikelihood
  OutlierRelevancePursuitNegativeBinomialGPModel
  OutlierRelevancePursuitNegativeBinomialMixedGPModel

models/regression/robust/negative_binomial_heteroscedastic.py
  HeteroscedasticNegativeBinomialGPModel
  HeteroscedasticNegativeBinomialMixedGPModel
```

## posterior の意味

```python
post = model.posterior(X)
```

`post.mean` は Negative Binomial mean `μ(x)` の近似平均です。

```python
mu = model.predict_mean(X)
count = model.predict_count(X)
r = model.predict_total_count()
logits = model.predict_logits(X)
```

latent GP posterior が必要な場合は以下を使います。

```python
latent_post = model.latent_posterior(X)
```

## fit 例

```python
import torch
from bochan.models.regression.non_gaussian.negative_binomial import NegativeBinomialGPModel
from bochan.models.regression.non_gaussian.fit import fit_non_gaussian_gp

train_X = torch.rand(40, 2, dtype=torch.double)
true_mu = 2.0 + 10.0 * train_X[:, 0]
true_r = torch.tensor(5.0, dtype=torch.double)
logits = (true_mu / true_r).log()
train_Y = torch.distributions.NegativeBinomial(
    total_count=true_r,
    logits=logits,
).sample()

model = NegativeBinomialGPModel(
    train_X=train_X,
    train_Y=train_Y,
    link="softplus",
    init_total_count=10.0,
    learn_total_count=True,
    num_inducing_points=min(128, train_X.shape[-2]),
)

result = fit_non_gaussian_gp(
    model,
    lr=0.01,
    num_epochs=300,
)

post = result.model.posterior(train_X)
print(post.mean)
print(result.model.predict_total_count())
```

## MLL を明示する場合

通常 SVGP 系は以下です。

```python
from gpytorch.mlls import VariationalELBO

mll = VariationalELBO(
    likelihood=model.likelihood,
    model=model.model,
    num_data=model.train_inputs_raw[0].shape[-2],
)
```

PCA / REMBO wrapper では `base_model` 側に MLL を作ります。

```python
mll = VariationalELBO(
    likelihood=model.base_model.likelihood,
    model=model.base_model.model,
    num_data=model.train_inputs_raw[0].shape[-2],
)
```

DeepGP では `DeepApproximateMLL` で包みます。

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

- target は 0 以上の整数 count である必要があります。
- `total_count` は過分散を制御します。
- `total_count -> ∞` で Poisson に近づきます。
- `posterior().rsample()` は differentiable な mean sample を返します。
- count sample が必要な場合は `posterior.sample_counts()` を使います。
- mixed model ではカテゴリ列を `input_transform` で変換しないでください。
