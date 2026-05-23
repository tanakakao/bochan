# Gamma regression models

このフォルダは、正の連続値 target 用の Gamma GP 回帰モデル群です。

## 基本方針

Gamma 回帰は非ガウス尤度なので、ExactGP ではなく SVGP / VariationalELBO を基本にします。

```text
latent f(x)
  -> link function
  -> mean μ(x)
  -> y ~ Gamma(concentration=κ, rate=κ/μ)
```

Gamma 分布の平均と分散は以下です。

```text
E[y|x] = μ(x)
Var[y|x] = μ(x)^2 / κ
```

## モデル一覧

```text
models/components/gamma.py
  GammaLogLikelihood
  GammaPosterior

models/regression/non_gaussian/gamma.py
  GammaGPModel
  GammaMixedGPModel

models/regression/high_dim/gamma_saas.py
  SaasGammaGPModel
  SaasGammaMixedGPModel
```

追加で PCA / REMBO / DeepKernel / DeepGP / RRP / heteroscedastic も同じ API 方針で拡張できます。

## posterior の意味

```python
post = model.posterior(X)
```

`post.mean` は Gamma mean μ の近似平均です。

```python
mu = model.predict_mean(X)
kappa = model.predict_concentration()
rate = model.predict_rate_parameter(X)
```

latent GP posterior が必要な場合は以下を使います。

```python
latent_post = model.latent_posterior(X)
```

## fit 例

```python
import torch
from bochan.models.regression.non_gaussian.gamma import GammaGPModel

train_X = torch.rand(30, 2, dtype=torch.double)
mu = 1.0 + 5.0 * train_X[:, 0]
train_Y = torch.distributions.Gamma(concentration=5.0, rate=5.0 / mu).sample()

model = GammaGPModel(
    train_X=train_X,
    train_Y=train_Y,
    link="softplus",
    init_concentration=5.0,
)

mll = model.make_mll()
# 通常の variational training loop で mll を最適化する
```

## 注意点

- target は strictly positive である必要があります。
- `link="softplus"` が数値安定性の面で推奨です。
- `link="exp"` を使う場合は `exp_clip` を適切に設定してください。
- `concentration` は scalar parameter として学習されます。
- heteroscedastic Gamma を作る場合は、concentration を input-dependent にする拡張も可能です。
