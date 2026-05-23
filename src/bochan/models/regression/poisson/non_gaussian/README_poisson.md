# Poisson regression models

このフォルダは、count target 用の Poisson GP 回帰モデル群です。

## 基本方針

Poisson 回帰は非ガウス尤度なので、ExactGP ではなく SVGP / VariationalELBO を基本にします。

```text
latent f(x)
  -> link function
  -> rate λ(x)
  -> y ~ Poisson(λ(x))
```

link function は以下を選べます。

```python
link="softplus"  # 推奨。数値的に安定
link="exp"       # 古典的な log-link。exp_clip で clip
```

## モデル一覧

```text
models/components/poisson.py
  PoissonLogLikelihood
  PoissonPosterior

models/regression/non_gaussian/poisson.py
  PoissonGPModel
  PoissonMixedGPModel

models/regression/high_dim/poisson_decomposition.py
  PCAPoissonGPModel
  REMBOPoissonGPModel
  PCAPoissonMixedGPModel
  REMBOPoissonMixedGPModel

models/regression/high_dim/poisson_saas.py
  SaasPoissonGPModel
  SaasPoissonMixedGPModel

models/regression/deep/poisson_deepkernel.py
  DeepKernelPoissonGPModel
  DeepKernelPoissonMixedGPModel
```

## posterior の意味

```python
post = model.posterior(X)
```

`post.mean` は Poisson rate λ の近似平均です。

```python
rate = model.predict_rate(X)
count = model.predict_count(X)
```

Poisson では期待 count は rate と同じです。

latent GP posterior が必要な場合は以下を使います。

```python
latent_post = model.latent_posterior(X)
```

## fit 例

```python
import torch
from bochan.models.regression.non_gaussian.poisson import PoissonGPModel

train_X = torch.rand(30, 2, dtype=torch.double)
rate = 2.0 + 8.0 * train_X[:, 0]
train_Y = torch.poisson(rate).long()

model = PoissonGPModel(
    train_X=train_X,
    train_Y=train_Y,
    link="softplus",
)

mll = model.make_mll()
# 通常の variational training loop で mll を最適化する
```

## 注意点

- target は 0 以上の整数 count である必要があります。
- `posterior().rsample()` は differentiable な rate sample を返します。
- 実際の count sample が必要な場合は `posterior.sample_counts()` を使います。
- mixed model ではカテゴリ列を `input_transform` で変換しないでください。
- `InputPerturbation` がある場合、学習時は点数を増やさず、posterior 評価時だけ q 展開を許します。
