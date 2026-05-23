# Poisson robust / heteroscedastic / DeepGP models

このファイルは Poisson 回帰の追加モデル群のメモです。

## Outlier RRP

```python
from bochan.models.regression.robust.poisson_relevance_pursuit import (
    OutlierRelevancePursuitPoissonGPModel,
    OutlierRelevancePursuitPoissonMixedGPModel,
)
```

Poisson RRP は Gaussian regression の feature RRP ではなく、classification / ordinal と同じ **train-point outlier RRP** として扱います。

```text
y_i ~ Poisson(rate(f_i + delta_i))
```

予測時には `delta_i` は使いません。

## Heteroscedastic Poisson

```python
from bochan.models.regression.robust.poisson_heteroscedastic import (
    HeteroscedasticPoissonGPModel,
    HeteroscedasticPoissonMixedGPModel,
)
```

Poisson には元々 `Var[y|x] = lambda` があります。  
このモデルでは、rate 残差から追加分散を推定し、`posterior(..., observation_noise=True)` のときだけ base variance に足します。

```text
variance ≒ Poisson variance + latent uncertainty + extra local noise
```

## Poisson DeepGP

```python
from bochan.models.regression.deep.poisson_deepgp import (
    PoissonDeepGPModel,
    PoissonMixedDeepGPModel,
)
```

true DeepGP + Poisson likelihood のモデルです。

```python
model = PoissonDeepGPModel(
    train_X=train_X,
    train_Y=train_Y,
    hidden_dim=4,
    num_inducing=64,
)

mll = model.make_mll()
```

## fit

RRP / base Poisson は `make_mll()` で `VariationalELBO` を返します。  
DeepGP は `make_mll()` で `DeepApproximateMLL(VariationalELBO(...))` を返します。

```python
mll = model.make_mll()
```

通常の torch training loop で `-mll(output, train_Y)` を最小化してください。
