# Gamma extended models

Gamma 回帰の追加モデル群です。

## RRP

```python
from bochan.models.regression.robust.gamma_relevance_pursuit import (
    OutlierRelevancePursuitGammaGPModel,
    OutlierRelevancePursuitGammaMixedGPModel,
)
```

Gamma RRP は Gaussian regression の feature RRP ではなく、学習点ごとの outlier offset を扱います。

```text
y_i ~ Gamma(mean=mean(f_i + δ_i), concentration=κ)
```

## Heteroscedastic Gamma

```python
from bochan.models.regression.robust.gamma_heteroscedastic import (
    HeteroscedasticGammaGPModel,
    HeteroscedasticGammaMixedGPModel,
)
```

Gamma 分布はもともと `Var[y|x] = μ^2 / κ` を持ちます。  
ここでは mean 残差から追加分散を推定し、`posterior(..., observation_noise=True)` で base variance に足します。

## Gamma DeepGP

```python
from bochan.models.regression.deep.gamma_deepgp import (
    GammaDeepGPModel,
    GammaMixedDeepGPModel,
)
```

true DeepGP + Gamma likelihood の正値回帰モデルです。

## fit

通常の SVGP モデルは `make_mll()` で `VariationalELBO` を返します。  
DeepGP は `DeepApproximateMLL(VariationalELBO(...))` を返します。
