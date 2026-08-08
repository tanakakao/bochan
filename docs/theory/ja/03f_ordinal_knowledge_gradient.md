# Ordinal Knowledge Gradient

`qOrdinalKnowledgeGradient` は、ordinal outcome の **離散観測 likelihood と expected utility を考慮した one-step look-ahead acquisition** です。

## Terminal utility

ordinal class `k` ごとに utility `u_k` を定義し、最終意思決定では posterior expected utility が最大の条件を選びます。

```math
U(x)=\sum_k u_k P(y=k\mid x,D)
```

```math
V(D)=\max_x U(x)
```

候補 `x` の観測結果は ordinal class のいずれかです。Ordinal KG は各仮想classで posterior function samplesをlikelihood再重み付けし、観測後の最大 expected utility を求めます。

```math
KG(x)=\sum_kP(y=k\mid x,D)V(D\cup\{(x,k)\})-V(D)
```

## Utility values

未指定時は `[0, 1, ..., K-1]` を使用します。クラス間の価値差が等間隔でない場合は明示してください。

```python
utility_values = [0.0, 0.2, 1.0, 5.0]
```

`OrdinalExpectedUtilityMCObjective` を渡した場合は、その `utility_values` も利用できます。

## 利用例

```python
from bochan.acquisition.ordinal.bayesian_optimization import qOrdinalKnowledgeGradient
from bochan.api import AcquisitionConfig, OptimizeConfig

acq_config = AcquisitionConfig(
    name="ordinal_kg",
    acqf_cls=qOrdinalKnowledgeGradient,
    acqf_kwargs={
        "bounds": bounds,
        "utility_values": [0.0, 0.5, 2.0, 5.0],
        "terminal_size": 128,
        "num_samples": 64,
    },
)

X_next, value = optimizer.candidate(
    acq_config,
    OptimizeConfig(q=1),
)
```

## v1 の範囲

- single-output ordinal model
- ordinal likelihood が `class_probs_from_f` を提供すること
- q=1
- expected utility 最大化
- continuous input は bounds から Sobol terminal set を自動生成可能
- mixed/categorical input は terminal set の明示指定を要求
- pending class の近似は行わない

回帰qKGのGaussian fantasyをordinal labelへ流用せず、ordinal likelihoodによる離散観測更新を明示的に扱います。
