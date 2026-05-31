# Feasible acquisition helpers

`bochan.acquisition.feasible` は、既存 acquisition を変更せずに、分類・順序回帰・回帰出力を feasible constraint として扱うための小さな補助パッケージです。

## 1. BoTorch 標準 acquisition の `constraints` に渡す

BoTorch の MC acquisition に渡す constraint callable は、feasible なときに `constraint(samples) <= 0` を返す必要があります。

```python
from bochan.acquisition.feasible import (
    FeasibilityConstraintSpec,
    make_sample_constraints,
)

constraints = make_sample_constraints(
    [
        FeasibilityConstraintSpec("safe_prob", threshold=0.8, sense="ge"),
        FeasibilityConstraintSpec("quality_utility", threshold=1.5, sense="ge"),
    ],
    output_names=hybrid_model.output_names,
)

# 例: qExpectedImprovement / qNoisyExpectedImprovement / qEHVI / qNEHVI など
# acqf = qExpectedImprovement(
#     model=hybrid_model,
#     best_f=best_f,
#     sampler=sampler,
#     constraints=constraints,
# )
```

## 2. 既存 acquisition に feasibility weight を掛ける

自作 active learning / level-set / UCB 系など、`constraints=` に直接乗せにくい acquisition には `FeasibilityWeightedAcquisition` を使います。

```python
from bochan.acquisition.feasible import (
    FeasibilityConstraintSpec,
    FeasibilityWeightedAcquisition,
)

base_acqf = SomeExistingAcquisition(model=hybrid_model)

acqf = FeasibilityWeightedAcquisition(
    acqf=base_acqf,
    model=hybrid_model,
    constraints=[
        FeasibilityConstraintSpec("safe_prob", threshold=0.8, sense="ge"),
        FeasibilityConstraintSpec("quality_utility", threshold=1.5, sense="ge"),
    ],
    eta=0.05,
    reduce_constraints="prod",
    reduce_q="min",
)
```

## 3. Ordinal rank probability を制約にする

`OrdinalRankConstraintSpec` を使うと、順序回帰の rank そのものをしきい値として扱えます。

```python
from bochan.acquisition.feasible import (
    FeasibilityWeightedAcquisition,
    OrdinalRankConstraintSpec,
)

base_acqf = SomeExistingAcquisition(model=hybrid_model)

acqf = FeasibilityWeightedAcquisition(
    acqf=base_acqf,
    model=hybrid_model,
    constraints=[
        # P(quality_rank >= 2) >= 0.8
        OrdinalRankConstraintSpec(
            output="quality_rank",
            rank=2,
            sense="ge",
            probability_threshold=0.8,
        ),
    ],
    eta=0.05,
    reduce_constraints="prod",
    reduce_q="min",
)
```

`OrdinalRankConstraintSpec` の意味は以下です。

- `sense="ge", rank=k`: `P(y >= k) >= probability_threshold`
- `sense="le", rank=k`: `P(y <= k) >= probability_threshold`
- `sense="eq", rank=k`: `P(y == k) >= probability_threshold`

`OrdinalRankConstraintSpec` は `HybridMultiOutputModel.class_probs_list()` から class probability を取得して評価するため、基本的には `FeasibilityWeightedAcquisition` と組み合わせて使います。

BoTorch 標準 acquisition の `constraints=` に直接渡す場合は、通常の `HybridMultiOutputModel.objective_posterior` には class probability 全体が残らないため、`FeasibilityConstraintSpec` による expected utility 制約を使うのが基本です。

## Constraint sense

- `sense="ge"`: `y >= threshold` を feasible とする。
- `sense="le"`: `y <= threshold` を feasible とする。
- `sense="eq"`: `abs(y - threshold) <= margin` を feasible とする。

## HybridMultiOutputModel との関係

`HybridMultiOutputModel.objective_posterior(X)` は、回帰・分類・順序回帰・多クラス分類を objective scale の `[..., q, m]` にそろえます。

そのため、分類や順序回帰を制約にしたい場合は、以下のようにあらかじめ出力名を分けておくと扱いやすくなります。

```python
OutputSpec(name="safe_prob", task_type="binary", model=binary_model)
OutputSpec(name="quality_rank", task_type="ordinal", model=ordinal_model, utility_values=[0.0, 1.0, 2.0])
```
