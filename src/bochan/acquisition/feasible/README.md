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

## Constraint sense

- `sense="ge"`: `y >= threshold` を feasible とする。
- `sense="le"`: `y <= threshold` を feasible とする。
- `sense="eq"`: `abs(y - threshold) <= margin` を feasible とする。

## HybridMultiOutputModel との関係

`HybridMultiOutputModel.objective_posterior(X)` は、回帰・分類・順序回帰・多クラス分類を objective scale の `[..., q, m]` にそろえます。

そのため、分類や順序回帰を制約にしたい場合は、以下のようにあらかじめ出力名を分けておくと扱いやすくなります。

```python
OutputSpec(name="safe_prob", task_type="binary", model=binary_model)
OutputSpec(name="quality_utility", task_type="ordinal", model=ordinal_model, utility_values=[0.0, 1.0, 2.0])
```
