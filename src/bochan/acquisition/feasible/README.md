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

High-level API の `AcquisitionConfig` では、`acqf_kwargs["constraints"]` ではなく、トップレベルの `constraints` に渡します。

```python
acq_config = AcquisitionConfig(
    name="ei",
    objective_config=ObjectiveConfig(
        mode="scalar",
        output="strength",
        direction="maximize",
    ),
    constraints=constraints,
)
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

## 3. Binary output を制約にする

Binary 出力は `positive_class` で指定したクラスの確率として扱えます。例えば、`defect=1` が不良を表す場合は、`P(defect=1) <= 0.2` を feasible 条件にできます。

```python
model_config = ModelConfig(
    task_type="hybrid",
    model_type="base",
    multi_output_config=MultiOutputConfig(
        output_configs=[
            OutputConfig(task_type="regression", model_type="base", name="strength"),
            OutputConfig(
                task_type="binary",
                model_type="base",
                name="defect",
                output_spec_kwargs={"positive_class": 1},
            ),
        ],
        use_hybrid=True,
    ),
)

constraints = make_sample_constraints(
    [
        # P(defect=1) <= 0.2
        FeasibilityConstraintSpec(output="defect", threshold=0.2, sense="le"),
    ],
    output_names=bo.bundle.model.output_names,
)

acq_config = AcquisitionConfig(
    name="ei",
    objective_config=ObjectiveConfig(mode="scalar", output="strength"),
    constraints=constraints,
)
```

`1=feasible` のように正例が実行可能を表す場合は、`P(feasible=1) >= 0.8` の形にします。

```python
OutputConfig(
    task_type="binary",
    model_type="base",
    name="safe",
    output_spec_kwargs={"positive_class": 1},
)

constraints = make_sample_constraints(
    [
        FeasibilityConstraintSpec(output="safe", threshold=0.8, sense="ge"),
    ],
    output_names=bo.bundle.model.output_names,
)
```

## 4. Multiclass output を制約にする

Multiclass 出力は `positive_class` で対象クラスを指定すると、そのクラス確率を `probability` scale で扱えます。例えば、class 2 が「良品」や「合格」を表す場合は、`P(class=2) >= 0.7` を feasible 条件にできます。

```python
model_config = ModelConfig(
    task_type="hybrid",
    model_type="base",
    multi_output_config=MultiOutputConfig(
        output_configs=[
            OutputConfig(task_type="regression", model_type="base", name="strength"),
            OutputConfig(
                task_type="multiclass",
                model_type="base",
                name="quality_class",
                output_spec_kwargs={"positive_class": 2},
            ),
        ],
        use_hybrid=True,
    ),
)

constraints = make_sample_constraints(
    [
        # P(quality_class == 2) >= 0.7
        FeasibilityConstraintSpec(output="quality_class", threshold=0.7, sense="ge"),
    ],
    output_names=bo.bundle.model.output_names,
)

acq_config = AcquisitionConfig(
    name="ei",
    objective_config=ObjectiveConfig(mode="scalar", output="strength"),
    constraints=constraints,
)
```

複数クラスにまたがる条件、例えば `P(class in {1, 2}) >= 0.8` のような条件を直接使いたい場合は、`class_probs_list()` から確率を取得する custom constraint、または `FeasibilityWeightedAcquisition` の拡張が必要です。標準の `FeasibilityConstraintSpec` は、1つの出力値または1つの対象クラス確率をしきい値判定する用途を想定しています。

## 5. Ordinal output を制約にする

Ordinal 出力には2通りの使い方があります。

1つ目は、expected utility / probability scale の1出力値を `FeasibilityConstraintSpec` でしきい値判定する方法です。例えば rank utility の期待値が `2.0` 以上なら feasible とします。

```python
model_config = ModelConfig(
    task_type="hybrid",
    model_type="base",
    multi_output_config=MultiOutputConfig(
        output_configs=[
            OutputConfig(task_type="regression", model_type="base", name="strength"),
            OutputConfig(
                task_type="ordinal",
                model_type="base",
                name="quality_rank",
                output_spec_kwargs={"utility_values": [0.0, 1.0, 2.0, 3.0]},
            ),
        ],
        use_hybrid=True,
    ),
)

constraints = make_sample_constraints(
    [
        # E[quality_rank utility] >= 2.0
        FeasibilityConstraintSpec(output="quality_rank", threshold=2.0, sense="ge"),
    ],
    output_names=bo.bundle.model.output_names,
)

acq_config = AcquisitionConfig(
    name="ei",
    objective_config=ObjectiveConfig(mode="scalar", output="strength"),
    constraints=constraints,
)
```

2つ目は、`OrdinalRankConstraintSpec` で rank probability を制約にする方法です。これは `HybridMultiOutputModel.class_probs_list()` から class probability を取得して評価するため、基本的には `FeasibilityWeightedAcquisition` と組み合わせて使います。

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

BoTorch 標準 acquisition の `constraints=` に直接渡す場合は、通常の `HybridMultiOutputModel.objective_posterior` には class probability 全体が残らないため、`FeasibilityConstraintSpec` による expected utility 制約を使うのが基本です。

## 6. Constraint sense

- `sense="ge"`: `y >= threshold` を feasible とする。
- `sense="le"`: `y <= threshold` を feasible とする。
- `sense="eq"`: `abs(y - threshold) <= margin` を feasible とする。

## 7. HybridMultiOutputModel との関係

`HybridMultiOutputModel.objective_posterior(X)` は、回帰・分類・順序回帰・多クラス分類を objective scale の `[..., q, m]` にそろえます。

そのため、分類や順序回帰を制約にしたい場合は、以下のようにあらかじめ出力名を分けておくと扱いやすくなります。

```python
OutputSpec(name="safe_prob", task_type="binary", model=binary_model)
OutputSpec(name="quality_class", task_type="multiclass", model=multiclass_model, positive_class=2)
OutputSpec(name="quality_rank", task_type="ordinal", model=ordinal_model, utility_values=[0.0, 1.0, 2.0])
```
