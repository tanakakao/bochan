# Feasible acquisition helpers

`bochan.acquisition.feasible` は、分類・順序回帰・回帰出力を feasible constraint として扱うための補助パッケージです。

## 1. 推奨: `outcome_constraint_config` に書く

通常のノートブックやアプリでは、`constraints` ではなく `outcome_constraint_config` を使います。`outcome_constraint_config` はユーザー向けの高レベル設定で、制約の意味をそのままデータとして残せます。

```python
from bochan.api import AcquisitionConfig, ObjectiveConfig, OutcomeConstraintConfig
from bochan.acquisition.feasible import FeasibilityConstraintSpec

acq_config = AcquisitionConfig(
    name="ei",
    objective_config=ObjectiveConfig(
        mode="scalar",
        output="strength",
        direction="maximize",
    ),
    outcome_constraint_config=OutcomeConstraintConfig(
        constraints=[
            FeasibilityConstraintSpec(
                output="quality_class",
                target_class=2,
                threshold=0.7,
                sense="ge",
            ),
        ],
    ),
)
```

これは以下を意味します。

```text
P(quality_class == 2) >= 0.7
```

この書き方では、モデル作成時の `OutputConfig` に `positive_class` を固定する必要はありません。

```python
model_config = ModelConfig(
    task_type="hybrid",
    model_type="base",
    multi_output_config=MultiOutputConfig(
        output_configs=[
            OutputConfig(task_type="regression", model_type="base", name="strength"),
            OutputConfig(task_type="multiclass", model_type="base", name="quality_class"),
        ],
        use_hybrid=True,
    ),
)
```

## 2. `constraints` と `outcome_constraint_config` の違い

- `outcome_constraint_config`: ユーザー向けの高レベルAPI。名前付き出力、`target_class`、`target_classes`、ordinal rank制約を表現する。
- `constraints`: BoTorchへ直接渡す低レベルcallable。すでに `samples -> constraint_value` の関数になっているものを渡す。

BoTorchの制約callableでは、返り値が `<= 0` なら feasible です。

```python
constraints = make_sample_constraints(
    [FeasibilityConstraintSpec(output="strength", threshold=0.5, sense="ge")],
    output_names=bo.bundle.model.output_names,
)

acq_config = AcquisitionConfig(
    name="ei",
    objective_config=ObjectiveConfig(mode="scalar", output="strength"),
    constraints=constraints,
)
```

`constraints` は高度な上書き・デバッグ用です。通常は `outcome_constraint_config` を推奨します。

## 3. Binary output を制約にする

`defect=1` が不良を表す場合は、`target_class=1` を制約側で指定します。

```python
acq_config = AcquisitionConfig(
    name="ei",
    objective_config=ObjectiveConfig(mode="scalar", output="strength"),
    outcome_constraint_config=OutcomeConstraintConfig(
        constraints=[
            FeasibilityConstraintSpec(
                output="defect",
                target_class=1,
                threshold=0.2,
                sense="le",
            ),
        ],
    ),
)
```

意味は以下です。

```text
P(defect == 1) <= 0.2
```

`safe=1` が安全を表すなら、次のようにします。

```python
FeasibilityConstraintSpec(
    output="safe",
    target_class=1,
    threshold=0.8,
    sense="ge",
)
```

意味は以下です。

```text
P(safe == 1) >= 0.8
```

## 4. Multiclass output を制約にする

class 2 が「良品」や「合格」を表す場合は、次のようにします。

```python
FeasibilityConstraintSpec(
    output="quality_class",
    target_class=2,
    threshold=0.7,
    sense="ge",
)
```

意味は以下です。

```text
P(quality_class == 2) >= 0.7
```

複数クラスを許容したい場合は、`target_classes` を使います。

```python
FeasibilityConstraintSpec(
    output="quality_class",
    target_classes=[1, 2],
    threshold=0.8,
    sense="ge",
)
```

意味は以下です。

```text
P(quality_class in {1, 2}) >= 0.8
```

## 5. Ordinal output を制約にする

期待utilityを制約にするだけなら、通常の `FeasibilityConstraintSpec` を使えます。

```python
FeasibilityConstraintSpec(
    output="quality_rank",
    threshold=2.0,
    sense="ge",
)
```

意味は以下です。

```text
E[quality_rank utility] >= 2.0
```

rank確率を直接使う場合は、`OrdinalRankConstraintSpec` を使います。

```python
from bochan.acquisition.feasible import OrdinalRankConstraintSpec

OrdinalRankConstraintSpec(
    output="quality_rank",
    rank=2,
    sense="ge",
    probability_threshold=0.8,
)
```

意味は以下です。

```text
P(quality_rank >= 2) >= 0.8
```

`OrdinalRankConstraintSpec` の意味は以下です。

- `sense="ge", rank=k`: `P(y >= k) >= probability_threshold`
- `sense="le", rank=k`: `P(y <= k) >= probability_threshold`
- `sense="eq", rank=k`: `P(y == k) >= probability_threshold`

## 6. 内部の使い分け

`OutcomeConstraintConfig` は acquisition class が解決された後、通常の acquisition composition として処理されます。`factory.build_acquisition` や `engine.build_acquisition` をimport時に差し替える処理は使いません。

数値出力のsample constraintをacquisition classが明示的に `constraints=` として受け取れる場合は、BoTorch互換のnative constraintとしてそのまま渡します。

一方、次のケースでは `FeasibilityWeightedAcquisition` を使います。

- `target_class` / `target_classes` のように `model.class_probs_list()` が必要な制約
- `OrdinalRankConstraintSpec`
- acquisition class自体が `constraints=` をサポートしない場合

この分岐により、native constrained acquisitionの意味論を維持しつつ、Active LearningやLevel-setなどの非native acquisitionにも同じ高位constraint APIを利用できます。

## 7. Constraint sense

- `sense="ge"`: `y >= threshold` を feasible とする。
- `sense="le"`: `y <= threshold` を feasible とする。
- `sense="eq"`: `abs(y - threshold) <= margin` を feasible とする。
