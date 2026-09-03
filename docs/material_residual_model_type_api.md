# Material Residual GP public `model_type` routing

CHGNet, M3GNet, and MACE residual Gaussian models can be selected through the
high-level `bochan.api.ModelConfig` without passing `model_cls` manually.
Concrete optional material backends remain lazily imported.

## Public model types

Normal inputs:

- `chgnet_residual_gp`
- `chgnet_multitask_residual_gp`
- `m3gnet_residual_gp`
- `m3gnet_multitask_residual_gp`
- `mace_residual_gp`
- `mace_multitask_residual_gp`

Mixed process inputs:

- `chgnet_mixed_residual_gp`
- `chgnet_mixed_multitask_residual_gp`
- `m3gnet_mixed_residual_gp`
- `m3gnet_mixed_multitask_residual_gp`
- `mace_mixed_residual_gp`
- `mace_mixed_multitask_residual_gp`

The scalar residual variants are registered for `task_type="regression"`.
Correlated multitask variants are available for both `regression` and
`multi_objective`. A scalar residual model is intentionally not exposed as a
multi-objective model because its pretrained predictor has one output.

## Examples

Single-output MACE residual:

```python
from bochan.api import ModelConfig

config = ModelConfig(
    task_type="regression",
    model_type="mace_residual_gp",
    outcome_transform=False,
    model_kwargs={
        "structures": structures,
        "model_name": "medium-mpa-0",
        "head": "Default",
    },
)
```

Mixed CHGNet residual:

```python
config = ModelConfig(
    task_type="regression",
    model_type="chgnet_mixed_residual_gp",
    cat_dims=[3],
    outcome_transform=False,
    model_kwargs={"structures": structures},
)
```

Correlated multi-output M3GNet residual:

```python
config = ModelConfig(
    task_type="multi_objective",
    model_type="m3gnet_multitask_residual_gp",
    outcome_transform=False,
    model_kwargs={
        "structures": structures,
        "pretrained_output_index": 0,
    },
)
```

For wide targets such as `[energy, strength, conductivity]`, setting
`pretrained_output_index=0` subtracts the pretrained scalar baseline from the
energy column only. The remaining columns use a zero baseline and the existing
correlated multitask GP learns their joint residual covariance.

## Compatibility and serialization

Existing public `model_type` names are not replaced. Registration is idempotent
and fails if a new material name would collide with a different existing path.
The new entries point to classes under
`bochan.models.regression.gaussian.materials.structure`; historical
`gaussian.deep.*` classes are not moved or renamed.

The common `.bochan.pt` artifact format is unchanged because it serializes the
fitted optimizer/model object. The residual model classes retain stable canonical
module paths and are covered by pickle class round-trip contract tests.

## Scope

This phase only integrates the core high-level Python model factory and public
`model_type` routing. FastAPI request schemas/endpoints and Web model selectors
are separate follow-up phases.
