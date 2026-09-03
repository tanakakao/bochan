# Multiple pretrained material baselines

Bochan can represent independent material outputs where more than one target uses a deterministic pretrained baseline.

## Example

For three outputs:

```text
energy       -> MACE baseline + residual GP
band_gap     -> CHGNet-compatible baseline + residual GP
strength     -> ordinary GP
```

`MaterialBaselinePlan` resolves `MaterialBaselineSpec` values against stable output names and rejects duplicate assignments. `MultipleBaselineModelListGP` then validates that every assigned output is backed by a `ResidualMaterialGPModel` carrying the same baseline specification.

```python
from bochan.models.regression.gaussian.materials.common import (
    MaterialBaselinePlan,
    MaterialBaselineSpec,
    MaterialPropertyContract,
)

specs = [
    MaterialBaselineSpec(
        family="mace",
        output_name="energy",
        property=MaterialPropertyContract("energy", "eV", "total"),
    ),
    MaterialBaselineSpec(
        family="chgnet",
        output_name="band_gap",
        property=MaterialPropertyContract("band_gap", "eV", "intensive"),
    ),
]

plan = MaterialBaselinePlan.resolve(
    output_names=["energy", "band_gap", "strength"],
    baseline_specs=specs,
)
```

The plan reports which outputs use pretrained baselines and which remain ordinary GP outputs. Baseline families may differ between outputs.

## Safety contract

Multiple baselines do not imply automatic unit conversion. Each residual submodel still owns a `MaterialBaselineSpec`, and the physical quantity/unit/aggregation contract introduced for residual baselines remains authoritative.

## Scope

This phase establishes the model-layer composition and validation contract. FastAPI request schemas and automatic family-specific predictor construction are intentionally left to the next API-routing phase so the model contract remains independent of serving concerns.
