# Unified MLIP workflows

bochan provides a backend-neutral materials layer for pretrained machine-learning interatomic potentials (MLIPs), Gaussian-process residual correction, structure relaxation, and relaxed-structure selection.

This page is the current user-facing reference for the MLIP work that was implemented incrementally across the older phase documents.

## Supported backends

| Backend | Direct energy | Direct force | Direct stress | Residual GP | Relaxation | Relax + rank | Relax + acquisition |
|---|---:|---:|---:|---:|---:|---:|---:|
| MACE | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| CHGNet | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| M3GNet / MatGL | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| ALIGNN-FF | Yes | Yes | Yes | Yes | Yes | Yes | Yes |

Canonical backend names are `mace`, `chgnet`, `m3gnet`, and `alignn-ff`. `alignn_ff` and `alignnff` are accepted aliases.

## Core concepts

The unified material API is defined by four orthogonal axes:

- **backend**: which pretrained MLIP family supplies the physical baseline;
- **quantity**: `energy`, `force`, or `stress`;
- **model mode**: `direct` or `residual_gp`;
- **workflow mode**: `model_only`, `relax_rank`, or `relax_acquisition`.

The normalized identity can be represented with `MaterialWorkflowSpec`.

```python
from bochan.models.regression.gaussian.materials.structure.workflow_factory import (
    MaterialWorkflowSpec,
)

spec = MaterialWorkflowSpec(
    backend="alignn_ff",
    quantity="energy",
    model_mode="residual-gp",
    workflow_mode="bo",
)

assert spec.as_dict() == {
    "backend": "alignn-ff",
    "quantity": "energy",
    "model_mode": "residual_gp",
    "workflow_mode": "relax_acquisition",
}
```

## Direct prediction

Use a direct predictor when the pretrained model itself is the desired surrogate.

```python
from bochan.models.regression.gaussian.materials.structure.property_factory import (
    create_direct_material_predictor,
)

predictor = create_direct_material_predictor(
    "mace",
    "energy",
    structures=structures,
)
```

The first model input column is the structure index. Additional columns may represent process variables, but they do not change a purely pretrained baseline unless the selected model explicitly uses them.

## Residual Gaussian process

Residual-GP mode learns a correction on top of the deterministic MLIP prediction:

```text
observed property = pretrained MLIP baseline + GP residual
```

Use `create_material_residual_gp` or the higher-level `create_material_model` factory.

```python
from bochan.models.regression.gaussian.materials.structure.model_factory import (
    create_material_model,
)

model = create_material_model(
    "chgnet",
    "energy",
    "residual_gp",
    structures=structures,
    train_X=train_X,
    train_Y=train_Y,
)
```

For residual models, `train_X[:, 0]` identifies the structure in the supplied structure bank. The remaining columns can represent process conditions.

ALIGNN-FF residual GPs additionally require `structure_graphs`, in the same order as `structures`, because the physical baseline is evaluated through ALIGNN-FF while the GP correction reuses the existing ALIGNN graph encoder.

```python
model = create_material_model(
    "alignn-ff",
    "energy",
    "residual_gp",
    structures=structures,
    structure_graphs=structure_graphs,
    train_X=train_X,
    train_Y=train_Y,
)
```

## Force and stress targets

Force residuals currently use a fixed-topology representation. For a structure bank with `N` atoms, force targets are flattened to `3N` correlated outputs. All structures in the bank must therefore use the same atom count.

Stress uses the full 3 x 3 tensor and is represented as 9 outputs.

bochan does not silently convert stress units or stress signs between backend conventions. Training targets must use the same physical definition as the selected pretrained baseline.

## Structure relaxation

The common structure-relaxation factory is `create_structure_relaxer`.

```python
from bochan.models.regression.gaussian.materials.structure.factory import (
    create_structure_relaxer,
)

relaxer = create_structure_relaxer("mace")
result = relaxer.relax(
    structure,
    optimizer="FIRE",
    fmax=0.05,
    max_steps=200,
    relax_cell=False,
)
```

All four backends expose the same common relaxation controls:

| Parameter | Default | Meaning |
|---|---:|---|
| `optimizer` | `FIRE` | ASE optimizer: `FIRE`, `BFGS`, or `LBFGS` |
| `fmax` | `0.05` | Force convergence threshold |
| `max_steps` | `200` | Maximum optimization steps |
| `relax_cell` | `False` | Optimize cell degrees of freedom through `FrechetCellFilter` |

The result contains the relaxed structure, final and initial energy, forces, full stress tensor, maximum force, step count, convergence status, optimizer settings, backend, and model name.

## Relax and rank

`MaterialRelaxationRanker` first relaxes all candidates, rebuilds the final relaxed structure bank, constructs the model against that exact bank, and then ranks posterior predictions.

```python
from bochan.models.regression.gaussian.materials.structure.factory import (
    create_relaxation_ranker,
)

ranker = create_relaxation_ranker("mace")
result = ranker.run(
    candidate_structures,
    model_factory=model_factory,
    direction="minimize",
    criterion="posterior_mean",
)
```

Supported ranking criteria are `posterior_mean` and `ucb`.

A critical ordering contract applies: the model factory receives the final relaxed structures in exactly the order used for structure indices `0..n-1`.

## Relax and acquisition-select

`MaterialRelaxationAcquisitionSelector` combines relaxation with discrete Bayesian-optimization or active-learning selection.

```python
from bochan.api import AcquisitionConfig
from bochan.models.regression.gaussian.materials.structure.factory import (
    create_relaxation_acquisition_selector,
)

selector = create_relaxation_acquisition_selector("chgnet")
result = selector.run(
    candidate_structures,
    bundle_factory=bundle_factory,
    acquisition_config=AcquisitionConfig(name="qEI"),
    q=3,
)
```

The relaxed-structure selector currently supports EI/logEI, PI, UCB, NEI/logNEI, and active-learning variance, predictive entropy, BALD, and NIPV. KG, MES, JES, multi-step lookahead, and multi-objective relaxed-structure selection are intentionally not enabled because they require specialized discrete semantics.

## Capability discovery

Use the capability registry when building a UI or service that needs to know what a backend supports.

```python
from bochan.models.regression.gaussian.materials.structure.capabilities import (
    get_material_backend_capabilities,
    get_material_capability_catalog,
)

catalog = get_material_capability_catalog()
requirements = get_material_backend_capabilities("alignn_ff").requirements(
    quantity="force",
    model_mode="residual_gp",
)
```

The requirement set for ALIGNN-FF residual force includes `structures`, `train_X`, `train_Y`, `structure_graphs`, and the fixed-atom-count constraint.

## FastAPI workflow endpoints

The unified serving surface uses these endpoints:

```text
GET  /api/v1/materials/mlip/capabilities
GET  /api/v1/materials/mlip/capabilities/{backend}
POST /api/v1/materials/mlip/workflows/validate
POST /api/v1/materials/mlip/workflows/configure
POST /api/v1/materials/mlip/workflows/execute/relaxation
```

`validate` performs dependency-light normalization and requirement discovery. `configure` adds validated common relaxation settings. `execute/relaxation` is the runtime endpoint that lazily creates the real selected MLIP backend and executes ASE relaxation.

See [FastAPI reference](../reference/fastapi.md) for request-shape examples.

## Installation notes

Core MLIP packages are optional because they are heavy and may download pretrained weights.

```bash
# MACE
pip install -e ".[mace]"

# CHGNet / MACE / MatGL and other materials dependencies
pip install -e ".[materials]"

# ALIGNN-FF is currently installed explicitly
pip install "alignn==2026.8.11"
```

ALIGNN is intentionally not currently included in the `materials` extra.

## Historical design documents

The following older documents are retained as implementation history rather than as the primary API reference:

- `material_models_architecture_phase*.md`
- `material_mixed_residual_gp_phase4.md`
- `material_baseline_spec_phase9.md`
- `mace_phase*_*.md`
- `chgnet_residual_gp_phase1.md`
- `m3gnet_residual_gp.md`
- `mace_relax_and_rank.md`

When current behavior and a historical phase note differ, this consolidated guide and the public source code should be treated as authoritative.
