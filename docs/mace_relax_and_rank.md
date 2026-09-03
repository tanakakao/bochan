# MACE relax and bochan rank

`MACERelaxationRanker` separates two responsibilities:

1. pretrained MACE + ASE relaxes each initial crystal structure;
2. a bochan / BoTorch scalar posterior ranks the relaxed structure bank.

The relaxed structures are always passed to `model_factory` in their final order. This avoids silently evaluating relaxed geometries with a stale structure-index bank.

```python
from bochan.models.regression.gaussian.materials.structure import MACERelaxationRanker

ranker = MACERelaxationRanker(model_name="medium-mpa-0", device="cpu")
result = ranker.run(
    initial_structures,
    model_factory=build_bochan_model_for_relaxed_bank,
    direction="minimize",
    criterion="posterior_mean",
    optimizer="FIRE",
    fmax=0.05,
    max_steps=200,
)

best = result.best
print(best.posterior_mean)
print(best.relaxation.structure)
```

## Ranking criteria

- `posterior_mean`: exploitation-only ranking using the scalar posterior mean.
- `ucb`: uncertainty-aware utility. For maximization it uses `mean + beta * std`; for minimization it uses `-mean + beta * std`, equivalent to favoring a low optimistic confidence bound while keeping the internal ranking score larger-is-better.

`beta=0` makes `ucb` equivalent to posterior-mean ordering.

## Acquisition-function selection

`MACERelaxationAcquisitionSelector` evaluates bochan acquisition functions over the finite relaxed candidate bank. The supplied `bundle_factory` receives the final relaxed structures and must return a `ModelBundle` configured for that bank.

```python
from bochan.api.configs import AcquisitionConfig, ObjectiveConfig
from bochan.models.regression.gaussian.materials.structure import (
    MACERelaxationAcquisitionSelector,
)

selector = MACERelaxationAcquisitionSelector(
    model_name="medium-mpa-0",
    device="cpu",
)

result = selector.run(
    initial_structures,
    bundle_factory=build_bundle_for_relaxed_bank,
    acquisition_config=AcquisitionConfig(
        name="qlogei",
        objective_config=ObjectiveConfig(direction="minimize"),
    ),
    q=3,
)
```

The selector resolves the acquisition through bochan's acquisition registry, builds it through the public acquisition construction path, and then uses BoTorch `optimize_acqf_discrete` over the relaxed structures. This supports true batch selection rather than sorting only pointwise posterior means.

### Bayesian optimization acquisitions

The supported finite-candidate BO family is:

- qEI / qLogEI
- qPI
- qUCB
- qNEI / qLogNEI

### Active learning acquisitions

The same relaxed structure bank can be selected by information value rather than predicted optimum value.

```python
result = selector.run(
    initial_structures,
    bundle_factory=build_bundle_for_relaxed_bank,
    acquisition_config=AcquisitionConfig(name="variance"),
    q=2,
)
```

Supported active-learning names are:

- `variance` / posterior variance: prioritize uncertain relaxed structures;
- `predictiveentropy`: prioritize high predictive observation entropy;
- `bald`: Gaussian-regression mutual-information acquisition, with bochan's variance fallback when noisy posterior semantics are unavailable;
- `nipv`: negative integrated posterior variance. When `DataContext.mc_points` is not supplied, the relaxed candidate bank itself is used as the integration set.

This enables a workflow such as:

```text
initial structures
  -> MACE + ASE relaxation
  -> relaxed structure bank
  -> bochan surrogate
  -> variance / predictive entropy / BALD / NIPV
  -> choose informative structures
  -> DFT or experiment
```

Acquisitions that require additional terminal sets or specialized one-shot state, such as KG, MES, JES, and multi-step lookahead, are deliberately rejected by this selector. Multi-objective acquisitions are also outside this phase.

## Process variables

Optional `process_X` must have one row per relaxed structure. Both the posterior ranker and acquisition selector prepend the canonical structure selector column:

```text
X = [structure_index, process variables ...]
```

The structure index is rebuilt as `0..n-1` for the relaxed bank.

## Scope

Phase 15 is the feature boundary for this workflow. It provides:

- pretrained MACE + ASE relaxation;
- posterior ranking of relaxed structures;
- finite-candidate Bayesian-optimization selection;
- finite-candidate active-learning selection with variance, predictive entropy, BALD, and NIPV.

The relaxation step itself still uses MACE + ASE. bochan decides which relaxed candidates should be evaluated next. Multi-objective selection, automatic DFT execution / feedback loops, and differentiable `MACE + GP residual` relaxation are intentionally left outside this feature set.
