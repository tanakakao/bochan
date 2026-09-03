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

`MACERelaxationAcquisitionSelector` goes one step further and evaluates bochan acquisition functions over the finite relaxed candidate bank. The supplied `bundle_factory` receives the final relaxed structures and must return a `ModelBundle` configured for that bank.

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

The intended Phase 15 acquisition family is the finite-candidate BO set: qEI / qLogEI, qPI, qUCB, qNEI / qLogNEI. Acquisitions that require additional terminal sets, fantasies, or specialized one-shot state should be integrated explicitly rather than assumed to work through this simple discrete selector.

## Process variables

Optional `process_X` must have one row per relaxed structure. Both the posterior ranker and acquisition selector prepend the canonical structure selector column:

```text
X = [structure_index, process variables ...]
```

The structure index is rebuilt as `0..n-1` for the relaxed bank.

## Scope

Phase 15 handles scalar posterior ranking and scalar acquisition-based discrete selection after MACE relaxation. Multi-objective acquisition selection and differentiable `MACE + GP residual` relaxation remain separate future work. The relaxation step itself still uses MACE + ASE; bochan decides which relaxed candidates should be evaluated next.
