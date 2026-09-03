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

## Process variables

Optional `process_X` must have one row per relaxed structure. The ranker prepends the canonical structure selector column:

```text
X = [structure_index, process variables ...]
```

The structure index is rebuilt as `0..n-1` for the relaxed bank.

## Scope

Phase 15 ranks one scalar posterior. Multi-objective acquisition ranking and differentiable `MACE + GP residual` relaxation remain separate future work. The relaxation step itself still uses MACE + ASE; the bochan contribution in this phase is posterior-based selection of the relaxed candidate bank.
