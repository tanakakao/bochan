# Material `train_Yvar` Phase 2: correlated multitask fixed noise

Phase 2 extends known observation variance to bochan's **correlated multitask** material GP/DKL models.

## Contract

- `train_Yvar` means **variance**, not standard deviation.
- Correlated multitask targets use wide `train_Y` and `train_Yvar` with shape `[n_samples, n_tasks]`, with at least two tasks.
- Variances must be finite and strictly positive.
- The default outcome transform is applied to `train_Y` and `train_Yvar` before constructing the likelihood, so fixed variance is expressed in the same transformed space used by the GP.
- A user-supplied `likelihood` remains authoritative. Automatic fixed noise is selected only when `likelihood=None` and `train_Yvar` is supplied.
- Omitting `train_Yvar` preserves the existing learned-noise `MultitaskGaussianLikelihood` path.

## Event ordering

The public variance tensor always remains wide as `[..., n, m]`; callers never flatten it. `MultitaskFixedNoiseGaussianLikelihood` converts the wide tensor only at the likelihood/covariance boundary according to `MultitaskMultivariateNormal._interleaved`:

- interleaved layout: `x0/task0, x0/task1, ..., x1/task0, ...`
- non-interleaved layout: `task0/x0, task0/x1, ..., task1/x0, ...`

Both event layouts are covered by the Phase 2 regression tests.

## Supported correlated material families

Phase 2 enables this contract for MACE, CHGNet, M3GNet, ALIGNN, and CrabNet, including correlated mixed-process and DKL variants where those classes already exist. Roost is not included because bochan does not currently expose a correlated Roost multitask model.

## Posterior observation noise

`posterior(..., observation_noise=False)` remains the latent-function posterior. To add known test-time observation noise, pass a wide tensor with shape `[q, m]` plus optional batch dimensions. Calling the fixed-noise likelihood on a different event size without explicit test noise is treated as a no-op, matching GPyTorch's scalar fixed-noise behavior.

Fantasy observations likewise use wide `noise=[..., q, m]`; the likelihood appends that variance along the data axis while preserving task ordering.

## Validation

The focused Phase 2 suite validates wide fixed-noise construction, interleaved and non-interleaved covariance ordering, transformed variance scaling, Exact MLL/backward, posterior observation noise, invalid variance handling, fantasy-noise concatenation, and pass-through from all five correlated material model families.

```bash
python -m pytest tests/test_material_train_yvar_phase2.py -q
```

The normal repository pull-request CI remains responsible for broader model and API regressions.

## Scope boundary

This phase establishes the model-layer correlated multitask contract. High-level tabular variance-column ingestion and FastAPI request/schema plumbing remain separate work.
