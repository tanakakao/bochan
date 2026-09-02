# Material `train_Yvar` Phase 2: correlated multitask fixed noise

Phase 2 extends known observation variance to bochan's **correlated multitask** material GP/DKL models.

## Contract

- `train_Yvar` means **variance**, not standard deviation.
- Correlated multitask targets use wide `train_Y` and `train_Yvar` with shape `[n_samples, n_tasks]`, with at least two tasks.
- The default outcome transform is applied to `train_Y` and `train_Yvar` before constructing the likelihood, so fixed variance is expressed in the same transformed space used by the GP.
- A user-supplied `likelihood` remains authoritative. Automatic fixed noise is selected only when `likelihood=None` and `train_Yvar` is supplied.

## Event ordering

The correlated DeepKernel models return GPyTorch's default interleaved `MultitaskMultivariateNormal`. Its covariance event order is `x0/task0, x0/task1, ..., x1/task0, ...`. The public variance tensor remains wide; `MultitaskFixedNoiseGaussianLikelihood` flattens it only at the likelihood boundary so diagonal noise is aligned with the covariance event order.

## Supported correlated material families

Phase 2 enables this contract for MACE, CHGNet, M3GNet, ALIGNN, and CrabNet, including correlated mixed-process and DKL variants where those classes already exist. Roost is not included because bochan does not currently expose a correlated Roost multitask model.

## Posterior observation noise

`posterior(..., observation_noise=False)` remains the latent-function posterior. To add known test-time observation noise, pass a wide tensor with shape `[q, m]` plus optional batch dimensions. Calling the fixed-noise likelihood on a different event size without explicit test noise is treated as a no-op, matching GPyTorch's scalar fixed-noise behavior.

## Scope boundary

This phase establishes the model-layer correlated multitask contract. High-level tabular variance-column ingestion and FastAPI request/schema plumbing remain separate work.
