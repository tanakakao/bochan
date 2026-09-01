# Material GP known observation variance — Phase 1

Phase 1 adds a shared fixed-noise contract for bochan's Gaussian material GP/DKL families.

## Meaning of `train_Yvar`

`train_Yvar` is the **known observation variance** for each training target, not a standard deviation. When it is supplied and no custom likelihood is passed, the shared Gaussian DeepKernel layer constructs `gpytorch.likelihoods.FixedNoiseGaussianLikelihood` from the outcome-transformed variance.

When `train_Yvar` is omitted, existing learned-noise behavior is unchanged.

## Phase 1 support

The scalar-output GP/DKL paths now accept `train_Yvar` for:

- MACE
- CHGNet
- M3GNet
- ALIGNN
- CrabNet
- Roost

Mixed-input variants that inherit the shared scalar Gaussian DeepKernel contract use the same fixed-noise behavior. Independent multi-output construction can reuse these scalar-output models once its caller provides one variance column per output.

## Explicit scope boundaries

Phase 1 intentionally does **not** add known-noise support to correlated multitask material models. Those models use a multitask covariance/likelihood event structure and need an explicit task-wise fixed-noise design rather than treating a wide `train_Yvar` tensor as scalar noise. That is Phase 2 work.

The high-level `TabularBayesianOptimizer` / FastAPI data contract also does not yet expose observation-variance columns. Phase 1 establishes the model-layer capability; high-level noise-column plumbing is a separate integration step.

This feature is also distinct from a learned heteroscedastic GP. `train_Yvar` means the observation variances are known inputs to the model rather than inferred as a second latent noise process.

## Custom likelihoods

An explicitly supplied likelihood remains authoritative. Automatic `FixedNoiseGaussianLikelihood` selection occurs only when `likelihood=None`.
