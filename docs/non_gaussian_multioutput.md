# Non-Gaussian multi-output regression

Bochan distinguishes correlated multitask models from independent objective
lists. The Gamma, Poisson, Negative Binomial, and Beta implementations share
these contracts; they are sparse variational non-Gaussian models and must not be
confused with BoTorch's exact Gaussian `MultiTaskGP` or
`KroneckerMultiTaskGP`.

| model type | input format | missing targets | output correlation | intended use |
| --- | --- | ---: | ---: | --- |
| multitask | long + task feature | no | yes | irregular task observations |
| wide multitask | wide `[n, m]` | yes | yes | partially observed wide data |
| Kronecker | wide complete block | no | yes | complete aligned observations |
| model list | wide or per-output | per submodel | no | independent objectives |

`*_multitask` requires scalar long targets and an integer `task_feature`.
`*_wide_multitask` converts only finite cells to long observations; it never
imputes a target. `*_kronecker` uses the separable ICM covariance
`K_x ⊗ K_task` with sparse variational inference, rather than exact Gaussian
Kronecker inference. `NonGaussianModelList` retains each submodel's native
response posterior in a `PosteriorList`.

Target supports are strict: Beta is `0 < y < 1`, Gamma is `y > 0`, and Poisson
and Negative Binomial are non-negative integer-valued (integer-like floating
point values are accepted). Validation ignores NaNs only for wide multitask
models.

The registry keys are `beta|gamma|poisson|negative_binomial` followed by
`_multitask`, `_wide_multitask`, or `_kronecker`. The former `*_multitask`
wide-input interpretation is a breaking change; callers with wide data must use
the explicit `*_wide_multitask` key.

Fantasy-dependent acquisitions remain available only when every underlying
model implements valid conditioning/fantasizing. `NonGaussianModelList` reports
unsupported submodels instead of substituting a Gaussian proxy.
