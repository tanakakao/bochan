# BoTorch-style fit helpers

This folder contains model-specific fit helpers organized by model family.

## Layout

```text
src/bochan/fit/
  __init__.py
  common.py

  classification/
    __init__.py
    binary.py
    multiclass.py

  ordinal.py

  deep/
    __init__.py
    common.py
    deepgp.py
    deepkernel.py

  robust/
    __init__.py
    rrp_binary.py
    rrp_ordinal.py

  non_gaussian.py
```

## Design

- Use `mll` as the primary fitting input where possible.
- Keep model-family-specific helpers under subpackages.
- Keep shared tensor/target/train-mode utilities in `common.py`.
- Keep DeepGP / DeepKernel full-batch loop sharing in `deep/common.py`.
- Keep RRP helpers separate from ordinary classification / ordinal fitting.
- Keep Poisson / Gamma / Beta / NegativeBinomial aliases in `non_gaussian.py`.
- Avoid changing the actual optimization behavior more than necessary.
- Return `mll` for BoTorch-style helpers when the existing behavior already did so.

## Public imports

The root package keeps the main helpers available from `bochan.fit`:

```python
from bochan.fit import (
    fit_binary_classifier_mll,
    fit_multiclass_mll,
    fit_deepgp_mll,
    fit_deepkernel_mll,
    make_ordinal_mll,
    fit_ordinal_mll,
    fit_ordinal_gp,
    fit_rrp_binary_classifier_mll,
    fit_rrp_ordinal_mll,
    fit_non_gaussian_mll,
    fit_beta_mll,
    fit_gamma_mll,
    fit_poisson_mll,
    fit_negative_binomial_mll,
)
```

Subpackage imports are also available when a more explicit path is useful:

```python
from bochan.fit.classification import fit_binary_classifier_mll, fit_multiclass_mll
from bochan.fit.deep import fit_deepgp_mll, fit_deepkernel_mll
from bochan.fit.robust import fit_rrp_binary_classifier_mll, fit_rrp_ordinal_mll
from bochan.fit.non_gaussian import fit_poisson_mll
```

## Classification

```python
from gpytorch.mlls import VariationalELBO
from bochan.fit import fit_binary_classifier_mll

mll = VariationalELBO(
    likelihood=model.likelihood,
    model=model,
    num_data=train_X.shape[-2],
)
fit_binary_classifier_mll(mll, lr=0.01, num_epochs=300)
```

For multiclass classification:

```python
from bochan.fit import fit_multiclass_mll

losses = fit_multiclass_mll(mll, lr=0.01, num_epochs=300)
```

## DeepGP / DeepKernel

```python
from bochan.fit import fit_deepgp_mll, fit_deepkernel_mll

fit_deepgp_mll(mll, lr=0.01, num_epochs=100)
fit_deepkernel_mll(mll, lr=0.01, num_epochs=100)
```

`epoch=` is still accepted as a backward-compatible alias.

## Ordinal GP

Recommended mll-first usage:

```python
from bochan.fit import make_ordinal_mll, fit_ordinal_mll

mll = make_ordinal_mll(model, use_predictive_log_likelihood=False)
fit_ordinal_mll(mll, fit_model=model, lr=0.03, num_epochs=300)
```

For PCA / random-projection wrappers, pass the wrapper to `fit_model`:

```python
mll = make_ordinal_mll(wrapper_model)
fit_ordinal_mll(mll, fit_model=wrapper_model)
```

Reason:
- The MLL often needs the underlying approximate GP (`wrapper_model.model`).
- The training loop should call the wrapper forward with raw X.

Backward-compatible usage is still available:

```python
from bochan.fit import fit_ordinal_gp

fit_ordinal_gp(model)
```

## RRP binary classification

```python
from bochan.fit import fit_rrp_binary_classifier_mll

fit_rrp_binary_classifier_mll(
    mll,
    method="backward",
    sparsity_levels=[1, 2, 4, 8],
    optimizer_kwargs={"lr": 0.01, "num_epochs": 300},
)
```

The RRP optimizer intentionally uses full-batch training because the sparse outlier likelihood can be tied to the full training set.

## RRP ordinal

Use this when your ordinal likelihood / sparse module inherits `RelevancePursuitMixin`.

```python
from bochan.fit import make_ordinal_mll, fit_rrp_ordinal_mll

mll = make_ordinal_mll(model)

fit_rrp_ordinal_mll(
    mll,
    fit_model=model,  # important for PCA / RP wrappers
    method="backward",
    sparsity_levels=[1, 2, 4, 8],
    optimizer_kwargs={"lr": 0.03, "num_epochs": 300},
)
```

The RRP ordinal optimizer intentionally uses full-batch training, matching the RRP classification helper.

## Non-Gaussian GP

Poisson / Gamma / Beta / NegativeBinomial fit helpers are consolidated in `non_gaussian.py`.

```python
from bochan.fit import fit_non_gaussian_mll, fit_poisson_mll, fit_gamma_mll

losses = fit_non_gaussian_mll(mll, lr=0.01, num_epochs=300)
losses = fit_poisson_mll(mll, lr=0.01, num_epochs=300)
losses = fit_gamma_mll(mll, lr=0.01, num_epochs=300)
```
