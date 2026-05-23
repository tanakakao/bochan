# BoTorch-style fit helpers

This folder contains model-specific fit helpers organized by model family.

## Design

- Use `mll` as the primary fitting input where possible.
- Keep model-family-specific files:
  - `classification.py`
  - `deepgp.py`
  - `deepkernel.py`
  - `ordinal.py`
  - `rrp_classification.py`
- Keep shared tensor/target/train-mode utilities in `common.py`.
- Avoid changing the actual optimization behavior more than necessary.
- Return `mll` for new BoTorch-style helpers.

## Classification

```python
from gpytorch.mlls import VariationalELBO
from botorch_fit_helpers import fit_classifier_mll

mll = VariationalELBO(
    likelihood=model.likelihood,
    model=model,
    num_data=train_X.shape[-2],
)
fit_classifier_mll(mll, lr=0.01, num_epochs=300)
```

## DeepGP

```python
from botorch_fit_helpers import fit_deepgp_mll

fit_deepgp_mll(mll, lr=0.01, num_epochs=100)
```

`epoch=` is still accepted as a backward-compatible alias.

## DeepKernel

```python
from botorch_fit_helpers import fit_deepkernel_mll

fit_deepkernel_mll(mll, lr=0.01, num_epochs=100)
```

`epoch=` is still accepted as a backward-compatible alias.

## Ordinal GP

Recommended mll-first usage:

```python
from botorch_fit_helpers import make_ordinal_mll, fit_ordinal_mll

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
from botorch_fit_helpers import fit_ordinal_gp

fit_ordinal_gp(model)
```

## RRP classification

```python
from botorch_fit_helpers import fit_rrp_classifier_mll

fit_rrp_classifier_mll(
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
from botorch_fit_helpers import make_ordinal_mll, fit_rrp_ordinal_mll

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
