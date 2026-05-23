from __future__ import annotations

from typing import Optional

import torch
from gpytorch.mlls import MarginalLogLikelihood, PredictiveLogLikelihood, VariationalELBO

from .common import (
    build_tensor_dataloader,
    get_fit_train_X,
    get_fit_train_Y,
    get_likelihood_from_mll_or_model,
    maybe_clip_grad_norm,
    move_batch_like,
    set_model_and_likelihood_eval_mode,
    set_model_and_likelihood_train_mode,
)


def _get_ordinal_cutpoints(model, likelihood=None):
    """Return ordinal cutpoints when available, otherwise None."""
    if hasattr(model, "ordinal_likelihood") and hasattr(model.ordinal_likelihood, "cutpoints"):
        return model.ordinal_likelihood.cutpoints
    if likelihood is not None and hasattr(likelihood, "cutpoints"):
        return likelihood.cutpoints
    if hasattr(model, "likelihood") and hasattr(model.likelihood, "cutpoints"):
        return model.likelihood.cutpoints
    return None


def _get_mll_model_for_ordinal(model):
    """
    Return the model object to be passed to VariationalELBO / PredictiveLogLikelihood.

    Wrappers such as PCAOrdinalGPModel often expose the underlying ApproximateGP
    as `model.model`, while their own forward accepts raw X.  The MLL should keep
    the underlying approximate GP, but fitting should call the wrapper forward.
    """
    return getattr(model, "model", model)


def make_ordinal_mll(
    model,
    *,
    use_predictive_log_likelihood: bool = False,
    num_data: Optional[int] = None,
):
    """
    Build an ordinal approximate MLL from a model or wrapper.

    For PCA / random-projection wrappers:
        mll = make_ordinal_mll(wrapper)
        fit_ordinal_mll(mll, fit_model=wrapper)

    For ordinary ordinal models:
        mll = make_ordinal_mll(model)
        fit_ordinal_mll(mll)
    """
    train_X = get_fit_train_X(model)
    if num_data is None:
        num_data = int(train_X.shape[-2])

    likelihood = getattr(model, "likelihood", None)
    if likelihood is None and hasattr(model, "ordinal_likelihood"):
        likelihood = model.ordinal_likelihood
    if likelihood is None:
        raise AttributeError("Could not find `likelihood` or `ordinal_likelihood` on ordinal model.")

    mll_model = _get_mll_model_for_ordinal(model)
    mll_cls = PredictiveLogLikelihood if use_predictive_log_likelihood else VariationalELBO

    return mll_cls(
        likelihood=likelihood,
        model=mll_model,
        num_data=int(num_data),
    )


def fit_ordinal_mll(
    mll,
    *,
    fit_model=None,
    lr: Optional[float] = None,
    num_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    verbose: Optional[bool] = None,
    optimizer_cls= torch.optim.Adam,
    clip_grad_norm: Optional[float] = None,
    **ignore,
):
    """
    Fit an ordinal variational GP from an MLL.

    Important:
        `mll.model` may be the underlying ApproximateGP, while `fit_model` may be
        the wrapper that accepts raw X.  This is useful for PCA / random-projection
        ordinal wrappers.

    Args:
        mll:
            VariationalELBO / PredictiveLogLikelihood.
        fit_model:
            Optional wrapper/model to call in the training loop.
            If omitted, `mll.model` is used.
        lr:
            Learning rate. Default is 0.03 to preserve the existing ordinal helper.
        num_epochs:
            Number of epochs. Default is 300 to preserve the existing ordinal helper.
        batch_size:
            Mini-batch size. Defaults to full-batch.
        shuffle:
            Whether to shuffle the TensorDataset.
        verbose:
            If True, prints loss and cutpoints.
        optimizer_cls:
            Optimizer class. Defaults to Adam.
        clip_grad_norm:
            Optional gradient clipping value.

    Returns:
        The input `mll`.
    """
    if num_epochs is None:
        num_epochs = 300
    else:
        num_epochs = int(num_epochs)

    if lr is None:
        lr = 0.03
    else:
        lr = float(lr)

    if verbose is None:
        verbose = False
    else:
        verbose = bool(verbose)

    model = mll.model if fit_model is None else fit_model
    likelihood = get_likelihood_from_mll_or_model(mll, model)

    if num_epochs <= 0:
        set_model_and_likelihood_eval_mode(model, likelihood)
        if hasattr(mll, "eval"):
            mll.eval()
        return mll

    train_X = get_fit_train_X(model)
    train_Y = get_fit_train_Y(model)

    loader = build_tensor_dataloader(
        train_X=train_X,
        train_Y=train_Y,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    set_model_and_likelihood_train_mode(model, likelihood)
    if hasattr(mll, "train"):
        mll.train()

    # optimizer = optimizer_cls(model.parameters(), lr=lr)
    optimizer = optimizer_cls(mll.parameters(), lr=lr)
    num_data = int(train_X.shape[-2])

    for epoch in range(num_epochs):
        total_loss = 0.0

        for xb, yb in loader:
            xb, yb = move_batch_like(xb, yb, train_X=train_X, train_Y=train_Y)

            optimizer.zero_grad()

            # Wrapper models should receive raw X here.
            latent_dist = model(xb)

            loss = -mll(latent_dist, yb)
            if loss.ndim > 0:
                loss = loss.sum()

            loss.backward()
            maybe_clip_grad_norm(model.parameters(), clip_grad_norm)
            optimizer.step()

            total_loss += float(loss.detach().item()) * xb.shape[0]

        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0 or epoch == num_epochs - 1):
            avg_loss = total_loss / num_data
            cutpoints = _get_ordinal_cutpoints(model, likelihood)
            if cutpoints is not None:
                cuts = cutpoints.detach().cpu().numpy()
                print(f"[fit_ordinal_mll] epoch={epoch + 1:03d} loss={avg_loss:.4f} cutpoints={cuts}")
            else:
                print(f"[fit_ordinal_mll] epoch={epoch + 1:03d} loss={avg_loss:.4f}")

    set_model_and_likelihood_eval_mode(model, likelihood)
    if hasattr(mll, "eval"):
        mll.eval()

    return mll


def fit_ordinal_gp(
    model_or_mll,
    *,
    num_epochs: Optional[int] = None,
    lr: Optional[float] = None,
    batch_size: Optional[int] = None,
    verbose: Optional[bool] = None,
    use_predictive_log_likelihood: Optional[bool] = None,
    **kwargs,
):
    """
    Backward-compatible ordinal fitting helper.

    New recommended usage:
        mll = make_ordinal_mll(model)
        fit_ordinal_mll(mll, fit_model=model)  # fit_model is needed for PCA/RP wrappers

    Old usage still works:
        fit_ordinal_gp(model)

    If `model_or_mll` is an MLL, this function delegates to `fit_ordinal_mll`.
    If `model_or_mll` is a model/wrapper, this function builds the MLL and returns the model,
    preserving the previous helper's return style.
    """
    if use_predictive_log_likelihood is None:
        use_predictive_log_likelihood = False
    else:
        use_predictive_log_likelihood = bool(use_predictive_log_likelihood)

    if isinstance(model_or_mll, MarginalLogLikelihood):
        return fit_ordinal_mll(
            model_or_mll,
            num_epochs=num_epochs,
            lr=lr,
            batch_size=batch_size,
            verbose=verbose,
            **kwargs,
        )

    model = model_or_mll
    mll = make_ordinal_mll(
        model,
        use_predictive_log_likelihood=use_predictive_log_likelihood,
    )
    fit_ordinal_mll(
        mll,
        fit_model=model,
        num_epochs=num_epochs,
        lr=lr,
        batch_size=batch_size,
        verbose=verbose,
        **kwargs,
    )
    return model


# Optional placeholder retained for future SAAS ordinal fitting.
# Fully Bayesian SAAS ordinal wrappers often need a two-stage procedure:
#   1. fit the underlying SAAS regression model with NUTS
#   2. calibrate ordinal cutpoints
# Keep that implementation model-specific, because the cutpoint calibration API
# differs across wrappers.
