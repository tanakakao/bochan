from __future__ import annotations

from typing import Optional, Sequence

import torch
from botorch.models.relevance_pursuit import (
    RelevancePursuitMixin,
    backward_relevance_pursuit,
    forward_relevance_pursuit,
)

from ..common import (
    get_fit_train_X,
    get_fit_train_Y,
    get_likelihood_from_mll_or_model,
    maybe_clip_grad_norm,
    set_model_and_likelihood_eval_mode,
)


def _unique_trainable_parameters(*modules):
    """
    複数 module から trainable parameters を重複なく集める。

    RRP ordinal では sparse correction / cutpoints / likelihood parameters が
    likelihood 側に存在することがあるため、model.parameters() だけでは不十分な
    場合がある。mll / likelihood / model をまとめて optimizer 対象にする。
    """
    params = []
    seen = set()

    for module in modules:
        if module is None or not hasattr(module, "parameters"):
            continue

        for p in module.parameters():
            if not p.requires_grad:
                continue
            if id(p) in seen:
                continue
            params.append(p)
            seen.add(id(p))

    return params


def _resolve_rrp_ordinal_sparse_module(mll, fit_model=None):
    """
    Resolve the sparse module used by relevance pursuit.

    In most RRP ordinal models this should be `mll.likelihood`.
    This helper also supports wrappers where the likelihood is attached to
    `fit_model`.
    """
    if hasattr(mll, "likelihood") and isinstance(mll.likelihood, RelevancePursuitMixin):
        return mll.likelihood

    if fit_model is not None:
        likelihood = get_likelihood_from_mll_or_model(mll, fit_model)
        if isinstance(likelihood, RelevancePursuitMixin):
            return likelihood

        if hasattr(fit_model, "ordinal_likelihood") and isinstance(
            fit_model.ordinal_likelihood,
            RelevancePursuitMixin,
        ):
            return fit_model.ordinal_likelihood

    raise TypeError(
        "Could not find an ordinal sparse module inheriting RelevancePursuitMixin. "
        "Expected `mll.likelihood`, `fit_model.likelihood`, or "
        "`fit_model.ordinal_likelihood` to inherit RelevancePursuitMixin."
    )


def fit_rrp_ordinal_mll_optimizer(
    mll,
    closure=None,
    *,
    fit_model=None,
    lr: float = 0.03,
    num_epochs: int = 300,
    batch_size=None,   # kept for compatibility; intentionally unused
    shuffle: bool = True,      # kept for compatibility; intentionally unused
    optimizer_cls= torch.optim.Adam,
    clip_grad_norm: Optional[float] = None,
    verbose: bool = False,
    **ignore,
):
    """
    Optimizer callable for RRP ordinal models.

    Notes:
        RRP likelihoods usually keep sparse correction parameters associated with
        the whole training set. As in the RRP classification helper, this
        optimizer intentionally uses full-batch training.

        The optimizer includes trainable parameters from model, likelihood, and mll
        with duplicate removal. This is important because ordinal RRP parameters may
        live on the likelihood / ordinal_likelihood side.

        `fit_model` is important for PCA / random-projection wrappers:
            - `mll.model` can be the underlying ApproximateGP
            - `fit_model` should be the wrapper that accepts raw X
    """
    model = mll.model if fit_model is None else fit_model
    likelihood = get_likelihood_from_mll_or_model(mll, model)

    ref_param = next(model.parameters())
    ref_dtype = ref_param.dtype
    ref_device = ref_param.device

    model.to(device=ref_device, dtype=ref_dtype)
    if likelihood is not None:
        likelihood.to(device=ref_device, dtype=ref_dtype)
    if hasattr(mll, "to"):
        mll.to(device=ref_device, dtype=ref_dtype)

    model.train()
    if likelihood is not None:
        likelihood.train()
    mll.train()

    train_X = get_fit_train_X(model).to(device=ref_device, dtype=ref_dtype)
    train_Y = get_fit_train_Y(model).to(device=ref_device)

    if train_X.shape[-2] != train_Y.shape[0]:
        raise RuntimeError(
            "train_X and train_Y have inconsistent data sizes. "
            f"train_X.shape={tuple(train_X.shape)}, train_Y.shape={tuple(train_Y.shape)}. "
            "For wrapper models, get_fit_train_X(fit_model) should return raw X "
            "that matches get_fit_train_Y(fit_model). For inner models, "
            "mll.model.train_inputs should match mll.model.train_targets."
        )

    params = _unique_trainable_parameters(model, likelihood, mll)
    if len(params) == 0:
        raise RuntimeError("No trainable parameters were found for RRP ordinal fitting.")

    optimizer = optimizer_cls(params, lr=lr)

    num_epochs = int(num_epochs)
    num_data = int(train_X.shape[-2])

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        latent_dist = model(train_X)

        loss = -mll(latent_dist, train_Y)
        if loss.ndim > 0:
            loss = loss.sum()

        loss.backward()
        maybe_clip_grad_norm(model.parameters(), clip_grad_norm)
        optimizer.step()

        if verbose and ((epoch + 1) % 50 == 0 or epoch == 0 or epoch == num_epochs - 1):
            print(
                "[fit_rrp_ordinal_mll_optimizer] "
                f"epoch={epoch + 1:04d} loss={float(loss.item()) / num_data:.6f}"
            )

    return mll


def fit_rrp_ordinal_mll(
    mll,
    *,
    fit_model=None,
    method: str = "backward",
    sparsity_levels: Optional[Sequence[int]] = None,
    initial_support: Optional[list[int]] = None,
    reset_parameters: bool = True,
    reset_dense_parameters: bool = False,
    record_model_trace: Optional[bool] = None,
    return_all: bool = False,
    optimizer=fit_rrp_ordinal_mll_optimizer,
    optimizer_kwargs: Optional[dict] = None,
    closure=None,
    closure_kwargs: Optional[dict] = None,
):
    """
    Fit an RRP ordinal MLL via forward/backward relevance pursuit.
    """
    if method not in {"forward", "backward"}:
        raise ValueError("method must be 'forward' or 'backward'.")

    sparse_module = _resolve_rrp_ordinal_sparse_module(mll, fit_model=fit_model)

    if record_model_trace is None:
        record_model_trace = bool(return_all)

    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    else:
        optimizer_kwargs = dict(optimizer_kwargs)

    if fit_model is not None and "fit_model" not in optimizer_kwargs:
        optimizer_kwargs["fit_model"] = fit_model

    rp_fn = forward_relevance_pursuit if method == "forward" else backward_relevance_pursuit

    sparse_module, model_trace = rp_fn(
        sparse_module=sparse_module,
        mll=mll,
        sparsity_levels=None if sparsity_levels is None else list(sparsity_levels),
        reset_parameters=reset_parameters,
        reset_dense_parameters=reset_dense_parameters,
        record_model_trace=record_model_trace,
        initial_support=initial_support,
        closure=closure,
        optimizer=optimizer,
        closure_kwargs=closure_kwargs,
        optimizer_kwargs=optimizer_kwargs,
    )

    model = mll.model if fit_model is None else fit_model
    likelihood = get_likelihood_from_mll_or_model(mll, model)
    set_model_and_likelihood_eval_mode(model, likelihood)
    if hasattr(mll, "eval"):
        mll.eval()

    if return_all:
        return mll, sparse_module, model_trace
    return mll
