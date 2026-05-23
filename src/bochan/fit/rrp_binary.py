from __future__ import annotations

from typing import Optional, Sequence

import torch
from botorch.models.relevance_pursuit import (
    RelevancePursuitMixin,
    backward_relevance_pursuit,
    forward_relevance_pursuit,
)

from .common import (
    get_train_inputs_tensor,
    get_train_targets_tensor,
    maybe_clip_grad_norm,
    set_mll_eval_mode,
)


def fit_rrp_binary_classifier_mll_optimizer(
    mll,
    closure=None,
    lr: float = 0.01,
    num_epochs: int = 300,
    batch_size=None,   # kept for compatibility; intentionally unused
    shuffle: bool = True,      # kept for compatibility; intentionally unused
    optimizer_cls= torch.optim.Adam,
    clip_grad_norm: Optional[float] = None,
    verbose: bool = False,
    **ignore,
):
    """
    Optimizer callable for RRP binary classification.

    Notes:
        SparseOutlierBernoulliLikelihood keeps correction terms associated with
        the full training set.  Mini-batch training can therefore cause shape
        mismatches, so this optimizer intentionally uses full-batch training.

        The optimizer uses `model.parameters()` to preserve the behavior of the
        existing implementation.
    """
    model = mll.model
    likelihood = mll.likelihood

    ref_param = next(model.parameters())
    ref_dtype = ref_param.dtype
    ref_device = ref_param.device

    model.to(device=ref_device, dtype=ref_dtype)
    likelihood.to(device=ref_device, dtype=ref_dtype)

    model.train()
    likelihood.train()
    mll.train()

    train_X = get_train_inputs_tensor(model).to(device=ref_device, dtype=ref_dtype)
    train_Y = get_train_targets_tensor(model).to(device=ref_device, dtype=ref_dtype)

    optimizer = optimizer_cls(model.parameters(), lr=lr)

    num_epochs = int(num_epochs)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        output = model(train_X)
        loss = -mll(output, train_Y)

        if loss.ndim > 0:
            loss = loss.sum()

        loss.backward()
        maybe_clip_grad_norm(model.parameters(), clip_grad_norm)
        optimizer.step()

        if verbose and ((epoch + 1) % 50 == 0 or epoch == 0 or epoch == num_epochs - 1):
            print(f"[fit_rrp_classifier_mll_optimizer] epoch={epoch + 1:04d} loss={float(loss.item()):.5f}")

    return mll


def fit_rrp_binary_classifier_mll(
    mll,
    *,
    method: str = "backward",
    sparsity_levels: Optional[Sequence[int]] = None,
    initial_support: Optional[list[int]] = None,
    reset_parameters: bool = True,
    reset_dense_parameters: bool = False,
    record_model_trace: Optional[bool] = None,
    return_all: bool = False,
    optimizer=fit_rrp_binary_classifier_mll_optimizer,
    optimizer_kwargs: Optional[dict] = None,
    closure=None,
    closure_kwargs: Optional[dict] = None,
):
    """
    Fit an RRP classification MLL via forward/backward relevance pursuit.

    Args:
        mll:
            VariationalELBO / PredictiveLogLikelihood-like approximate MLL.
            `mll.likelihood` must inherit RelevancePursuitMixin.
        method:
            "forward" or "backward".
        sparsity_levels:
            Candidate support sizes.
        initial_support:
            Initial active support.
        reset_parameters:
            Whether to reset sparse parameters between support changes.
        reset_dense_parameters:
            Whether to reset dense hyperparameters between support changes.
        record_model_trace:
            Whether to store model snapshots for each support.
            Defaults to `return_all`.
        return_all:
            If True, returns (mll, sparse_module, model_trace).
            If False, returns mll.
        optimizer:
            Relevance-pursuit-compatible optimizer callable.
        optimizer_kwargs:
            Keyword arguments passed to optimizer.
        closure, closure_kwargs:
            Passed through to the relevance pursuit routine.

    Returns:
        mll, or (mll, sparse_module, model_trace) if return_all=True.
    """
    sparse_module = mll.likelihood
    if not isinstance(sparse_module, RelevancePursuitMixin):
        raise TypeError(
            "mll.likelihood must inherit RelevancePursuitMixin for RRP fitting. "
            f"Got: {type(sparse_module)}"
        )

    if method not in {"forward", "backward"}:
        raise ValueError("method must be 'forward' or 'backward'.")

    if record_model_trace is None:
        record_model_trace = bool(return_all)

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

    set_mll_eval_mode(mll)

    if return_all:
        return mll, sparse_module, model_trace
    return mll
