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

    multiclass RRP では sparse softmax offset が likelihood 側に存在するため、
    model.parameters() だけではなく mll / likelihood / wrapper も optimizer 対象にする。
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


def _base_mll(mll):
    """DeepApproximateMLL などで wrap されている場合は内側の mll を返す。"""
    return getattr(mll, "base_mll", mll)


def _resolve_rrp_multiclass_sparse_module(mll, fit_model=None):
    """
    Relevance Pursuit が扱う sparse softmax likelihood を解決する。

    通常は `mll.likelihood` が SparseOutlierSoftmaxLikelihood になる。
    wrapper を明示した場合は `fit_model.likelihood` も見る。
    """
    base_mll = _base_mll(mll)

    if hasattr(base_mll, "likelihood") and isinstance(base_mll.likelihood, RelevancePursuitMixin):
        return base_mll.likelihood

    if fit_model is not None:
        likelihood = get_likelihood_from_mll_or_model(base_mll, fit_model)
        if isinstance(likelihood, RelevancePursuitMixin):
            return likelihood

        if hasattr(fit_model, "likelihood") and isinstance(fit_model.likelihood, RelevancePursuitMixin):
            return fit_model.likelihood

    raise TypeError(
        "Could not find a multiclass sparse module inheriting RelevancePursuitMixin. "
        "Expected `mll.likelihood` or `fit_model.likelihood` to inherit RelevancePursuitMixin. "
        "For multiclass RRP, use SparseOutlierSoftmaxLikelihood."
    )


def _call_latent_multiclass_model(model, train_X):
    """
    multiclass wrapper / inner latent model のどちらでも latent distribution を返す。

    `fit_model` に wrapper を渡した場合は raw X を input_transform してから
    内側の class-wise latent SVGP を呼び出す。`mll.model` を直接使う場合は
    そのまま呼び出す。
    """
    if hasattr(model, "model") and hasattr(model, "transform_inputs"):
        train_X = model.transform_inputs(train_X)
        return model.model(train_X)
    return model(train_X)


def fit_rrp_multiclass_mll_optimizer(
    mll,
    closure=None,
    *,
    fit_model=None,
    lr: float = 0.01,
    num_epochs: int = 300,
    batch_size=None,   # kept for compatibility; intentionally unused
    shuffle: bool = True,      # kept for compatibility; intentionally unused
    optimizer_cls=torch.optim.Adam,
    clip_grad_norm: Optional[float] = None,
    verbose: bool = False,
    **ignore,
):
    """
    Optimizer callable for RRP multiclass classification models.

    Notes:
        SparseOutlierSoftmaxLikelihood keeps class-wise sparse offset parameters
        associated with the full training set. Mini-batch training can break this
        correspondence, so this optimizer intentionally uses full-batch training.

        `fit_model` can be supplied when `mll.model` is the inner latent SVGP and
        the outer wrapper is needed to apply raw-space input transforms.
    """
    _ = closure, batch_size, shuffle, ignore

    base_mll = _base_mll(mll)
    model = base_mll.model if fit_model is None else fit_model
    likelihood = get_likelihood_from_mll_or_model(base_mll, model)

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
    if train_Y.ndim > 1 and train_Y.shape[-1] == 1:
        train_Y = train_Y.squeeze(-1)
    train_Y = train_Y.long().reshape(-1)

    if train_X.shape[-2] != train_Y.shape[0]:
        raise RuntimeError(
            "train_X and train_Y have inconsistent data sizes. "
            f"train_X.shape={tuple(train_X.shape)}, train_Y.shape={tuple(train_Y.shape)}. "
            "For wrapper models, pass fit_model=wrapper_model so raw X and targets match."
        )

    params = _unique_trainable_parameters(model, likelihood, mll)
    if len(params) == 0:
        raise RuntimeError("No trainable parameters were found for RRP multiclass fitting.")

    optimizer = optimizer_cls(params, lr=lr)

    num_epochs = int(num_epochs)
    num_data = int(train_X.shape[-2])

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        latent_dist = _call_latent_multiclass_model(model, train_X)
        loss = -mll(latent_dist, train_Y)
        if loss.ndim > 0:
            loss = loss.sum()

        loss.backward()
        maybe_clip_grad_norm(params, clip_grad_norm)
        optimizer.step()

        if verbose and ((epoch + 1) % 50 == 0 or epoch == 0 or epoch == num_epochs - 1):
            print(
                "[fit_rrp_multiclass_mll_optimizer] "
                f"epoch={epoch + 1:04d} loss={float(loss.item()) / num_data:.6f}"
            )

    return mll


def fit_rrp_multiclass_mll(
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
    optimizer=fit_rrp_multiclass_mll_optimizer,
    optimizer_kwargs: Optional[dict] = None,
    closure=None,
    closure_kwargs: Optional[dict] = None,
):
    """
    Fit an RRP multiclass classification MLL via forward/backward relevance pursuit.
    """
    if method not in {"forward", "backward"}:
        raise ValueError("method must be 'forward' or 'backward'.")

    sparse_module = _resolve_rrp_multiclass_sparse_module(mll, fit_model=fit_model)

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

    base_mll = _base_mll(mll)
    model = base_mll.model if fit_model is None else fit_model
    likelihood = get_likelihood_from_mll_or_model(base_mll, model)
    set_model_and_likelihood_eval_mode(model, likelihood)
    if hasattr(base_mll, "model") and base_mll.model is not model:
        base_mll.model.eval()
    if hasattr(mll, "eval"):
        mll.eval()

    if return_all:
        return mll, sparse_module, model_trace
    return mll


# Binary 側の命名に寄せたい場合の alias。
fit_rrp_multiclass_classifier_mll_optimizer = fit_rrp_multiclass_mll_optimizer
fit_rrp_multiclass_classifier_mll = fit_rrp_multiclass_mll


__all__ = [
    "fit_rrp_multiclass_mll",
    "fit_rrp_multiclass_mll_optimizer",
    "fit_rrp_multiclass_classifier_mll",
    "fit_rrp_multiclass_classifier_mll_optimizer",
]
