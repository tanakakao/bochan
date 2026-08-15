"""Ordinal single-output feasibility and observed-utility helpers.

Standard ordinal qExpectedUtility / qEI / qPI / qUCB live in ``standard.py`` and
use BoTorch joint q-batch semantics. This module contains only the task-specific
helpers that are not provided by BoTorch.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.utils.transforms import average_over_ensemble_models, t_batch_mode_transform
from torch import Tensor

QFeasMode = Literal["prod", "mean", "min", "max"]
OrdinalFeasibilityMode = Literal[
    "class_ge",
    "class_le",
    "class_interval",
    "expected_utility_ge",
]


def _canonicalize_utility_values(
    utility_values: Sequence[float] | Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    utilities = torch.as_tensor(utility_values, device=device, dtype=dtype)
    if utilities.ndim != 1:
        raise ValueError(
            f"utility_values must be 1D. Got shape={tuple(utilities.shape)}."
        )
    return utilities


def compute_ordinal_expected_utility_values(
    model: Model,
    X: Tensor,
    utility_values: Sequence[float] | Tensor,
    maximize: bool = True,
) -> Tensor:
    """Compute expected ordinal utility on observed candidate points."""
    utilities = _canonicalize_utility_values(
        utility_values,
        device=X.device,
        dtype=X.dtype,
    )
    with torch.no_grad():
        if hasattr(model, "expected_utility"):
            values = model.expected_utility(X, utilities)
        elif hasattr(model, "class_probs"):
            probs = model.class_probs(X)
            if probs.shape[-1] != utilities.numel():
                raise RuntimeError(
                    "Number of classes in model.class_probs(X) does not match "
                    "utility_values."
                )
            values = (probs * utilities).sum(dim=-1)
        else:
            raise TypeError(
                "model must expose expected_utility(X, utilities) or class_probs(X)."
            )
        if values.ndim >= 1 and values.shape[-1] == 1:
            values = values.squeeze(-1)
        if not maximize:
            values = -values
    return values.detach()


def compute_ordinal_expected_utility_best_f(
    model: Model,
    train_X: Tensor,
    utility_values: Sequence[float] | Tensor,
    maximize: bool = True,
) -> Tensor:
    """Return the best observed expected utility on ``train_X``."""
    values = compute_ordinal_expected_utility_values(
        model=model,
        X=train_X,
        utility_values=utility_values,
        maximize=maximize,
    )
    return values.max().detach()


def ensure_q_batch(X: Tensor) -> Tensor:
    if not torch.is_tensor(X):
        raise TypeError(f"X must be Tensor. Got {type(X)}.")
    if X.ndim == 1:
        return X.view(1, 1, -1)
    if X.ndim == 2:
        return X.unsqueeze(0)
    return X


def _coerce_pending_to_tensor(
    X_pending,
    *,
    ref: Optional[Tensor] = None,
) -> Optional[Tensor]:
    if X_pending is None:
        return None
    if torch.is_tensor(X_pending):
        out = X_pending
    elif isinstance(X_pending, (list, tuple)):
        tensors = [
            tensor
            for item in X_pending
            if item is not None
            and (tensor := _coerce_pending_to_tensor(item, ref=ref)) is not None
            and tensor.numel() > 0
        ]
        if not tensors:
            return None
        out = torch.cat(
            [tensor.reshape(-1, tensor.shape[-1]) for tensor in tensors],
            dim=-2,
        )
    else:
        raise TypeError(
            "X_pending must be None, Tensor, list, or tuple. "
            f"Got {type(X_pending)}."
        )
    if ref is not None:
        out = out.to(device=ref.device, dtype=ref.dtype)
    return out


def _apply_input_transform_for_pending(model: Model, X: Tensor) -> Tensor:
    X = ensure_q_batch(X)
    input_transform = getattr(model, "input_transform", None)
    if input_transform is not None:
        Xt = input_transform(X)
        return ensure_q_batch(Xt[0] if isinstance(Xt, tuple) else Xt)

    models = getattr(model, "models", None)
    if models is not None and len(models) > 0:
        input_transform = getattr(models[0], "input_transform", None)
        if input_transform is not None:
            Xt = input_transform(X)
            return ensure_q_batch(Xt[0] if isinstance(Xt, tuple) else Xt)
    return X


def _transform_reference_like_candidate(
    model: Model,
    X_ref,
    *,
    ref: Tensor,
) -> Optional[Tensor]:
    Xr = _coerce_pending_to_tensor(X_ref, ref=ref)
    if Xr is None or Xr.numel() == 0:
        return None
    return _apply_input_transform_for_pending(model, Xr).to(
        device=ref.device,
        dtype=ref.dtype,
    )


def _cat_dims_from_model(model: Model, d: int) -> list[int]:
    cat_dims = getattr(model, "cat_dims", [])
    try:
        return [int(j) for j in cat_dims if 0 <= int(j) < d]
    except TypeError:
        return []


def _pairwise_distance2(
    A: Tensor,
    B: Tensor,
    *,
    cat_dims: Optional[Sequence[int]] = None,
) -> Tensor:
    d = A.shape[-1]
    cat_set = set(int(j) for j in (cat_dims or []) if 0 <= int(j) < d)
    cont_dims = [j for j in range(d) if j not in cat_set]
    cat_dims_valid = sorted(cat_set)

    distance: Tensor | float = 0.0
    if cont_dims:
        distance = distance + (
            A[..., cont_dims].unsqueeze(-2) - B[..., cont_dims].unsqueeze(-3)
        ).pow(2).sum(dim=-1)
    if cat_dims_valid:
        distance = distance + (
            A[..., cat_dims_valid].unsqueeze(-2)
            != B[..., cat_dims_valid].unsqueeze(-3)
        ).to(A.dtype).sum(dim=-1)
    if isinstance(distance, float):
        raise RuntimeError("No valid dimensions found for pairwise distance.")
    return distance


def _reference_repulsion_penalty(
    X: Tensor,
    X_ref: Optional[Tensor],
    *,
    beta: float,
    weight: float,
    cat_dims: Optional[Sequence[int]] = None,
) -> Tensor:
    X = ensure_q_batch(X)
    batch_shape = X.shape[:-2]
    if weight <= 0.0 or X_ref is None or X_ref.numel() == 0:
        return X.new_zeros(batch_shape)

    d = X.shape[-1]
    Xb = X.reshape(-1, X.shape[-2], d)
    Xr = X_ref.reshape(-1, X_ref.shape[-1]).to(device=X.device, dtype=X.dtype)
    Xr = Xr.view(1, Xr.shape[0], d).expand(Xb.shape[0], -1, -1)
    nearest = _pairwise_distance2(Xb, Xr, cat_dims=cat_dims).min(dim=-1).values
    penalty = weight * torch.exp(-float(beta) * nearest).sum(dim=-1)
    return penalty.reshape(*batch_shape)


def _same_batch_repulsion_penalty(
    X: Tensor,
    *,
    beta: float,
    weight: float,
    cat_dims: Optional[Sequence[int]] = None,
) -> Tensor:
    X = ensure_q_batch(X)
    batch_shape = X.shape[:-2]
    q = X.shape[-2]
    if q <= 1 or weight <= 0.0:
        return X.new_zeros(batch_shape)

    Xb = X.reshape(-1, q, X.shape[-1])
    d2 = _pairwise_distance2(Xb, Xb, cat_dims=cat_dims)
    eye = torch.eye(q, device=X.device, dtype=torch.bool).unsqueeze(0)
    d2 = d2.masked_fill(eye, float("inf"))
    penalty = 0.5 * weight * torch.exp(-float(beta) * d2).sum(dim=(-1, -2))
    return penalty.reshape(*batch_shape)


def _is_ordinal_likelihood(obj: Any) -> bool:
    return obj is not None and (
        hasattr(obj, "marginal_class_probs") or hasattr(obj, "class_probs_from_f")
    )


def _resolve_ordinal_likelihood(model: Model, ordinal_likelihood: Optional[Any] = None) -> Any:
    if ordinal_likelihood is not None:
        return ordinal_likelihood

    candidates: list[Any] = [
        getattr(model, "ordinal_likelihood", None),
        getattr(model, "likelihood", None),
    ]
    for attr in ("latent_model", "base_model", "model"):
        inner = getattr(model, attr, None)
        if inner is not None:
            candidates.extend(
                [
                    getattr(inner, "ordinal_likelihood", None),
                    getattr(inner, "likelihood", None),
                ]
            )
    for candidate in candidates:
        if _is_ordinal_likelihood(candidate):
            return candidate

    models = getattr(model, "models", None)
    if models is not None:
        likelihoods = [
            candidate
            for submodel in models
            for candidate in (
                getattr(submodel, "ordinal_likelihood", None),
                getattr(submodel, "likelihood", None),
            )
            if _is_ordinal_likelihood(candidate)
        ]
        if len(likelihoods) == 1:
            return likelihoods[0]
        if len(likelihoods) > 1:
            raise ValueError(
                "Multiple ordinal likelihoods were found. Pass ordinal_likelihood explicitly."
            )
    raise ValueError(
        "ordinal_likelihood was not provided and could not be inferred from model."
    )


def _reduce_extra_batch_dims(tensor: Tensor, X: Tensor, n_trailing_keep: int) -> Tensor:
    out = tensor
    x_batch_shape = tuple(ensure_q_batch(X).shape[:-2])
    target_ndim = len(x_batch_shape) + n_trailing_keep
    while out.ndim > target_ndim:
        prefix = tuple(out.shape[:-n_trailing_keep]) if n_trailing_keep > 0 else tuple(out.shape)
        if not x_batch_shape:
            reduce_dim = 0
        else:
            match_start = next(
                (
                    start
                    for start in range(len(prefix) - len(x_batch_shape) + 1)
                    if tuple(prefix[start : start + len(x_batch_shape)]) == x_batch_shape
                ),
                None,
            )
            if match_start is None:
                reduce_dim = max(out.ndim - n_trailing_keep - 1, 0)
            else:
                protected = set(range(match_start, match_start + len(x_batch_shape)))
                extra_dims = [i for i in range(len(prefix)) if i not in protected]
                if not extra_dims:
                    break
                reduce_dim = extra_dims[0]
        out = out.mean(dim=reduce_dim)
    return out


def _class_probs_from_model_or_likelihood(
    model: Model,
    X: Tensor,
    ordinal_likelihood: Any,
    eps: float,
) -> Tensor:
    Xq = ensure_q_batch(X)
    posterior = model.posterior(Xq)

    if hasattr(ordinal_likelihood, "marginal_class_probs"):
        try:
            probs = ordinal_likelihood.marginal_class_probs(posterior.distribution)
            probs = _reduce_extra_batch_dims(probs, X=Xq, n_trailing_keep=2)
            if probs.requires_grad or not Xq.requires_grad:
                probs = probs.clamp_min(eps)
                return probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)
        except Exception:
            pass

    mean_f = posterior.mean
    if mean_f.ndim >= 1 and mean_f.shape[-1] == 1:
        mean_f = mean_f.squeeze(-1)
    mean_f = _reduce_extra_batch_dims(mean_f, X=Xq, n_trailing_keep=1)
    if not hasattr(ordinal_likelihood, "class_probs_from_f"):
        raise RuntimeError(
            "ordinal_likelihood must expose marginal_class_probs or class_probs_from_f "
            "for differentiable qOrdinalProbabilityOfFeasibility."
        )
    probs = ordinal_likelihood.class_probs_from_f(mean_f)
    probs = _reduce_extra_batch_dims(probs, X=Xq, n_trailing_keep=2).clamp_min(eps)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)


class qOrdinalProbabilityOfFeasibility(AcquisitionFunction):
    """Probability of feasibility for ordinal constraints."""

    def __init__(
        self,
        model: Model,
        ordinal_likelihood: Optional[Any] = None,
        mode: OrdinalFeasibilityMode = "class_ge",
        min_class: Optional[int] = None,
        max_class: Optional[int] = None,
        utility_values: Optional[Sequence[float] | Tensor] = None,
        utility_threshold: Optional[float | Tensor] = None,
        tau: float = 1e-3,
        q_feas_mode: QFeasMode = "prod",
        eps: float = 1e-8,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        super().__init__(model=model)
        self.ordinal_likelihood = _resolve_ordinal_likelihood(model, ordinal_likelihood)
        self.mode = mode
        self.min_class = min_class
        self.max_class = max_class
        self.utility_values = utility_values
        self.utility_threshold = utility_threshold
        self.tau = float(tau)
        self.q_feas_mode = q_feas_mode
        self.eps = float(eps)
        self.pending_penalty_weight = float(pending_penalty_weight)
        self.pending_penalty_beta = float(pending_penalty_beta)
        self.same_batch_penalty_weight = float(same_batch_penalty_weight)
        self.same_batch_penalty_beta = float(same_batch_penalty_beta)
        self.X_pending: Optional[Tensor] = None
        self.set_X_pending(X_pending)

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        self.X_pending = _coerce_pending_to_tensor(X_pending)

    def _pointwise_feasibility(self, X: Tensor) -> Tensor:
        probs = _class_probs_from_model_or_likelihood(
            self.model,
            X,
            self.ordinal_likelihood,
            eps=self.eps,
        )
        num_classes = probs.shape[-1]

        if self.mode == "class_ge":
            if self.min_class is None:
                raise ValueError("min_class must be specified for mode='class_ge'.")
            k = int(self.min_class)
            if not 0 <= k < num_classes:
                raise ValueError(f"min_class must be in [0, {num_classes - 1}].")
            return probs[..., k:].sum(dim=-1)

        if self.mode == "class_le":
            if self.max_class is None:
                raise ValueError("max_class must be specified for mode='class_le'.")
            k = int(self.max_class)
            if not 0 <= k < num_classes:
                raise ValueError(f"max_class must be in [0, {num_classes - 1}].")
            return probs[..., : k + 1].sum(dim=-1)

        if self.mode == "class_interval":
            if self.min_class is None or self.max_class is None:
                raise ValueError(
                    "min_class and max_class must be specified for mode='class_interval'."
                )
            lo, hi = int(self.min_class), int(self.max_class)
            if lo > hi:
                raise ValueError("min_class must be <= max_class.")
            if not (0 <= lo < num_classes and 0 <= hi < num_classes):
                raise ValueError(f"class bounds must be in [0, {num_classes - 1}].")
            return probs[..., lo : hi + 1].sum(dim=-1)

        if self.mode == "expected_utility_ge":
            if self.utility_values is None:
                raise ValueError(
                    "utility_values must be specified for mode='expected_utility_ge'."
                )
            if self.utility_threshold is None:
                raise ValueError(
                    "utility_threshold must be specified for mode='expected_utility_ge'."
                )
            utilities = _canonicalize_utility_values(
                self.utility_values,
                device=probs.device,
                dtype=probs.dtype,
            )
            if utilities.numel() != num_classes:
                raise ValueError(
                    f"utility_values must have length {num_classes}, got {utilities.numel()}."
                )
            expected_u = (probs * utilities).sum(dim=-1)
            threshold = torch.as_tensor(
                self.utility_threshold,
                device=expected_u.device,
                dtype=expected_u.dtype,
            )
            tau = torch.as_tensor(self.tau, device=expected_u.device, dtype=expected_u.dtype)
            return torch.sigmoid((expected_u - threshold) / tau.clamp_min(1e-9))

        raise ValueError(f"Unknown ordinal feasibility mode: {self.mode}")

    def _reduce_q_feasibility(self, point_feas: Tensor) -> Tensor:
        point_feas = point_feas.clamp(self.eps, 1.0 - self.eps)
        if self.q_feas_mode == "prod":
            return point_feas.prod(dim=-1)
        if self.q_feas_mode == "mean":
            return point_feas.mean(dim=-1)
        if self.q_feas_mode == "min":
            return point_feas.min(dim=-1).values
        if self.q_feas_mode == "max":
            return point_feas.max(dim=-1).values
        raise ValueError(f"Unknown q_feas_mode: {self.q_feas_mode}")

    def _repulsion_penalty(self, X: Tensor, value: Tensor) -> Tensor:
        Xt = _apply_input_transform_for_pending(self.model, X)
        cat_dims = _cat_dims_from_model(self.model, Xt.shape[-1])
        penalty = _same_batch_repulsion_penalty(
            Xt,
            beta=self.same_batch_penalty_beta,
            weight=self.same_batch_penalty_weight,
            cat_dims=cat_dims,
        )
        Xp_t = _transform_reference_like_candidate(self.model, self.X_pending, ref=Xt)
        penalty = penalty + _reference_repulsion_penalty(
            Xt,
            Xp_t,
            beta=self.pending_penalty_beta,
            weight=self.pending_penalty_weight,
            cat_dims=cat_dims,
        )
        return value - penalty.to(device=value.device, dtype=value.dtype)

    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        value = self._reduce_q_feasibility(self._pointwise_feasibility(X))
        return self._repulsion_penalty(X, value)


__all__ = [
    "compute_ordinal_expected_utility_values",
    "compute_ordinal_expected_utility_best_f",
    "qOrdinalProbabilityOfFeasibility",
]
