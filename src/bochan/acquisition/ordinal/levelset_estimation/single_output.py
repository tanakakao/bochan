from __future__ import annotations

import math
from typing import Callable, Literal, Optional, Sequence

import torch
from torch import Tensor

from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import average_over_ensemble_models, t_batch_mode_transform

from bochan.acquisition._duplicate_exclusion import (
    hard_reference_duplicate_penalty_per_point,
    hard_same_batch_duplicate_penalty_per_point,
    resolve_observed_X,
    unwrap_single_output_model,
)


RiskType = Optional[Literal["var", "cvar"]]
PerturbationJointReduction = Literal["block_mean", "diagonal_mean"]
ReductionType = Literal["sum", "mean"]
BoundaryReduction = Literal["sum", "mean", "max", "min"]


class _OrdinalLevelSetScoreObjective(torch.nn.Module):
    """ordinal level-set acquisition の pointwise score に作用する objective。"""

    def __init__(
        self,
        n_w: Optional[int] = None,
        risk_type: RiskType = None,
        alpha: float = 0.5,
        maximize: bool = True,
        weight: float = 1.0,
        sign: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_w = None if n_w is None else int(n_w)
        self.risk_type = risk_type
        self.alpha = float(alpha)
        self.maximize = bool(maximize)
        self.weight = float(weight)
        self.sign = float(sign)

        if self.risk_type not in (None, "var", "cvar"):
            raise ValueError(f"Unknown risk_type: {self.risk_type}")
        if self.risk_type is not None and self.n_w is None:
            raise ValueError("risk_type is specified, but n_w is None.")
        if self.risk_type is not None and not (0.0 < self.alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1].")

    def forward(self, score: Tensor, X: Optional[Tensor] = None) -> Tensor:
        score = score * self.sign * self.weight
        if self.n_w is None or self.n_w <= 1:
            return score

        if X is not None:
            X_in = X if X.ndim > 2 else X.unsqueeze(0)
            if tuple(score.shape) == tuple(X_in.shape[:-2]):
                return score

        q_expanded = score.shape[-1]
        if q_expanded % self.n_w != 0:
            raise RuntimeError(
                f"score.shape[-1] must be divisible by n_w. "
                f"Got score.shape={tuple(score.shape)}, n_w={self.n_w}."
            )
        q = q_expanded // self.n_w
        score_w = score.reshape(*score.shape[:-1], q, self.n_w)

        if self.risk_type is None:
            return score_w.mean(dim=-1)

        descending = not self.maximize
        sorted_score = torch.sort(score_w, dim=-1, descending=descending).values
        k = max(1, int(math.ceil(self.n_w * self.alpha)))
        tail = sorted_score[..., :k]

        if self.risk_type == "var":
            return tail[..., -1]
        if self.risk_type == "cvar":
            return tail.mean(dim=-1)
        raise ValueError(f"Unknown risk_type: {self.risk_type}")


def _apply_ordinal_levelset_objective_to_score(
    owner,
    score: Tensor,
    X: Optional[Tensor] = None,
    name: str = "OrdinalLevelSetAcquisition",
) -> Tensor:
    objective = getattr(owner, "objective", None)
    if objective is None:
        return score
    try:
        out = objective(score, X=X)
    except TypeError:
        out = objective(score)
    if not torch.is_tensor(out):
        raise RuntimeError(f"{name}: objective must return a Tensor. Got {type(out)}.")
    return out


def _infer_n_w_from_objective_or_owner(owner) -> Optional[int]:
    n_w = getattr(owner, "input_perturbation_n_w", None)
    if n_w is not None:
        return int(n_w)
    objective = getattr(owner, "objective", None)
    if objective is not None and getattr(objective, "n_w", None) is not None:
        return int(objective.n_w)
    return None


def _try_call_zero_arg(obj):
    return obj() if callable(obj) else obj


def _get_ordinal_likelihood(model: Model):
    if hasattr(model, "ordinal_likelihood"):
        return getattr(model, "ordinal_likelihood")
    if hasattr(model, "likelihood"):
        return getattr(model, "likelihood")
    raise ValueError("model must expose ordinal_likelihood or likelihood")


def _get_cutpoints_from_likelihood(ordinal_likelihood) -> Tensor:
    if hasattr(ordinal_likelihood, "get_cutpoints"):
        cutpoints = _try_call_zero_arg(getattr(ordinal_likelihood, "get_cutpoints"))
        return torch.as_tensor(cutpoints).detach().clone().reshape(-1)

    for name in ("transformed_cutpoints", "cutpoints", "thresholds", "cuts", "cutoffs"):
        if hasattr(ordinal_likelihood, name):
            cutpoints = _try_call_zero_arg(getattr(ordinal_likelihood, name))
            return torch.as_tensor(cutpoints).detach().clone().reshape(-1)

    if hasattr(ordinal_likelihood, "raw_cutpoints"):
        raw = torch.as_tensor(_try_call_zero_arg(getattr(ordinal_likelihood, "raw_cutpoints"))).detach().clone()
        if hasattr(ordinal_likelihood, "transform_cutpoints"):
            cutpoints = ordinal_likelihood.transform_cutpoints(raw)
            return torch.as_tensor(cutpoints).detach().clone().reshape(-1)
        return raw.reshape(-1)

    raise ValueError(
        "Could not find cutpoints on ordinal likelihood. "
        "Expected one of: get_cutpoints / transformed_cutpoints / cutpoints / thresholds / raw_cutpoints."
    )


def _posterior_mvn(posterior):
    if hasattr(posterior, "distribution"):
        dist = posterior.distribution
        if hasattr(dist, "covariance_matrix"):
            return dist
    if hasattr(posterior, "mvn"):
        dist = posterior.mvn
        if hasattr(dist, "covariance_matrix"):
            return dist
    return None


def _find_matching_batch_slice(prefix_shape: tuple[int, ...], x_batch_shape: tuple[int, ...]) -> int | None:
    if len(x_batch_shape) == 0:
        return 0
    max_start = len(prefix_shape) - len(x_batch_shape)
    for s in range(max_start + 1):
        if tuple(prefix_shape[s : s + len(x_batch_shape)]) == x_batch_shape:
            return s
    return None


def _reduce_extra_batch_dims(tensor: Tensor, X: Tensor, n_trailing_keep: int) -> Tensor:
    out = tensor
    x_batch_shape = tuple(X.shape[:-2])
    target_ndim = len(x_batch_shape) + n_trailing_keep

    while out.ndim > target_ndim:
        prefix = tuple(out.shape[:-n_trailing_keep]) if n_trailing_keep > 0 else tuple(out.shape)
        if len(x_batch_shape) == 0:
            reduce_dim = 0
        else:
            match_start = _find_matching_batch_slice(prefix, x_batch_shape)
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


def _reduce_probs_to_match_X(probs: Tensor, X: Tensor) -> Tensor:
    out = _reduce_extra_batch_dims(probs, X=X, n_trailing_keep=2)
    out = out.clamp_min(1e-12)
    return out / out.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _posterior_mean_var(posterior, X: Tensor) -> tuple[Tensor, Tensor]:
    mean = posterior.mean
    if mean.shape[-1] == 1:
        mean = mean.squeeze(-1)

    if hasattr(posterior, "variance"):
        var = posterior.variance
        if var.shape[-1] == 1:
            var = var.squeeze(-1)
        var = var.clamp_min(1e-12)
    else:
        mvn = _posterior_mvn(posterior)
        if mvn is None:
            raise ValueError("posterior must expose variance or covariance_matrix")
        var = mvn.covariance_matrix.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12)

    mean = _reduce_extra_batch_dims(mean, X=X, n_trailing_keep=1)
    var = _reduce_extra_batch_dims(var, X=X, n_trailing_keep=1)
    return mean, var


def _posterior_covariance(posterior, X: Tensor) -> Tensor:
    mvn = _posterior_mvn(posterior)
    if mvn is not None:
        cov = mvn.covariance_matrix
    elif hasattr(posterior, "variance"):
        var = posterior.variance
        if var.shape[-1] == 1:
            var = var.squeeze(-1)
        cov = torch.diag_embed(var.clamp_min(1e-12))
    else:
        raise ValueError("posterior must expose covariance_matrix or variance")
    return _reduce_extra_batch_dims(cov, X=X, n_trailing_keep=2)


def _reduce_input_perturbation_mean_cov(
    mean: Tensor,
    cov: Tensor,
    X: Tensor,
    n_w: Optional[int],
    *,
    mode: PerturbationJointReduction = "block_mean",
    jitter: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    if n_w is None or n_w <= 1:
        return mean, cov

    X_in = X if X.ndim > 2 else X.unsqueeze(0)
    batch_shape = X_in.shape[:-2]
    q = X_in.shape[-2]

    expected_mean = batch_shape + torch.Size([q])
    expected_cov = batch_shape + torch.Size([q, q])
    if mean.shape == expected_mean and cov.shape == expected_cov:
        eye = torch.eye(q, dtype=cov.dtype, device=cov.device)
        return mean, cov + jitter * eye

    q_expanded = q * n_w
    expanded_mean = batch_shape + torch.Size([q_expanded])
    expanded_cov = batch_shape + torch.Size([q_expanded, q_expanded])
    if mean.shape != expanded_mean or cov.shape != expanded_cov:
        return mean, cov

    mean_q = mean.reshape(*batch_shape, q, n_w).mean(dim=-1)
    cov_blocks = cov.reshape(*batch_shape, q, n_w, q, n_w)

    if mode == "block_mean":
        cov_q = cov_blocks.mean(dim=(-3, -1))
    elif mode == "diagonal_mean":
        diag = torch.diagonal(cov, dim1=-2, dim2=-1)
        var_q = diag.reshape(*batch_shape, q, n_w).mean(dim=-1).clamp_min(0.0)
        cov_q = torch.diag_embed(var_q)
    else:
        raise ValueError(f"Unknown perturbation_joint_reduction: {mode}")

    cov_q = 0.5 * (cov_q + cov_q.transpose(-1, -2))
    eye = torch.eye(q, dtype=cov_q.dtype, device=cov_q.device)
    return mean_q, cov_q + jitter * eye


def ordinal_class_probs_from_f(f: Tensor, ordinal_likelihood) -> Tensor:
    for name in (
        "class_probs_from_f",
        "probs_from_f",
        "predict_proba_from_f",
        "class_probabilities_from_f",
        "marginal_probs_from_f",
        "latent_to_probs",
    ):
        if hasattr(ordinal_likelihood, name):
            probs = getattr(ordinal_likelihood, name)(f)
            if hasattr(probs, "probs"):
                probs = probs.probs
            probs = torch.as_tensor(probs, device=f.device, dtype=f.dtype)
            probs = probs.clamp_min(1e-12)
            return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    cutpoints = _get_cutpoints_from_likelihood(ordinal_likelihood).to(device=f.device, dtype=f.dtype)
    z = cutpoints.view(*([1] * f.ndim), -1) - f.unsqueeze(-1)
    cdf = torch.sigmoid(z)
    p0 = cdf[..., :1]
    if cutpoints.numel() > 1:
        pmid = cdf[..., 1:] - cdf[..., :-1]
        plast = 1.0 - cdf[..., -1:]
        probs = torch.cat([p0, pmid, plast], dim=-1)
    else:
        plast = 1.0 - cdf[..., -1:]
        probs = torch.cat([p0, plast], dim=-1)
    probs = probs.clamp_min(1e-12)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def ordinal_cumulative_ge_probs_from_class_probs(class_probs: Tensor) -> Tensor:
    rev_cumsum = torch.flip(
        torch.cumsum(torch.flip(class_probs, dims=[-1]), dim=-1),
        dims=[-1],
    )
    return rev_cumsum[..., 1:]


def ordinal_boundary_uncertainty(ge_probs: Tensor) -> Tensor:
    return 4.0 * ge_probs * (1.0 - ge_probs)


def _prepare_boundary_weights(
    boundary_weights: Optional[Tensor | Sequence[float]],
    n_boundaries: int,
    *,
    device,
    dtype,
) -> Tensor:
    if boundary_weights is None:
        return torch.ones(n_boundaries, device=device, dtype=dtype)
    w = torch.as_tensor(boundary_weights, device=device, dtype=dtype).detach().reshape(-1)
    if w.numel() != n_boundaries:
        raise ValueError(f"boundary_weights must have length {n_boundaries}, got {w.numel()}")
    return w


def _validate_target_boundary_idx(target_boundary_idx: Optional[int], n_boundaries: int) -> Optional[int]:
    if target_boundary_idx is None:
        return None
    idx = int(target_boundary_idx)
    if not (0 <= idx < n_boundaries):
        raise ValueError(
            f"target_boundary_idx must satisfy 0 <= idx < {n_boundaries}. "
            f"Got {target_boundary_idx}."
        )
    return idx


def _cutpoint_distances_by_boundary(values: Tensor, cutpoints: Tensor) -> Tensor:
    cp = cutpoints.detach().to(device=values.device, dtype=values.dtype).reshape(-1)
    return (values.unsqueeze(-1) - cp.view(*([1] * values.ndim), -1)).abs()


def _boundary_kernel_weights_by_boundary(values: Tensor, cutpoints: Tensor, tau: float) -> Tensor:
    cp = cutpoints.detach().to(device=values.device, dtype=values.dtype).reshape(-1)
    tau_t = torch.as_tensor(tau, device=values.device, dtype=values.dtype).clamp_min(1e-8)
    z2 = ((values.unsqueeze(-1) - cp.view(*([1] * values.ndim), -1)) / tau_t) ** 2
    return torch.exp(-0.5 * z2)


def _aggregate_boundary_scores(
    boundary_scores: Tensor,
    *,
    target_boundary_idx: Optional[int] = None,
    boundary_weights: Optional[Tensor | Sequence[float]] = None,
    boundary_reduction: BoundaryReduction = "sum",
) -> Tensor:
    if boundary_scores.ndim < 1:
        raise RuntimeError("boundary_scores must have a boundary dimension.")

    n_boundaries = boundary_scores.shape[-1]
    idx = _validate_target_boundary_idx(target_boundary_idx, n_boundaries)
    if idx is not None:
        return boundary_scores[..., idx]

    if boundary_weights is not None:
        w = _prepare_boundary_weights(
            boundary_weights,
            n_boundaries=n_boundaries,
            device=boundary_scores.device,
            dtype=boundary_scores.dtype,
        )
        boundary_scores = boundary_scores * w.view(*([1] * (boundary_scores.ndim - 1)), -1)

    if boundary_reduction == "sum":
        return boundary_scores.sum(dim=-1)
    if boundary_reduction == "mean":
        return boundary_scores.mean(dim=-1)
    if boundary_reduction == "max":
        return boundary_scores.max(dim=-1).values
    if boundary_reduction == "min":
        return boundary_scores.min(dim=-1).values
    raise ValueError(f"Unknown boundary_reduction: {boundary_reduction}")


def _flatten_ref_points(X_ref: Optional[Tensor], *, device, dtype) -> Optional[Tensor]:
    if X_ref is None or X_ref.numel() == 0:
        return None
    X_ref = X_ref.to(device=device, dtype=dtype)
    return X_ref.reshape(-1, X_ref.shape[-1])


def _same_batch_penalty(X: Tensor, lengthscale: float) -> Tensor:
    if X.shape[-2] <= 1:
        return torch.zeros(X.shape[:-2], device=X.device, dtype=X.dtype)
    ls2 = float(lengthscale) ** 2 + 1e-12
    diff = X.unsqueeze(-2) - X.unsqueeze(-3)
    d2 = diff.pow(2).sum(dim=-1)
    eye = torch.eye(X.shape[-2], device=X.device, dtype=torch.bool)
    d2 = d2.masked_fill(eye, float("inf"))
    return torch.exp(-0.5 * d2 / ls2).sum(dim=(-1, -2))


def _pointwise_same_batch_penalty(
    X: Tensor,
    lengthscale: float,
    n_w: Optional[int] = None,
) -> Tensor:
    q_expanded = X.shape[-2]
    if q_expanded <= 1:
        return torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)

    ls2 = float(lengthscale) ** 2 + 1e-12
    diff = X.unsqueeze(-2) - X.unsqueeze(-3)
    d2 = diff.pow(2).sum(dim=-1)
    eye = torch.eye(q_expanded, device=X.device, dtype=torch.bool)
    mask = eye

    if n_w is not None and n_w > 1 and q_expanded % n_w == 0:
        group = torch.arange(q_expanded, device=X.device) // int(n_w)
        mask = mask | (group.unsqueeze(0) == group.unsqueeze(1))

    d2 = d2.masked_fill(mask, float("inf"))
    return torch.exp(-0.5 * d2 / ls2).sum(dim=-1)


def _pointwise_ref_penalty(
    X: Tensor,
    X_ref: Optional[Tensor],
    lengthscale: float,
) -> Tensor:
    X_ref = _flatten_ref_points(X_ref, device=X.device, dtype=X.dtype)
    if X_ref is None:
        return torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)

    ls2 = float(lengthscale) ** 2 + 1e-12
    ref = X_ref.view(*([1] * (X.ndim - 2)), X_ref.shape[0], X_ref.shape[1])
    diff = X.unsqueeze(-2) - ref
    d2 = diff.pow(2).sum(dim=-1)
    return torch.exp(-0.5 * d2 / ls2).sum(dim=-1)


def _ensure_q_batch_for_pending(X: Tensor) -> Tensor:
    return X.unsqueeze(-2) if torch.is_tensor(X) and X.ndim == 2 else X


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
        tensors = []
        for item in X_pending:
            if item is None:
                continue
            t = _coerce_pending_to_tensor(item, ref=ref)
            if t is not None and t.numel() > 0:
                tensors.append(t)
        if len(tensors) == 0:
            return None
        if len(tensors) == 1:
            out = tensors[0]
        else:
            try:
                out = torch.cat(tensors, dim=-2)
            except RuntimeError:
                out = torch.cat([t.reshape(-1, t.shape[-1]) for t in tensors], dim=-2)
    else:
        raise TypeError(
            "X_pending must be None, Tensor, list, or tuple. "
            f"Got {type(X_pending)}."
        )
    if ref is not None:
        out = out.to(device=ref.device, dtype=ref.dtype)
    return out.detach()


def _apply_input_transform_for_pending(model: Model, X: Tensor) -> Tensor:
    X = _ensure_q_batch_for_pending(X)

    it = getattr(model, "input_transform", None)
    if it is not None:
        Xt = it(X)
        if isinstance(Xt, tuple):
            Xt = Xt[0]
        return _ensure_q_batch_for_pending(Xt)

    models = getattr(model, "models", None)
    if models is not None and len(models) > 0:
        it = getattr(models[0], "input_transform", None)
        if it is not None:
            Xt = it(X)
            if isinstance(Xt, tuple):
                Xt = Xt[0]
            return _ensure_q_batch_for_pending(Xt)

    return X


def _transform_pending_like_candidate(
    model: Model,
    X_pending,
    *,
    ref: Tensor,
) -> Optional[Tensor]:
    Xp = _coerce_pending_to_tensor(X_pending, ref=ref)
    if Xp is None or Xp.numel() == 0:
        return None
    Xp_t = _apply_input_transform_for_pending(model, Xp)
    return Xp_t.to(device=ref.device, dtype=ref.dtype)


def _ref_penalty(X: Tensor, X_ref: Optional[Tensor], lengthscale: float) -> Tensor:
    X_ref = _flatten_ref_points(X_ref, device=X.device, dtype=X.dtype)
    if X_ref is None:
        return torch.zeros(X.shape[:-2], device=X.device, dtype=X.dtype)
    ls2 = float(lengthscale) ** 2 + 1e-12
    ref = X_ref.view(*([1] * (X.ndim - 2)), X_ref.shape[0], X_ref.shape[1])
    diff = X.unsqueeze(-2) - ref
    d2 = diff.pow(2).sum(dim=-1)
    return torch.exp(-0.5 * d2 / ls2).sum(dim=(-1, -2))


class _qOrdinalBoundaryBase(MCAcquisitionFunction):
    """Common ordinal LSE base with Active-Learning-compatible duplicate controls."""

    def __init__(
        self,
        model: Model,
        sampler: Optional[MCSampler] = None,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        reduction: ReductionType = "sum",
        same_batch_penalty_weight: float = 0.0,
        pending_penalty_weight: float = 0.0,
        observed_penalty_weight: float = 0.0,
        penalty_lengthscale: float = 0.1,
        hard_duplicate_tol: float = 1e-8,
        exclude_same_batch_duplicates: bool = True,
        exclude_pending_duplicates: bool = True,
        exclude_observed_duplicates: bool = True,
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
    ) -> None:
        model = unwrap_single_output_model(model)
        super().__init__(model=model)
        self.ordinal_likelihood = _get_ordinal_likelihood(model)
        cutpoints = _get_cutpoints_from_likelihood(self.ordinal_likelihood)
        self.register_buffer("cutpoints", torch.as_tensor(cutpoints).detach().clone())
        self.sampler = sampler or SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        self.objective = objective

        if reduction not in ("sum", "mean"):
            raise ValueError(f"Unknown reduction: {reduction}")
        if float(hard_duplicate_tol) < 0.0:
            raise ValueError("hard_duplicate_tol must be non-negative.")
        self.reduction = reduction

        self.same_batch_penalty_weight = float(same_batch_penalty_weight)
        self.pending_penalty_weight = float(pending_penalty_weight)
        self.observed_penalty_weight = float(observed_penalty_weight)
        self.penalty_lengthscale = float(penalty_lengthscale)
        self.hard_duplicate_tol = float(hard_duplicate_tol)
        self.exclude_same_batch_duplicates = bool(exclude_same_batch_duplicates)
        self.exclude_pending_duplicates = bool(exclude_pending_duplicates)
        self.exclude_observed_duplicates = bool(exclude_observed_duplicates)
        self.X_pending = None
        self.X_observed = None
        self.set_X_pending(X_pending)
        self.set_X_observed(X_observed)

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        self.X_pending = _coerce_pending_to_tensor(X_pending)

    def set_X_observed(self, X_observed: Optional[Tensor] = None) -> None:
        self.X_observed = _coerce_pending_to_tensor(
            resolve_observed_X(self.model, X_observed)
        )

    def _apply_objective_to_score(self, score: Tensor, X: Tensor, name: str) -> Tensor:
        return _apply_ordinal_levelset_objective_to_score(self, score, X=X, name=name)

    def _reduce_q(self, score: Tensor) -> Tensor:
        if score.ndim == 0:
            return score
        if score.shape[-1] == 1:
            return score.squeeze(-1)
        if self.reduction == "mean":
            return score.mean(dim=-1)
        if self.reduction == "sum":
            return score.sum(dim=-1)
        raise ValueError(f"Unknown reduction: {self.reduction}")

    def _reference_points_in_distance_space(
        self,
        X_ref: Optional[Tensor],
        *,
        Xt: Tensor,
    ) -> Optional[Tensor]:
        return _transform_pending_like_candidate(self.model, X_ref, ref=Xt)

    def _pointwise_repulsion_penalty(self, X: Tensor) -> Tensor:
        Xt = _apply_input_transform_for_pending(self.model, X)
        penalty = torch.zeros(Xt.shape[:-1], device=Xt.device, dtype=Xt.dtype)
        n_w = _infer_n_w_from_objective_or_owner(self)

        if self.same_batch_penalty_weight > 0.0:
            penalty = penalty + self.same_batch_penalty_weight * _pointwise_same_batch_penalty(
                Xt,
                self.penalty_lengthscale,
                n_w=n_w,
            )

        Xp_t = self._reference_points_in_distance_space(self.X_pending, Xt=Xt)
        if self.pending_penalty_weight > 0.0:
            penalty = penalty + self.pending_penalty_weight * _pointwise_ref_penalty(
                Xt,
                Xp_t,
                self.penalty_lengthscale,
            )

        Xobs_t = self._reference_points_in_distance_space(self.X_observed, Xt=Xt)
        if self.observed_penalty_weight > 0.0:
            penalty = penalty + self.observed_penalty_weight * _pointwise_ref_penalty(
                Xt,
                Xobs_t,
                self.penalty_lengthscale,
            )

        penalty = penalty + hard_same_batch_duplicate_penalty_per_point(
            Xt,
            enabled=self.exclude_same_batch_duplicates,
            tolerance=self.hard_duplicate_tol,
        )
        penalty = penalty + hard_reference_duplicate_penalty_per_point(
            Xt,
            Xp_t,
            enabled=self.exclude_pending_duplicates,
            tolerance=self.hard_duplicate_tol,
        )
        penalty = penalty + hard_reference_duplicate_penalty_per_point(
            Xt,
            Xobs_t,
            enabled=self.exclude_observed_duplicates,
            tolerance=self.hard_duplicate_tol,
        )
        return penalty

    def _joint_repulsion_penalty(self, X: Tensor) -> Tensor:
        Xt = _apply_input_transform_for_pending(self.model, X)
        Xp_t = self._reference_points_in_distance_space(self.X_pending, Xt=Xt)
        Xobs_t = self._reference_points_in_distance_space(self.X_observed, Xt=Xt)
        penalty = torch.zeros(Xt.shape[:-2], device=Xt.device, dtype=Xt.dtype)

        if self.same_batch_penalty_weight > 0.0:
            penalty = penalty + self.same_batch_penalty_weight * _same_batch_penalty(
                Xt,
                self.penalty_lengthscale,
            )
        if self.pending_penalty_weight > 0.0:
            penalty = penalty + self.pending_penalty_weight * _ref_penalty(
                Xt,
                Xp_t,
                self.penalty_lengthscale,
            )
        if self.observed_penalty_weight > 0.0:
            penalty = penalty + self.observed_penalty_weight * _ref_penalty(
                Xt,
                Xobs_t,
                self.penalty_lengthscale,
            )

        hard = hard_same_batch_duplicate_penalty_per_point(
            Xt,
            enabled=self.exclude_same_batch_duplicates,
            tolerance=self.hard_duplicate_tol,
        )
        hard = hard + hard_reference_duplicate_penalty_per_point(
            Xt,
            Xp_t,
            enabled=self.exclude_pending_duplicates,
            tolerance=self.hard_duplicate_tol,
        )
        hard = hard + hard_reference_duplicate_penalty_per_point(
            Xt,
            Xobs_t,
            enabled=self.exclude_observed_duplicates,
            tolerance=self.hard_duplicate_tol,
        )
        return penalty + hard.sum(dim=-1)

    def _latent_mean_var(self, X: Tensor) -> tuple[Tensor, Tensor]:
        posterior = self.model.posterior(X)
        return _posterior_mean_var(posterior, X)

    def _latent_covariance(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        return _posterior_covariance(posterior, X)

    def _latent_samples(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        return self.get_posterior_samples(posterior).squeeze(-1)

    def _predictive_class_probs(self, X: Tensor) -> Tensor:
        f_samples = self._latent_samples(X)
        probs = ordinal_class_probs_from_f(f_samples, self.ordinal_likelihood)
        return _reduce_probs_to_match_X(probs, X)


class qOrdinalLatentStraddleAcquisition(_qOrdinalBoundaryBase):
    """ordinal 用 straddle acquisition。境界に近く、かつ不確実な点を選びます。"""

    def __init__(
        self,
        model: Model,
        beta: float = 1.0,
        sampler: Optional[MCSampler] = None,
        objective=None,
        reduction: ReductionType = "sum",
        target_boundary_idx: Optional[int] = None,
        boundary_weights: Optional[Tensor | Sequence[float]] = None,
        boundary_reduction: BoundaryReduction = "max",
        same_batch_penalty_weight: float = 0.0,
        pending_penalty_weight: float = 0.0,
        observed_penalty_weight: float = 0.0,
        penalty_lengthscale: float = 0.1,
        hard_duplicate_tol: float = 1e-8,
        exclude_same_batch_duplicates: bool = True,
        exclude_pending_duplicates: bool = True,
        exclude_observed_duplicates: bool = True,
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
    ) -> None:
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            reduction=reduction,
            same_batch_penalty_weight=same_batch_penalty_weight,
            pending_penalty_weight=pending_penalty_weight,
            observed_penalty_weight=observed_penalty_weight,
            penalty_lengthscale=penalty_lengthscale,
            hard_duplicate_tol=hard_duplicate_tol,
            exclude_same_batch_duplicates=exclude_same_batch_duplicates,
            exclude_pending_duplicates=exclude_pending_duplicates,
            exclude_observed_duplicates=exclude_observed_duplicates,
            X_pending=X_pending,
            X_observed=X_observed,
        )
        self.beta = float(beta)
        self.target_boundary_idx = target_boundary_idx
        self.boundary_weights = boundary_weights
        self.boundary_reduction = boundary_reduction

    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        mean_f, var_f = self._latent_mean_var(X)
        sigma = var_f.sqrt()
        dist_b = _cutpoint_distances_by_boundary(mean_f, self.cutpoints)
        score_b = self.beta * sigma.unsqueeze(-1) - dist_b
        score = _aggregate_boundary_scores(
            score_b,
            target_boundary_idx=self.target_boundary_idx,
            boundary_weights=self.boundary_weights,
            boundary_reduction=self.boundary_reduction,
        )
        score = score - self._pointwise_repulsion_penalty(X)
        score = self._apply_objective_to_score(score, X=X, name="qOrdinalLatentStraddle")
        return self._reduce_q(score)


class qOrdinalJointLatentStraddleAcquisition(_qOrdinalBoundaryBase):
    """ordinal 用 joint straddle acquisition。q-batch 全体の境界不確実性を評価します。"""

    def __init__(
        self,
        model: Model,
        beta: float = 1.0,
        tau: float = 1.0,
        uncertainty_measure: Literal["logdet", "trace"] = "logdet",
        target_boundary_idx: Optional[int] = None,
        boundary_weights: Optional[Tensor | Sequence[float]] = None,
        boundary_reduction: BoundaryReduction = "max",
        same_batch_penalty_weight: float = 0.0,
        pending_penalty_weight: float = 0.0,
        observed_penalty_weight: float = 0.0,
        penalty_lengthscale: float = 0.1,
        hard_duplicate_tol: float = 1e-8,
        exclude_same_batch_duplicates: bool = True,
        exclude_pending_duplicates: bool = True,
        exclude_observed_duplicates: bool = True,
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        sampler: Optional[MCSampler] = None,
        objective=None,
        input_perturbation_n_w: Optional[int] = None,
        perturbation_joint_reduction: PerturbationJointReduction = "block_mean",
        jitter: float = 1e-6,
    ) -> None:
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            reduction="sum",
            same_batch_penalty_weight=same_batch_penalty_weight,
            pending_penalty_weight=pending_penalty_weight,
            observed_penalty_weight=observed_penalty_weight,
            penalty_lengthscale=penalty_lengthscale,
            hard_duplicate_tol=hard_duplicate_tol,
            exclude_same_batch_duplicates=exclude_same_batch_duplicates,
            exclude_pending_duplicates=exclude_pending_duplicates,
            exclude_observed_duplicates=exclude_observed_duplicates,
            X_pending=X_pending,
            X_observed=X_observed,
        )
        self.beta = float(beta)
        self.tau = float(tau)
        self.uncertainty_measure = uncertainty_measure
        self.target_boundary_idx = target_boundary_idx
        self.boundary_weights = boundary_weights
        self.boundary_reduction = boundary_reduction
        self.input_perturbation_n_w = None if input_perturbation_n_w is None else int(input_perturbation_n_w)
        self.perturbation_joint_reduction = perturbation_joint_reduction
        self.jitter = float(jitter)

    def _uncertainty_score(self, cov: Tensor) -> Tensor:
        if self.uncertainty_measure == "trace":
            return cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        q = cov.shape[-1]
        eye = torch.eye(q, device=cov.device, dtype=cov.dtype)
        mat = eye + cov / (self.tau ** 2 + 1e-12)
        sign, logabsdet = torch.linalg.slogdet(mat)
        if not torch.all(sign > 0):
            raise RuntimeError("Non-positive definite covariance encountered in ordinal joint straddle.")
        return 0.5 * logabsdet

    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        mean_f, _ = self._latent_mean_var(X)
        cov_f = self._latent_covariance(X)
        n_w = _infer_n_w_from_objective_or_owner(self)
        mean_f, cov_f = _reduce_input_perturbation_mean_cov(
            mean_f,
            cov_f,
            X,
            n_w=n_w,
            mode=self.perturbation_joint_reduction,
            jitter=self.jitter,
        )
        uncertainty = self._uncertainty_score(cov_f)

        dist_b = _cutpoint_distances_by_boundary(mean_f, self.cutpoints)
        boundary_score = _aggregate_boundary_scores(
            -dist_b,
            target_boundary_idx=self.target_boundary_idx,
            boundary_weights=self.boundary_weights,
            boundary_reduction=self.boundary_reduction,
        ).mean(dim=-1)

        score = self.beta * uncertainty + boundary_score - self._joint_repulsion_penalty(X)
        return self._apply_objective_to_score(score, X=X, name="qOrdinalJointLatentStraddle")


class qOrdinalICUAcquisition(_qOrdinalBoundaryBase):
    """ordinal 用 ICU acquisition。contour / boundary 周辺の不確実性を評価します。"""

    def __init__(
        self,
        model: Model,
        boundary_weights: Optional[Tensor | Sequence[float]] = None,
        target_boundary_idx: Optional[int] = None,
        boundary_reduction: BoundaryReduction = "sum",
        sampler: Optional[MCSampler] = None,
        objective=None,
        reduction: ReductionType = "sum",
        same_batch_penalty_weight: float = 0.0,
        pending_penalty_weight: float = 0.0,
        observed_penalty_weight: float = 0.0,
        penalty_lengthscale: float = 0.1,
        hard_duplicate_tol: float = 1e-8,
        exclude_same_batch_duplicates: bool = True,
        exclude_pending_duplicates: bool = True,
        exclude_observed_duplicates: bool = True,
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
    ) -> None:
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            reduction=reduction,
            same_batch_penalty_weight=same_batch_penalty_weight,
            pending_penalty_weight=pending_penalty_weight,
            observed_penalty_weight=observed_penalty_weight,
            penalty_lengthscale=penalty_lengthscale,
            hard_duplicate_tol=hard_duplicate_tol,
            exclude_same_batch_duplicates=exclude_same_batch_duplicates,
            exclude_pending_duplicates=exclude_pending_duplicates,
            exclude_observed_duplicates=exclude_observed_duplicates,
            X_pending=X_pending,
            X_observed=X_observed,
        )
        self.boundary_weights = boundary_weights
        self.target_boundary_idx = target_boundary_idx
        self.boundary_reduction = boundary_reduction

    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        probs = self._predictive_class_probs(X)
        ge_probs = ordinal_cumulative_ge_probs_from_class_probs(probs)
        u = ordinal_boundary_uncertainty(ge_probs)
        score = _aggregate_boundary_scores(
            u,
            target_boundary_idx=self.target_boundary_idx,
            boundary_weights=self.boundary_weights,
            boundary_reduction=self.boundary_reduction,
        )
        score = score - self._pointwise_repulsion_penalty(X)
        score = self._apply_objective_to_score(score, X=X, name="qOrdinalICU")
        return self._reduce_q(score)


class qOrdinalBoundaryVarianceAcquisition(_qOrdinalBoundaryBase):
    """ordinal 用 boundary variance acquisition。境界近傍の posterior variance を重視します。"""

    def __init__(
        self,
        model: Model,
        tau: float = 1.0,
        reduce: Optional[Literal["sum", "max"]] = None,
        target_boundary_idx: Optional[int] = None,
        boundary_weights: Optional[Tensor | Sequence[float]] = None,
        boundary_reduction: BoundaryReduction = "sum",
        sampler: Optional[MCSampler] = None,
        objective=None,
        reduction: ReductionType = "sum",
        same_batch_penalty_weight: float = 0.0,
        pending_penalty_weight: float = 0.0,
        observed_penalty_weight: float = 0.0,
        penalty_lengthscale: float = 0.1,
        hard_duplicate_tol: float = 1e-8,
        exclude_same_batch_duplicates: bool = True,
        exclude_pending_duplicates: bool = True,
        exclude_observed_duplicates: bool = True,
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
    ) -> None:
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            reduction=reduction,
            same_batch_penalty_weight=same_batch_penalty_weight,
            pending_penalty_weight=pending_penalty_weight,
            observed_penalty_weight=observed_penalty_weight,
            penalty_lengthscale=penalty_lengthscale,
            hard_duplicate_tol=hard_duplicate_tol,
            exclude_same_batch_duplicates=exclude_same_batch_duplicates,
            exclude_pending_duplicates=exclude_pending_duplicates,
            exclude_observed_duplicates=exclude_observed_duplicates,
            X_pending=X_pending,
            X_observed=X_observed,
        )
        self.tau = float(tau)
        if reduce is not None:
            boundary_reduction = "max" if reduce == "max" else "sum"
        self.target_boundary_idx = target_boundary_idx
        self.boundary_weights = boundary_weights
        self.boundary_reduction = boundary_reduction

    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        mean_f, var_f = self._latent_mean_var(X)
        w_b = _boundary_kernel_weights_by_boundary(mean_f, self.cutpoints, tau=self.tau)
        score_b = var_f.unsqueeze(-1) * w_b
        score = _aggregate_boundary_scores(
            score_b,
            target_boundary_idx=self.target_boundary_idx,
            boundary_weights=self.boundary_weights,
            boundary_reduction=self.boundary_reduction,
        )
        score = score - self._pointwise_repulsion_penalty(X)
        score = self._apply_objective_to_score(score, X=X, name="qOrdinalBoundaryVariance")
        return self._reduce_q(score)


class qOrdinalClassEntropyAcquisition(_qOrdinalBoundaryBase):
    """ordinal 用 class entropy acquisition。class probability の entropy を評価します。"""

    def __init__(
        self,
        model: Model,
        sampler: Optional[MCSampler] = None,
        objective=None,
        reduction: ReductionType = "sum",
        same_batch_penalty_weight: float = 0.0,
        pending_penalty_weight: float = 0.0,
        observed_penalty_weight: float = 0.0,
        penalty_lengthscale: float = 0.1,
        hard_duplicate_tol: float = 1e-8,
        exclude_same_batch_duplicates: bool = True,
        exclude_pending_duplicates: bool = True,
        exclude_observed_duplicates: bool = True,
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
    ) -> None:
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            reduction=reduction,
            same_batch_penalty_weight=same_batch_penalty_weight,
            pending_penalty_weight=pending_penalty_weight,
            observed_penalty_weight=observed_penalty_weight,
            penalty_lengthscale=penalty_lengthscale,
            hard_duplicate_tol=hard_duplicate_tol,
            exclude_same_batch_duplicates=exclude_same_batch_duplicates,
            exclude_pending_duplicates=exclude_pending_duplicates,
            exclude_observed_duplicates=exclude_observed_duplicates,
            X_pending=X_pending,
            X_observed=X_observed,
        )

    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        probs = self._predictive_class_probs(X)
        score = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
        score = score - self._pointwise_repulsion_penalty(X)
        score = self._apply_objective_to_score(score, X=X, name="qOrdinalClassEntropy")
        return self._reduce_q(score)


__all__ = [
    "qOrdinalLatentStraddleAcquisition",
    "qOrdinalJointLatentStraddleAcquisition",
    "qOrdinalICUAcquisition",
    "qOrdinalBoundaryVarianceAcquisition",
    "qOrdinalClassEntropyAcquisition",
]
