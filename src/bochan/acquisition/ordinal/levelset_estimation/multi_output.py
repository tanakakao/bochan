from __future__ import annotations

import math
from typing import Callable, Literal, Optional, Sequence

import torch
from torch import Tensor
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform

from bochan.likelihoods.ordinal import OrdinalLogitLikelihood


RiskType = Optional[Literal["var", "cvar"]]
ReductionType = Literal["mean", "sum"]
MultiOutputMode = Literal["mean", "sum", "max", "min", "weighted_mean"]
BoundaryReduction = Literal["sum", "mean", "max", "min"]
PerturbationJointReduction = Literal["block_mean", "diagonal_mean"]


# =========================================================
# Score objective
# =========================================================
def _validate_n_w_risk(
    *,
    n_w: Optional[int],
    risk_type: RiskType,
    alpha: float,
) -> None:
    if n_w is not None and int(n_w) <= 0:
        raise ValueError("n_w must be a positive integer or None.")
    if risk_type not in (None, "var", "cvar"):
        raise ValueError(f"Unknown risk_type: {risk_type!r}.")
    if risk_type is not None and n_w is None:
        raise ValueError("risk_type is specified, but n_w is None.")
    if risk_type is not None and not (0.0 < float(alpha) <= 1.0):
        raise ValueError("alpha must be in (0, 1].")


def _aggregate_scalar_axis(
    values_w: Tensor,
    *,
    n_w: int,
    risk_type: RiskType,
    alpha: float,
    risk_dim: int,
    maximize: bool = True,
) -> Tensor:
    if risk_type is None:
        return values_w.mean(dim=risk_dim)

    descending = not maximize
    sorted_values = torch.sort(values_w, dim=risk_dim, descending=descending).values
    k = max(1, int(math.ceil(int(n_w) * float(alpha))))
    tail = sorted_values.narrow(dim=risk_dim, start=0, length=k)

    if risk_type == "var":
        return tail.select(dim=risk_dim, index=k - 1)
    if risk_type == "cvar":
        return tail.mean(dim=risk_dim)
    raise ValueError(f"Unknown risk_type: {risk_type!r}.")


class MultiOutputOrdinalLevelSetScoreObjective(torch.nn.Module):
    """
    multi-output ordinal level-set acquisition の計算済み score に作用する objective。

    InputPerturbation 使用時は、pointwise score の q * n_w を q に戻します。
    joint scalar score に対しては、デフォルトでは何もせずそのまま返します。
    """

    def __init__(
        self,
        n_w: Optional[int] = None,
        risk_type: RiskType = None,
        alpha: float = 0.5,
        maximize: bool = True,
        weight: float = 1.0,
        sign: float = 1.0,
        aggregated_risk_mode: Literal["ignore", "error"] = "ignore",
    ) -> None:
        super().__init__()
        self.n_w = None if n_w is None else int(n_w)
        self.risk_type = risk_type
        self.alpha = float(alpha)
        self.maximize = bool(maximize)
        self.weight = float(weight)
        self.sign = float(sign)
        self.aggregated_risk_mode = aggregated_risk_mode

        _validate_n_w_risk(n_w=self.n_w, risk_type=self.risk_type, alpha=self.alpha)
        if self.aggregated_risk_mode not in ("ignore", "error"):
            raise ValueError("aggregated_risk_mode must be 'ignore' or 'error'.")

    @staticmethod
    def _ensure_q_batch(X: Tensor) -> Tensor:
        return X if X.ndim > 2 else X.unsqueeze(0)

    def _is_aggregated_score(self, score: Tensor, X: Optional[Tensor]) -> bool:
        if X is None or score.ndim == 0:
            return False
        Xq = self._ensure_q_batch(X)
        return tuple(score.shape) == tuple(Xq.shape[:-2])

    def forward(self, score: Tensor, X: Optional[Tensor] = None) -> Tensor:
        if not torch.is_tensor(score):
            raise TypeError(f"score must be Tensor. Got {type(score)}.")

        score = score * self.sign * self.weight

        if score.ndim == 0 or self.n_w is None or self.n_w <= 1:
            return score

        if self._is_aggregated_score(score, X):
            if self.aggregated_risk_mode == "error":
                raise RuntimeError(
                    "MultiOutputOrdinalLevelSetScoreObjective received an aggregated score. "
                    "n_w aggregation is valid only for pointwise score."
                )
            return score

        q_expanded = score.shape[-1]
        if q_expanded % int(self.n_w) != 0:
            raise RuntimeError(
                "score.shape[-1] must be divisible by n_w. "
                f"Got score.shape={tuple(score.shape)}, n_w={self.n_w}."
            )

        q = q_expanded // int(self.n_w)
        score_w = score.reshape(*score.shape[:-1], q, int(self.n_w))
        return _aggregate_scalar_axis(
            score_w,
            n_w=int(self.n_w),
            risk_type=self.risk_type,
            alpha=self.alpha,
            risk_dim=-1,
            maximize=self.maximize,
        )


# Backward compatible internal name.
_MultiOutputOrdinalLevelSetScoreObjective = MultiOutputOrdinalLevelSetScoreObjective


def _apply_score_objective(
    owner,
    score: Tensor,
    X: Optional[Tensor],
    *,
    name: str,
) -> Tensor:
    objective = getattr(owner, "objective", None)
    if objective is None:
        return score

    try:
        out = objective(score, X=X)
    except TypeError:
        out = objective(score)

    if not torch.is_tensor(out):
        raise TypeError(f"{name}: objective must return Tensor. Got {type(out)}.")
    return out


# =========================================================
# Ordinal / posterior helpers
# =========================================================
def _try_call_zero_arg(obj):
    return obj() if callable(obj) else obj


def _get_submodels(model: Model) -> list[Model]:
    if isinstance(model, (list, tuple)):
        submodels = list(model)
    elif hasattr(model, "models"):
        submodels = list(model.models)
    else:
        raise ValueError(
            "This multi-output acquisition expects model.models or a list/tuple of ordinal models."
        )
    if len(submodels) == 0:
        raise ValueError("No submodels found.")
    return submodels


def _is_ordinal_likelihood(obj) -> bool:
    return obj is not None and (
        hasattr(obj, "marginal_class_probs") or hasattr(obj, "class_probs_from_f")
    )


def _get_ordinal_likelihood(model: Model) -> OrdinalLogitLikelihood:
    for cand in (getattr(model, "ordinal_likelihood", None), getattr(model, "likelihood", None)):
        if _is_ordinal_likelihood(cand):
            return cand
    raise ValueError("Each submodel must expose ordinal_likelihood or likelihood.")


def _get_cutpoints_from_likelihood(ordinal_likelihood) -> Tensor:
    if hasattr(ordinal_likelihood, "get_cutpoints"):
        cutpoints = _try_call_zero_arg(getattr(ordinal_likelihood, "get_cutpoints"))
        return torch.as_tensor(cutpoints).reshape(-1)

    for name in ("transformed_cutpoints", "cutpoints", "thresholds", "cuts", "cutoffs"):
        if hasattr(ordinal_likelihood, name):
            cutpoints = _try_call_zero_arg(getattr(ordinal_likelihood, name))
            return torch.as_tensor(cutpoints).reshape(-1)

    if hasattr(ordinal_likelihood, "raw_cutpoints"):
        raw = torch.as_tensor(_try_call_zero_arg(getattr(ordinal_likelihood, "raw_cutpoints")))
        if hasattr(ordinal_likelihood, "transform_cutpoints"):
            cutpoints = ordinal_likelihood.transform_cutpoints(raw)
            return torch.as_tensor(cutpoints).reshape(-1)
        return raw.reshape(-1)

    raise ValueError(
        "Could not find cutpoints on ordinal likelihood. Expected get_cutpoints, "
        "transformed_cutpoints, cutpoints, thresholds, cuts, cutoffs, or raw_cutpoints."
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


def _posterior_mean_var(posterior) -> tuple[Tensor, Tensor]:
    mean = posterior.mean
    if mean.ndim >= 1 and mean.shape[-1] == 1:
        mean = mean.squeeze(-1)

    if hasattr(posterior, "variance"):
        var = posterior.variance
        if var.ndim >= 1 and var.shape[-1] == 1:
            var = var.squeeze(-1)
        var = var.clamp_min(1e-12)
    else:
        mvn = _posterior_mvn(posterior)
        if mvn is None:
            raise ValueError("posterior must expose variance or covariance_matrix.")
        var = mvn.covariance_matrix.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12)

    return mean, var


def _posterior_covariance(posterior) -> Tensor:
    mvn = _posterior_mvn(posterior)
    if mvn is not None:
        return mvn.covariance_matrix

    if hasattr(posterior, "variance"):
        var = posterior.variance
        if var.ndim >= 1 and var.shape[-1] == 1:
            var = var.squeeze(-1)
        return torch.diag_embed(var.clamp_min(1e-12))

    raise ValueError("posterior must expose covariance_matrix or variance.")


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


def ordinal_entropy_from_probs(probs: Tensor, eps: float = 1e-12) -> Tensor:
    probs = probs.clamp_min(eps)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    return -(probs * probs.log()).sum(dim=-1)


def _reduce_extra_batch_dims(
    tensor: Tensor,
    X_like: Tensor,
    n_trailing_keep: int,
    *,
    reduce_extra: Literal["mean", "sum"] = "mean",
) -> Tensor:
    out = tensor
    X_like = _ensure_q_batch(X_like)
    x_batch_shape = tuple(X_like.shape[:-2])
    target_ndim = len(x_batch_shape) + n_trailing_keep

    while out.ndim > target_ndim:
        prefix = tuple(out.shape[:-n_trailing_keep]) if n_trailing_keep > 0 else tuple(out.shape)

        if len(x_batch_shape) == 0:
            reduce_dim = 0
        else:
            match_start = None
            max_start = len(prefix) - len(x_batch_shape)
            for s in range(max_start + 1):
                if tuple(prefix[s : s + len(x_batch_shape)]) == x_batch_shape:
                    match_start = s
                    break

            if match_start is None:
                reduce_dim = max(out.ndim - n_trailing_keep - 1, 0)
            else:
                protected = set(range(match_start, match_start + len(x_batch_shape)))
                extra_dims = [i for i in range(len(prefix)) if i not in protected]
                if not extra_dims:
                    break
                reduce_dim = extra_dims[0]

        if reduce_extra == "mean":
            out = out.mean(dim=reduce_dim)
        elif reduce_extra == "sum":
            out = out.sum(dim=reduce_dim)
        else:
            raise ValueError(f"Unknown reduce_extra: {reduce_extra}")

    return out


def _align_pointwise_score_to_X(score: Tensor, X_like: Tensor, *, name: str) -> Tensor:
    X_like = _ensure_q_batch(X_like)
    expected = X_like.shape[:-1]

    if score.shape == expected:
        return score

    if score.ndim == len(expected) + 1 and score.shape[-1] == 1:
        squeezed = score.squeeze(-1)
        if squeezed.shape == expected:
            return squeezed

    score = _reduce_extra_batch_dims(score, X_like, n_trailing_keep=1, reduce_extra="mean")
    if score.shape == expected:
        return score

    if score.numel() == math.prod(expected):
        return score.reshape(*expected)

    raise RuntimeError(
        f"{name}: failed to align score to X_like. "
        f"score.shape={tuple(score.shape)}, X_like.shape={tuple(X_like.shape)}."
    )


def _align_probs_to_X(probs: Tensor, X_like: Tensor, *, eps: float) -> Tensor:
    out = _reduce_extra_batch_dims(probs, X_like, n_trailing_keep=2, reduce_extra="mean")
    out = out.clamp_min(eps)
    return out / out.sum(dim=-1, keepdim=True).clamp_min(eps)


# =========================================================
# Boundary aggregation
# =========================================================
def _to_optional_list(value, n: int, *, name: str) -> list:
    if value is None:
        return [None] * n

    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"{name} length must match number of outputs. Expected {n}, got {len(value)}.")
        return list(value)

    return [value] * n


def _prepare_boundary_weights(
    boundary_weights: Optional[Tensor | Sequence[float]],
    n_boundaries: int,
    *,
    device,
    dtype,
) -> Optional[Tensor]:
    if boundary_weights is None:
        return None

    w = torch.as_tensor(boundary_weights, device=device, dtype=dtype).reshape(-1)
    if w.numel() != n_boundaries:
        raise ValueError(f"boundary_weights must have length {n_boundaries}, got {w.numel()}.")
    return w


def _aggregate_boundary_scores(
    boundary_scores: Tensor,
    *,
    target_boundary_idx: Optional[int] = None,
    boundary_weights: Optional[Tensor | Sequence[float]] = None,
    boundary_reduction: BoundaryReduction = "sum",
) -> Tensor:
    """
    Args:
        boundary_scores: shape = (..., n_boundaries)

    Returns:
        Tensor: shape = (...)
    """
    n_boundaries = boundary_scores.shape[-1]

    if target_boundary_idx is not None:
        idx = int(target_boundary_idx)
        if not (0 <= idx < n_boundaries):
            raise ValueError(
                f"target_boundary_idx must be in [0, {n_boundaries - 1}], got {idx}."
            )
        return boundary_scores[..., idx]

    w = _prepare_boundary_weights(
        boundary_weights,
        n_boundaries,
        device=boundary_scores.device,
        dtype=boundary_scores.dtype,
    )
    if w is not None:
        boundary_scores = boundary_scores * w.view(*([1] * (boundary_scores.ndim - 1)), -1)

    if boundary_reduction == "sum":
        return boundary_scores.sum(dim=-1)
    if boundary_reduction == "mean":
        return boundary_scores.mean(dim=-1)
    if boundary_reduction == "max":
        return boundary_scores.max(dim=-1).values
    if boundary_reduction == "min":
        return boundary_scores.min(dim=-1).values

    raise ValueError(f"Unknown boundary_reduction: {boundary_reduction}.")


def _boundary_kernel_scores(values: Tensor, cutpoints: Tensor, tau: float) -> Tensor:
    cp = cutpoints.detach().to(device=values.device, dtype=values.dtype)
    tau_t = torch.as_tensor(tau, device=values.device, dtype=values.dtype).clamp_min(1e-8)
    z2 = ((values.unsqueeze(-1) - cp.view(*([1] * values.ndim), -1)) / tau_t) ** 2
    return torch.exp(-0.5 * z2)


# =========================================================
# Distance / transform / penalty utilities
# =========================================================
def _ensure_q_batch(X: Tensor) -> Tensor:
    if not torch.is_tensor(X):
        raise TypeError(f"X must be Tensor. Got {type(X)}.")
    if X.ndim == 1:
        return X.view(1, 1, -1)
    if X.ndim == 2:
        return X.unsqueeze(0)
    return X


def _coerce_reference_to_tensor(
    X_ref,
    *,
    ref: Optional[Tensor] = None,
) -> Optional[Tensor]:
    if X_ref is None:
        return None

    if torch.is_tensor(X_ref):
        out = X_ref
    elif isinstance(X_ref, (list, tuple)):
        tensors = []
        for item in X_ref:
            if item is None:
                continue
            t = _coerce_reference_to_tensor(item, ref=ref)
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
            "X_pending / X_observed must be None, Tensor, list, or tuple. "
            f"Got {type(X_ref)}."
        )

    if ref is not None:
        out = out.to(device=ref.device, dtype=ref.dtype)
    return out


# Backward-compatible helper name.
_coerce_pending_to_tensor = _coerce_reference_to_tensor


def _apply_input_transform_for_reference(model: Model, X: Tensor) -> Tensor:
    X = _ensure_q_batch(X)

    it = getattr(model, "input_transform", None)
    if it is not None:
        Xt = it(X)
        if isinstance(Xt, tuple):
            Xt = Xt[0]
        return _ensure_q_batch(Xt)

    models = getattr(model, "models", None)
    if models is not None and len(models) > 0:
        it = getattr(models[0], "input_transform", None)
        if it is not None:
            Xt = it(X)
            if isinstance(Xt, tuple):
                Xt = Xt[0]
            return _ensure_q_batch(Xt)

    return X


_apply_input_transform_for_pending = _apply_input_transform_for_reference


def _transform_reference_like_candidate(
    model: Model,
    X_ref,
    *,
    ref: Tensor,
) -> Optional[Tensor]:
    Xr = _coerce_reference_to_tensor(X_ref, ref=ref)
    if Xr is None or Xr.numel() == 0:
        return None

    Xr_t = _apply_input_transform_for_reference(model, Xr)
    return Xr_t.to(device=ref.device, dtype=ref.dtype)


_transform_pending_like_candidate = _transform_reference_like_candidate


def _resolve_observed_X(model: Model, X_observed: Optional[Tensor] = None) -> Optional[Tensor]:
    if X_observed is not None:
        return X_observed

    for attr in ("train_X_original", "train_X", "train_inputs_raw"):
        x = getattr(model, attr, None)
        if x is not None:
            return x

    train_inputs = getattr(model, "train_inputs", None)
    if isinstance(train_inputs, tuple) and len(train_inputs) > 0:
        return train_inputs[0]

    models = getattr(model, "models", None)
    if models is not None and len(models) > 0:
        sm = models[0]
        for attr in ("train_X_original", "train_X", "train_inputs_raw"):
            x = getattr(sm, attr, None)
            if x is not None:
                return x
        train_inputs = getattr(sm, "train_inputs", None)
        if isinstance(train_inputs, tuple) and len(train_inputs) > 0:
            return train_inputs[0]

    return None


def _resolve_cat_dims(model: Model) -> list[int]:
    cat_dims = getattr(model, "cat_dims", None)
    if cat_dims is not None:
        return [int(i) for i in cat_dims]

    models = getattr(model, "models", None)
    if models is not None and len(models) > 0:
        cat_dims = getattr(models[0], "cat_dims", None)
        if cat_dims is not None:
            return [int(i) for i in cat_dims]
    return []


def _split_cont_cat(X: Tensor, cat_dims: Sequence[int]) -> tuple[Optional[Tensor], Optional[Tensor]]:
    d = X.shape[-1]
    cat_dims = [i for i in cat_dims if 0 <= i < d]
    cont_dims = [i for i in range(d) if i not in cat_dims]

    X_cont = X[..., cont_dims] if len(cont_dims) > 0 else None
    X_cat = X[..., cat_dims] if len(cat_dims) > 0 else None
    return X_cont, X_cat


def _pairwise_distance_proxy(A: Tensor, B: Tensor, cat_dims: Sequence[int]) -> Tensor:
    A_cont, A_cat = _split_cont_cat(A, cat_dims)
    B_cont, B_cat = _split_cont_cat(B, cat_dims)

    dist2: Tensor | float = 0.0

    if A_cont is not None:
        diff = A_cont.unsqueeze(-2) - B_cont.unsqueeze(-3)
        dist2 = dist2 + (diff**2).sum(dim=-1)

    if A_cat is not None:
        mismatch = (A_cat.unsqueeze(-2) != B_cat.unsqueeze(-3)).to(A.dtype)
        dist2 = dist2 + mismatch.sum(dim=-1)

    if isinstance(dist2, float):
        raise RuntimeError("No valid dimensions found for distance computation.")
    return dist2


def _broadcast_reference_to_batch(X_ref: Tensor, batch_shape: torch.Size) -> Tensor:
    X_ref = _ensure_q_batch(X_ref)

    if X_ref.shape[:-2] == batch_shape:
        return X_ref

    try:
        return X_ref.expand(*batch_shape, X_ref.shape[-2], X_ref.shape[-1])
    except RuntimeError:
        X2d = X_ref.reshape(-1, X_ref.shape[-1])
        return X2d.view(*([1] * len(batch_shape)), X2d.shape[-2], X2d.shape[-1]).expand(
            *batch_shape,
            X2d.shape[-2],
            X2d.shape[-1],
        )


def _reference_penalty_per_point(
    X: Tensor,
    X_ref: Optional[Tensor],
    *,
    beta: float,
    weight: float,
    cat_dims: Sequence[int],
) -> Tensor:
    X = _ensure_q_batch(X)
    if weight <= 0.0 or X_ref is None or X_ref.numel() == 0:
        return X.new_zeros(X.shape[:-1])

    X_ref = _broadcast_reference_to_batch(
        X_ref.to(device=X.device, dtype=X.dtype),
        X.shape[:-2],
    )

    dist2 = _pairwise_distance_proxy(X, X_ref, cat_dims)
    nearest = dist2.min(dim=-1).values
    return weight * torch.exp(-float(beta) * nearest)


def _same_batch_penalty_per_point(
    X: Tensor,
    *,
    beta: float,
    weight: float,
    cat_dims: Sequence[int],
) -> Tensor:
    X = _ensure_q_batch(X)
    batch_shape = X.shape[:-2]
    q = X.shape[-2]

    if q <= 1 or weight <= 0.0:
        return X.new_zeros(X.shape[:-1])

    d2 = _pairwise_distance_proxy(X, X, cat_dims)
    eye = torch.eye(q, device=X.device, dtype=torch.bool)
    d2 = d2.masked_fill(eye, float("inf"))

    return weight * torch.exp(-float(beta) * d2).sum(dim=-1).reshape(*batch_shape, q)


def _reference_penalty_aggregated(
    X: Tensor,
    X_ref: Optional[Tensor],
    *,
    beta: float,
    weight: float,
    cat_dims: Sequence[int],
) -> Tensor:
    per_point = _reference_penalty_per_point(
        X=X,
        X_ref=X_ref,
        beta=beta,
        weight=weight,
        cat_dims=cat_dims,
    )
    return per_point.sum(dim=-1)


def _same_batch_penalty_aggregated(
    X: Tensor,
    *,
    beta: float,
    weight: float,
    cat_dims: Sequence[int],
) -> Tensor:
    per_point = _same_batch_penalty_per_point(X, beta=beta, weight=weight, cat_dims=cat_dims)
    if per_point.shape[-1] <= 1:
        return per_point.new_zeros(per_point.shape[:-1])
    return 0.5 * per_point.sum(dim=-1)


# =========================================================
# InputPerturbation joint reduction
# =========================================================
def _infer_n_w_from_objective_or_owner(owner) -> Optional[int]:
    n_w = getattr(owner, "input_perturbation_n_w", None)
    if n_w is not None:
        return int(n_w)

    objective = getattr(owner, "objective", None)
    if objective is not None and getattr(objective, "n_w", None) is not None:
        return int(objective.n_w)

    return None


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

    X_in = _ensure_q_batch(X)
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


# =========================================================
# Base classes
# =========================================================
class _qMultiOutputOrdinalBoundaryBase(AcquisitionFunction):
    """
    classification multi-output level-set API に寄せた ordinal multi-output base。

    Pointwise acquisition の標準順序:
        boundary/output score -> output_mode -> pointwise penalty -> objective -> reduction
    """

    def __init__(
        self,
        model: Model,
        output_weights: Optional[Tensor | Sequence[float]] = None,
        reduction: ReductionType = "mean",
        output_mode: MultiOutputMode = "mean",
        sampler: Optional[MCSampler] = None,
        eps: float = 1e-6,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(model=model)

        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'.")
        if output_mode not in ("mean", "sum", "max", "min", "weighted_mean"):
            raise ValueError(
                "output_mode must be one of 'mean', 'sum', 'max', 'min', 'weighted_mean'."
            )

        self.submodels = _get_submodels(model)
        self.n_outputs = len(self.submodels)
        self.ordinal_likelihoods = [_get_ordinal_likelihood(m) for m in self.submodels]
        # cutpoints は likelihood parameter から計算されることがあり、
        # transformed_cutpoints / transform_cutpoints が grad_fn を持つ Tensor を返す場合がある。
        # acquisition 最適化では X だけを最適化対象にしたいので、cutpoints は定数として detach する。
        # これをしないと torch optimizer の複数 step backward で
        # "Trying to backward through the graph a second time" が起きやすい。
        self.cutpoints_list = [
            torch.as_tensor(_get_cutpoints_from_likelihood(lik)).detach().clone()
            for lik in self.ordinal_likelihoods
        ]

        ref_train_X = getattr(self.submodels[0], "train_X", None)
        ref_device = ref_train_X.device if torch.is_tensor(ref_train_X) else self.cutpoints_list[0].device
        ref_dtype = ref_train_X.dtype if torch.is_tensor(ref_train_X) else self.cutpoints_list[0].dtype

        self.register_buffer(
            "output_weights",
            _to_1d_float_tensor(
                output_weights,
                self.n_outputs,
                device=ref_device,
                dtype=ref_dtype,
                default=1.0,
            ),
        )

        self.reduction = reduction
        self.output_mode = output_mode
        self.sampler = sampler or SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        self.eps = float(eps)

        self.pending_penalty_weight = float(pending_penalty_weight)
        self.pending_penalty_beta = float(pending_penalty_beta)
        self.observed_penalty_weight = float(observed_penalty_weight)
        self.observed_penalty_beta = float(observed_penalty_beta)
        self.same_batch_penalty_weight = float(same_batch_penalty_weight)
        self.same_batch_penalty_beta = float(same_batch_penalty_beta)

        self.cat_dims = _resolve_cat_dims(model)
        self.objective = objective

        self.X_pending: Optional[Tensor] = None
        self.X_observed: Optional[Tensor] = None
        self.set_X_pending(X_pending)
        self.set_X_observed(X_observed)

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        self.X_pending = _coerce_reference_to_tensor(X_pending)

    def set_X_observed(self, X_observed: Optional[Tensor] = None) -> None:
        self.X_observed = _coerce_reference_to_tensor(_resolve_observed_X(self.model, X_observed))

    def _set_eval_mode(self) -> None:
        self.model.eval()
        for sm in self.submodels:
            sm.eval()
            for attr in ("likelihood", "ordinal_likelihood"):
                lik = getattr(sm, attr, None)
                if lik is not None and hasattr(lik, "eval"):
                    lik.eval()

    def _weights_like(self, X: Tensor) -> Tensor:
        return self.output_weights.to(device=X.device, dtype=X.dtype)

    def _aggregate_outputs(self, score_per_output: Tensor) -> Tensor:
        """
        Args:
            score_per_output: shape = (*batch, q_like, m)

        Returns:
            Tensor: shape = (*batch, q_like)
        """
        if self.output_mode == "mean":
            return score_per_output.mean(dim=-1)
        if self.output_mode == "sum":
            return score_per_output.sum(dim=-1)
        if self.output_mode == "max":
            return score_per_output.max(dim=-1).values
        if self.output_mode == "min":
            return score_per_output.min(dim=-1).values
        if self.output_mode == "weighted_mean":
            w = self.output_weights.to(device=score_per_output.device, dtype=score_per_output.dtype)
            if w.ndim != 1 or w.numel() != score_per_output.shape[-1]:
                raise ValueError(
                    f"output_weights must have shape ({score_per_output.shape[-1]},), got {tuple(w.shape)}."
                )
            w = w / w.sum().clamp_min(self.eps)
            view_shape = (1,) * (score_per_output.ndim - 1) + (w.numel(),)
            return (score_per_output * w.view(*view_shape)).sum(dim=-1)
        raise ValueError(f"Unknown output_mode: {self.output_mode}.")

    def _aggregate_output_scalars(self, score_per_output: Tensor) -> Tensor:
        """
        Args:
            score_per_output: shape = (*batch, m)

        Returns:
            Tensor: shape = (*batch,)
        """
        if self.output_mode == "mean":
            return score_per_output.mean(dim=-1)
        if self.output_mode == "sum":
            return score_per_output.sum(dim=-1)
        if self.output_mode == "max":
            return score_per_output.max(dim=-1).values
        if self.output_mode == "min":
            return score_per_output.min(dim=-1).values
        if self.output_mode == "weighted_mean":
            w = self.output_weights.to(device=score_per_output.device, dtype=score_per_output.dtype)
            if w.ndim != 1 or w.numel() != score_per_output.shape[-1]:
                raise ValueError(
                    f"output_weights must have shape ({score_per_output.shape[-1]},), got {tuple(w.shape)}."
                )
            w = w / w.sum().clamp_min(self.eps)
            view_shape = (1,) * (score_per_output.ndim - 1) + (w.numel(),)
            return (score_per_output * w.view(*view_shape)).sum(dim=-1)
        raise ValueError(f"Unknown output_mode: {self.output_mode}.")

    def _reduce_q(self, score: Tensor) -> Tensor:
        if score.ndim == 0:
            return score
        if score.shape[-1] == 1:
            return score.squeeze(-1)
        if self.reduction == "mean":
            return score.mean(dim=-1)
        if self.reduction == "sum":
            return score.sum(dim=-1)
        raise ValueError(f"Unknown reduction: {self.reduction}.")

    def _check_output_shape(self, out: Tensor, expected: torch.Size, name: str) -> None:
        if out.shape != expected:
            raise RuntimeError(
                f"{name}: output shape mismatch. Expected {tuple(expected)}, got {tuple(out.shape)}."
            )

    def _latent_mean_var_list(self, X: Tensor, X_like: Optional[Tensor] = None) -> list[tuple[Tensor, Tensor]]:
        if X_like is None:
            X_like = _apply_input_transform_for_reference(self.model, X)

        outs: list[tuple[Tensor, Tensor]] = []
        for m in self.submodels:
            mean, var = _posterior_mean_var(m.posterior(X))
            mean = _align_pointwise_score_to_X(mean, X_like, name="latent mean")
            var = _align_pointwise_score_to_X(var, X_like, name="latent variance").clamp_min(self.eps)
            outs.append((mean, var))
        return outs

    def _latent_covariance_list(self, X: Tensor) -> list[Tensor]:
        return [_posterior_covariance(m.posterior(X)) for m in self.submodels]

    def _predictive_class_probs_list(self, X: Tensor, X_like: Optional[Tensor] = None) -> list[Tensor]:
        if X_like is None:
            X_like = _apply_input_transform_for_reference(self.model, X)

        outs: list[Tensor] = []
        for m, lik in zip(self.submodels, self.ordinal_likelihoods):
            posterior = m.posterior(X)
            if hasattr(lik, "marginal_class_probs"):
                probs = lik.marginal_class_probs(posterior.distribution)
            else:
                f_samples = self.sampler(posterior)
                if f_samples.ndim >= 1 and f_samples.shape[-1] == 1:
                    f_samples = f_samples.squeeze(-1)
                probs = ordinal_class_probs_from_f(f_samples, lik).mean(dim=0)
            outs.append(_align_probs_to_X(probs, X_like, eps=self.eps))
        return outs

    def _pointwise_repulsion_penalty(self, Xt: Tensor) -> Tensor:
        Xt = _ensure_q_batch(Xt)
        penalty = torch.zeros(Xt.shape[:-1], device=Xt.device, dtype=Xt.dtype)

        if self.pending_penalty_weight > 0.0:
            Xp_t = _transform_reference_like_candidate(self.model, self.X_pending, ref=Xt)
            penalty = penalty + _reference_penalty_per_point(
                Xt,
                Xp_t,
                beta=self.pending_penalty_beta,
                weight=self.pending_penalty_weight,
                cat_dims=self.cat_dims,
            )

        if self.observed_penalty_weight > 0.0:
            Xobs_t = _transform_reference_like_candidate(self.model, self.X_observed, ref=Xt)
            penalty = penalty + _reference_penalty_per_point(
                Xt,
                Xobs_t,
                beta=self.observed_penalty_beta,
                weight=self.observed_penalty_weight,
                cat_dims=self.cat_dims,
            )

        if self.same_batch_penalty_weight > 0.0:
            penalty = penalty + _same_batch_penalty_per_point(
                Xt,
                beta=self.same_batch_penalty_beta,
                weight=self.same_batch_penalty_weight,
                cat_dims=self.cat_dims,
            )

        return penalty

    def _aggregated_repulsion_penalty(self, Xt: Tensor) -> Tensor:
        Xt = _ensure_q_batch(Xt)
        penalty = torch.zeros(Xt.shape[:-2], device=Xt.device, dtype=Xt.dtype)

        if self.same_batch_penalty_weight > 0.0:
            penalty = penalty + _same_batch_penalty_aggregated(
                Xt,
                beta=self.same_batch_penalty_beta,
                weight=self.same_batch_penalty_weight,
                cat_dims=self.cat_dims,
            )

        if self.pending_penalty_weight > 0.0:
            Xp_t = _transform_reference_like_candidate(self.model, self.X_pending, ref=Xt)
            penalty = penalty + _reference_penalty_aggregated(
                Xt,
                Xp_t,
                beta=self.pending_penalty_beta,
                weight=self.pending_penalty_weight,
                cat_dims=self.cat_dims,
            )

        if self.observed_penalty_weight > 0.0:
            Xobs_t = _transform_reference_like_candidate(self.model, self.X_observed, ref=Xt)
            penalty = penalty + _reference_penalty_aggregated(
                Xt,
                Xobs_t,
                beta=self.observed_penalty_beta,
                weight=self.observed_penalty_weight,
                cat_dims=self.cat_dims,
            )

        return penalty

    def _finalize_pointwise_scores(
        self,
        score_per_output: Tensor,
        X: Tensor,
        *,
        name: str,
    ) -> Tensor:
        raw_X = _ensure_q_batch(X)
        original_batch_shape = raw_X.shape[:-2]
        Xt = _apply_input_transform_for_reference(self.model, raw_X)

        if score_per_output.shape[:-1] != Xt.shape[:-1]:
            expected = Xt.shape[:-1] + torch.Size([score_per_output.shape[-1]])
            if score_per_output.numel() == math.prod(expected):
                score_per_output = score_per_output.reshape(*expected)
            else:
                raise RuntimeError(
                    f"{name}: score_per_output shape mismatch. "
                    f"score_per_output.shape={tuple(score_per_output.shape)}, Xt.shape={tuple(Xt.shape)}."
                )

        score = self._aggregate_outputs(score_per_output)
        score = score - self._pointwise_repulsion_penalty(Xt)
        score = _apply_score_objective(self, score, raw_X, name=name)
        out = self._reduce_q(score)
        self._check_output_shape(out, original_batch_shape, name)
        return out


# Backward-compatible non-q base name.
_MultiOutputOrdinalBoundaryBase = _qMultiOutputOrdinalBoundaryBase


def _to_1d_float_tensor(
    value: float | Sequence[float] | Tensor | None,
    length: int,
    *,
    device,
    dtype,
    default: float = 1.0,
) -> Tensor:
    if value is None:
        return torch.full((length,), float(default), device=device, dtype=dtype)
    if isinstance(value, (float, int)):
        return torch.full((length,), float(value), device=device, dtype=dtype)
    out = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
    if out.numel() != length:
        raise ValueError(f"Expected length {length}, got {out.numel()}.")
    return out


# =========================================================
# Pointwise multi-output acquisitions
# =========================================================
class qMultiOutputOrdinalLatentStraddleAcquisition(_qMultiOutputOrdinalBoundaryBase):
    """multi-output ordinal latent straddle acquisition."""

    def __init__(
        self,
        model: Model,
        beta: float | Sequence[float] | Tensor = 1.0,
        output_weights: Optional[Tensor | Sequence[float]] = None,
        reduction: ReductionType = "mean",
        output_mode: MultiOutputMode = "mean",
        boundary_weights_list: Optional[Sequence[Optional[Tensor | Sequence[float]]]] = None,
        target_boundary_idx_list: Optional[Sequence[Optional[int]] | int] = None,
        boundary_reduction: BoundaryReduction = "sum",
        sampler: Optional[MCSampler] = None,
        eps: float = 1e-6,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(
            model=model,
            output_weights=output_weights,
            reduction=reduction,
            output_mode=output_mode,
            sampler=sampler,
            eps=eps,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            observed_penalty_weight=observed_penalty_weight,
            observed_penalty_beta=observed_penalty_beta,
            same_batch_penalty_weight=same_batch_penalty_weight,
            same_batch_penalty_beta=same_batch_penalty_beta,
            X_pending=X_pending,
            X_observed=X_observed,
            objective=objective,
        )
        self.register_buffer(
            "beta_vec",
            _to_1d_float_tensor(
                beta,
                self.n_outputs,
                device=self.output_weights.device,
                dtype=self.output_weights.dtype,
                default=1.0,
            ),
        )
        self.boundary_weights_list = _to_optional_list(
            boundary_weights_list,
            self.n_outputs,
            name="boundary_weights_list",
        )
        self.target_boundary_idx_list = _to_optional_list(
            target_boundary_idx_list,
            self.n_outputs,
            name="target_boundary_idx_list",
        )
        self.boundary_reduction = boundary_reduction

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        X = _ensure_q_batch(X)
        Xt = _apply_input_transform_for_reference(self.model, X)

        scores: list[Tensor] = []
        for o, ((mean_f, var_f), cp) in enumerate(zip(self._latent_mean_var_list(X, X_like=Xt), self.cutpoints_list)):
            std_f = var_f.sqrt()
            cp = cp.detach().to(device=mean_f.device, dtype=mean_f.dtype)
            dist_b = (mean_f.unsqueeze(-1) - cp.view(*([1] * mean_f.ndim), -1)).abs()
            score_b = self.beta_vec[o].to(mean_f) * std_f.unsqueeze(-1) - dist_b

            score_o = _aggregate_boundary_scores(
                score_b,
                target_boundary_idx=self.target_boundary_idx_list[o],
                boundary_weights=self.boundary_weights_list[o],
                boundary_reduction=self.boundary_reduction,
            )
            scores.append(score_o)

        score_per_output = torch.stack(scores, dim=-1)
        return self._finalize_pointwise_scores(
            score_per_output,
            X,
            name="qMultiOutputOrdinalLatentStraddle",
        )


class qMultiOutputOrdinalICUAcquisition(_qMultiOutputOrdinalBoundaryBase):
    """multi-output ordinal ICU acquisition."""

    def __init__(
        self,
        model: Model,
        boundary_weights_list: Optional[Sequence[Optional[Tensor | Sequence[float]]]] = None,
        target_boundary_idx_list: Optional[Sequence[Optional[int]] | int] = None,
        boundary_reduction: BoundaryReduction = "sum",
        output_weights: Optional[Tensor | Sequence[float]] = None,
        reduction: ReductionType = "mean",
        output_mode: MultiOutputMode = "mean",
        sampler: Optional[MCSampler] = None,
        eps: float = 1e-6,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(
            model=model,
            output_weights=output_weights,
            reduction=reduction,
            output_mode=output_mode,
            sampler=sampler,
            eps=eps,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            observed_penalty_weight=observed_penalty_weight,
            observed_penalty_beta=observed_penalty_beta,
            same_batch_penalty_weight=same_batch_penalty_weight,
            same_batch_penalty_beta=same_batch_penalty_beta,
            X_pending=X_pending,
            X_observed=X_observed,
            objective=objective,
        )
        self.boundary_weights_list = _to_optional_list(
            boundary_weights_list,
            self.n_outputs,
            name="boundary_weights_list",
        )
        self.target_boundary_idx_list = _to_optional_list(
            target_boundary_idx_list,
            self.n_outputs,
            name="target_boundary_idx_list",
        )
        self.boundary_reduction = boundary_reduction

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        X = _ensure_q_batch(X)
        Xt = _apply_input_transform_for_reference(self.model, X)

        scores: list[Tensor] = []
        for o, probs in enumerate(self._predictive_class_probs_list(X, X_like=Xt)):
            ge_probs = ordinal_cumulative_ge_probs_from_class_probs(probs)
            u = ordinal_boundary_uncertainty(ge_probs)

            score_o = _aggregate_boundary_scores(
                u,
                target_boundary_idx=self.target_boundary_idx_list[o],
                boundary_weights=self.boundary_weights_list[o],
                boundary_reduction=self.boundary_reduction,
            )
            scores.append(score_o)

        score_per_output = torch.stack(scores, dim=-1)
        return self._finalize_pointwise_scores(score_per_output, X, name="qMultiOutputOrdinalICU")


class qMultiOutputOrdinalBoundaryVarianceAcquisition(_qMultiOutputOrdinalBoundaryBase):
    """multi-output ordinal boundary variance acquisition."""

    def __init__(
        self,
        model: Model,
        tau: float = 1.0,
        boundary_weights_list: Optional[Sequence[Optional[Tensor | Sequence[float]]]] = None,
        target_boundary_idx_list: Optional[Sequence[Optional[int]] | int] = None,
        boundary_reduction: BoundaryReduction = "sum",
        output_weights: Optional[Tensor | Sequence[float]] = None,
        reduction: ReductionType = "mean",
        output_mode: MultiOutputMode = "mean",
        sampler: Optional[MCSampler] = None,
        eps: float = 1e-6,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        # Compatibility with older API. If provided, it overrides boundary_reduction
        # for boundary aggregation with "sum" or "max".
        reduce: Optional[Literal["sum", "max"]] = None,
    ) -> None:
        if reduce is not None:
            boundary_reduction = reduce

        super().__init__(
            model=model,
            output_weights=output_weights,
            reduction=reduction,
            output_mode=output_mode,
            sampler=sampler,
            eps=eps,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            observed_penalty_weight=observed_penalty_weight,
            observed_penalty_beta=observed_penalty_beta,
            same_batch_penalty_weight=same_batch_penalty_weight,
            same_batch_penalty_beta=same_batch_penalty_beta,
            X_pending=X_pending,
            X_observed=X_observed,
            objective=objective,
        )
        self.tau = float(tau)
        self.boundary_weights_list = _to_optional_list(
            boundary_weights_list,
            self.n_outputs,
            name="boundary_weights_list",
        )
        self.target_boundary_idx_list = _to_optional_list(
            target_boundary_idx_list,
            self.n_outputs,
            name="target_boundary_idx_list",
        )
        self.boundary_reduction = boundary_reduction

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        X = _ensure_q_batch(X)
        Xt = _apply_input_transform_for_reference(self.model, X)

        scores: list[Tensor] = []
        for o, ((mean_f, var_f), cp) in enumerate(zip(self._latent_mean_var_list(X, X_like=Xt), self.cutpoints_list)):
            cp = cp.to(device=mean_f.device, dtype=mean_f.dtype)
            w_b = _boundary_kernel_scores(mean_f, cp, tau=self.tau)
            score_b = var_f.unsqueeze(-1) * w_b

            score_o = _aggregate_boundary_scores(
                score_b,
                target_boundary_idx=self.target_boundary_idx_list[o],
                boundary_weights=self.boundary_weights_list[o],
                boundary_reduction=self.boundary_reduction,
            )
            scores.append(score_o)

        score_per_output = torch.stack(scores, dim=-1)
        return self._finalize_pointwise_scores(
            score_per_output,
            X,
            name="qMultiOutputOrdinalBoundaryVariance",
        )


class qMultiOutputOrdinalBoundaryEntropyAcquisition(_qMultiOutputOrdinalBoundaryBase):
    """multi-output ordinal boundary entropy acquisition.

    Class entropy ではなく、各 boundary の binary entropy
    H[1(y >= k)] を使うため、target_boundary_idx_list による境界指定が可能。
    """

    def __init__(
        self,
        model: Model,
        boundary_weights_list: Optional[Sequence[Optional[Tensor | Sequence[float]]]] = None,
        target_boundary_idx_list: Optional[Sequence[Optional[int]] | int] = None,
        boundary_reduction: BoundaryReduction = "sum",
        output_weights: Optional[Tensor | Sequence[float]] = None,
        reduction: ReductionType = "mean",
        output_mode: MultiOutputMode = "mean",
        sampler: Optional[MCSampler] = None,
        eps: float = 1e-6,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(
            model=model,
            output_weights=output_weights,
            reduction=reduction,
            output_mode=output_mode,
            sampler=sampler,
            eps=eps,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            observed_penalty_weight=observed_penalty_weight,
            observed_penalty_beta=observed_penalty_beta,
            same_batch_penalty_weight=same_batch_penalty_weight,
            same_batch_penalty_beta=same_batch_penalty_beta,
            X_pending=X_pending,
            X_observed=X_observed,
            objective=objective,
        )
        self.boundary_weights_list = _to_optional_list(
            boundary_weights_list,
            self.n_outputs,
            name="boundary_weights_list",
        )
        self.target_boundary_idx_list = _to_optional_list(
            target_boundary_idx_list,
            self.n_outputs,
            name="target_boundary_idx_list",
        )
        self.boundary_reduction = boundary_reduction

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        X = _ensure_q_batch(X)
        Xt = _apply_input_transform_for_reference(self.model, X)

        scores: list[Tensor] = []
        for o, probs in enumerate(self._predictive_class_probs_list(X, X_like=Xt)):
            ge_probs = ordinal_cumulative_ge_probs_from_class_probs(probs).clamp(
                self.eps,
                1.0 - self.eps,
            )
            boundary_entropy = -(
                ge_probs * ge_probs.log()
                + (1.0 - ge_probs) * (1.0 - ge_probs).log()
            )

            score_o = _aggregate_boundary_scores(
                boundary_entropy,
                target_boundary_idx=self.target_boundary_idx_list[o],
                boundary_weights=self.boundary_weights_list[o],
                boundary_reduction=self.boundary_reduction,
            )
            scores.append(score_o)

        score_per_output = torch.stack(scores, dim=-1)
        return self._finalize_pointwise_scores(
            score_per_output,
            X,
            name="qMultiOutputOrdinalBoundaryEntropy",
        )


class qMultiOutputOrdinalClassEntropyAcquisition(_qMultiOutputOrdinalBoundaryBase):
    """multi-output ordinal class entropy acquisition.

    This acquisition measures entropy of the whole class distribution H[y].
    It does not use target_boundary_idx_list; use qMultiOutputOrdinalBoundaryEntropyAcquisition
    if you need boundary-wise entropy.
    """

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        X = _ensure_q_batch(X)
        Xt = _apply_input_transform_for_reference(self.model, X)

        scores: list[Tensor] = []
        for probs in self._predictive_class_probs_list(X, X_like=Xt):
            scores.append(ordinal_entropy_from_probs(probs, eps=self.eps))

        score_per_output = torch.stack(scores, dim=-1)
        return self._finalize_pointwise_scores(
            score_per_output,
            X,
            name="qMultiOutputOrdinalClassEntropy",
        )


# =========================================================
# Joint multi-output acquisition
# =========================================================
class qMultiOutputOrdinalJointLatentStraddleAcquisition(_qMultiOutputOrdinalBoundaryBase):
    """multi-output ordinal joint latent straddle acquisition.

    The score is already batch-level, so InputPerturbation risk objective should
    usually not be used to perform q*n_w -> q aggregation here.
    """

    def __init__(
        self,
        model: Model,
        beta: float | Sequence[float] | Tensor = 1.0,
        tau: float = 1.0,
        uncertainty_measure: Literal["logdet", "trace"] = "logdet",
        output_weights: Optional[Tensor | Sequence[float]] = None,
        output_mode: MultiOutputMode = "weighted_mean",
        same_batch_penalty_weight: float = 0.0,
        pending_penalty_weight: float = 0.0,
        observed_penalty_weight: float = 0.0,
        penalty_lengthscale: Optional[float] = None,
        distance_beta: Optional[float] = None,
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        sampler: Optional[MCSampler] = None,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
        input_perturbation_n_w: Optional[int] = None,
        perturbation_joint_reduction: PerturbationJointReduction = "block_mean",
        jitter: float = 1e-6,
        boundary_weights_list: Optional[Sequence[Optional[Tensor | Sequence[float]]]] = None,
        target_boundary_idx_list: Optional[Sequence[Optional[int]] | int] = None,
        boundary_reduction: BoundaryReduction = "sum",
    ) -> None:
        beta_for_penalty = 10.0 if distance_beta is None else float(distance_beta)
        if penalty_lengthscale is not None:
            beta_for_penalty = 0.5 / (float(penalty_lengthscale) ** 2 + 1e-12)

        super().__init__(
            model=model,
            output_weights=output_weights,
            reduction="sum",
            output_mode=output_mode,
            sampler=sampler,
            eps=1e-8,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=beta_for_penalty,
            observed_penalty_weight=observed_penalty_weight,
            observed_penalty_beta=beta_for_penalty,
            same_batch_penalty_weight=same_batch_penalty_weight,
            same_batch_penalty_beta=beta_for_penalty,
            X_pending=X_pending,
            X_observed=X_observed,
            objective=objective,
        )
        self.register_buffer(
            "beta_vec",
            _to_1d_float_tensor(
                beta,
                self.n_outputs,
                device=self.output_weights.device,
                dtype=self.output_weights.dtype,
                default=1.0,
            ),
        )
        self.tau = float(tau)
        self.uncertainty_measure = uncertainty_measure
        self.input_perturbation_n_w = (
            None if input_perturbation_n_w is None else int(input_perturbation_n_w)
        )
        self.perturbation_joint_reduction = perturbation_joint_reduction
        self.jitter = float(jitter)
        self.boundary_weights_list = _to_optional_list(
            boundary_weights_list,
            self.n_outputs,
            name="boundary_weights_list",
        )
        self.target_boundary_idx_list = _to_optional_list(
            target_boundary_idx_list,
            self.n_outputs,
            name="target_boundary_idx_list",
        )
        self.boundary_reduction = boundary_reduction

    def _uncertainty_score(self, cov: Tensor) -> Tensor:
        q = cov.shape[-1]
        eye = torch.eye(q, device=cov.device, dtype=cov.dtype)

        if self.uncertainty_measure == "logdet":
            mat = cov + self.jitter * eye
            sign, logdet = torch.linalg.slogdet(mat)
            if not torch.all(sign > 0):
                # Fall back to numerically safer logdet1p-like score.
                tau2 = max(self.tau**2, self.jitter)
                sign, logdet = torch.linalg.slogdet(eye + cov / tau2)
            return logdet.clamp_min(-50.0)

        if self.uncertainty_measure == "trace":
            return torch.diagonal(cov, dim1=-2, dim2=-1).sum(dim=-1).clamp_min(0.0).sqrt()

        raise ValueError(f"Unknown uncertainty_measure: {self.uncertainty_measure}.")

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        X = _ensure_q_batch(X)
        Xt = _apply_input_transform_for_reference(self.model, X)
        n_w = _infer_n_w_from_objective_or_owner(self)

        scores: list[Tensor] = []
        for o, (m, cp) in enumerate(zip(self.submodels, self.cutpoints_list)):
            posterior = m.posterior(X)
            mean_f, var_f = _posterior_mean_var(posterior)
            cov_f = _posterior_covariance(posterior)

            mean_f = _align_pointwise_score_to_X(mean_f, Xt, name="joint latent mean")
            # cov_f may already be expanded to q_like x q_like.
            mean_f, cov_f = _reduce_input_perturbation_mean_cov(
                mean_f,
                cov_f,
                X,
                n_w,
                mode=self.perturbation_joint_reduction,
                jitter=self.jitter,
            )

            uncertainty = self._uncertainty_score(cov_f)
            cp = cp.detach().to(device=mean_f.device, dtype=mean_f.dtype)
            dist_b = (mean_f.unsqueeze(-1) - cp.view(*([1] * mean_f.ndim), -1)).abs()
            boundary_distance_score = -dist_b.mean(dim=-2)  # (*batch, n_boundaries)

            boundary_score = self.beta_vec[o].to(mean_f) * uncertainty.unsqueeze(-1) + boundary_distance_score
            score_o = _aggregate_boundary_scores(
                boundary_score,
                target_boundary_idx=self.target_boundary_idx_list[o],
                boundary_weights=self.boundary_weights_list[o],
                boundary_reduction=self.boundary_reduction,
            )
            scores.append(score_o)

        score_per_output = torch.stack(scores, dim=-1)  # (*batch, m)
        score = self._aggregate_output_scalars(score_per_output)

        # Joint score is scalar over q, so use aggregated penalty.
        score = score - self._aggregated_repulsion_penalty(Xt)
        score = _apply_score_objective(self, score, X, name="qMultiOutputOrdinalJointLatentStraddle")
        return score


__all__ = [
    "MultiOutputOrdinalLevelSetScoreObjective",
    "_MultiOutputOrdinalLevelSetScoreObjective",
    "qMultiOutputOrdinalLatentStraddleAcquisition",
    "qMultiOutputOrdinalJointLatentStraddleAcquisition",
    "qMultiOutputOrdinalICUAcquisition",
    "qMultiOutputOrdinalBoundaryVarianceAcquisition",
    "qMultiOutputOrdinalBoundaryEntropyAcquisition",
    "qMultiOutputOrdinalClassEntropyAcquisition",
]
