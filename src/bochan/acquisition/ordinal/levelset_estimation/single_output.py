from __future__ import annotations

import math
from typing import Callable, Literal, Optional, Sequence

import torch
from torch import Tensor

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import average_over_ensemble_models, t_batch_mode_transform


RiskType = Optional[Literal["var", "cvar"]]
PerturbationJointReduction = Literal["block_mean", "diagonal_mean"]
ReductionType = Literal["sum", "mean"]
BoundaryReduction = Literal["sum", "mean", "max", "min"]


class _OrdinalLevelSetScoreObjective(torch.nn.Module):
    """ordinal level-set acquisition の score に作用する objective。"""

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
        raw = _try_call_zero_arg(getattr(ordinal_likelihood, "raw_cutpoints"))
        raw = torch.as_tensor(raw).detach().clone()
        if hasattr(ordinal_likelihood, "transform_cutpoints"):
            cutpoints = ordinal_likelihood.transform_cutpoints(raw)
            return torch.as_tensor(cutpoints).detach().clone().reshape(-1)
        return raw.detach().clone().reshape(-1)

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


def _sigmoid(x: Tensor) -> Tensor:
    return torch.sigmoid(x)


def ordinal_class_probs_from_f(f: Tensor, ordinal_likelihood) -> Tensor:
    for name in (
        "class_probs_from_f",
        "probs_from_f",
        "predict_proba_from_f",
        "class_probabilities_from_f",
        "marginal_probs_from_f",
    ):
        if hasattr(ordinal_likelihood, name):
            probs = getattr(ordinal_likelihood, name)(f)
            if hasattr(probs, "probs"):
                probs = probs.probs
            probs = torch.as_tensor(probs, device=f.device, dtype=f.dtype)
            probs = probs.clamp_min(1e-12)
            return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    cutpoints = _get_cutpoints_from_likelihood(ordinal_likelihood).detach().to(device=f.device, dtype=f.dtype)
    z = cutpoints.view(*([1] * f.ndim), -1) - f.unsqueeze(-1)
    cdf = _sigmoid(z)
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
    rev_cumsum = torch.flip(torch.cumsum(torch.flip(class_probs, dims=[-1]), dim=-1), dims=[-1])
    return rev_cumsum[..., 1:]


def ordinal_boundary_uncertainty(ge_probs: Tensor) -> Tensor:
    return 4.0 * ge_probs * (1.0 - ge_probs)


def _nearest_cutpoint_distance(values: Tensor, cutpoints: Tensor) -> Tensor:
    cp = cutpoints.detach().to(device=values.device, dtype=values.dtype)
    dist = (values.unsqueeze(-1) - cp.view(*([1] * values.ndim), -1)).abs()
    return dist.min(dim=-1).values


def _boundary_kernel_weight(values: Tensor, cutpoints: Tensor, tau: float, reduce: Literal["sum", "max"] = "sum") -> Tensor:
    cp = cutpoints.detach().to(device=values.device, dtype=values.dtype)
    tau_t = torch.as_tensor(tau, device=values.device, dtype=values.dtype).clamp_min(1e-8)
    z2 = ((values.unsqueeze(-1) - cp.view(*([1] * values.ndim), -1)) / tau_t) ** 2
    w = torch.exp(-0.5 * z2)
    if reduce == "max":
        return w.max(dim=-1).values
    return w.sum(dim=-1)


def _prepare_boundary_weights(boundary_weights: Optional[Tensor | Sequence[float]], n_boundaries: int, *, device, dtype) -> Tensor:
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
    """Return distance from latent values to each ordinal cutpoint.

    Args:
        values: latent mean values. shape = batch_shape x q_like.
        cutpoints: ordinal cutpoints. shape = n_boundaries.

    Returns:
        Tensor: shape = batch_shape x q_like x n_boundaries.
    """
    cp = cutpoints.detach().to(device=values.device, dtype=values.dtype).reshape(-1)
    return (values.unsqueeze(-1) - cp.view(*([1] * values.ndim), -1)).abs()


def _boundary_kernel_weights_by_boundary(values: Tensor, cutpoints: Tensor, tau: float) -> Tensor:
    """Return Gaussian boundary weights for each ordinal cutpoint.

    Args:
        values: latent mean values. shape = batch_shape x q_like.
        cutpoints: ordinal cutpoints. shape = n_boundaries.
        tau: boundary width.

    Returns:
        Tensor: shape = batch_shape x q_like x n_boundaries.
    """
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
    """Aggregate boundary-wise scores into a pointwise score.

    ``target_boundary_idx=k`` corresponds to the boundary between class ``k`` and
    class ``k + 1``. For example, in classes 0/1/2, idx=0 is the 0/1 boundary
    and idx=1 is the 1/2 boundary.

    Args:
        boundary_scores: shape = batch_shape x q_like x n_boundaries.
        target_boundary_idx: specific boundary to target. If specified,
            ``boundary_weights`` and ``boundary_reduction`` are ignored.
        boundary_weights: optional weights for each boundary.
        boundary_reduction: aggregation over boundaries when target is not specified.

    Returns:
        Tensor: shape = batch_shape x q_like.
    """
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
    """q-batch 内の近接 candidate を各点ごとに penalty する。

    InputPerturbation により ``q * n_w`` に展開されている場合は、同じ
    元 candidate に由来する摂動 replica 同士を penalty しない。

    Args:
        X: candidate points. shape は ``batch_shape x q x d`` または
            ``batch_shape x (q * n_w) x d``。
        lengthscale: RBF 型距離 penalty の lengthscale。
        n_w: 1 candidate あたりの摂動 replica 数。None の場合は通常の q として扱う。

    Returns:
        Tensor: ``batch_shape x q`` または ``batch_shape x (q * n_w)`` の penalty。
    """
    q_expanded = X.shape[-2]
    if q_expanded <= 1:
        return torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)

    ls2 = float(lengthscale) ** 2 + 1e-12
    diff = X.unsqueeze(-2) - X.unsqueeze(-3)
    d2 = diff.pow(2).sum(dim=-1)

    eye = torch.eye(q_expanded, device=X.device, dtype=torch.bool)
    mask = eye

    if n_w is not None and n_w > 1 and q_expanded % n_w == 0:
        q = q_expanded // n_w
        group = torch.arange(q_expanded, device=X.device) // int(n_w)
        same_group = group.unsqueeze(0) == group.unsqueeze(1)
        mask = mask | same_group

    d2 = d2.masked_fill(mask, float("inf"))
    return torch.exp(-0.5 * d2 / ls2).sum(dim=-1)


def _pointwise_ref_penalty(
    X: Tensor,
    X_ref: Optional[Tensor],
    lengthscale: float,
) -> Tensor:
    """各 candidate 点ごとの reference penalty を計算する。

    Args:
        X: candidate points. shape は ``batch_shape x q x d``。
        X_ref: pending / observed points. shape は ``m x d`` または
            ``batch_shape x m x d``。
        lengthscale: RBF 型距離 penalty の lengthscale。

    Returns:
        Tensor: ``batch_shape x q`` の penalty。
    """
    X_ref = _flatten_ref_points(X_ref, device=X.device, dtype=X.dtype)
    if X_ref is None:
        return torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)

    ls2 = float(lengthscale) ** 2 + 1e-12
    ref = X_ref.view(*([1] * (X.ndim - 2)), X_ref.shape[0], X_ref.shape[1])
    diff = X.unsqueeze(-2) - ref
    d2 = diff.pow(2).sum(dim=-1)
    return torch.exp(-0.5 * d2 / ls2).sum(dim=-1)




def _ensure_q_batch_for_pending(X: Tensor) -> Tensor:
    """pending penalty 用に X を `(..., q, d)` へ揃える。"""
    return X.unsqueeze(-2) if torch.is_tensor(X) and X.ndim == 2 else X


def _coerce_pending_to_tensor(
    X_pending,
    *,
    ref: Optional[Tensor] = None,
) -> Optional[Tensor]:
    """X_pending を Tensor または None に正規化する。"""
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
    """candidate / pending を同じ距離計算空間へ写す。"""
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
    """raw-space の X_pending を candidate と同じ transformed space へ写す。"""
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


class _OrdinalBoundaryBase(AcquisitionFunction):
    def __init__(
        self,
        model: Model,
        sampler: Optional[MCSampler] = None,
        objective: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> None:
        super().__init__(model=model)
        self.ordinal_likelihood = _get_ordinal_likelihood(model)
        # cutpoints は ordinal likelihood の parameter 変換から得られるため、
        # grad_fn を持つ Tensor になり得る。acquisition 最適化では X のみを
        # 最適化対象にするので、cutpoints は定数として detach して保持する。
        # cutpoints は ordinal likelihood の parameter 変換から得られるため、
        # grad_fn を持つ Tensor になり得る。acquisition 最適化では X のみを
        # 最適化対象にするので、cutpoints は定数として detach して保持する。
        cutpoints = _get_cutpoints_from_likelihood(self.ordinal_likelihood)
        self.register_buffer("cutpoints", torch.as_tensor(cutpoints).detach().clone())
        self.sampler = sampler or SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        self.objective = objective

    def _apply_objective_to_score(self, score: Tensor, X: Tensor, name: str) -> Tensor:
        return _apply_ordinal_levelset_objective_to_score(self, score, X=X, name=name)

    def _latent_mean_var(self, X: Tensor) -> tuple[Tensor, Tensor]:
        posterior = self.model.posterior(X)
        return _posterior_mean_var(posterior, X)

    def _latent_samples(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        return self.sampler(posterior).squeeze(-1)

    def _predictive_class_probs(self, X: Tensor) -> Tensor:
        f_samples = self._latent_samples(X)
        probs = ordinal_class_probs_from_f(f_samples, self.ordinal_likelihood)
        return _reduce_probs_to_match_X(probs, X)


class _qOrdinalBoundaryBase(MCAcquisitionFunction):
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
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
    ) -> None:
        super().__init__(model=model)
        self.ordinal_likelihood = _get_ordinal_likelihood(model)
        cutpoints = _get_cutpoints_from_likelihood(self.ordinal_likelihood)
        self.register_buffer("cutpoints", torch.as_tensor(cutpoints).detach().clone())
        self.sampler = sampler or SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        self.objective = objective

        if reduction not in ("sum", "mean"):
            raise ValueError(f"Unknown reduction: {reduction}")
        self.reduction = reduction

        self.same_batch_penalty_weight = float(same_batch_penalty_weight)
        self.pending_penalty_weight = float(pending_penalty_weight)
        self.observed_penalty_weight = float(observed_penalty_weight)
        self.penalty_lengthscale = float(penalty_lengthscale)
        self.X_pending = _coerce_pending_to_tensor(X_pending)
        self.X_observed = _coerce_pending_to_tensor(X_observed)

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        """BoTorch の sequential optimization などから X_pending を更新する。"""
        self.X_pending = _coerce_pending_to_tensor(X_pending)

    def set_X_observed(self, X_observed: Optional[Tensor] = None) -> None:
        """観測済み点を更新する。"""
        self.X_observed = _coerce_pending_to_tensor(X_observed)

    def _apply_objective_to_score(self, score: Tensor, X: Tensor, name: str) -> Tensor:
        return _apply_ordinal_levelset_objective_to_score(self, score, X=X, name=name)

    def _reduce_q(self, score: Tensor) -> Tensor:
        """pointwise score を q-batch 方向に集約する。"""
        if score.ndim == 0:
            return score
        if score.shape[-1] == 1:
            return score.squeeze(-1)
        if self.reduction == "mean":
            return score.mean(dim=-1)
        if self.reduction == "sum":
            return score.sum(dim=-1)
        raise ValueError(f"Unknown reduction: {self.reduction}")

    def _pointwise_repulsion_penalty(self, X: Tensor) -> Tensor:
        """pointwise acquisition 用の重複回避 penalty を返す。

        candidate / pending / observed は input_transform 後の同じ距離計算空間へ
        写してから比較する。InputPerturbation が含まれる場合は、score と同じ
        ``q * n_w`` の形に揃え、objective 側で ``q`` に戻す。
        """
        Xt = _apply_input_transform_for_pending(self.model, X)
        penalty = torch.zeros(Xt.shape[:-1], device=Xt.device, dtype=Xt.dtype)
        n_w = _infer_n_w_from_objective_or_owner(self)

        if self.same_batch_penalty_weight > 0.0:
            penalty = penalty + self.same_batch_penalty_weight * _pointwise_same_batch_penalty(
                Xt,
                self.penalty_lengthscale,
                n_w=n_w,
            )

        if self.pending_penalty_weight > 0.0:
            Xp_t = _transform_pending_like_candidate(self.model, self.X_pending, ref=Xt)
            penalty = penalty + self.pending_penalty_weight * _pointwise_ref_penalty(
                Xt,
                Xp_t,
                self.penalty_lengthscale,
            )

        if self.observed_penalty_weight > 0.0:
            Xobs_t = _transform_pending_like_candidate(self.model, self.X_observed, ref=Xt)
            penalty = penalty + self.observed_penalty_weight * _pointwise_ref_penalty(
                Xt,
                Xobs_t,
                self.penalty_lengthscale,
            )

        return penalty

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


class _OrdinalLatentStraddleAcquisition(_OrdinalBoundaryBase):
    def __init__(self, model: Model, beta: float = 1.0, sampler: Optional[MCSampler] = None, objective=None) -> None:
        super().__init__(model=model, sampler=sampler, objective=objective)
        self.beta = float(beta)

    @t_batch_mode_transform(expected_q=1)
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        mean_f, var_f = self._latent_mean_var(X)
        score = self.beta * var_f.sqrt() - _nearest_cutpoint_distance(mean_f, self.cutpoints)
        score = self._apply_objective_to_score(score, X=X, name="OrdinalLatentStraddle")
        expected = X.shape[:-2]
        if score.shape == expected:
            return score
        if score.shape[-1] == 1:
            return score.squeeze(-1)
        return score.mean(dim=-1)


class qOrdinalLatentStraddleAcquisition(_qOrdinalBoundaryBase):
    """ordinal 用 straddle acquisition。境界に近く、かつ不確実な点を選びます。

    target_boundary_idx は class k / class k+1 境界を直接指定するための引数です。
    例: 3 クラス 0/1/2 では 0 が 0/1 境界、1 が 1/2 境界です。
    """

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
    """ordinal 用 joint straddle acquisition。q-batch 全体の境界不確実性を評価します。

    target_boundary_idx は class k / class k+1 境界を直接指定します。
    デフォルトの boundary_reduction="max" は従来の nearest cutpoint に近い挙動です。
    """

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
        X_pending: Optional[Tensor] = None,
        X_observed: Optional[Tensor] = None,
        sampler: Optional[MCSampler] = None,
        objective=None,
        input_perturbation_n_w: Optional[int] = None,
        perturbation_joint_reduction: PerturbationJointReduction = "block_mean",
        jitter: float = 1e-6,
    ) -> None:
        super().__init__(model=model, sampler=sampler, objective=objective)
        self.beta = float(beta)
        self.tau = float(tau)
        self.uncertainty_measure = uncertainty_measure
        self.target_boundary_idx = target_boundary_idx
        self.boundary_weights = boundary_weights
        self.boundary_reduction = boundary_reduction
        self.same_batch_penalty_weight = float(same_batch_penalty_weight)
        self.pending_penalty_weight = float(pending_penalty_weight)
        self.observed_penalty_weight = float(observed_penalty_weight)
        self.penalty_lengthscale = float(penalty_lengthscale)
        self.X_pending = _coerce_pending_to_tensor(X_pending)
        self.X_observed = _coerce_pending_to_tensor(X_observed)
        self.input_perturbation_n_w = None if input_perturbation_n_w is None else int(input_perturbation_n_w)
        self.perturbation_joint_reduction = perturbation_joint_reduction
        self.jitter = float(jitter)

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        self.X_pending = _coerce_pending_to_tensor(X_pending)

    def set_X_observed(self, X_observed: Optional[Tensor] = None) -> None:
        self.X_observed = _coerce_pending_to_tensor(X_observed)

    def _uncertainty_score(self, cov: Tensor) -> Tensor:
        if self.uncertainty_measure == "trace":
            return cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        q = cov.shape[-1]
        eye = torch.eye(q, device=cov.device, dtype=cov.dtype)
        mat = eye + cov / (self.tau ** 2 + 1e-12)
        sign, logabsdet = torch.linalg.slogdet(mat)
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
        boundary_score_b = -dist_b
        boundary_score = _aggregate_boundary_scores(
            boundary_score_b,
            target_boundary_idx=self.target_boundary_idx,
            boundary_weights=self.boundary_weights,
            boundary_reduction=self.boundary_reduction,
        ).mean(dim=-1)

        Xt = _apply_input_transform_for_pending(self.model, X)
        Xp_t = _transform_pending_like_candidate(self.model, self.X_pending, ref=Xt)
        Xobs_t = _transform_pending_like_candidate(self.model, self.X_observed, ref=Xt)

        penalty = torch.zeros_like(uncertainty)
        if self.same_batch_penalty_weight > 0:
            penalty = penalty + self.same_batch_penalty_weight * _same_batch_penalty(Xt, self.penalty_lengthscale)
        if self.pending_penalty_weight > 0:
            penalty = penalty + self.pending_penalty_weight * _ref_penalty(Xt, Xp_t, self.penalty_lengthscale)
        if self.observed_penalty_weight > 0:
            penalty = penalty + self.observed_penalty_weight * _ref_penalty(Xt, Xobs_t, self.penalty_lengthscale)

        score = self.beta * uncertainty + boundary_score - penalty
        return self._apply_objective_to_score(score, X=X, name="qOrdinalJointLatentStraddle")



class _OrdinalICUAcquisition(_OrdinalBoundaryBase):
    def __init__(self, model: Model, boundary_weights: Optional[Tensor | Sequence[float]] = None, sampler: Optional[MCSampler] = None, objective=None) -> None:
        super().__init__(model=model, sampler=sampler, objective=objective)
        self.boundary_weights = boundary_weights

    @t_batch_mode_transform(expected_q=1)
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        probs = self._predictive_class_probs(X)
        ge_probs = ordinal_cumulative_ge_probs_from_class_probs(probs)
        u = ordinal_boundary_uncertainty(ge_probs)
        w = _prepare_boundary_weights(self.boundary_weights, n_boundaries=u.shape[-1], device=u.device, dtype=u.dtype)
        score = (u * w).sum(dim=-1)
        score = self._apply_objective_to_score(score, X=X, name="OrdinalICU")
        expected = X.shape[:-2]
        if score.shape == expected:
            return score
        if score.shape[-1] == 1:
            return score.squeeze(-1)
        return score.mean(dim=-1)


class qOrdinalICUAcquisition(_qOrdinalBoundaryBase):
    """ordinal 用 ICU acquisition。contour / boundary 周辺の不確実性を評価します。

    target_boundary_idx は class k / class k+1 境界を直接指定します。
    例: 3 クラス 0/1/2 では 0 が 0/1 境界、1 が 1/2 境界です。
    """

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



class _OrdinalBoundaryVarianceAcquisition(_OrdinalBoundaryBase):
    def __init__(self, model: Model, tau: float = 1.0, reduce: Literal["sum", "max"] = "sum", sampler: Optional[MCSampler] = None, objective=None) -> None:
        super().__init__(model=model, sampler=sampler, objective=objective)
        self.tau = float(tau)
        self.reduce = reduce

    @t_batch_mode_transform(expected_q=1)
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        mean_f, var_f = self._latent_mean_var(X)
        w = _boundary_kernel_weight(mean_f, self.cutpoints, tau=self.tau, reduce=self.reduce)
        score = var_f * w
        score = self._apply_objective_to_score(score, X=X, name="OrdinalBoundaryVariance")
        expected = X.shape[:-2]
        if score.shape == expected:
            return score
        if score.shape[-1] == 1:
            return score.squeeze(-1)
        return score.mean(dim=-1)


class qOrdinalBoundaryVarianceAcquisition(_qOrdinalBoundaryBase):
    """ordinal 用 boundary variance acquisition。境界近傍の posterior variance を重視します。

    target_boundary_idx は class k / class k+1 境界を直接指定します。
    boundary_reduction は target_boundary_idx 未指定時の boundary score 集約方法です。
    """

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



class _OrdinalClassEntropyAcquisition(_OrdinalBoundaryBase):
    @t_batch_mode_transform(expected_q=1)
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        probs = self._predictive_class_probs(X)
        score = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
        score = self._apply_objective_to_score(score, X=X, name="OrdinalClassEntropy")
        expected = X.shape[:-2]
        if score.shape == expected:
            return score
        if score.shape[-1] == 1:
            return score.squeeze(-1)
        return score.mean(dim=-1)


class qOrdinalClassEntropyAcquisition(_qOrdinalBoundaryBase):
    """ordinal 用 class entropy acquisition。class probability の entropy を評価します。

    Args:
        model: BoTorch 互換の surrogate model。`posterior(X)` を実装していることを想定します。
        sampler: posterior samples を生成する BoTorch sampler。省略時は SobolQMCNormalSampler を使います。
        objective: 計算済み score に作用する objective。InputPerturbation の q*n_w -> q 集約にも使えます。
        reduction: q-batch 方向の集約方法。`sum` または `mean`。
        same_batch_penalty_weight: 同一 q-batch 内の候補点同士が近すぎる場合の penalty の強さ。
        pending_penalty_weight: X_pending 近傍を避ける penalty の強さ。
        observed_penalty_weight: X_observed 近傍を避ける penalty の強さ。
        penalty_lengthscale: RBF 型距離 penalty の lengthscale。
        X_pending: 評価中で、まだ結果が返っていない候補点。
        X_observed: 既に観測済みの点。

    Forward Args:
        X: 候補点。shape は通常 `batch_shape x q x d` です。

    Returns:
        Tensor: `batch_shape` の acquisition value。`optimize_acqf` はこの値を最大化します。
    """

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
