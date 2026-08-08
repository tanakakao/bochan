from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Literal, Optional

import torch
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bochan.acquisition.ordinal.active_learning.multi_output import (
    MultiOutputMode,
    ReductionType,
    _apply_input_transform_for_reference,
    _ensure_q_batch,
    _qMultiOutputOrdinalActiveLearningBase,
    ordinal_entropy_from_probs,
)

RiskType = Optional[Literal["var", "cvar"]]
BoundaryReduction = Literal["sum", "mean", "max", "min"]
PerturbationJointReduction = Literal["block_mean", "diagonal_mean"]


# =========================================================
# Score objective
# =========================================================
def _validate_n_w_risk(
    *,
    n_w: int | None,
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
    """multi-output ordinal LSE の pointwise score objective。

    InputPerturbation で ``q * n_w`` に展開された pointwise score を ``q`` に
    戻します。joint acquisition のように既に batch-level scalar へ集約済みの
    score はデフォルトでそのまま返します。
    """

    def __init__(
        self,
        n_w: int | None = None,
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

    def _is_aggregated_score(self, score: Tensor, X: Tensor | None) -> bool:
        if X is None or score.ndim == 0:
            return False
        Xq = self._ensure_q_batch(X)
        return tuple(score.shape) == tuple(Xq.shape[:-2])

    def forward(self, score: Tensor, X: Tensor | None = None) -> Tensor:
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

        q_expanded = int(score.shape[-1])
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


# Backward-supported internal name.
_MultiOutputOrdinalLevelSetScoreObjective = MultiOutputOrdinalLevelSetScoreObjective


# =========================================================
# Ordinal / posterior helpers
# =========================================================
def _try_call_zero_arg(obj):
    return obj() if callable(obj) else obj


def _get_cutpoints_from_likelihood(ordinal_likelihood) -> Tensor:
    if hasattr(ordinal_likelihood, "get_cutpoints"):
        cutpoints = _try_call_zero_arg(ordinal_likelihood.get_cutpoints)
        return torch.as_tensor(cutpoints).detach().clone().reshape(-1)

    for name in ("transformed_cutpoints", "cutpoints", "thresholds", "cuts", "cutoffs"):
        if hasattr(ordinal_likelihood, name):
            cutpoints = _try_call_zero_arg(getattr(ordinal_likelihood, name))
            return torch.as_tensor(cutpoints).detach().clone().reshape(-1)

    if hasattr(ordinal_likelihood, "raw_cutpoints"):
        raw = torch.as_tensor(_try_call_zero_arg(ordinal_likelihood.raw_cutpoints)).detach().clone()
        if hasattr(ordinal_likelihood, "transform_cutpoints"):
            cutpoints = ordinal_likelihood.transform_cutpoints(raw)
            return torch.as_tensor(cutpoints).detach().clone().reshape(-1)
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


def _reduce_extra_batch_dims(
    tensor: Tensor,
    X_like: Tensor,
    n_trailing_keep: int,
) -> Tensor:
    out = tensor
    X_like = _ensure_q_batch(X_like)
    target_ndim = len(X_like.shape[:-2]) + n_trailing_keep
    while out.ndim > target_ndim:
        out = out.mean(dim=0)
    return out


def _align_pointwise_score_to_X(score: Tensor, X_like: Tensor, *, name: str) -> Tensor:
    X_like = _ensure_q_batch(X_like)
    expected = X_like.shape[:-1]
    out = score

    if out.shape == expected:
        return out
    if out.ndim == len(expected) + 1 and out.shape[-1] == 1:
        out = out.squeeze(-1)
        if out.shape == expected:
            return out

    out = _reduce_extra_batch_dims(out, X_like, n_trailing_keep=1)
    if out.shape == expected:
        return out
    if out.numel() == math.prod(expected):
        return out.reshape(*expected)
    raise RuntimeError(
        f"{name}: failed to align score to X_like. "
        f"score.shape={tuple(score.shape)}, X_like.shape={tuple(X_like.shape)}."
    )


def _align_probs_to_X(probs: Tensor, X_like: Tensor, *, eps: float) -> Tensor:
    out = _reduce_extra_batch_dims(probs, X_like, n_trailing_keep=2)
    expected_prefix = _ensure_q_batch(X_like).shape[:-1]
    if out.shape[:-1] != expected_prefix:
        c = int(out.shape[-1])
        expected_numel = math.prod(expected_prefix) * c
        if out.numel() == expected_numel:
            out = out.reshape(*expected_prefix, c)
        else:
            raise RuntimeError(
                "Could not align ordinal class probabilities to candidates. "
                f"probs.shape={tuple(probs.shape)}, X_like.shape={tuple(X_like.shape)}."
            )
    out = out.clamp_min(eps)
    return out / out.sum(dim=-1, keepdim=True).clamp_min(eps)


def _to_optional_list(value, n: int, *, name: str) -> list:
    if value is None:
        return [None] * n
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"{name} length must match number of outputs. Expected {n}, got {len(value)}.")
        return list(value)
    return [value] * n


def _prepare_boundary_weights(
    boundary_weights: Tensor | Sequence[float] | None,
    n_boundaries: int,
    *,
    device,
    dtype,
) -> Tensor | None:
    if boundary_weights is None:
        return None
    w = torch.as_tensor(boundary_weights, device=device, dtype=dtype).reshape(-1)
    if w.numel() != n_boundaries:
        raise ValueError(f"boundary_weights must have length {n_boundaries}, got {w.numel()}.")
    return w


def _aggregate_boundary_scores(
    boundary_scores: Tensor,
    *,
    target_boundary_idx: int | None = None,
    boundary_weights: Tensor | Sequence[float] | None = None,
    boundary_reduction: BoundaryReduction = "sum",
) -> Tensor:
    n_boundaries = int(boundary_scores.shape[-1])
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


def _infer_n_w_from_objective_or_owner(owner) -> int | None:
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
    n_w: int | None,
    *,
    mode: PerturbationJointReduction = "block_mean",
    jitter: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    if n_w is None or n_w <= 1:
        return mean, cov

    X_in = _ensure_q_batch(X)
    batch_shape = X_in.shape[:-2]
    q = int(X_in.shape[-2])
    if mean.shape == batch_shape + torch.Size([q]) and cov.shape == batch_shape + torch.Size([q, q]):
        eye = torch.eye(q, dtype=cov.dtype, device=cov.device)
        return mean, cov + jitter * eye

    q_expanded = q * int(n_w)
    if mean.shape != batch_shape + torch.Size([q_expanded]) or cov.shape != batch_shape + torch.Size([q_expanded, q_expanded]):
        return mean, cov

    mean_q = mean.reshape(*batch_shape, q, int(n_w)).mean(dim=-1)
    cov_blocks = cov.reshape(*batch_shape, q, int(n_w), q, int(n_w))
    if mode == "block_mean":
        cov_q = cov_blocks.mean(dim=(-3, -1))
    elif mode == "diagonal_mean":
        diag = torch.diagonal(cov, dim1=-2, dim2=-1)
        var_q = diag.reshape(*batch_shape, q, int(n_w)).mean(dim=-1).clamp_min(0.0)
        cov_q = torch.diag_embed(var_q)
    else:
        raise ValueError(f"Unknown perturbation_joint_reduction: {mode}")

    cov_q = 0.5 * (cov_q + cov_q.transpose(-1, -2))
    eye = torch.eye(q, dtype=cov_q.dtype, device=cov_q.device)
    return mean_q, cov_q + jitter * eye


# =========================================================
# Shared LSE base
# =========================================================
class _qMultiOutputOrdinalBoundaryBase(_qMultiOutputOrdinalActiveLearningBase):
    """Ordinal multi-output LSE base sharing AL duplicate controls permanently.

    Duplicate exclusion, observed-X resolution, mixed continuous/categorical
    distance handling, X_pending updates, and objective/q reduction are inherited
    from ``_qMultiOutputOrdinalActiveLearningBase``.  This class adds only the
    ordinal level-set specific posterior/boundary helpers.
    """

    def __init__(
        self,
        model: Model,
        output_weights: Tensor | Sequence[float] | None = None,
        reduction: ReductionType = "mean",
        output_mode: MultiOutputMode = "mean",
        sampler: MCSampler | None = None,
        eps: float = 1e-6,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        hard_duplicate_tol: float = 1e-8,
        exclude_same_batch_duplicates: bool = True,
        exclude_pending_duplicates: bool = True,
        exclude_observed_duplicates: bool = True,
        X_pending: Tensor | None = None,
        X_observed: Tensor | None = None,
        objective: Callable[[Tensor, Tensor | None], Tensor] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            reduction=reduction,
            output_mode=output_mode,
            output_weights=output_weights,
            sampler=sampler,
            eps=eps,
            pending_penalty_weight=pending_penalty_weight,
            pending_penalty_beta=pending_penalty_beta,
            observed_penalty_weight=observed_penalty_weight,
            observed_penalty_beta=observed_penalty_beta,
            same_batch_penalty_weight=same_batch_penalty_weight,
            same_batch_penalty_beta=same_batch_penalty_beta,
            hard_duplicate_tol=hard_duplicate_tol,
            exclude_same_batch_duplicates=exclude_same_batch_duplicates,
            exclude_pending_duplicates=exclude_pending_duplicates,
            exclude_observed_duplicates=exclude_observed_duplicates,
            X_pending=X_pending,
            X_observed=X_observed,
            objective=objective,
        )
        self.n_outputs = len(self.submodels)
        self.cutpoints_list = [
            _get_cutpoints_from_likelihood(lik).detach().clone()
            for lik in self.ordinal_likelihoods
        ]

    def _latent_mean_var_list(self, X: Tensor, *, X_like: Tensor) -> list[tuple[Tensor, Tensor]]:
        outs: list[tuple[Tensor, Tensor]] = []
        for submodel in self.submodels:
            mean, var = _posterior_mean_var(submodel.posterior(X))
            mean = _align_pointwise_score_to_X(mean, X_like, name="ordinal LSE latent mean")
            var = _align_pointwise_score_to_X(var, X_like, name="ordinal LSE latent variance").clamp_min(self.eps)
            outs.append((mean, var))
        return outs

    def _predictive_class_probs_list(self, X: Tensor, *, X_like: Tensor) -> list[Tensor]:
        outs: list[Tensor] = []
        for submodel, likelihood in zip(self.submodels, self.ordinal_likelihoods):
            posterior = submodel.posterior(X)
            if hasattr(likelihood, "marginal_class_probs"):
                probs = likelihood.marginal_class_probs(posterior.distribution)
            else:
                samples = self.sampler(posterior)
                if samples.ndim >= 1 and samples.shape[-1] == 1:
                    samples = samples.squeeze(-1)
                probs = ordinal_class_probs_from_f(samples, likelihood).mean(dim=0)
            outs.append(_align_probs_to_X(probs, X_like, eps=self.eps))
        return outs

    def _aggregate_output_scalars(self, score_per_output: Tensor) -> Tensor:
        return self._aggregate_outputs(score_per_output.unsqueeze(-2)).squeeze(-1)

    def _finalize_pointwise_scores(self, score_per_output: Tensor, X: Tensor, *, name: str) -> Tensor:
        score = self._aggregate_outputs(score_per_output)
        return self._finalize_pointwise_score(score, X, name=name)

    def _aggregated_repulsion_penalty(self, Xt: Tensor) -> Tensor:
        """Aggregate shared pointwise penalties for a joint q-batch score."""
        pending = self._pending_penalty_per_point(Xt).sum(dim=-1)
        observed = self._observed_penalty_per_point(Xt).sum(dim=-1)
        same_batch = 0.5 * self._same_batch_penalty_per_point(Xt).sum(dim=-1)
        return pending + observed + same_batch


# Backward-supported non-q base name.
_MultiOutputOrdinalBoundaryBase = _qMultiOutputOrdinalBoundaryBase


# =========================================================
# Pointwise multi-output acquisitions
# =========================================================
class qMultiOutputOrdinalLatentStraddleAcquisition(_qMultiOutputOrdinalBoundaryBase):
    """multi-output ordinal latent straddle acquisition."""

    def __init__(
        self,
        model: Model,
        beta: float | Sequence[float] | Tensor = 1.0,
        output_weights: Tensor | Sequence[float] | None = None,
        reduction: ReductionType = "mean",
        output_mode: MultiOutputMode = "mean",
        boundary_weights_list: Sequence[Tensor | Sequence[float] | None] | None = None,
        target_boundary_idx_list: Sequence[int | None] | int | None = None,
        boundary_reduction: BoundaryReduction = "sum",
        sampler: MCSampler | None = None,
        eps: float = 1e-6,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        hard_duplicate_tol: float = 1e-8,
        exclude_same_batch_duplicates: bool = True,
        exclude_pending_duplicates: bool = True,
        exclude_observed_duplicates: bool = True,
        X_pending: Tensor | None = None,
        X_observed: Tensor | None = None,
        objective: Callable[[Tensor, Tensor | None], Tensor] | None = None,
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
            hard_duplicate_tol=hard_duplicate_tol,
            exclude_same_batch_duplicates=exclude_same_batch_duplicates,
            exclude_pending_duplicates=exclude_pending_duplicates,
            exclude_observed_duplicates=exclude_observed_duplicates,
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
        for o, ((mean_f, var_f), cp) in enumerate(
            zip(self._latent_mean_var_list(X, X_like=Xt), self.cutpoints_list)
        ):
            std_f = var_f.sqrt()
            cp = cp.to(device=mean_f.device, dtype=mean_f.dtype)
            dist_b = (mean_f.unsqueeze(-1) - cp.view(*([1] * mean_f.ndim), -1)).abs()
            score_b = self.beta_vec[o].to(mean_f) * std_f.unsqueeze(-1) - dist_b
            scores.append(
                _aggregate_boundary_scores(
                    score_b,
                    target_boundary_idx=self.target_boundary_idx_list[o],
                    boundary_weights=self.boundary_weights_list[o],
                    boundary_reduction=self.boundary_reduction,
                )
            )
        return self._finalize_pointwise_scores(
            torch.stack(scores, dim=-1),
            X,
            name="qMultiOutputOrdinalLatentStraddle",
        )


class qMultiOutputOrdinalICUAcquisition(_qMultiOutputOrdinalBoundaryBase):
    """multi-output ordinal ICU acquisition."""

    def __init__(
        self,
        model: Model,
        boundary_weights_list: Sequence[Tensor | Sequence[float] | None] | None = None,
        target_boundary_idx_list: Sequence[int | None] | int | None = None,
        boundary_reduction: BoundaryReduction = "sum",
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
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
            u = ordinal_boundary_uncertainty(
                ordinal_cumulative_ge_probs_from_class_probs(probs)
            )
            scores.append(
                _aggregate_boundary_scores(
                    u,
                    target_boundary_idx=self.target_boundary_idx_list[o],
                    boundary_weights=self.boundary_weights_list[o],
                    boundary_reduction=self.boundary_reduction,
                )
            )
        return self._finalize_pointwise_scores(
            torch.stack(scores, dim=-1),
            X,
            name="qMultiOutputOrdinalICU",
        )


class qMultiOutputOrdinalBoundaryVarianceAcquisition(_qMultiOutputOrdinalBoundaryBase):
    """multi-output ordinal boundary variance acquisition."""

    def __init__(
        self,
        model: Model,
        tau: float = 1.0,
        boundary_weights_list: Sequence[Tensor | Sequence[float] | None] | None = None,
        target_boundary_idx_list: Sequence[int | None] | int | None = None,
        boundary_reduction: BoundaryReduction = "sum",
        reduce: Literal["sum", "max"] | None = None,
        **kwargs,
    ) -> None:
        if reduce is not None:
            boundary_reduction = reduce
        super().__init__(model=model, **kwargs)
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
        for o, ((mean_f, var_f), cp) in enumerate(
            zip(self._latent_mean_var_list(X, X_like=Xt), self.cutpoints_list)
        ):
            cp = cp.to(device=mean_f.device, dtype=mean_f.dtype)
            score_b = var_f.unsqueeze(-1) * _boundary_kernel_scores(
                mean_f,
                cp,
                tau=self.tau,
            )
            scores.append(
                _aggregate_boundary_scores(
                    score_b,
                    target_boundary_idx=self.target_boundary_idx_list[o],
                    boundary_weights=self.boundary_weights_list[o],
                    boundary_reduction=self.boundary_reduction,
                )
            )
        return self._finalize_pointwise_scores(
            torch.stack(scores, dim=-1),
            X,
            name="qMultiOutputOrdinalBoundaryVariance",
        )


class qMultiOutputOrdinalBoundaryEntropyAcquisition(_qMultiOutputOrdinalBoundaryBase):
    """各 ordinal boundary の binary entropy を使う multi-output LSE。"""

    def __init__(
        self,
        model: Model,
        boundary_weights_list: Sequence[Tensor | Sequence[float] | None] | None = None,
        target_boundary_idx_list: Sequence[int | None] | int | None = None,
        boundary_reduction: BoundaryReduction = "sum",
        **kwargs,
    ) -> None:
        super().__init__(model=model, **kwargs)
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
            entropy_b = -(
                ge_probs * ge_probs.log()
                + (1.0 - ge_probs) * (1.0 - ge_probs).log()
            )
            scores.append(
                _aggregate_boundary_scores(
                    entropy_b,
                    target_boundary_idx=self.target_boundary_idx_list[o],
                    boundary_weights=self.boundary_weights_list[o],
                    boundary_reduction=self.boundary_reduction,
                )
            )
        return self._finalize_pointwise_scores(
            torch.stack(scores, dim=-1),
            X,
            name="qMultiOutputOrdinalBoundaryEntropy",
        )


class qMultiOutputOrdinalClassEntropyAcquisition(_qMultiOutputOrdinalBoundaryBase):
    """whole-class entropy を使う multi-output ordinal LSE。"""

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        self._set_eval_mode()
        X = _ensure_q_batch(X)
        Xt = _apply_input_transform_for_reference(self.model, X)
        scores = [
            ordinal_entropy_from_probs(probs, eps=self.eps)
            for probs in self._predictive_class_probs_list(X, X_like=Xt)
        ]
        return self._finalize_pointwise_scores(
            torch.stack(scores, dim=-1),
            X,
            name="qMultiOutputOrdinalClassEntropy",
        )


# =========================================================
# Joint multi-output acquisition
# =========================================================
class qMultiOutputOrdinalJointLatentStraddleAcquisition(_qMultiOutputOrdinalBoundaryBase):
    """multi-output ordinal joint latent straddle acquisition。"""

    def __init__(
        self,
        model: Model,
        beta: float | Sequence[float] | Tensor = 1.0,
        tau: float = 1.0,
        uncertainty_measure: Literal["logdet", "trace"] = "logdet",
        output_weights: Tensor | Sequence[float] | None = None,
        output_mode: MultiOutputMode = "weighted_mean",
        same_batch_penalty_weight: float = 0.0,
        pending_penalty_weight: float = 0.0,
        observed_penalty_weight: float = 0.0,
        penalty_lengthscale: float | None = None,
        distance_beta: float | None = None,
        hard_duplicate_tol: float = 1e-8,
        exclude_same_batch_duplicates: bool = True,
        exclude_pending_duplicates: bool = True,
        exclude_observed_duplicates: bool = True,
        X_pending: Tensor | None = None,
        X_observed: Tensor | None = None,
        sampler: MCSampler | None = None,
        objective: Callable[[Tensor, Tensor | None], Tensor] | None = None,
        input_perturbation_n_w: int | None = None,
        perturbation_joint_reduction: PerturbationJointReduction = "block_mean",
        jitter: float = 1e-6,
        boundary_weights_list: Sequence[Tensor | Sequence[float] | None] | None = None,
        target_boundary_idx_list: Sequence[int | None] | int | None = None,
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
            hard_duplicate_tol=hard_duplicate_tol,
            exclude_same_batch_duplicates=exclude_same_batch_duplicates,
            exclude_pending_duplicates=exclude_pending_duplicates,
            exclude_observed_duplicates=exclude_observed_duplicates,
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
        q = int(cov.shape[-1])
        eye = torch.eye(q, device=cov.device, dtype=cov.dtype)
        if self.uncertainty_measure == "logdet":
            mat = 0.5 * (cov + cov.transpose(-1, -2)) + self.jitter * eye
            sign, logdet = torch.linalg.slogdet(mat)
            if not torch.all(sign > 0):
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

        for o, (submodel, cp) in enumerate(zip(self.submodels, self.cutpoints_list)):
            posterior = submodel.posterior(X)
            mean_f, _ = _posterior_mean_var(posterior)
            mean_f = _align_pointwise_score_to_X(
                mean_f,
                Xt,
                name="ordinal joint latent mean",
            )
            cov_f = _posterior_covariance(posterior)
            mean_f, cov_f = _reduce_input_perturbation_mean_cov(
                mean_f,
                cov_f,
                X,
                n_w,
                mode=self.perturbation_joint_reduction,
                jitter=self.jitter,
            )

            uncertainty = self._uncertainty_score(cov_f)
            cp = cp.to(device=mean_f.device, dtype=mean_f.dtype)
            dist_b = (mean_f.unsqueeze(-1) - cp.view(*([1] * mean_f.ndim), -1)).abs()
            boundary_distance_score = -dist_b.mean(dim=-2)
            boundary_score = (
                self.beta_vec[o].to(mean_f) * uncertainty.unsqueeze(-1)
                + boundary_distance_score
            )
            scores.append(
                _aggregate_boundary_scores(
                    boundary_score,
                    target_boundary_idx=self.target_boundary_idx_list[o],
                    boundary_weights=self.boundary_weights_list[o],
                    boundary_reduction=self.boundary_reduction,
                )
            )

        score = self._aggregate_output_scalars(torch.stack(scores, dim=-1))
        score = score - self._aggregated_repulsion_penalty(Xt)
        if self.objective is not None:
            try:
                score = self.objective(score, X=X)
            except TypeError:
                score = self.objective(score)
            if not torch.is_tensor(score):
                raise TypeError(
                    "qMultiOutputOrdinalJointLatentStraddle objective must return Tensor."
                )
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
