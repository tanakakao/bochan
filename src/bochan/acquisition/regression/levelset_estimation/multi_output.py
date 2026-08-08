from __future__ import annotations

"""Multi-output regression level-set estimation acquisition functions.

Design policy:
    - Public names follow the classification / ordinal multi-output naming style.
    - Straddle / margin / boundary / contour acquisitions live here, not in
      active learning.
    - Pointwise acquisitions use a common pipeline:

        posterior mean / variance per output
        -> level-set score per output
        -> output reduction
        -> same-batch / pending / observed penalty
        -> optional score objective / input-perturbation aggregation
        -> q reduction

    - Old ``...Acquisition`` public names are intentionally not kept.
      This file is for the development-stage aligned API.
"""

from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bochan.acquisition._duplicate_exclusion import resolve_observed_X

ReductionType = Literal["mean", "sum", "max", "min"]
OutputReductionType = Literal[
    "mean",
    "sum",
    "max",
    "min",
    "weighted_sum",
    "weighted_mean",
]
BoundaryMode = Literal[
    "distance_to_threshold",
    "common_satisfaction",
    "all_above",
    "all_below",
]
ProbabilityMode = Literal["above", "below", "interval"]


# ============================================================
# Generic helpers
# ============================================================


def _reduce(t: Tensor, dim: int, mode: str) -> Tensor:
    if mode == "mean":
        return t.mean(dim=dim)
    if mode == "sum":
        return t.sum(dim=dim)
    if mode == "max":
        return t.max(dim=dim).values
    if mode == "min":
        return t.min(dim=dim).values
    raise ValueError(f"Unknown reduction mode: {mode!r}.")


def _ensure_q_batch(X: Tensor) -> Tensor:
    if not torch.is_tensor(X):
        raise TypeError(f"X must be a Tensor. Got {type(X)}.")
    if X.ndim == 1:
        return X.view(1, 1, -1)
    if X.ndim == 2:
        return X.unsqueeze(0)
    return X


def _safe_prod(shape: torch.Size | tuple[int, ...]) -> int:
    out = 1
    for s in shape:
        out *= int(s)
    return out


def _objective_call(objective: Callable, score: Tensor, X: Tensor | None):
    try:
        return objective(score, X=X)
    except TypeError:
        return objective(score)


def _objective_X_for_score(score: Tensor, X: Tensor | None) -> Tensor | None:
    """Return an ``X`` argument compatible with a pointwise score objective.

    Args:
        score: Pointwise score tensor passed to an optional objective.
        X: Raw candidate tensor originally passed to the acquisition function.

    Returns:
        The original ``X`` when its q dimension already matches ``score``;
        otherwise ``None`` so BoTorch objectives skip q-shape validation for
        internally transformed pointwise scores.
    """
    if X is None or X.ndim < 3 or score.ndim == 0:
        return X

    if int(score.shape[-1]) == int(X.shape[-2]):
        return X

    return None


def _safe_normal_cdf(z: Tensor) -> Tensor:
    two = torch.as_tensor(2.0, device=z.device, dtype=z.dtype)
    return 0.5 * (1.0 + torch.erf(z / torch.sqrt(two)))


def _safe_logdet(covar: Tensor, jitter: float = 1e-6) -> Tensor:
    q = covar.shape[-1]
    eye = torch.eye(q, device=covar.device, dtype=covar.dtype)
    while eye.ndim < covar.ndim:
        eye = eye.unsqueeze(0)
    covar = 0.5 * (covar + covar.transpose(-1, -2))
    return torch.linalg.slogdet(covar + jitter * eye).logabsdet


# ============================================================
# Score objective
# ============================================================


class MultiOutputRegressionLevelSetScoreObjective(torch.nn.Module):
    """Objective applied to pointwise multi-output regression level-set scores.

    This mirrors the classification / ordinal score-objective pattern.  It is
    mainly used to aggregate InputPerturbation-expanded scores from ``q * n_w``
    back to ``q`` after output reduction.

    Args:
        n_w:
            Number of perturbation samples per candidate.
        risk_type:
            None, "var", or "cvar".
        alpha:
            Tail fraction for VaR / CVaR.
        maximize:
            If True, lower-score tail is treated as worst-case.  This matches
            maximization acquisitions.
        weight:
            Multiplicative weight.
        sign:
            Sign flip.  Keep 1.0 for maximization.
        aggregated_risk_mode:
            If ``"ignore"``, already aggregated batch scores are returned as-is.
            If ``"error"``, receiving an already aggregated score raises.
    """

    def __init__(
        self,
        n_w: int | None = None,
        risk_type: Literal["var", "cvar"] | None = None,
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

        if self.n_w is not None and self.n_w <= 0:
            raise ValueError("n_w must be positive or None.")
        if self.risk_type not in (None, "var", "cvar"):
            raise ValueError(f"Unknown risk_type: {self.risk_type!r}.")
        if self.risk_type is not None and self.n_w is None:
            raise ValueError("risk_type is specified, but n_w is None.")
        if self.risk_type is not None and not (0.0 < self.alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1].")
        if self.aggregated_risk_mode not in ("ignore", "error"):
            raise ValueError("aggregated_risk_mode must be 'ignore' or 'error'.")

    @staticmethod
    def _is_aggregated_score(score: Tensor, X: Tensor | None) -> bool:
        if X is None or score.ndim == 0:
            return False
        Xq = _ensure_q_batch(X)
        return tuple(score.shape) == tuple(Xq.shape[:-2])

    def forward(self, score: Tensor, X: Tensor | None = None) -> Tensor:
        if not torch.is_tensor(score):
            raise TypeError(f"score must be a Tensor. Got {type(score)}.")

        score = score * self.sign * self.weight

        if score.ndim == 0:
            return score

        if self.n_w is None or self.n_w <= 1:
            return score

        if self._is_aggregated_score(score, X):
            if self.aggregated_risk_mode == "error":
                raise RuntimeError(
                    "MultiOutputRegressionLevelSetScoreObjective received an aggregated score. "
                    "InputPerturbation aggregation requires pointwise score."
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

        if self.risk_type is None:
            return score_w.mean(dim=-1)

        descending = not self.maximize
        sorted_score = torch.sort(score_w, dim=-1, descending=descending).values
        k = max(1, int(torch.ceil(torch.as_tensor(self.n_w * self.alpha)).item()))
        tail = sorted_score[..., :k]

        if self.risk_type == "var":
            return tail[..., -1]
        if self.risk_type == "cvar":
            return tail.mean(dim=-1)
        raise ValueError(f"Unknown risk_type: {self.risk_type!r}.")


# ============================================================
# Base class
# ============================================================


class _MultiOutputRegressionLevelSetBase(AcquisitionFunction):
    """Base class aligned with classification / ordinal multi-output level-set APIs."""

    def __init__(
        self,
        model: Model,
        *,
        thresholds: Sequence[float] | Tensor | None = None,
        threshold: float | Tensor | None = None,
        # Backward-supported local alias only inside this new API.
        # Public docs should prefer thresholds / threshold.
        h: Sequence[float] | Tensor | None = None,
        reduction: ReductionType = "mean",
        output_reduction: OutputReductionType = "weighted_mean",
        output_weights: Tensor | Sequence[float] | None = None,
        normalize_output_weights: bool = True,
        X_pending: Tensor | None = None,
        X_observed: Tensor | None = None,
        same_batch_penalty_weight: float = 0.0,
        same_batch_penalty_beta: float = 10.0,
        pending_penalty_weight: float = 0.0,
        pending_penalty_beta: float = 10.0,
        observed_penalty_weight: float = 0.0,
        observed_penalty_beta: float = 10.0,
        hard_duplicate_penalty: float = 0.0,
        hard_duplicate_tol: float = 1e-8,
        exclude_same_batch_duplicates: bool = True,
        exclude_pending_duplicates: bool = True,
        exclude_observed_duplicates: bool = True,
        objective: Callable[[Tensor, Tensor | None], Tensor] | None = None,
        n_w: int | None = None,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(model=model)

        if h is not None:
            thresholds = h
        if thresholds is None:
            thresholds = threshold if threshold is not None else 0.0

        if reduction not in ("mean", "sum", "max", "min"):
            raise ValueError("reduction must be one of 'mean', 'sum', 'max', 'min'.")
        if output_reduction not in (
            "mean",
            "sum",
            "max",
            "min",
            "weighted_sum",
            "weighted_mean",
        ):
            raise ValueError(
                "output_reduction must be one of "
                "'mean', 'sum', 'max', 'min', 'weighted_sum', 'weighted_mean'."
            )
        if float(hard_duplicate_tol) < 0.0:
            raise ValueError("hard_duplicate_tol must be non-negative.")

        self.register_buffer("thresholds", torch.as_tensor(thresholds).reshape(-1))
        self.reduction = reduction
        self.output_reduction = output_reduction
        self.normalize_output_weights = bool(normalize_output_weights)

        if output_weights is not None:
            w = torch.as_tensor(output_weights)
            if w.ndim != 1:
                raise ValueError("output_weights must have shape [m].")
            self.register_buffer("output_weights", w.detach().clone())
        else:
            self.output_weights = None

        self.same_batch_penalty_weight = float(same_batch_penalty_weight)
        self.same_batch_penalty_beta = float(same_batch_penalty_beta)
        self.pending_penalty_weight = float(pending_penalty_weight)
        self.pending_penalty_beta = float(pending_penalty_beta)
        self.observed_penalty_weight = float(observed_penalty_weight)
        self.observed_penalty_beta = float(observed_penalty_beta)
        self.hard_duplicate_penalty = float(hard_duplicate_penalty)
        self.hard_duplicate_tol = float(hard_duplicate_tol)
        self.exclude_same_batch_duplicates = bool(exclude_same_batch_duplicates)
        self.exclude_pending_duplicates = bool(exclude_pending_duplicates)
        self.exclude_observed_duplicates = bool(exclude_observed_duplicates)
        self.objective = objective
        self.eps = float(eps)

        if n_w is None and objective is not None:
            n_w = getattr(objective, "n_w", None)
        self.n_w = None if n_w is None else int(n_w)
        if self.n_w is not None and self.n_w <= 0:
            raise ValueError("n_w must be positive or None.")

        self.X_pending: Tensor | None = None
        self.X_observed: Tensor | None = None
        self.set_X_pending(X_pending)
        self.set_X_observed(X_observed)

    # ------------------------------------------------------------
    # Reference handling
    # ------------------------------------------------------------
    def _coerce_reference_to_tensor(
        self,
        ref,
        *,
        like: Tensor | None = None,
    ) -> Tensor | None:
        if ref is None:
            return None

        if torch.is_tensor(ref):
            out = ref
        elif isinstance(ref, (list, tuple)):
            tensors = []
            for item in ref:
                if item is None:
                    continue
                t = self._coerce_reference_to_tensor(item, like=like)
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
                "Reference points must be None, Tensor, list, or tuple. "
                f"Got {type(ref)}."
            )

        if like is not None:
            out = out.to(device=like.device, dtype=like.dtype)

        return out.detach()

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        self.X_pending = self._coerce_reference_to_tensor(X_pending)

    def set_X_observed(self, X_observed: Tensor | None = None) -> None:
        self.X_observed = self._coerce_reference_to_tensor(
            resolve_observed_X(self.model, X_observed)
        )

    # ------------------------------------------------------------
    # Shape / transform helpers
    # ------------------------------------------------------------
    def _prepare_eval(self) -> None:
        self.model.eval()
        likelihood = getattr(self.model, "likelihood", None)
        if likelihood is not None and hasattr(likelihood, "eval"):
            likelihood.eval()

    def _apply_input_transform_for_distance(self, X: Tensor) -> Tensor:
        X = _ensure_q_batch(X)

        it = getattr(self.model, "input_transform", None)
        if it is not None:
            Xt = it(X)
            if isinstance(Xt, tuple):
                Xt = Xt[0]
            return _ensure_q_batch(Xt)

        models = getattr(self.model, "models", None)
        if models is not None and len(models) > 0:
            it = getattr(models[0], "input_transform", None)
            if it is not None:
                Xt = it(X)
                if isinstance(Xt, tuple):
                    Xt = Xt[0]
                return _ensure_q_batch(Xt)

        return X

    def _reference_to_distance_space(self, ref, *, like: Tensor) -> Tensor | None:
        ref = self._coerce_reference_to_tensor(ref, like=like)
        if ref is None or ref.numel() == 0:
            return None
        ref_t = self._apply_input_transform_for_distance(ref)
        return _ensure_q_batch(ref_t).to(device=like.device, dtype=like.dtype)

    def _thresholds_like(self, value: Tensor) -> Tensor:
        """Return thresholds broadcastable to ``value[..., m]``."""
        if value.ndim < 1:
            raise RuntimeError("value must have an output dimension.")
        m = int(value.shape[-1])
        thresholds = self.thresholds.to(device=value.device, dtype=value.dtype)

        if thresholds.numel() == 1:
            thresholds = thresholds.expand(m)
        elif thresholds.numel() != m:
            raise ValueError(
                f"Number of thresholds ({thresholds.numel()}) does not match "
                f"number of outputs ({m})."
            )

        return thresholds.view(*((1,) * (value.ndim - 1)), m)

    def _align_output_tensor_to_X(
        self,
        value: Tensor,
        Xt: Tensor,
        *,
        name: str,
    ) -> Tensor:
        """Align posterior mean / variance to ``Xt.shape[:-1] + (m,)``."""
        Xt = _ensure_q_batch(Xt)
        target_prefix = torch.Size(Xt.shape[:-1])
        out = value

        # Already scalar per point: add output dimension m=1.
        if out.shape == target_prefix:
            return out.unsqueeze(-1)

        # Reduce leading MCMC / ensemble dims until at most output dim remains.
        while out.ndim > len(target_prefix) + 1:
            out = out.mean(dim=0)
            if out.shape == target_prefix:
                return out.unsqueeze(-1)

        if out.ndim == len(target_prefix) + 1 and out.shape[:-1] == target_prefix:
            return out

        if out.ndim == len(target_prefix) and out.shape == target_prefix:
            return out.unsqueeze(-1)

        # Last-resort reshape if possible.
        if out.numel() % max(_safe_prod(target_prefix), 1) == 0:
            m = out.numel() // max(_safe_prod(target_prefix), 1)
            return out.reshape(*target_prefix, m)

        raise RuntimeError(
            f"{name}: could not align tensor to output shape. "
            f"value.shape={tuple(value.shape)}, Xt.shape={tuple(Xt.shape)}."
        )

    def _output_weights_like(self, value: Tensor) -> Tensor | None:
        weights = self.output_weights
        if weights is None:
            return None
        if value.shape[-1] != weights.numel():
            raise ValueError(
                f"Mismatch between output dim {value.shape[-1]} and "
                f"output_weights {weights.numel()}."
            )
        w = weights.to(device=value.device, dtype=value.dtype)
        if self.normalize_output_weights:
            w = w / w.sum().clamp_min(self.eps)
        return w

    def _reduce_outputs(self, value: Tensor) -> Tensor:
        """Reduce output dimension ``m`` to pointwise scalar score."""
        if value.ndim < 1:
            return value

        if value.shape[-1] == 1:
            return value.squeeze(-1)

        if self.output_reduction == "weighted_sum":
            w = self._output_weights_like(value)
            if w is None:
                raise ValueError("output_reduction='weighted_sum' requires output_weights.")
            return (value * w).sum(dim=-1)

        if self.output_reduction == "weighted_mean":
            w = self._output_weights_like(value)
            if w is None:
                return value.mean(dim=-1)
            return (value * w).sum(dim=-1)

        return _reduce(value, dim=-1, mode=self.output_reduction)

    def _align_pointwise_score_to_X(
        self,
        score: Tensor,
        Xt: Tensor,
        *,
        name: str,
        reduce_extra: ReductionType = "mean",
    ) -> Tensor:
        Xt = _ensure_q_batch(Xt)
        target = torch.Size(Xt.shape[:-1])
        out = score

        if out.shape == target:
            return out

        if out.ndim >= 1 and out.shape[-1] == 1:
            out_s = out.squeeze(-1)
            if out_s.shape == target:
                return out_s
            out = out_s

        if out.shape == target:
            return out

        while out.ndim > len(target):
            out = _reduce(out, dim=0, mode=reduce_extra)
            if out.shape == target:
                return out

        if out.shape == target:
            return out

        if out.numel() == _safe_prod(target):
            return out.reshape(target)

        raise RuntimeError(
            f"{name}: score shape mismatch. "
            f"score.shape={tuple(score.shape)}, expected={tuple(target)}, Xt.shape={tuple(Xt.shape)}."
        )

    # ------------------------------------------------------------
    # Posterior helpers
    # ------------------------------------------------------------
    def _posterior_mean_variance_outputs(self, X: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        Xq = _ensure_q_batch(X)
        self._prepare_eval()

        posterior = self.model.posterior(Xq, observation_noise=False)
        Xt = self._apply_input_transform_for_distance(Xq)

        mean = self._align_output_tensor_to_X(posterior.mean, Xt, name="posterior.mean")
        var = self._align_output_tensor_to_X(posterior.variance, Xt, name="posterior.variance")
        var = var.clamp_min(self.eps)
        return mean, var, Xt

    def _posterior_covariance(self, X: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return output-reduced mean and covariance over q-like points.

        For true multi-output joint covariance, posterior covariance may have
        shape ``q*m x q*m``.  To keep the API robust across custom wrappers, this
        function falls back to diagonal covariance from output-reduced variance
        when the covariance shape cannot be aligned to ``q_like x q_like``.
        """
        mean_outputs, var_outputs, Xt = self._posterior_mean_variance_outputs(X)
        mean = self._reduce_outputs(mean_outputs)
        var = self._reduce_outputs(var_outputs)

        posterior = self.model.posterior(_ensure_q_batch(X), observation_noise=False)

        covar = None
        mvn = getattr(posterior, "mvn", None)
        if mvn is not None and hasattr(mvn, "covariance_matrix"):
            covar = mvn.covariance_matrix
        elif hasattr(posterior, "distribution") and hasattr(posterior.distribution, "covariance_matrix"):
            covar = posterior.distribution.covariance_matrix

        q_like = int(Xt.shape[-2])
        target_covar_shape = torch.Size(Xt.shape[:-2]) + torch.Size([q_like, q_like])

        if covar is None:
            return mean, torch.diag_embed(var), Xt

        while covar.ndim > len(target_covar_shape):
            covar = covar.mean(dim=0)
            if covar.shape == target_covar_shape:
                break

        if covar.shape != target_covar_shape:
            if covar.numel() == _safe_prod(target_covar_shape):
                covar = covar.reshape(target_covar_shape)
            else:
                covar = torch.diag_embed(var)

        covar = 0.5 * (covar + covar.transpose(-1, -2))
        return mean, covar, Xt

    # ------------------------------------------------------------
    # Penalty helpers
    # ------------------------------------------------------------
    def _same_batch_penalty_per_point(self, Xt: Tensor) -> Tensor:
        Xt = _ensure_q_batch(Xt)
        q = int(Xt.shape[-2])
        zeros = Xt.new_zeros(Xt.shape[:-1])
        if q <= 1:
            return zeros
        if (
            self.same_batch_penalty_weight <= 0.0
            and self.hard_duplicate_penalty <= 0.0
            and not self.exclude_same_batch_duplicates
        ):
            return zeros

        d2 = (Xt.unsqueeze(-2) - Xt.unsqueeze(-3)).pow(2).sum(dim=-1)
        eye = torch.eye(q, dtype=torch.bool, device=Xt.device)
        while eye.ndim < d2.ndim:
            eye = eye.unsqueeze(0)
        valid = ~eye

        penalty = zeros
        if self.same_batch_penalty_weight > 0.0:
            soft = torch.exp(-self.same_batch_penalty_beta * d2)
            soft = torch.where(valid, soft, torch.zeros_like(soft))
            penalty = self.same_batch_penalty_weight * soft.sum(dim=-1)

        duplicate_pairs = valid & (d2 <= self.hard_duplicate_tol**2)
        if self.hard_duplicate_penalty > 0.0:
            penalty = penalty + self.hard_duplicate_penalty * duplicate_pairs.to(
                dtype=Xt.dtype
            ).sum(dim=-1)

        if self.exclude_same_batch_duplicates:
            duplicate_batch = duplicate_pairs.any(dim=-1).any(dim=-1, keepdim=True)
            penalty = torch.where(
                duplicate_batch.expand_as(penalty),
                torch.full_like(penalty, torch.inf),
                penalty,
            )
        return penalty

    def _reference_penalty_per_point(
        self,
        Xt: Tensor,
        ref,
        *,
        weight: float,
        beta: float,
        exclude_duplicates: bool = False,
    ) -> Tensor:
        Xt = _ensure_q_batch(Xt)
        zeros = Xt.new_zeros(Xt.shape[:-1])
        if weight <= 0.0 and not exclude_duplicates:
            return zeros

        ref_t = self._reference_to_distance_space(ref, like=Xt)
        if ref_t is None or ref_t.numel() == 0:
            return zeros

        ref2d = ref_t.reshape(-1, ref_t.shape[-1])
        if ref2d.shape[-1] != Xt.shape[-1]:
            raise RuntimeError(
                "Reference feature dimension mismatch after transform: "
                f"Xt.shape={tuple(Xt.shape)}, ref_transformed.shape={tuple(ref_t.shape)}."
            )

        dist = torch.cdist(Xt.reshape(-1, Xt.shape[-1]), ref2d)
        min_dist = dist.min(dim=-1).values.reshape(*Xt.shape[:-1])
        penalty = (
            weight * torch.exp(-beta * min_dist)
            if weight > 0.0
            else zeros
        )
        if exclude_duplicates:
            duplicate_batch = (min_dist <= self.hard_duplicate_tol).any(
                dim=-1,
                keepdim=True,
            )
            penalty = torch.where(
                duplicate_batch.expand_as(penalty),
                torch.full_like(penalty, torch.inf),
                penalty,
            )
        return penalty

    def _total_penalty_per_point(self, Xt: Tensor) -> Tensor:
        return (
            self._same_batch_penalty_per_point(Xt)
            + self._reference_penalty_per_point(
                Xt,
                self.X_pending,
                weight=self.pending_penalty_weight,
                beta=self.pending_penalty_beta,
                exclude_duplicates=self.exclude_pending_duplicates,
            )
            + self._reference_penalty_per_point(
                Xt,
                self.X_observed,
                weight=self.observed_penalty_weight,
                beta=self.observed_penalty_beta,
                exclude_duplicates=self.exclude_observed_duplicates,
            )
        )

    # ------------------------------------------------------------
    # Objective / reduction
    # ------------------------------------------------------------
    def _apply_objective_to_score(self, score: Tensor, X: Tensor, name: str) -> Tensor:
        if self.objective is None:
            return score

        X_for_objective = _objective_X_for_score(score, X)
        out = _objective_call(self.objective, score, X_for_objective)
        if not torch.is_tensor(out):
            raise RuntimeError(f"{name}: objective must return Tensor. Got {type(out)}.")
        return out

    def _aggregate_n_w_if_needed(self, score: Tensor, *, q: int, context: str) -> Tensor:
        """Aggregate one-to-many perturbation scores back to raw q points.

        Args:
            score: Pointwise acquisition scores with the candidate dimension last.
            q: Number of raw candidate points passed to the acquisition function.
            context: Human-readable acquisition name used in error messages.

        Returns:
            A score tensor whose last dimension is the raw candidate dimension.

        Raises:
            RuntimeError: If the score shape is incompatible with the raw q size,
                the configured perturbation count, and known sequential q behavior.
        """
        if self.n_w is None:
            return score

        expected = q * int(self.n_w)
        actual = int(score.shape[-1])
        if actual == q:
            return score
        if actual == expected:
            return score.reshape(*score.shape[:-1], q, int(self.n_w)).mean(dim=-1)

        # BoTorch sequential q optimization evaluates a one-point acquisition
        # while keeping already selected points in X_pending. Some one-to-many
        # input transforms can expose those pending points in the transformed
        # q-like dimension, so the internal score has more points than raw X.
        # Only the leading raw q points correspond to the candidate currently
        # optimized; pending-point interactions are still handled by the pending
        # penalty path.
        if q == 1 and actual > q:
            return score[..., :q]

        raise RuntimeError(
            f"{context}: expected last dimension q={q} or q*n_w={expected}, "
            f"got score.shape={tuple(score.shape)}."
        )

    def _reduce_q(self, score: Tensor) -> Tensor:
        return _reduce(score, dim=-1, mode=self.reduction)

    def _finalize_pointwise_score(
        self,
        score: Tensor,
        X: Tensor,
        Xt: Tensor,
        *,
        name: str,
    ) -> Tensor:
        raw_X = _ensure_q_batch(X)
        original_batch_shape = torch.Size(raw_X.shape[:-2])
        q = int(raw_X.shape[-2])

        score = self._align_pointwise_score_to_X(score, Xt, name=f"{name} score before penalty")
        score = score - self._total_penalty_per_point(Xt)

        score = self._align_pointwise_score_to_X(score, Xt, name=f"{name} score before objective")
        score = self._apply_objective_to_score(score, raw_X, name=name)

        score = self._aggregate_n_w_if_needed(score, q=q, context=name)
        out = self._reduce_q(score)

        if out.shape == original_batch_shape:
            return out

        while out.ndim > len(original_batch_shape):
            out = out.mean(dim=0)

        if out.shape == original_batch_shape:
            return out

        if out.numel() == 1 and _safe_prod(original_batch_shape) > 1:
            return out.reshape(*((1,) * len(original_batch_shape))).expand(
                original_batch_shape
            )

        if out.numel() == _safe_prod(original_batch_shape):
            return out.reshape(original_batch_shape)

        raise RuntimeError(
            f"{name}: output shape mismatch. "
            f"Expected {tuple(original_batch_shape)}, got {tuple(out.shape)}."
        )

    def _finalize_joint_score(self, score: Tensor, X: Tensor, Xt: Tensor, *, name: str) -> Tensor:
        raw_X = _ensure_q_batch(X)
        original_batch_shape = torch.Size(raw_X.shape[:-2])

        if score.shape != Xt.shape[:-2]:
            while score.ndim > len(Xt.shape[:-2]):
                score = score.mean(dim=0)
            if score.shape != Xt.shape[:-2] and score.numel() == _safe_prod(Xt.shape[:-2]):
                score = score.reshape(Xt.shape[:-2])
            if score.shape != Xt.shape[:-2]:
                raise RuntimeError(
                    f"{name}: joint score shape mismatch. "
                    f"score.shape={tuple(score.shape)}, expected={tuple(Xt.shape[:-2])}."
                )

        penalty = self._total_penalty_per_point(Xt)
        penalty = self._reduce_q(penalty)
        score = score - penalty

        score = self._apply_objective_to_score(score, raw_X, name=name)

        if score.shape == original_batch_shape:
            return score

        while score.ndim > len(original_batch_shape):
            score = score.mean(dim=0)

        if score.shape == original_batch_shape:
            return score

        if score.numel() == _safe_prod(original_batch_shape):
            return score.reshape(original_batch_shape)

        raise RuntimeError(
            f"{name}: output shape mismatch. "
            f"Expected {tuple(original_batch_shape)}, got {tuple(score.shape)}."
        )


# ============================================================
# Acquisition implementations
# ============================================================


class qMultiOutputRegressionStraddle(_MultiOutputRegressionLevelSetBase):
    """Multi-output regression straddle acquisition.

    Per output:
        score_j(x) = beta * std_j(x) - |mean_j(x) - threshold_j|

    Then the output dimension is reduced by ``output_reduction``.
    """

    def __init__(
        self,
        model: Model,
        *,
        beta: float | Tensor = 1.96,
        boundary_mode: BoundaryMode = "distance_to_threshold",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        if boundary_mode not in (
            "distance_to_threshold",
            "common_satisfaction",
            "all_above",
            "all_below",
        ):
            raise ValueError(
                "boundary_mode must be 'distance_to_threshold', "
                "'common_satisfaction', 'all_above', or 'all_below'."
            )
        self.register_buffer("beta", torch.as_tensor(beta))
        self.boundary_mode = boundary_mode

    def _boundary_distance(self, mean: Tensor, thresholds: Tensor) -> Tensor:
        if self.boundary_mode == "distance_to_threshold":
            return (mean - thresholds).abs()
        if self.boundary_mode in ("common_satisfaction", "all_above"):
            return torch.relu(thresholds - mean)
        if self.boundary_mode == "all_below":
            return torch.relu(mean - thresholds)
        raise ValueError(f"Unknown boundary_mode: {self.boundary_mode!r}.")

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        mean, var, Xt = self._posterior_mean_variance_outputs(X)
        std = var.sqrt()
        thresholds = self._thresholds_like(mean)
        beta = self.beta.to(device=mean.device, dtype=mean.dtype)

        score_per_output = beta * std - self._boundary_distance(mean, thresholds)
        score = self._reduce_outputs(score_per_output)

        return self._finalize_pointwise_score(
            score,
            X,
            Xt,
            name="qMultiOutputRegressionStraddle",
        )


class qMultiOutputRegressionJointStraddle(_MultiOutputRegressionLevelSetBase):
    """Joint multi-output regression straddle acquisition.

    This scores the q-batch jointly by combining average boundary proximity
    across outputs with joint covariance uncertainty.
    """

    def __init__(
        self,
        model: Model,
        *,
        beta: float | Tensor = 1.0,
        uncertainty_measure: Literal["logdet", "logdet1p", "trace"] = "logdet1p",
        boundary_mode: BoundaryMode = "distance_to_threshold",
        covariance_jitter: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        if uncertainty_measure not in ("logdet", "logdet1p", "trace"):
            raise ValueError("uncertainty_measure must be 'logdet', 'logdet1p', or 'trace'.")
        if boundary_mode not in (
            "distance_to_threshold",
            "common_satisfaction",
            "all_above",
            "all_below",
        ):
            raise ValueError(
                "boundary_mode must be 'distance_to_threshold', "
                "'common_satisfaction', 'all_above', or 'all_below'."
            )
        self.register_buffer("beta", torch.as_tensor(beta))
        self.uncertainty_measure = uncertainty_measure
        self.boundary_mode = boundary_mode
        self.covariance_jitter = float(covariance_jitter)

    def _boundary_distance(self, mean_outputs: Tensor, thresholds: Tensor) -> Tensor:
        if self.boundary_mode == "distance_to_threshold":
            return (mean_outputs - thresholds).abs()
        if self.boundary_mode in ("common_satisfaction", "all_above"):
            return torch.relu(thresholds - mean_outputs)
        if self.boundary_mode == "all_below":
            return torch.relu(mean_outputs - thresholds)
        raise ValueError(f"Unknown boundary_mode: {self.boundary_mode!r}.")

    def _uncertainty_score(self, covar: Tensor) -> Tensor:
        if self.uncertainty_measure == "trace":
            return covar.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        if self.uncertainty_measure == "logdet":
            return _safe_logdet(covar, jitter=self.covariance_jitter)

        q = covar.shape[-1]
        eye = torch.eye(q, device=covar.device, dtype=covar.dtype)
        while eye.ndim < covar.ndim:
            eye = eye.unsqueeze(0)
        return _safe_logdet(eye + covar, jitter=self.covariance_jitter)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        mean_outputs, _, Xt = self._posterior_mean_variance_outputs(X)
        thresholds = self._thresholds_like(mean_outputs)

        _, covar, Xt = self._posterior_covariance(X)
        beta = self.beta.to(device=mean_outputs.device, dtype=mean_outputs.dtype)

        boundary = self._boundary_distance(mean_outputs, thresholds)
        boundary_score = -self._reduce_outputs(boundary).mean(dim=-1)
        uncertainty = self._uncertainty_score(covar)
        score = boundary_score + beta * uncertainty

        return self._finalize_joint_score(
            score,
            X,
            Xt,
            name="qMultiOutputRegressionJointStraddle",
        )


class qMultiOutputRegressionICU(_MultiOutputRegressionLevelSetBase):
    """Multi-output integrated contour uncertainty style acquisition.

    For each output this uses a smooth threshold-density style score and then
    reduces over outputs.
    """

    def __init__(
        self,
        model: Model,
        *,
        bandwidth: float | Tensor | None = None,
        joint_boundary: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.bandwidth = None if bandwidth is None else torch.as_tensor(bandwidth)
        self.joint_boundary = bool(joint_boundary)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        mean, var, Xt = self._posterior_mean_variance_outputs(X)
        std = var.sqrt().clamp_min(self.eps)
        thresholds = self._thresholds_like(mean)

        if self.bandwidth is None:
            bw = std
        else:
            bw = self.bandwidth.to(device=mean.device, dtype=mean.dtype).clamp_min(self.eps)

        z = (mean - thresholds) / bw
        score_per_output = torch.exp(-0.5 * z.pow(2)) * std

        if self.joint_boundary:
            # All outputs near their thresholds simultaneously.
            score = score_per_output.prod(dim=-1)
        else:
            score = self._reduce_outputs(score_per_output)

        return self._finalize_pointwise_score(
            score,
            X,
            Xt,
            name="qMultiOutputRegressionICU",
        )


class qMultiOutputRegressionBoundaryVariance(_MultiOutputRegressionLevelSetBase):
    """Boundary-weighted posterior variance acquisition for multi-output regression."""

    def __init__(
        self,
        model: Model,
        *,
        tau: float | Tensor = 1.0,
        joint_boundary: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.register_buffer("tau", torch.as_tensor(tau))
        self.joint_boundary = bool(joint_boundary)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        mean, var, Xt = self._posterior_mean_variance_outputs(X)
        thresholds = self._thresholds_like(mean)
        tau = self.tau.to(device=mean.device, dtype=mean.dtype).clamp_min(self.eps)

        boundary_weight = torch.exp(-0.5 * ((mean - thresholds) / tau).pow(2))
        score_per_output = var * boundary_weight

        if self.joint_boundary:
            # Joint boundary uncertainty: all outputs near boundary and uncertain.
            score = score_per_output.prod(dim=-1)
        else:
            score = self._reduce_outputs(score_per_output)

        return self._finalize_pointwise_score(
            score,
            X,
            Xt,
            name="qMultiOutputRegressionBoundaryVariance",
        )


class qMultiOutputRegressionProbabilityOfExceedance(_MultiOutputRegressionLevelSetBase):
    """Probability-of-exceedance / feasibility style acquisition.

    Modes:
        - ``above``:    P(f_j(x) >= threshold_j)
        - ``below``:    P(f_j(x) <= threshold_j)
        - ``interval``: P(lower_j <= f_j(x) <= upper_j)

    If ``joint=True``, output probabilities are multiplied before q reduction.
    Otherwise they are reduced by ``output_reduction``.
    """

    def __init__(
        self,
        model: Model,
        *,
        mode: ProbabilityMode = "above",
        lower: Sequence[float] | Tensor | None = None,
        upper: Sequence[float] | Tensor | None = None,
        temperature: float | Tensor | None = None,
        joint: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        if mode not in ("above", "below", "interval"):
            raise ValueError("mode must be 'above', 'below', or 'interval'.")
        self.mode = mode
        self.lower = None if lower is None else torch.as_tensor(lower).reshape(-1)
        self.upper = None if upper is None else torch.as_tensor(upper).reshape(-1)
        self.temperature = None if temperature is None else torch.as_tensor(temperature)
        self.joint = bool(joint)

    def _bounds_like(self, value: Tensor, which: str) -> Tensor:
        bound = self.lower if which == "lower" else self.upper
        if bound is None:
            raise ValueError(f"{which} must be provided when mode='interval'.")
        m = int(value.shape[-1])
        bound = bound.to(device=value.device, dtype=value.dtype)
        if bound.numel() == 1:
            bound = bound.expand(m)
        elif bound.numel() != m:
            raise ValueError(
                f"{which} length ({bound.numel()}) does not match output dim ({m})."
            )
        return bound.view(*((1,) * (value.ndim - 1)), m)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        mean, var, Xt = self._posterior_mean_variance_outputs(X)
        std = var.sqrt().clamp_min(self.eps)
        thresholds = self._thresholds_like(mean)

        if self.temperature is not None:
            temp = self.temperature.to(device=mean.device, dtype=mean.dtype).clamp_min(self.eps)
            if self.mode == "above":
                score_per_output = torch.sigmoid((mean - thresholds) / temp)
            elif self.mode == "below":
                score_per_output = torch.sigmoid((thresholds - mean) / temp)
            else:
                lo = self._bounds_like(mean, "lower")
                hi = self._bounds_like(mean, "upper")
                score_per_output = torch.sigmoid((mean - lo) / temp) * torch.sigmoid((hi - mean) / temp)
        else:
            if self.mode == "above":
                score_per_output = _safe_normal_cdf((mean - thresholds) / std)
            elif self.mode == "below":
                score_per_output = _safe_normal_cdf((thresholds - mean) / std)
            else:
                lo = self._bounds_like(mean, "lower")
                hi = self._bounds_like(mean, "upper")
                score_per_output = _safe_normal_cdf((hi - mean) / std) - _safe_normal_cdf((lo - mean) / std)

        score_per_output = score_per_output.clamp_min(0.0)

        if self.joint:
            score = score_per_output.prod(dim=-1)
        else:
            score = self._reduce_outputs(score_per_output)

        return self._finalize_pointwise_score(
            score,
            X,
            Xt,
            name="qMultiOutputRegressionProbabilityOfExceedance",
        )


__all__ = [
    "MultiOutputRegressionLevelSetScoreObjective",
    "qMultiOutputRegressionStraddle",
    "qMultiOutputRegressionJointStraddle",
    "qMultiOutputRegressionICU",
    "qMultiOutputRegressionBoundaryVariance",
    "qMultiOutputRegressionProbabilityOfExceedance",
]
