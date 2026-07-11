from __future__ import annotations

"""Multi-output regression active-learning acquisition functions.

Design policy:
    - Public names follow the classification / ordinal multi-output naming style.
    - This module is for active learning / uncertainty reduction only.
      Straddle / margin / boundary / contour acquisitions belong to
      multi-output regression level-set estimation.
    - Pointwise proxy acquisitions use a common pipeline:

        posterior score per output
        -> output reduction
        -> same-batch / pending / observed penalty
        -> optional score objective / input-perturbation aggregation
        -> q reduction

    - If BoTorch already provides the true acquisition, this module follows its
      computation directly rather than nesting another decorated acquisition.

Notes:
    qMultiOutputRegressionIntegratedPosteriorVarianceProxy is intentionally a
    proxy for models that do not support fantasize(), such as many custom
    DeepGP wrappers.
"""

from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
from botorch import settings
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from torch import Tensor

try:
    from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
except Exception:  # pragma: no cover - depends on BoTorch version
    MCMultiOutputObjective = None  # type: ignore


ReductionType = Literal["mean", "sum", "max", "min"]
OutputReductionType = Literal[
    "mean",
    "sum",
    "max",
    "min",
    "weighted_sum",
    "weighted_mean",
]


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


def _is_mc_multi_output_objective(objective: Any) -> bool:
    return MCMultiOutputObjective is not None and isinstance(objective, MCMultiOutputObjective)


def _looks_like_score_objective(objective: Any) -> bool:
    if objective is None:
        return False
    if _is_mc_multi_output_objective(objective):
        return False
    return (
        hasattr(objective, "n_w")
        or hasattr(objective, "risk_type")
        or hasattr(objective, "alpha")
        or objective.__class__.__name__.endswith("ScoreObjective")
    )


# ============================================================
# Base class
# ============================================================


class _MultiOutputRegressionActiveLearningBase(AcquisitionFunction):
    """Base class aligned with classification / ordinal active-learning APIs.

    Args:
        model:
            BoTorch-supported multi-output regression model.
        reduction:
            q-batch reduction.  This matches the classification / ordinal API.
        output_reduction:
            Reduction over output dimension ``m``.
        output_weights:
            Optional output weights for weighted reductions.
        pending_penalty_weight:
            Weight for avoiding X_pending.
        observed_penalty_weight:
            Weight for avoiding X_observed.
        same_batch_penalty_weight:
            Weight for q-batch diversity penalty.
        objective:
            Optional score objective.  Classification / ordinal style score
            objectives receive pointwise scalar scores after output reduction.
            BoTorch MC multi-output objectives receive deterministic
            pseudo-samples before final reduction only when explicitly needed.
        n_w:
            Number of input perturbation samples.  If omitted but objective has
            ``n_w``, that value is used.
    """

    def __init__(
        self,
        model,
        *,
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
        objective: Callable[[Tensor, Tensor | None], Tensor] | None = None,
        n_w: int | None = None,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(model=model)

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

        # Reference points are constants during acquisition optimization.
        return out.detach()

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        self.X_pending = self._coerce_reference_to_tensor(X_pending)

    def set_X_observed(self, X_observed: Tensor | None = None) -> None:
        self.X_observed = self._coerce_reference_to_tensor(X_observed)

    # ------------------------------------------------------------
    # Transform / shape helpers
    # ------------------------------------------------------------
    def _prepare_eval(self) -> None:
        self.model.eval()
        likelihood = getattr(self.model, "likelihood", None)
        if likelihood is not None and hasattr(likelihood, "eval"):
            likelihood.eval()

    def _apply_input_transform_for_distance(self, X: Tensor) -> Tensor:
        """Apply model input transform for distance / penalty calculations."""
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

    def _reference_to_distance_space(
        self,
        ref,
        *,
        like: Tensor,
    ) -> Tensor | None:
        ref = self._coerce_reference_to_tensor(ref, like=like)
        if ref is None or ref.numel() == 0:
            return None
        ref_t = self._apply_input_transform_for_distance(ref)
        return _ensure_q_batch(ref_t).to(device=like.device, dtype=like.dtype)

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

        # Drop singleton output dim, not q=1.
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

    def _reduce_outputs_if_needed(self, value: Tensor, Xt: Tensor, *, name: str) -> Tensor:
        """Reduce output dimension ``m`` to a pointwise scalar score."""
        Xt = _ensure_q_batch(Xt)
        target_prefix = torch.Size(Xt.shape[:-1])
        out = value

        if out.shape == target_prefix:
            return out

        # Reduce leading MCMC / ensemble dims until only output dim can remain.
        while out.ndim > len(target_prefix) + 1:
            out = out.mean(dim=0)
            if out.shape == target_prefix:
                return out

        if out.ndim == len(target_prefix) + 1 and out.shape[:-1] == target_prefix:
            if out.shape[-1] == 1:
                return out.squeeze(-1)

            if self.output_reduction == "weighted_sum":
                w = self._output_weights_like(out)
                if w is None:
                    raise ValueError("output_reduction='weighted_sum' requires output_weights.")
                return (out * w).sum(dim=-1)

            if self.output_reduction == "weighted_mean":
                w = self._output_weights_like(out)
                if w is None:
                    return out.mean(dim=-1)
                return (out * w).sum(dim=-1)

            return _reduce(out, dim=-1, mode=self.output_reduction)

        if out.shape == target_prefix:
            return out

        # Last-resort reshape if shape is equivalent.
        if out.numel() % max(_safe_prod(target_prefix), 1) == 0:
            m = out.numel() // max(_safe_prod(target_prefix), 1)
            out = out.reshape(*target_prefix, m)
            if m == 1:
                return out.squeeze(-1)

            if self.output_reduction == "weighted_sum":
                w = self._output_weights_like(out)
                if w is None:
                    raise ValueError("output_reduction='weighted_sum' requires output_weights.")
                return (out * w).sum(dim=-1)

            if self.output_reduction == "weighted_mean":
                w = self._output_weights_like(out)
                if w is None:
                    return out.mean(dim=-1)
                return (out * w).sum(dim=-1)

            return _reduce(out, dim=-1, mode=self.output_reduction)

        raise RuntimeError(
            f"{name}: could not reduce output dimension. "
            f"value.shape={tuple(value.shape)}, Xt.shape={tuple(Xt.shape)}."
        )

    # ------------------------------------------------------------
    # Posterior scores
    # ------------------------------------------------------------
    def _posterior_mean_variance(
        self,
        X: Tensor,
        *,
        observation_noise: bool | Tensor = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        Xq = _ensure_q_batch(X)
        self._prepare_eval()

        post = self.model.posterior(Xq, observation_noise=observation_noise)
        Xt = self._apply_input_transform_for_distance(Xq)

        mean = self._reduce_outputs_if_needed(post.mean, Xt, name="posterior.mean")
        var = self._reduce_outputs_if_needed(post.variance, Xt, name="posterior.variance")
        var = var.clamp_min(self.eps)

        mean = self._align_pointwise_score_to_X(mean, Xt, name="posterior.mean")
        var = self._align_pointwise_score_to_X(var, Xt, name="posterior.variance")

        return mean, var, Xt

    def _posterior_variance_score(self, X: Tensor) -> tuple[Tensor, Tensor]:
        _, var, Xt = self._posterior_mean_variance(X, observation_noise=False)
        return var, Xt

    # ------------------------------------------------------------
    # Penalties
    # ------------------------------------------------------------
    def _same_batch_penalty_per_point(self, Xt: Tensor) -> Tensor:
        Xt = _ensure_q_batch(Xt)
        q = int(Xt.shape[-2])
        if self.same_batch_penalty_weight <= 0.0 or q <= 1:
            return Xt.new_zeros(Xt.shape[:-1])

        d2 = (Xt.unsqueeze(-2) - Xt.unsqueeze(-3)).pow(2).sum(dim=-1)
        eye = torch.eye(q, dtype=torch.bool, device=Xt.device)
        while eye.ndim < d2.ndim:
            eye = eye.unsqueeze(0)
        valid = ~eye

        soft = torch.exp(-self.same_batch_penalty_beta * d2)
        soft = torch.where(valid, soft, torch.zeros_like(soft))
        per_point = soft.sum(dim=-1)

        if self.hard_duplicate_penalty > 0.0:
            dup = (d2 <= self.hard_duplicate_tol).to(dtype=Xt.dtype)
            dup = torch.where(valid, dup, torch.zeros_like(dup))
            per_point = per_point + self.hard_duplicate_penalty * dup.sum(dim=-1)

        return self.same_batch_penalty_weight * per_point

    def _reference_penalty_per_point(
        self,
        Xt: Tensor,
        ref,
        *,
        weight: float,
        beta: float,
    ) -> Tensor:
        Xt = _ensure_q_batch(Xt)
        if weight <= 0.0:
            return Xt.new_zeros(Xt.shape[:-1])

        ref_t = self._reference_to_distance_space(ref, like=Xt)
        if ref_t is None or ref_t.numel() == 0:
            return Xt.new_zeros(Xt.shape[:-1])

        ref2d = ref_t.reshape(-1, ref_t.shape[-1])
        if ref2d.shape[-1] != Xt.shape[-1]:
            raise RuntimeError(
                "Reference feature dimension mismatch after transform: "
                f"Xt.shape={tuple(Xt.shape)}, ref_transformed.shape={tuple(ref_t.shape)}."
            )

        dist = torch.cdist(Xt.reshape(-1, Xt.shape[-1]), ref2d)
        min_dist = dist.min(dim=-1).values.reshape(*Xt.shape[:-1])
        return weight * torch.exp(-beta * min_dist)

    def _total_penalty_per_point(self, Xt: Tensor) -> Tensor:
        return (
            self._same_batch_penalty_per_point(Xt)
            + self._reference_penalty_per_point(
                Xt,
                self.X_pending,
                weight=self.pending_penalty_weight,
                beta=self.pending_penalty_beta,
            )
            + self._reference_penalty_per_point(
                Xt,
                self.X_observed,
                weight=self.observed_penalty_weight,
                beta=self.observed_penalty_beta,
            )
        )

    # ------------------------------------------------------------
    # Objective / q reduction
    # ------------------------------------------------------------
    def _apply_objective_to_score(
        self,
        score: Tensor,
        *,
        raw_X: Tensor,
        expanded_X: Tensor,
        name: str,
    ) -> Tensor:
        objective = self.objective
        if objective is None:
            return score

        # Classification / ordinal style score objective.
        if _looks_like_score_objective(objective):
            out = _objective_call(objective, score, raw_X)
            if not torch.is_tensor(out):
                raise TypeError(f"{name}: objective must return Tensor. Got {type(out)}.")
            return out

        # BoTorch MC multi-output objective.  Since active-learning pointwise
        # score is already output-reduced, treat it as deterministic scalar samples.
        if _is_mc_multi_output_objective(objective):
            pseudo = score
            if pseudo.ndim == expanded_X.ndim - 1:
                pseudo = pseudo.unsqueeze(-1)
            pseudo = pseudo.unsqueeze(0)
            out = _objective_call(objective, pseudo, raw_X)
            if not torch.is_tensor(out):
                raise TypeError(f"{name}: objective must return Tensor. Got {type(out)}.")
            if out.ndim >= 1 and out.shape[0] == 1:
                out = out.squeeze(0)
            if out.ndim == raw_X.ndim and out.shape[-1] == 1:
                out = out.squeeze(-1)
            return out

        # Generic callable: try score-objective style first.
        try:
            out = _objective_call(objective, score, raw_X)
            if torch.is_tensor(out):
                return out
        except Exception:
            pass

        pseudo = score
        if pseudo.ndim == expanded_X.ndim - 1:
            pseudo = pseudo.unsqueeze(-1)
        pseudo = pseudo.unsqueeze(0)
        out = _objective_call(objective, pseudo, raw_X)
        if not torch.is_tensor(out):
            raise TypeError(f"{name}: objective must return Tensor. Got {type(out)}.")
        if out.ndim >= 1 and out.shape[0] == 1:
            out = out.squeeze(0)
        if out.ndim == raw_X.ndim and out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out

    def _aggregate_n_w_if_needed(
        self,
        score: Tensor,
        *,
        q: int,
        context: str,
    ) -> Tensor:
        if self.n_w is None:
            return score

        expected = q * int(self.n_w)
        if score.shape[-1] == q:
            return score
        if score.shape[-1] != expected:
            raise RuntimeError(
                f"{context}: expected last dimension q={q} or q*n_w={expected}, "
                f"got score.shape={tuple(score.shape)}."
            )

        return score.reshape(*score.shape[:-1], q, int(self.n_w)).mean(dim=-1)

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
        score = self._apply_objective_to_score(
            score,
            raw_X=raw_X,
            expanded_X=Xt,
            name=name,
        )

        score = self._aggregate_n_w_if_needed(score, q=q, context=name)
        out = self._reduce_q(score)

        if out.shape == original_batch_shape:
            return out

        while out.ndim > len(original_batch_shape):
            out = out.mean(dim=0)
            if out.shape == original_batch_shape:
                return out

        if out.shape == original_batch_shape:
            return out

        if out.numel() == _safe_prod(original_batch_shape):
            return out.reshape(original_batch_shape)

        raise RuntimeError(
            f"{name}: output shape mismatch. "
            f"Expected {tuple(original_batch_shape)}, got {tuple(out.shape)}."
        )


# ============================================================
# Pointwise active-learning acquisitions
# ============================================================


class qMultiOutputRegressionPosteriorVariance(_MultiOutputRegressionActiveLearningBase):
    """Multi-output regression posterior-variance acquisition."""

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        var, Xt = self._posterior_variance_score(X)
        return self._finalize_pointwise_score(
            var,
            X,
            Xt,
            name="qMultiOutputRegressionPosteriorVariance",
        )


class qMultiOutputRegressionPredictiveEntropy(_MultiOutputRegressionActiveLearningBase):
    """Multi-output regression predictive entropy acquisition.

    Entropy is computed per output under Gaussian marginal approximation and
    then reduced over outputs by ``output_reduction``.
    """

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        _, var, Xt = self._posterior_mean_variance(X, observation_noise=True)
        entropy = 0.5 * torch.log(
            torch.as_tensor(
                2.0 * torch.pi * torch.e,
                device=var.device,
                dtype=var.dtype,
            )
            * var.clamp_min(self.eps)
        )
        return self._finalize_pointwise_score(
            entropy,
            X,
            Xt,
            name="qMultiOutputRegressionPredictiveEntropy",
        )


class qMultiOutputRegressionBALD(_MultiOutputRegressionActiveLearningBase):
    """Multi-output regression BALD / mutual-information acquisition.

    For Gaussian regression with observation noise this computes

        0.5 * log(total_variance / noise_variance)

    using ``posterior(observation_noise=True)`` and
    ``posterior(observation_noise=False)``.  If the model does not support
    noisy posteriors, it falls back to posterior variance by default.
    """

    def __init__(
        self,
        model,
        *,
        fallback_to_variance: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.fallback_to_variance = bool(fallback_to_variance)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        try:
            _, latent_var, Xt = self._posterior_mean_variance(X, observation_noise=False)
            _, total_var, _ = self._posterior_mean_variance(X, observation_noise=True)
            total_var = self._align_pointwise_score_to_X(
                total_var,
                Xt,
                name="qMultiOutputRegressionBALD total variance",
            )
            noise_var = (total_var - latent_var).clamp_min(self.eps)
            score = 0.5 * torch.log(total_var.clamp_min(self.eps) / noise_var)
        except Exception:
            if not self.fallback_to_variance:
                raise
            score, Xt = self._posterior_variance_score(X)

        return self._finalize_pointwise_score(
            score,
            X,
            Xt,
            name="qMultiOutputRegressionBALD",
        )


# ============================================================
# Integrated posterior variance
# ============================================================


class qMultiOutputRegressionNegIntegratedPosteriorVariance(AcquisitionFunction):
    """Multi-output negative integrated posterior variance.

    The fantasy / conditioning calculation follows BoTorch's
    ``qNegIntegratedPosteriorVariance`` directly. It is implemented here rather
    than wrapping a second acquisition instance, because the inner BoTorch
    ``t_batch_mode_transform`` validates its unreduced multi-output tensor before
    this class can aggregate the integration and output dimensions.
    """

    def __init__(
        self,
        model,
        mc_points: Tensor,
        *,
        sampler: Any | None = None,
        objective: Any | None = None,
        posterior_transform: Any | None = None,
        X_pending: Tensor | None = None,
        integration_reduction: ReductionType = "mean",
        output_reduction: OutputReductionType = "mean",
        output_weights: Tensor | Sequence[float] | None = None,
        normalize_output_weights: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model)

        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected NIPV keyword arguments: {unexpected}.")
        if integration_reduction not in ("mean", "sum", "max", "min"):
            raise ValueError(
                "integration_reduction must be one of 'mean', 'sum', 'max', or 'min'."
            )
        if output_reduction not in (
            "mean",
            "sum",
            "max",
            "min",
            "weighted_sum",
            "weighted_mean",
        ):
            raise ValueError(
                "output_reduction must be one of 'mean', 'sum', 'max', 'min', "
                "'weighted_sum', or 'weighted_mean'."
            )

        self.posterior_transform = posterior_transform
        self.objective = objective
        self.integration_reduction = integration_reduction
        self.output_reduction = output_reduction
        self.normalize_output_weights = bool(normalize_output_weights)

        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([1]))
        self.sampler = sampler
        self.X_pending = X_pending
        self.register_buffer("mc_points", mc_points)

        if output_weights is None:
            self.output_weights = None
        else:
            weights = torch.as_tensor(output_weights).reshape(-1)
            self.register_buffer("output_weights", weights.detach().clone())

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        self.X_pending = X_pending

    def _reduce_output_dimension(self, value: Tensor) -> Tensor:
        if value.shape[-1] == 1:
            return value.squeeze(-1)

        if self.output_reduction == "weighted_sum":
            if self.output_weights is None:
                raise ValueError("output_reduction='weighted_sum' requires output_weights.")
            weights = self.output_weights.to(device=value.device, dtype=value.dtype)
            if weights.numel() != value.shape[-1]:
                raise ValueError(
                    f"output_weights length={weights.numel()} does not match "
                    f"output dimension={value.shape[-1]}."
                )
            if self.normalize_output_weights:
                weights = weights / weights.sum().clamp_min(torch.finfo(value.dtype).eps)
            return (value * weights).sum(dim=-1)

        if self.output_reduction == "weighted_mean":
            if self.output_weights is None:
                return value.mean(dim=-1)
            weights = self.output_weights.to(device=value.device, dtype=value.dtype)
            if weights.numel() != value.shape[-1]:
                raise ValueError(
                    f"output_weights length={weights.numel()} does not match "
                    f"output dimension={value.shape[-1]}."
                )
            if self.normalize_output_weights:
                weights = weights / weights.sum().clamp_min(torch.finfo(value.dtype).eps)
            return (value * weights).sum(dim=-1)

        return _reduce(value, dim=-1, mode=self.output_reduction)

    def _finalize_output(self, value: Tensor, X: Tensor) -> Tensor:
        """Reduce non-t-batch dimensions while preserving ``X.shape[:-2]``."""
        target_shape = torch.Size(X.shape[:-2])
        if value.shape == target_shape:
            return value
        if value.ndim == 0:
            return value.expand(target_shape) if len(target_shape) > 0 else value
        if len(target_shape) == 0:
            return value.mean()

        shape = tuple(value.shape)
        target_tuple = tuple(target_shape)
        start = None
        for index in range(len(shape) - len(target_tuple) + 1):
            if shape[index : index + len(target_tuple)] == target_tuple:
                start = index
                break

        if start is not None:
            for _ in range(start):
                value = _reduce(value, dim=0, mode=self.integration_reduction)

            while value.ndim > len(target_shape) + 1:
                value = _reduce(
                    value,
                    dim=len(target_shape),
                    mode=self.integration_reduction,
                )

            if value.ndim == len(target_shape) + 1:
                value = self._reduce_output_dimension(value)

            if value.shape == target_shape:
                return value

        if value.numel() == _safe_prod(target_shape):
            return value.reshape(target_shape)

        if len(target_shape) == 1:
            batch_size = int(target_shape[0])
            matching_dims = [i for i, size in enumerate(value.shape) if int(size) == batch_size]
            if len(matching_dims) == 1:
                value = value.movedim(matching_dims[0], 0)
                while value.ndim > 2:
                    value = _reduce(value, dim=1, mode=self.integration_reduction)
                if value.ndim == 2:
                    value = self._reduce_output_dimension(value)
                if value.shape == target_shape:
                    return value

        raise RuntimeError(
            "qMultiOutputRegressionNegIntegratedPosteriorVariance could not align "
            f"posterior variance shape={tuple(value.shape)} to "
            f"t-batch shape={tuple(target_shape)} for X.shape={tuple(X.shape)}."
        )

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        fantasy_model = self.model.fantasize(X=X, sampler=self.sampler)

        bdims = tuple(1 for _ in X.shape[:-2])
        if int(getattr(self.model, "num_outputs", 1)) > 1:
            mc_points = self.mc_points.view(-1, *bdims, 1, X.size(-1))
        else:
            mc_points = self.mc_points.view(*bdims, -1, X.size(-1))

        with settings.propagate_grads(True):
            posterior = fantasy_model.posterior(
                mc_points,
                posterior_transform=self.posterior_transform,
            )

        neg_variance = posterior.variance.mul(-1.0)
        return self._finalize_output(neg_variance, X)


class qMultiOutputRegressionIntegratedPosteriorVarianceProxy(_MultiOutputRegressionActiveLearningBase):
    """Lightweight integrated-posterior-variance proxy for multi-output regression.

    This is not BoTorch qNegIntegratedPosteriorVariance.  It does not fantasize.
    It scores candidates by how much they cover high-variance reference regions.
    """

    def __init__(
        self,
        model,
        X_ref: Tensor,
        *,
        kernel_lengthscale: float = 0.2,
        normalize_weights: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        if X_ref.ndim != 2:
            raise ValueError(f"X_ref must have shape [n_ref, d]. Got {tuple(X_ref.shape)}.")
        self.register_buffer("X_ref", X_ref.detach().clone())
        self.kernel_lengthscale = float(kernel_lengthscale)
        self.normalize_weights = bool(normalize_weights)

    def _reference_variance(self) -> Tensor:
        _, ref_var, Xt_ref = self._posterior_mean_variance(self.X_ref, observation_noise=False)
        n_ref = int(self.X_ref.shape[-2])
        ref_var = self._aggregate_n_w_if_needed(
            ref_var,
            q=n_ref,
            context="qMultiOutputRegressionIntegratedPosteriorVarianceProxy reference variance",
        )
        if ref_var.shape[-1] != n_ref:
            raise RuntimeError(
                "Reference variance must have last dimension n_ref. "
                f"ref_var.shape={tuple(ref_var.shape)}, n_ref={n_ref}."
            )
        return ref_var

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        raw_X = _ensure_q_batch(X)
        Xt = self._apply_input_transform_for_distance(raw_X)

        ref_var = self._reference_variance()
        X_ref_t = self._reference_to_distance_space(self.X_ref, like=Xt)
        if X_ref_t is None:
            raise RuntimeError("X_ref unexpectedly became None after transform.")
        X_ref_2d = X_ref_t.reshape(-1, X_ref_t.shape[-1])

        if ref_var.ndim > 1:
            while ref_var.ndim > 1:
                ref_var = ref_var.mean(dim=0)

        if ref_var.shape[-1] != X_ref_2d.shape[-2]:
            n_ref = int(self.X_ref.shape[-2])
            if X_ref_2d.shape[-2] % n_ref == 0:
                n_w_ref = X_ref_2d.shape[-2] // n_ref
                X_ref_2d = X_ref_2d.reshape(n_ref, n_w_ref, X_ref_2d.shape[-1]).mean(dim=1)
            if ref_var.shape[-1] != X_ref_2d.shape[-2]:
                raise RuntimeError(
                    "Reference variance / reference point mismatch. "
                    f"ref_var.shape={tuple(ref_var.shape)}, X_ref_2d.shape={tuple(X_ref_2d.shape)}."
                )

        d2 = torch.cdist(Xt.reshape(-1, Xt.shape[-1]), X_ref_2d).pow(2)
        d2 = d2.reshape(*Xt.shape[:-1], X_ref_2d.shape[-2])

        ls2 = max(self.kernel_lengthscale ** 2, self.eps)
        weights = torch.exp(-0.5 * d2 / ls2)
        if self.normalize_weights:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        view_shape = (1,) * (weights.ndim - 1) + (ref_var.shape[-1],)
        score = (weights * ref_var.view(*view_shape)).sum(dim=-1)

        return self._finalize_pointwise_score(
            score,
            raw_X,
            Xt,
            name="qMultiOutputRegressionIntegratedPosteriorVarianceProxy",
        )


__all__ = [
    "qMultiOutputRegressionPredictiveEntropy",
    "qMultiOutputRegressionBALD",
    "qMultiOutputRegressionPosteriorVariance",
    "qMultiOutputRegressionNegIntegratedPosteriorVariance",
    "qMultiOutputRegressionIntegratedPosteriorVarianceProxy",
]
