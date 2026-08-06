from __future__ import annotations

"""Regression active-learning acquisition functions.

Pointwise bochan acquisitions share exact duplicate handling through the
regression active-learning base.  BoTorch-native acquisitions continue to
delegate to BoTorch without an external acquisition wrapper.
"""

from collections.abc import Callable
from typing import Any

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from ._base_common import (
    OutputReductionType,
    ReductionType,
    _ensure_q_batch,
    _objective_call,
    _is_mc_multi_output_objective,
    _looks_like_score_objective,
    _reduce,
    _safe_prod,
)
from ._base_objective import _RegressionObjectiveMixin
from ._base_reference import _RegressionReferenceMixin
from ._base_scoring import _RegressionScoringMixin

try:
    from botorch.acquisition.active_learning import (
        qNegIntegratedPosteriorVariance as _BoTorchQNegIntegratedPosteriorVariance,
    )
except Exception:  # pragma: no cover - depends on BoTorch version
    _BoTorchQNegIntegratedPosteriorVariance = None


class _RegressionActiveLearningBase(
    _RegressionReferenceMixin,
    _RegressionScoringMixin,
    _RegressionObjectiveMixin,
    AcquisitionFunction,
):
    """Base class aligned with classification / ordinal active-learning APIs.

    Args:
        model:
            BoTorch-supported regression model.
        reduction:
            q-batch reduction.  This is intentionally named ``reduction`` to
            match classification / ordinal APIs.
        output_reduction:
            Reduction over output dimension for multi-output regression.
        pending_penalty_weight:
            Weight for avoiding X_pending.
        observed_penalty_weight:
            Weight for avoiding X_observed.
        same_batch_penalty_weight:
            Weight for q-batch diversity penalty.
        exclude_same_batch_duplicates:
            Hard-exclude q-batches containing duplicate candidate points.
        exclude_pending_duplicates:
            Hard-exclude q-batches containing a point already in ``X_pending``.
        exclude_observed_duplicates:
            Hard-exclude q-batches containing a point already in ``X_observed``.
        objective:
            Optional score objective.  Classification / ordinal style score
            objectives receive pointwise scores.  BoTorch MC multi-output
            objectives receive deterministic pseudo-samples.
        n_w:
            Number of input perturbation samples.  If omitted but objective has
            ``n_w``, that value is used.
    """

    def __init__(
        self,
        model,
        *,
        reduction: ReductionType = "mean",
        output_reduction: OutputReductionType = "mean",
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
        exclude_observed_duplicates: bool = False,
        objective: Callable[[Tensor, Tensor | None], Tensor] | None = None,
        n_w: int | None = None,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(model=model)

        if reduction not in ("mean", "sum", "max", "min"):
            raise ValueError("reduction must be one of 'mean', 'sum', 'max', 'min'.")
        if output_reduction not in ("mean", "sum", "max", "min"):
            raise ValueError("output_reduction must be one of 'mean', 'sum', 'max', 'min'.")

        self.reduction = reduction
        self.output_reduction = output_reduction
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
        if self.hard_duplicate_tol < 0.0:
            raise ValueError("hard_duplicate_tol must be non-negative.")
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


class qRegressionPosteriorVariance(_RegressionActiveLearningBase):
    """Regression posterior-variance acquisition.

    This is the standard lightweight uncertainty-sampling acquisition for
    regression.  It maximizes posterior variance.
    """

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        var, Xt = self._posterior_variance_score(X)
        return self._finalize_pointwise_score(
            var,
            X,
            Xt,
            name="qRegressionPosteriorVariance",
        )


class qRegressionPredictiveEntropy(_RegressionActiveLearningBase):
    """Regression predictive-entropy acquisition for Gaussian predictive marginals."""

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        var, Xt = self._posterior_variance_score(X)
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
            name="qRegressionPredictiveEntropy",
        )


class qRegressionBALD(_RegressionActiveLearningBase):
    """Regression BALD / mutual-information acquisition.

    For Gaussian regression with observation noise this computes

        0.5 * log(total_variance / noise_variance)

    using ``posterior(observation_noise=True)`` and
    ``posterior(observation_noise=False)``.  If the model does not support
    noisy posteriors, it falls back to posterior variance.
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
            _, total_var, Xt_total = self._posterior_mean_variance(X, observation_noise=True)

            total_var = self._align_pointwise_score_to_X(
                total_var,
                Xt,
                name="qRegressionBALD total variance",
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
            name="qRegressionBALD",
        )


# ============================================================
# Integrated posterior variance
# ============================================================


class qRegressionNegIntegratedPosteriorVariance(AcquisitionFunction):
    """True BoTorch qNegIntegratedPosteriorVariance wrapper.

    This delegates to BoTorch's implementation and therefore requires a model
    that supports the operations expected by BoTorch, especially fantasize().
    Use qRegressionIntegratedPosteriorVarianceProxy for DeepGP / custom models
    that do not support fantasize().
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
        **kwargs: Any,
    ) -> None:
        if _BoTorchQNegIntegratedPosteriorVariance is None:
            raise ImportError(
                "botorch.acquisition.active_learning.qNegIntegratedPosteriorVariance "
                "is not available in this BoTorch version."
            )

        super().__init__(model=model)

        init_kwargs: dict[str, Any] = {
            "model": model,
            "mc_points": mc_points,
        }
        if sampler is not None:
            init_kwargs["sampler"] = sampler
        if objective is not None:
            init_kwargs["objective"] = objective
        if posterior_transform is not None:
            init_kwargs["posterior_transform"] = posterior_transform
        if X_pending is not None:
            init_kwargs["X_pending"] = X_pending
        init_kwargs.update(kwargs)

        # BoTorch signatures differ slightly across versions.  Try the most
        # complete call first, then progressively remove optional keywords.
        try:
            self.acqf = _BoTorchQNegIntegratedPosteriorVariance(**init_kwargs)
        except TypeError:
            for key in ("X_pending", "posterior_transform", "objective", "sampler"):
                init_kwargs.pop(key, None)
                try:
                    self.acqf = _BoTorchQNegIntegratedPosteriorVariance(**init_kwargs)
                    break
                except TypeError:
                    continue
            else:
                raise

    def set_X_pending(self, X_pending: Tensor | None = None) -> None:
        if hasattr(self.acqf, "set_X_pending"):
            self.acqf.set_X_pending(X_pending)
        else:
            self.acqf.X_pending = X_pending

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        return self.acqf(X)


class qRegressionIntegratedPosteriorVarianceProxy(_RegressionActiveLearningBase):
    """Lightweight integrated-posterior-variance proxy.

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
            context="qRegressionIntegratedPosteriorVarianceProxy reference variance",
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
            # If reference variance has extra leading dimensions, average them.
            while ref_var.ndim > 1:
                ref_var = ref_var.mean(dim=0)

        if ref_var.shape[-1] != X_ref_2d.shape[-2]:
            # InputPerturbation may expand X_ref in distance space.  Collapse
            # repeated reference points back to nominal reference count if possible.
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
            name="qRegressionIntegratedPosteriorVarianceProxy",
        )

__all__ = [
    "qRegressionPredictiveEntropy",
    "qRegressionBALD",
    "qRegressionPosteriorVariance",
    "qRegressionNegIntegratedPosteriorVariance",
    "qRegressionIntegratedPosteriorVarianceProxy",
]
