"""InputPerturbation-safe public regression level-set acquisitions.

The underlying level-set implementations intentionally use transformed inputs for
soft distance penalties.  BoTorch one-to-many input transforms can therefore
produce two valid posterior contracts:

- a posterior that still exposes ``q * n_w`` pointwise moments; or
- a wrapper posterior that has already aggregated perturbations back to nominal
  ``q`` points.

The public classes in this module preserve the first contract and detect the
second.  When posterior moments are already nominal, distance inputs use only
preprocessing transforms (for example normalization) and skip eval-only
one-to-many expansion.  This keeps posterior and distance geometry aligned
without disabling InputPerturbation for models that genuinely expose expanded
posterior rows.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .multi_output import (
    MultiOutputRegressionLevelSetScoreObjective as _MultiOutputRegressionLevelSetScoreObjective,
)
from .multi_output import (
    qMultiOutputRegressionBoundaryVariance as _qMultiOutputRegressionBoundaryVariance,
)
from .multi_output import qMultiOutputRegressionICU as _qMultiOutputRegressionICU
from .multi_output import (
    qMultiOutputRegressionJointStraddle as _qMultiOutputRegressionJointStraddle,
)
from .multi_output import (
    qMultiOutputRegressionProbabilityOfExceedance as _qMultiOutputRegressionProbabilityOfExceedance,
)
from .multi_output import qMultiOutputRegressionStraddle as _qMultiOutputRegressionStraddle
from .single_output import (
    RegressionLevelSetScoreObjective as _RegressionLevelSetScoreObjective,
)
from .single_output import _ensure_q_batch, _extract_covariance_matrix, _safe_prod
from .single_output import qRegressionBoundaryVariance as _qRegressionBoundaryVariance
from .single_output import qRegressionICU as _qRegressionICU
from .single_output import qRegressionJointStraddle as _qRegressionJointStraddle
from .single_output import (
    qRegressionProbabilityOfExceedance as _qRegressionProbabilityOfExceedance,
)
from .single_output import qRegressionStraddle as _qRegressionStraddle


def _unwrap_transform_result(value: Tensor | tuple[Tensor, ...]) -> Tensor:
    """Return the tensor component of a BoTorch input-transform result."""

    return value[0] if isinstance(value, tuple) else value


def _posterior_matches_nominal_q(value: Any, X: Tensor) -> bool:
    """Return whether posterior values retain the raw batch/q dimensions.

    Leading ensemble / MCMC axes are allowed.  The posterior may either expose
    scalar values with no explicit output axis or the ordinary ``... x q x m``
    layout.
    """

    if not torch.is_tensor(value):
        return False
    Xq = _ensure_q_batch(X)
    nominal_prefix = tuple(int(size) for size in Xq.shape[:-1])
    shape = tuple(int(size) for size in value.shape)
    prefix_ndim = len(nominal_prefix)

    if len(shape) >= prefix_ndim + 1:
        candidate_prefix = shape[-(prefix_ndim + 1) : -1]
        if candidate_prefix == nominal_prefix:
            return True
    if len(shape) >= prefix_ndim:
        return shape[-prefix_ndim:] == nominal_prefix
    return False


def _preprocess_distance_inputs(model: Any, X: Tensor) -> Tensor:
    """Apply one-to-one preprocessing while skipping eval-only perturbations."""

    Xq = _ensure_q_batch(X)
    candidates = [model]
    models = getattr(model, "models", None)
    if models is not None and len(models) > 0:
        candidates.append(models[0])

    for candidate in candidates:
        input_transform = getattr(candidate, "input_transform", None)
        if input_transform is None:
            continue
        preprocess = getattr(input_transform, "preprocess_transform", None)
        if not callable(preprocess):
            continue
        try:
            transformed = _unwrap_transform_result(preprocess(Xq))
        except Exception:
            continue
        transformed = _ensure_q_batch(transformed)
        if transformed.shape[:-1] == Xq.shape[:-1]:
            return transformed
    return Xq


def _distance_inputs_aligned_to_posterior(
    owner: Any,
    X: Tensor,
    posterior_mean: Tensor,
) -> Tensor:
    """Return distance-space inputs with the same q contract as the posterior."""

    Xq = _ensure_q_batch(X)
    Xt = owner._apply_input_transform_for_distance(Xq)
    Xt = _ensure_q_batch(Xt)

    n_w = getattr(owner, "n_w", None)
    try:
        n_w = None if n_w is None else int(n_w)
    except (TypeError, ValueError):
        n_w = None
    if n_w is None or n_w <= 1:
        return Xt

    same_batch_shape = Xt.shape[:-2] == Xq.shape[:-2]
    expected_expanded_q = int(Xq.shape[-2]) * n_w
    transformed_is_expanded = (
        same_batch_shape and int(Xt.shape[-2]) == expected_expanded_q
    )
    if not transformed_is_expanded:
        return Xt

    if not _posterior_matches_nominal_q(posterior_mean, Xq):
        return Xt

    # Hybrid / wrapper posteriors may already aggregate q*n_w evaluations to q.
    # In that case the perturbation replicas are not distinct posterior points,
    # so distance penalties must be evaluated on nominal candidates as well.
    return _preprocess_distance_inputs(owner.model, Xq)


class RegressionLevelSetScoreObjective(_RegressionLevelSetScoreObjective):
    """Score objective that accepts already-aggregated nominal q scores."""

    @staticmethod
    def _is_aggregated_score(score: Tensor, X: Tensor | None) -> bool:
        if _RegressionLevelSetScoreObjective._is_aggregated_score(score, X):
            return True
        if X is None or score.ndim == 0:
            return False
        Xq = _ensure_q_batch(X)
        return int(score.shape[-1]) == int(Xq.shape[-2])


class MultiOutputRegressionLevelSetScoreObjective(
    _MultiOutputRegressionLevelSetScoreObjective
):
    """Multi-output score objective accepting nominal q aggregate scores."""

    @staticmethod
    def _is_aggregated_score(score: Tensor, X: Tensor | None) -> bool:
        if _MultiOutputRegressionLevelSetScoreObjective._is_aggregated_score(
            score,
            X,
        ):
            return True
        if X is None or score.ndim == 0:
            return False
        Xq = _ensure_q_batch(X)
        return int(score.shape[-1]) == int(Xq.shape[-2])


class _SingleOutputInputPerturbationShapeMixin:
    """Align single-output LSE posterior and distance q dimensions."""

    def _posterior_mean_variance(self, X: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        Xq = _ensure_q_batch(X)
        self._prepare_eval()

        posterior = self.model.posterior(Xq, observation_noise=False)
        Xt = _distance_inputs_aligned_to_posterior(
            self,
            Xq,
            posterior.mean,
        )

        mean = self._reduce_outputs_if_needed(
            posterior.mean,
            Xt,
            name="posterior.mean",
        )
        var = self._reduce_outputs_if_needed(
            posterior.variance,
            Xt,
            name="posterior.variance",
        ).clamp_min(self.eps)
        mean = self._align_pointwise_score_to_X(mean, Xt, name="posterior.mean")
        var = self._align_pointwise_score_to_X(
            var,
            Xt,
            name="posterior.variance",
        )
        return mean, var, Xt

    def _posterior_covariance(self, X: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return covariance using the same nominal/expanded q contract."""

        Xq = _ensure_q_batch(X)
        self._prepare_eval()

        posterior = self.model.posterior(Xq, observation_noise=False)
        Xt = _distance_inputs_aligned_to_posterior(
            self,
            Xq,
            posterior.mean,
        )
        mean = self._reduce_outputs_if_needed(
            posterior.mean,
            Xt,
            name="posterior.mean",
        )
        mean = self._align_pointwise_score_to_X(mean, Xt, name="posterior.mean")

        q_like = int(Xt.shape[-2])
        target_covar_shape = torch.Size(Xt.shape[:-2]) + torch.Size(
            [q_like, q_like]
        )
        covar = _extract_covariance_matrix(posterior)
        if covar is not None:
            while covar.ndim > len(target_covar_shape):
                covar = covar.mean(dim=0)
                if covar.shape == target_covar_shape:
                    break
            if covar.shape != target_covar_shape:
                if covar.numel() == _safe_prod(target_covar_shape):
                    covar = covar.reshape(target_covar_shape)
                else:
                    covar = None

        if covar is None:
            var = self._reduce_outputs_if_needed(
                posterior.variance,
                Xt,
                name="posterior.variance",
            )
            var = self._align_pointwise_score_to_X(
                var,
                Xt,
                name="posterior.variance",
            )
            covar = torch.diag_embed(var.clamp_min(self.eps))

        covar = 0.5 * (covar + covar.transpose(-1, -2))
        return mean, covar, Xt


class _MultiOutputInputPerturbationShapeMixin:
    """Align multi-output LSE posterior and distance q dimensions."""

    def _posterior_mean_variance_outputs(
        self,
        X: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        Xq = _ensure_q_batch(X)
        self._prepare_eval()

        posterior = self.model.posterior(Xq, observation_noise=False)
        Xt = _distance_inputs_aligned_to_posterior(
            self,
            Xq,
            posterior.mean,
        )
        mean = self._align_output_tensor_to_X(
            posterior.mean,
            Xt,
            name="posterior.mean",
        )
        var = self._align_output_tensor_to_X(
            posterior.variance,
            Xt,
            name="posterior.variance",
        ).clamp_min(self.eps)
        return mean, var, Xt


class qRegressionStraddle(
    _SingleOutputInputPerturbationShapeMixin,
    _qRegressionStraddle,
):
    """InputPerturbation-safe regression Straddle."""


class qRegressionJointStraddle(
    _SingleOutputInputPerturbationShapeMixin,
    _qRegressionJointStraddle,
):
    """InputPerturbation-safe joint regression Straddle."""


class qRegressionICU(
    _SingleOutputInputPerturbationShapeMixin,
    _qRegressionICU,
):
    """InputPerturbation-safe regression ICU."""


class qRegressionBoundaryVariance(
    _SingleOutputInputPerturbationShapeMixin,
    _qRegressionBoundaryVariance,
):
    """InputPerturbation-safe regression Boundary Variance."""


class qRegressionProbabilityOfExceedance(
    _SingleOutputInputPerturbationShapeMixin,
    _qRegressionProbabilityOfExceedance,
):
    """InputPerturbation-safe regression probability of exceedance."""


class qMultiOutputRegressionStraddle(
    _MultiOutputInputPerturbationShapeMixin,
    _qMultiOutputRegressionStraddle,
):
    """InputPerturbation-safe multi-output regression Straddle."""


class qMultiOutputRegressionJointStraddle(
    _MultiOutputInputPerturbationShapeMixin,
    _qMultiOutputRegressionJointStraddle,
):
    """InputPerturbation-safe multi-output joint Straddle."""


class qMultiOutputRegressionICU(
    _MultiOutputInputPerturbationShapeMixin,
    _qMultiOutputRegressionICU,
):
    """InputPerturbation-safe multi-output regression ICU."""


class qMultiOutputRegressionBoundaryVariance(
    _MultiOutputInputPerturbationShapeMixin,
    _qMultiOutputRegressionBoundaryVariance,
):
    """InputPerturbation-safe multi-output Boundary Variance."""


class qMultiOutputRegressionProbabilityOfExceedance(
    _MultiOutputInputPerturbationShapeMixin,
    _qMultiOutputRegressionProbabilityOfExceedance,
):
    """InputPerturbation-safe multi-output probability of exceedance."""


__all__ = [
    "RegressionLevelSetScoreObjective",
    "MultiOutputRegressionLevelSetScoreObjective",
    "qRegressionStraddle",
    "qRegressionJointStraddle",
    "qRegressionICU",
    "qRegressionBoundaryVariance",
    "qRegressionProbabilityOfExceedance",
    "qMultiOutputRegressionStraddle",
    "qMultiOutputRegressionJointStraddle",
    "qMultiOutputRegressionICU",
    "qMultiOutputRegressionBoundaryVariance",
    "qMultiOutputRegressionProbabilityOfExceedance",
]
