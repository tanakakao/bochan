from __future__ import annotations

"""Common active-learning acquisitions for non-Gaussian regression models.

These classes are shared across Poisson, Beta, Gamma, and Negative Binomial GP
wrappers.  Standard BO acquisitions such as qEI / qNEI / qUCB are intentionally
not reimplemented here because BoTorch can use the model's response-scale
``posterior`` with an appropriate objective.
"""

from typing import Any, Literal

from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bochan.acquisition.regression.active_learning.single_output import _RegressionActiveLearningBase

from .._stats import non_gaussian_response_stats


ScoreType = Literal[
    "response_mean_variance",
    "expected_observation_variance",
    "total_observation_variance",
    "expected_observation_entropy",
    "predictive_entropy_proxy",
    "bald_proxy",
]


class _NonGaussianActiveLearningBase(_RegressionActiveLearningBase):
    """Base class for shared non-Gaussian active-learning acquisitions."""

    def __init__(
        self,
        model,
        *,
        num_samples: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.num_samples = int(num_samples)
        if self.num_samples <= 1:
            raise ValueError("num_samples must be greater than 1.")

    def _score(self, X: Tensor, score_type: ScoreType):
        Xq = self._ensure_candidate_X(X)
        stats = non_gaussian_response_stats(
            self.model,
            Xq,
            num_samples=self.num_samples,
            eps=self.eps,
        )
        score = stats[score_type]
        Xt = self._apply_input_transform_for_distance(Xq)
        return score, Xt

    @staticmethod
    def _ensure_candidate_X(X: Tensor) -> Tensor:
        if X.ndim == 1:
            return X.view(1, 1, -1)
        if X.ndim == 2:
            return X.unsqueeze(0)
        return X


class qNonGaussianResponseMeanVariance(_NonGaussianActiveLearningBase):
    """Variance of the response mean induced by latent GP uncertainty.

    This is the closest non-Gaussian analogue of posterior-mean uncertainty.  It
    ignores irreducible observation noise and is usually the safest first choice
    for active learning with Poisson / Beta / Gamma / Negative Binomial models.
    """

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        score, Xt = self._score(X, "response_mean_variance")
        return self._finalize_pointwise_score(
            score,
            X,
            Xt,
            name="qNonGaussianResponseMeanVariance",
        )


class qNonGaussianExpectedObservationVariance(_NonGaussianActiveLearningBase):
    """Expected conditional observation variance under the latent posterior.

    This emphasizes intrinsically noisy regions of the observation distribution.
    It is useful for diagnostics, but for sample-efficient learning it can over-
    select high-noise regions.
    """

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        score, Xt = self._score(X, "expected_observation_variance")
        return self._finalize_pointwise_score(
            score,
            X,
            Xt,
            name="qNonGaussianExpectedObservationVariance",
        )


class qNonGaussianTotalObservationVariance(_NonGaussianActiveLearningBase):
    """Total predictive observation variance proxy.

    Computes Var_f[E[y|f]] + E_f[Var[y|f]].  This is response-scale predictive
    variance including observation noise.
    """

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        score, Xt = self._score(X, "total_observation_variance")
        return self._finalize_pointwise_score(
            score,
            X,
            Xt,
            name="qNonGaussianTotalObservationVariance",
        )


class qNonGaussianExpectedObservationEntropy(_NonGaussianActiveLearningBase):
    """Expected entropy of the observation distribution E_f[H[p(y|f)]]."""

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        score, Xt = self._score(X, "expected_observation_entropy")
        return self._finalize_pointwise_score(
            score,
            X,
            Xt,
            name="qNonGaussianExpectedObservationEntropy",
        )


class qNonGaussianPredictiveEntropyProxy(_NonGaussianActiveLearningBase):
    """Gaussian-moment proxy for non-Gaussian predictive entropy.

    Exact entropy of the latent-mixture predictive distribution is generally not
    available in closed form.  This uses the total predictive variance as a
    Gaussian moment-matched proxy.
    """

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        score, Xt = self._score(X, "predictive_entropy_proxy")
        return self._finalize_pointwise_score(
            score,
            X,
            Xt,
            name="qNonGaussianPredictiveEntropyProxy",
        )


class qNonGaussianBALDProxy(_NonGaussianActiveLearningBase):
    """Moment-matched BALD / mutual-information proxy for non-Gaussian models.

    Uses ``0.5 * log(total_observation_variance / expected_observation_variance)``.
    This targets reducible uncertainty in the response distribution and avoids
    reimplementing distribution-specific exact mixture entropies.
    """

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        score, Xt = self._score(X, "bald_proxy")
        return self._finalize_pointwise_score(
            score,
            X,
            Xt,
            name="qNonGaussianBALDProxy",
        )


# Alias following the existing regression / binary / ordinal naming style.
qNonGaussianPosteriorVariance = qNonGaussianResponseMeanVariance


__all__ = [
    "qNonGaussianResponseMeanVariance",
    "qNonGaussianPosteriorVariance",
    "qNonGaussianExpectedObservationVariance",
    "qNonGaussianTotalObservationVariance",
    "qNonGaussianExpectedObservationEntropy",
    "qNonGaussianPredictiveEntropyProxy",
    "qNonGaussianBALDProxy",
]
