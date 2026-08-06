"""Pointwise regression active-learning acquisitions."""

from __future__ import annotations

from typing import Any

import torch
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from ._base import _RegressionActiveLearningBase


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
            _, total_var, _ = self._posterior_mean_variance(X, observation_noise=True)

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
