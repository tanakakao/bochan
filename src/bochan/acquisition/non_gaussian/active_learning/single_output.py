"""Active learning for non-Gaussian response posteriors."""
from __future__ import annotations

from typing import Any, Literal
import torch
from botorch.sampling.base import MCSampler
from botorch.sampling.get_sampler import get_sampler
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bochan.acquisition.regression.active_learning.single_output import _RegressionActiveLearningBase
from .._stats import ensure_q_batch, non_gaussian_response_stats, safe_logdet


class _NonGaussianActiveLearningBase(_RegressionActiveLearningBase):
    """Common fixed-base-sample response statistics acquisition base."""
    def __init__(self, model, *, sampler: MCSampler | None = None,
                 sample_shape: torch.Size = torch.Size([128]), seed: int | None = None,
                 num_samples: int | None = None, **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self.sampler = sampler
        self.sample_shape = torch.Size([num_samples]) if num_samples is not None else sample_shape
        self.seed = seed

    def _stats(self, X: Tensor):
        Xq = ensure_q_batch(X)
        if self.sampler is None and not getattr(self.model, "is_non_gaussian_model_list", False):
            self.sampler = get_sampler(self.model.posterior(Xq, observation_noise=False), self.sample_shape, seed=self.seed)
        return non_gaussian_response_stats(self.model, Xq, sampler=self.sampler,
                                           sample_shape=self.sample_shape, seed=self.seed, eps=self.eps)

    def _pointwise(self, X: Tensor, field: str) -> Tensor:
        Xq = ensure_q_batch(X)
        value = getattr(self._stats(Xq), field)
        Xt = self._apply_input_transform_for_distance(Xq)
        return self._finalize_pointwise_score(value, X, Xt, name=type(self).__name__)


def _field_class(name: str, field: str, doc: str):
    """Create a documented pointwise acquisition with a shared implementation."""
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate the response statistic at raw-space candidates."""
        return self._pointwise(X, field)
    return type(name, (_NonGaussianActiveLearningBase,), {"__doc__": doc, "forward": t_batch_mode_transform()(forward)})


qNonGaussianResponseMeanVariance = _field_class("qNonGaussianResponseMeanVariance", "response_mean_variance", """Evaluate Var_f[E(Y|f)], the recommended standard non-Gaussian AL score.""")
qNonGaussianExpectedObservationVariance = _field_class("qNonGaussianExpectedObservationVariance", "expected_observation_variance", """Evaluate aleatoric variance for noise diagnostics and high-variance-region exploration.""")
qNonGaussianTotalObservationVariance = _field_class("qNonGaussianTotalObservationVariance", "total_observation_variance", """Evaluate epistemic plus expected observation variance.""")
qNonGaussianExpectedObservationEntropy = _field_class("qNonGaussianExpectedObservationEntropy", "expected_observation_entropy", """Evaluate E_f[H(Y|f)], with an identified Gaussian entropy fallback only when unavailable.""")
qNonGaussianPredictiveEntropyProxy = _field_class("qNonGaussianPredictiveEntropyProxy", "predictive_entropy_proxy", """Moment-matched Gaussian entropy proxy based on total observation variance.""")


class qNonGaussianBALDProxy(_NonGaussianActiveLearningBase):
    """Moment proxy for BALD using an entropy difference or variance ratio."""
    def __init__(self, model, *, method: Literal["variance_ratio", "entropy_difference"] = "entropy_difference", **kwargs: Any) -> None:
        if method not in ("variance_ratio", "entropy_difference"):
            raise ValueError("method must be 'variance_ratio' or 'entropy_difference'.")
        super().__init__(model, **kwargs); self.method = method

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate the selected non-negative BALD moment proxy."""
        field = "bald_variance_ratio_proxy" if self.method == "variance_ratio" else "bald_entropy_difference_proxy"
        return self._pointwise(X, field)


class qNonGaussianIntegratedResponseMeanVarianceProxy(_NonGaussianActiveLearningBase):
    """Reference-set aggregation of response-mean variance, not fantasy reduction."""
    def __init__(self, model, *, mc_points: Tensor, integration_reduction: Literal["mean", "sum"] = "mean", **kwargs: Any) -> None:
        if integration_reduction not in ("mean", "sum"):
            raise ValueError("integration_reduction must be 'mean' or 'sum'.")
        super().__init__(model, **kwargs); self.register_buffer("mc_points", mc_points); self.integration_reduction = integration_reduction

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Return the integrated proxy, broadcast over candidate batch shape."""
        value = self._stats(self.mc_points).response_mean_variance
        value = value.mean() if self.integration_reduction == "mean" else value.sum()
        return value.expand(X.shape[:-2])


class qNonGaussianJointBALDProxy(_NonGaussianActiveLearningBase):
    """Joint moment BALD proxy 0.5 log det(I + D^-1/2 Sigma D^-1/2)."""
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Preserve q/output sample covariance in a joint information score."""
        stats = self._stats(X); samples = stats.response_mean_samples
        s = samples.shape[0]; flat = samples.reshape(s, *samples.shape[1:-2], -1)
        centered = flat - flat.mean(0)
        cov = torch.einsum("s...i,s...j->...ij", centered, centered) / max(s - 1, 1)
        noise = stats.expected_observation_variance.reshape(*stats.expected_observation_variance.shape[:-2], -1).clamp_min(self.eps)
        scaled = cov / torch.sqrt(noise.unsqueeze(-1) * noise.unsqueeze(-2))
        eye = torch.eye(scaled.shape[-1], device=X.device, dtype=X.dtype)
        return 0.5 * safe_logdet(eye + scaled, jitter=max(self.eps, 1e-6))


class qNonGaussianGreedyJointBALDProxy(qNonGaussianJointBALDProxy):
    """Greedy joint BALD proxy represented by the joint batch information gain."""


qNonGaussianPosteriorVariance = qNonGaussianResponseMeanVariance
qNonGaussianVariance = qNonGaussianResponseMeanVariance

__all__ = [n for n in globals() if n.startswith("qNonGaussian")]
