from __future__ import annotations

"""Common level-set acquisitions for non-Gaussian regression models.

These acquisitions operate on the response mean scale and share one
implementation across Poisson, Beta, Gamma, and Negative Binomial wrappers.
"""

from typing import Any, Literal, Optional

import torch
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from bochan.acquisition.regression.levelset_estimation.single_output import _RegressionLevelSetBase

from ._stats import non_gaussian_response_stats, safe_normal_cdf


class _NonGaussianLevelSetBase(_RegressionLevelSetBase):
    """Base for response-scale non-Gaussian level-set acquisitions."""

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

    def _response_mean_variance(self, X: Tensor):
        Xq = X if X.ndim > 2 else X.unsqueeze(0)
        stats = non_gaussian_response_stats(
            self.model,
            Xq,
            num_samples=self.num_samples,
            eps=self.eps,
        )
        Xt = self._apply_input_transform_for_distance(Xq)
        mean = stats["response_mean"]
        var = stats["response_mean_variance"].clamp_min(self.eps)
        return mean, var, Xt


class qNonGaussianStraddle(_NonGaussianLevelSetBase):
    """Response-scale non-Gaussian straddle acquisition.

    score(x) = beta * std_response_mean(x) - |E[y|x] - threshold|
    """

    def __init__(
        self,
        model,
        *,
        beta: float | Tensor = 1.96,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.register_buffer("beta", torch.as_tensor(beta))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        mean, var, Xt = self._response_mean_variance(X)
        threshold = self.threshold.to(device=mean.device, dtype=mean.dtype)
        beta = self.beta.to(device=mean.device, dtype=mean.dtype)
        score = beta * var.sqrt() - (mean - threshold).abs()
        return self._finalize_pointwise_score(score, X, Xt, name="qNonGaussianStraddle")


class qNonGaussianBoundaryVariance(_NonGaussianLevelSetBase):
    """Boundary-weighted response-mean variance acquisition."""

    def __init__(
        self,
        model,
        *,
        tau: float | Tensor = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.register_buffer("tau", torch.as_tensor(tau))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        mean, var, Xt = self._response_mean_variance(X)
        threshold = self.threshold.to(device=mean.device, dtype=mean.dtype)
        tau = self.tau.to(device=mean.device, dtype=mean.dtype).clamp_min(self.eps)
        boundary_weight = torch.exp(-0.5 * ((mean - threshold) / tau).pow(2))
        score = var * boundary_weight
        return self._finalize_pointwise_score(score, X, Xt, name="qNonGaussianBoundaryVariance")


class qNonGaussianICU(_NonGaussianLevelSetBase):
    """Integrated-contour-uncertainty style proxy for non-Gaussian regression."""

    def __init__(
        self,
        model,
        *,
        bandwidth: Optional[float | Tensor] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.bandwidth = None if bandwidth is None else torch.as_tensor(bandwidth)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        mean, var, Xt = self._response_mean_variance(X)
        std = var.sqrt().clamp_min(self.eps)
        threshold = self.threshold.to(device=mean.device, dtype=mean.dtype)
        if self.bandwidth is None:
            bw = std
        else:
            bw = self.bandwidth.to(device=mean.device, dtype=mean.dtype).clamp_min(self.eps)
        z = (mean - threshold) / bw
        score = torch.exp(-0.5 * z.pow(2)) * std
        return self._finalize_pointwise_score(score, X, Xt, name="qNonGaussianICU")


class qNonGaussianProbabilityOfExceedance(_NonGaussianLevelSetBase):
    """Response-mean probability of exceedance proxy.

    Modes:
        - ``above``:    P(E[y|x] >= threshold)
        - ``below``:    P(E[y|x] <= threshold)
        - ``interval``: P(lower <= E[y|x] <= upper)
    """

    def __init__(
        self,
        model,
        *,
        mode: Literal["above", "below", "interval"] = "above",
        lower: Optional[float | Tensor] = None,
        upper: Optional[float | Tensor] = None,
        temperature: Optional[float | Tensor] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        if mode not in ("above", "below", "interval"):
            raise ValueError("mode must be 'above', 'below', or 'interval'.")
        self.mode = mode
        self.lower = None if lower is None else torch.as_tensor(lower)
        self.upper = None if upper is None else torch.as_tensor(upper)
        self.temperature = None if temperature is None else torch.as_tensor(temperature)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        mean, var, Xt = self._response_mean_variance(X)
        std = var.sqrt().clamp_min(self.eps)
        threshold = self.threshold.to(device=mean.device, dtype=mean.dtype)

        if self.temperature is not None:
            temp = self.temperature.to(device=mean.device, dtype=mean.dtype).clamp_min(self.eps)
            if self.mode == "above":
                score = torch.sigmoid((mean - threshold) / temp)
            elif self.mode == "below":
                score = torch.sigmoid((threshold - mean) / temp)
            else:
                lo = self.lower.to(device=mean.device, dtype=mean.dtype) if self.lower is not None else threshold
                hi = self.upper.to(device=mean.device, dtype=mean.dtype) if self.upper is not None else threshold
                score = torch.sigmoid((mean - lo) / temp) * torch.sigmoid((hi - mean) / temp)
        else:
            if self.mode == "above":
                score = safe_normal_cdf((mean - threshold) / std)
            elif self.mode == "below":
                score = safe_normal_cdf((threshold - mean) / std)
            else:
                if self.lower is None or self.upper is None:
                    raise ValueError("lower and upper must be provided when mode='interval'.")
                lo = self.lower.to(device=mean.device, dtype=mean.dtype)
                hi = self.upper.to(device=mean.device, dtype=mean.dtype)
                score = safe_normal_cdf((hi - mean) / std) - safe_normal_cdf((lo - mean) / std)

        return self._finalize_pointwise_score(
            score.clamp_min(0.0),
            X,
            Xt,
            name="qNonGaussianProbabilityOfExceedance",
        )


__all__ = [
    "qNonGaussianStraddle",
    "qNonGaussianBoundaryVariance",
    "qNonGaussianICU",
    "qNonGaussianProbabilityOfExceedance",
]
