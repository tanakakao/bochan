from __future__ import annotations

from typing import Any, Literal, Optional, Sequence

import torch
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from .multi_output import (
    BoundaryMode,
    MultiOutputRegressionLevelSetScoreObjective,
    ProbabilityMode,
    _MultiOutputRegressionLevelSetBase,
    _ensure_q_batch,
    _safe_logdet,
    _safe_normal_cdf,
)

VarianceSource = Literal["latent", "total", "noise"]
NoiseCombineType = Literal["subtract", "multiply", "none"]
NoiseWeightMode = Literal["inverse_linear", "inverse_sqrt", "exp", "none"]


class HeteroMultiOutputRegressionLevelSetScoreObjective(
    MultiOutputRegressionLevelSetScoreObjective
):
    """Score objective for heteroscedastic multi-output regression LSE."""


class _HeteroMultiOutputRegressionLevelSetBase(_MultiOutputRegressionLevelSetBase):
    """Noise-aware multi-output LSE sharing the standard duplicate-control base."""

    def __init__(
        self,
        model: Model,
        *,
        variance_source: VarianceSource = "latent",
        noise_penalty: Optional[float] = None,
        noise_penalty_lambda: float = 1.0,
        noise_mode: NoiseWeightMode = "inverse_linear",
        noise_combine: NoiseCombineType = "subtract",
        noise_min_weight: float = 0.0,
        **kwargs: Any,
    ) -> None:
        if variance_source not in ("latent", "total", "noise"):
            raise ValueError("variance_source must be 'latent', 'total', or 'noise'.")
        if noise_mode not in ("inverse_linear", "inverse_sqrt", "exp", "none"):
            raise ValueError("noise_mode must be 'inverse_linear', 'inverse_sqrt', 'exp', or 'none'.")
        if noise_combine not in ("subtract", "multiply", "none"):
            raise ValueError("noise_combine must be 'subtract', 'multiply', or 'none'.")

        super().__init__(model=model, **kwargs)
        self.variance_source = variance_source
        self.noise_penalty_lambda = float(
            noise_penalty_lambda if noise_penalty is None else noise_penalty
        )
        self.noise_mode = noise_mode
        self.noise_combine = noise_combine
        self.noise_min_weight = float(noise_min_weight)

    def _posterior_mean_variance_outputs(
        self,
        X: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return mean, latent/total/noise variance per output and transformed X."""
        Xq = _ensure_q_batch(X)
        self._prepare_eval()

        try:
            post_latent = self.model.posterior(Xq, observation_noise=False)
            post_total = self.model.posterior(Xq, observation_noise=True)
        except Exception:
            post_latent = self.model.posterior(Xq)
            post_total = post_latent

        Xt = self._apply_input_transform_for_distance(Xq)
        mean = self._align_output_tensor_to_X(
            post_latent.mean,
            Xt,
            name="posterior.mean",
        )
        latent_var = self._align_output_tensor_to_X(
            post_latent.variance,
            Xt,
            name="latent variance",
        ).clamp_min(self.eps)
        total_var = self._align_output_tensor_to_X(
            post_total.variance,
            Xt,
            name="total variance",
        ).clamp_min(self.eps)
        noise_var = (total_var - latent_var).clamp_min(self.eps)

        noise_fn = getattr(self.model, "predict_noise_var", None)
        if callable(noise_fn):
            try:
                noise_raw = noise_fn(Xq)
                noise_var = self._align_output_tensor_to_X(
                    noise_raw,
                    Xt,
                    name="predict_noise_var",
                ).clamp_min(self.eps)
                total_var = (latent_var + noise_var).clamp_min(self.eps)
            except Exception:
                pass

        return mean, latent_var, total_var, noise_var, Xt

    def _select_variance_outputs(
        self,
        latent_var: Tensor,
        total_var: Tensor,
        noise_var: Tensor,
    ) -> Tensor:
        if self.variance_source == "latent":
            return latent_var
        if self.variance_source == "total":
            return total_var
        if self.variance_source == "noise":
            return noise_var
        raise ValueError(f"Unknown variance_source: {self.variance_source!r}.")

    def _noise_weight_outputs(self, noise_var: Tensor) -> Tensor:
        if self.noise_mode == "none":
            return torch.ones_like(noise_var)
        lam = max(self.noise_penalty_lambda, 0.0)
        if self.noise_mode == "inverse_linear":
            weight = 1.0 / (1.0 + lam * noise_var)
        elif self.noise_mode == "inverse_sqrt":
            weight = 1.0 / (1.0 + lam * noise_var.clamp_min(self.eps).sqrt())
        elif self.noise_mode == "exp":
            weight = torch.exp(-lam * noise_var)
        else:
            raise ValueError(f"Unknown noise_mode: {self.noise_mode!r}.")
        return weight.clamp_min(self.noise_min_weight)

    def _apply_noise_to_output_score(
        self,
        score_outputs: Tensor,
        noise_var: Tensor,
    ) -> Tensor:
        if self.noise_combine == "none":
            return score_outputs
        if self.noise_combine == "subtract":
            return score_outputs - self.noise_penalty_lambda * noise_var
        if self.noise_combine == "multiply":
            return score_outputs * self._noise_weight_outputs(noise_var)
        raise ValueError(f"Unknown noise_combine: {self.noise_combine!r}.")


class qHeteroMultiOutputRegressionStraddle(_HeteroMultiOutputRegressionLevelSetBase):
    """Noise-aware multi-output regression straddle acquisition."""

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
        mean, latent_var, total_var, noise_var, Xt = self._posterior_mean_variance_outputs(X)
        var = self._select_variance_outputs(latent_var, total_var, noise_var)
        thresholds = self._thresholds_like(mean)
        beta = self.beta.to(device=mean.device, dtype=mean.dtype)
        score_outputs = beta * var.sqrt() - self._boundary_distance(mean, thresholds)
        score_outputs = self._apply_noise_to_output_score(score_outputs, noise_var)
        score = self._reduce_outputs(score_outputs)
        return self._finalize_pointwise_score(
            score,
            X,
            Xt,
            name="qHeteroMultiOutputRegressionStraddle",
        )


class qHeteroMultiOutputRegressionJointStraddle(_HeteroMultiOutputRegressionLevelSetBase):
    """Joint noise-aware multi-output regression straddle acquisition."""

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
        mean_outputs, _, _, noise_var_outputs, _ = self._posterior_mean_variance_outputs(X)
        thresholds = self._thresholds_like(mean_outputs)
        _, covar, Xt = self._posterior_covariance(X)
        beta = self.beta.to(device=mean_outputs.device, dtype=mean_outputs.dtype)
        boundary = self._boundary_distance(mean_outputs, thresholds)
        boundary_score = -self._reduce_outputs(boundary).mean(dim=-1)
        uncertainty = self._uncertainty_score(covar)
        noise_score = self._reduce_outputs(noise_var_outputs).mean(dim=-1)

        if self.noise_combine == "subtract":
            score = boundary_score + beta * uncertainty - self.noise_penalty_lambda * noise_score
        elif self.noise_combine == "multiply":
            score = (boundary_score + beta * uncertainty) * self._noise_weight_outputs(noise_score)
        else:
            score = boundary_score + beta * uncertainty
        return self._finalize_joint_score(
            score,
            X,
            Xt,
            name="qHeteroMultiOutputRegressionJointStraddle",
        )


class qHeteroMultiOutputRegressionICU(_HeteroMultiOutputRegressionLevelSetBase):
    """Noise-aware multi-output integrated contour uncertainty acquisition."""

    def __init__(
        self,
        model: Model,
        *,
        bandwidth: Optional[float | Tensor] = None,
        joint_boundary: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.bandwidth = None if bandwidth is None else torch.as_tensor(bandwidth)
        self.joint_boundary = bool(joint_boundary)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        mean, latent_var, total_var, noise_var, Xt = self._posterior_mean_variance_outputs(X)
        var = self._select_variance_outputs(latent_var, total_var, noise_var)
        std = var.sqrt().clamp_min(self.eps)
        thresholds = self._thresholds_like(mean)
        bw = (
            std
            if self.bandwidth is None
            else self.bandwidth.to(device=mean.device, dtype=mean.dtype).clamp_min(self.eps)
        )
        z = (mean - thresholds) / bw
        score_outputs = torch.exp(-0.5 * z.pow(2)) * std
        score_outputs = self._apply_noise_to_output_score(score_outputs, noise_var)
        score = (
            score_outputs.prod(dim=-1)
            if self.joint_boundary
            else self._reduce_outputs(score_outputs)
        )
        return self._finalize_pointwise_score(
            score,
            X,
            Xt,
            name="qHeteroMultiOutputRegressionICU",
        )


class qHeteroMultiOutputRegressionBoundaryVariance(_HeteroMultiOutputRegressionLevelSetBase):
    """Noise-aware boundary-weighted variance acquisition."""

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
        mean, latent_var, total_var, noise_var, Xt = self._posterior_mean_variance_outputs(X)
        var = self._select_variance_outputs(latent_var, total_var, noise_var)
        thresholds = self._thresholds_like(mean)
        tau = self.tau.to(device=mean.device, dtype=mean.dtype).clamp_min(self.eps)
        boundary_weight = torch.exp(-0.5 * ((mean - thresholds) / tau).pow(2))
        score_outputs = self._apply_noise_to_output_score(var * boundary_weight, noise_var)
        score = (
            score_outputs.prod(dim=-1)
            if self.joint_boundary
            else self._reduce_outputs(score_outputs)
        )
        return self._finalize_pointwise_score(
            score,
            X,
            Xt,
            name="qHeteroMultiOutputRegressionBoundaryVariance",
        )


class qHeteroMultiOutputRegressionProbabilityOfExceedance(_HeteroMultiOutputRegressionLevelSetBase):
    """Noise-aware probability-of-exceedance / feasibility acquisition."""

    def __init__(
        self,
        model: Model,
        *,
        mode: ProbabilityMode = "above",
        lower: Optional[Sequence[float] | Tensor] = None,
        upper: Optional[Sequence[float] | Tensor] = None,
        temperature: Optional[float | Tensor] = None,
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
        mean, latent_var, total_var, noise_var, Xt = self._posterior_mean_variance_outputs(X)
        var = self._select_variance_outputs(latent_var, total_var, noise_var)
        std = var.sqrt().clamp_min(self.eps)
        thresholds = self._thresholds_like(mean)

        if self.temperature is not None:
            temp = self.temperature.to(device=mean.device, dtype=mean.dtype).clamp_min(self.eps)
            if self.mode == "above":
                score_outputs = torch.sigmoid((mean - thresholds) / temp)
            elif self.mode == "below":
                score_outputs = torch.sigmoid((thresholds - mean) / temp)
            else:
                lo = self._bounds_like(mean, "lower")
                hi = self._bounds_like(mean, "upper")
                score_outputs = torch.sigmoid((mean - lo) / temp) * torch.sigmoid((hi - mean) / temp)
        else:
            if self.mode == "above":
                score_outputs = _safe_normal_cdf((mean - thresholds) / std)
            elif self.mode == "below":
                score_outputs = _safe_normal_cdf((thresholds - mean) / std)
            else:
                lo = self._bounds_like(mean, "lower")
                hi = self._bounds_like(mean, "upper")
                score_outputs = _safe_normal_cdf((hi - mean) / std) - _safe_normal_cdf((lo - mean) / std)

        score_outputs = self._apply_noise_to_output_score(
            score_outputs.clamp_min(0.0),
            noise_var,
        )
        score = score_outputs.prod(dim=-1) if self.joint else self._reduce_outputs(score_outputs)
        return self._finalize_pointwise_score(
            score,
            X,
            Xt,
            name="qHeteroMultiOutputRegressionProbabilityOfExceedance",
        )


__all__ = [
    "HeteroMultiOutputRegressionLevelSetScoreObjective",
    "qHeteroMultiOutputRegressionStraddle",
    "qHeteroMultiOutputRegressionJointStraddle",
    "qHeteroMultiOutputRegressionICU",
    "qHeteroMultiOutputRegressionBoundaryVariance",
    "qHeteroMultiOutputRegressionProbabilityOfExceedance",
]
