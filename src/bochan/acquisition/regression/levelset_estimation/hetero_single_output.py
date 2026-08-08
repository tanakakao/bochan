from __future__ import annotations

import math
from typing import Any, Literal, Optional

import torch
from botorch.models.model import Model
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor

from .single_output import (
    RegressionLevelSetScoreObjective,
    _RegressionLevelSetBase,
    _ensure_q_batch,
    _safe_logdet,
    _safe_normal_cdf,
)

VarianceSource = Literal["latent", "total", "noise"]
BoundaryMode = Literal["distance_to_threshold", "above", "below"]
ProbabilityMode = Literal["above", "below", "interval"]


class HeteroRegressionLevelSetScoreObjective(RegressionLevelSetScoreObjective):
    """Score objective for heteroscedastic regression LSE.

    The objective contract is identical to standard regression LSE; the separate
    public class name is retained for API clarity and backward compatibility.
    """


class _HeteroRegressionLevelSetBase(_RegressionLevelSetBase):
    """Noise-aware regression LSE sharing the standard duplicate-control base."""

    def __init__(
        self,
        model: Model,
        *,
        variance_source: VarianceSource = "latent",
        noise_penalty: float = 0.0,
        **kwargs: Any,
    ) -> None:
        if variance_source not in ("latent", "total", "noise"):
            raise ValueError("variance_source must be 'latent', 'total', or 'noise'.")
        super().__init__(model=model, **kwargs)
        self.variance_source = variance_source
        self.noise_penalty = float(noise_penalty)

    def _posterior_mean_variances(
        self,
        X: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return mean, latent/total/noise variance and distance-space X."""
        Xq = _ensure_q_batch(X)
        self._prepare_eval()

        try:
            post_latent = self.model.posterior(Xq, observation_noise=False)
            post_total = self.model.posterior(Xq, observation_noise=True)
        except Exception:
            post_latent = self.model.posterior(Xq)
            post_total = post_latent

        Xt = self._apply_input_transform_for_distance(Xq)
        mean = self._reduce_outputs_if_needed(post_latent.mean, Xt, name="posterior.mean")
        latent_var = self._reduce_outputs_if_needed(
            post_latent.variance,
            Xt,
            name="latent variance",
        )
        total_var = self._reduce_outputs_if_needed(
            post_total.variance,
            Xt,
            name="total variance",
        )

        mean = self._align_pointwise_score_to_X(mean, Xt, name="posterior.mean")
        latent_var = self._align_pointwise_score_to_X(
            latent_var,
            Xt,
            name="latent variance",
        ).clamp_min(self.eps)
        total_var = self._align_pointwise_score_to_X(
            total_var,
            Xt,
            name="total variance",
        ).clamp_min(self.eps)
        noise_var = (total_var - latent_var).clamp_min(self.eps)

        noise_fn = getattr(self.model, "predict_noise_var", None)
        if callable(noise_fn):
            try:
                noise_raw = noise_fn(Xq)
                noise_var = self._reduce_outputs_if_needed(
                    noise_raw,
                    Xt,
                    name="predict_noise_var",
                )
                noise_var = self._align_pointwise_score_to_X(
                    noise_var,
                    Xt,
                    name="predict_noise_var",
                ).clamp_min(self.eps)
                total_var = (latent_var + noise_var).clamp_min(self.eps)
            except Exception:
                pass

        return mean, latent_var, total_var, noise_var, Xt

    def _select_variance(
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

    def _posterior_covariance(self, X: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return latent mean/covariance and transformed X for joint LSE."""
        mean, latent_var, _, _, Xt = self._posterior_mean_variances(X)
        posterior = self.model.posterior(_ensure_q_batch(X), observation_noise=False)

        covar = None
        mvn = getattr(posterior, "mvn", None)
        if mvn is not None and hasattr(mvn, "covariance_matrix"):
            covar = mvn.covariance_matrix
        elif hasattr(posterior, "distribution") and hasattr(
            posterior.distribution,
            "covariance_matrix",
        ):
            covar = posterior.distribution.covariance_matrix

        q_like = int(Xt.shape[-2])
        target_shape = torch.Size(Xt.shape[:-2]) + torch.Size([q_like, q_like])
        if covar is None:
            return mean, torch.diag_embed(latent_var), Xt

        while covar.ndim > len(target_shape):
            covar = covar.mean(dim=0)
            if covar.shape == target_shape:
                break

        if covar.shape != target_shape:
            if covar.numel() == math.prod(target_shape):
                covar = covar.reshape(target_shape)
            else:
                covar = torch.diag_embed(latent_var)

        covar = 0.5 * (covar + covar.transpose(-1, -2))
        return mean, covar, Xt


class qHeteroRegressionStraddle(_HeteroRegressionLevelSetBase):
    """Noise-aware regression straddle acquisition."""

    def __init__(
        self,
        model: Model,
        *,
        beta: float | Tensor = 1.96,
        boundary_mode: BoundaryMode = "distance_to_threshold",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        if boundary_mode not in ("distance_to_threshold", "above", "below"):
            raise ValueError("boundary_mode must be 'distance_to_threshold', 'above', or 'below'.")
        self.register_buffer("beta", torch.as_tensor(beta))
        self.boundary_mode = boundary_mode

    def _boundary_distance(self, mean: Tensor, threshold: Tensor) -> Tensor:
        if self.boundary_mode == "distance_to_threshold":
            return (mean - threshold).abs()
        if self.boundary_mode == "above":
            return torch.relu(threshold - mean)
        if self.boundary_mode == "below":
            return torch.relu(mean - threshold)
        raise ValueError(f"Unknown boundary_mode: {self.boundary_mode!r}.")

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        mean, latent_var, total_var, noise_var, Xt = self._posterior_mean_variances(X)
        var = self._select_variance(latent_var, total_var, noise_var)
        threshold = self.threshold.to(device=mean.device, dtype=mean.dtype)
        beta = self.beta.to(device=mean.device, dtype=mean.dtype)
        score = beta * var.sqrt() - self._boundary_distance(mean, threshold)
        score = score - self.noise_penalty * noise_var.sqrt()
        return self._finalize_pointwise_score(
            score,
            X,
            Xt,
            name="qHeteroRegressionStraddle",
        )


class qHeteroRegressionJointStraddle(_HeteroRegressionLevelSetBase):
    """Joint noise-aware regression straddle acquisition."""

    def __init__(
        self,
        model: Model,
        *,
        beta: float | Tensor = 1.0,
        uncertainty_measure: Literal["logdet", "logdet1p", "trace"] = "logdet1p",
        covariance_jitter: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        if uncertainty_measure not in ("logdet", "logdet1p", "trace"):
            raise ValueError("uncertainty_measure must be 'logdet', 'logdet1p', or 'trace'.")
        self.register_buffer("beta", torch.as_tensor(beta))
        self.uncertainty_measure = uncertainty_measure
        self.covariance_jitter = float(covariance_jitter)

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
        mean, covar, Xt = self._posterior_covariance(X)
        _, _, _, noise_var, _ = self._posterior_mean_variances(X)
        threshold = self.threshold.to(device=mean.device, dtype=mean.dtype)
        beta = self.beta.to(device=mean.device, dtype=mean.dtype)
        proximity = -(mean - threshold).abs().mean(dim=-1)
        uncertainty = self._uncertainty_score(covar)
        noise_pen = self.noise_penalty * noise_var.sqrt().mean(dim=-1)
        score = proximity + beta * uncertainty - noise_pen
        return self._finalize_joint_score(
            score,
            X,
            Xt,
            name="qHeteroRegressionJointStraddle",
        )


class qHeteroRegressionICU(_HeteroRegressionLevelSetBase):
    """Noise-aware integrated contour uncertainty style acquisition."""

    def __init__(
        self,
        model: Model,
        *,
        bandwidth: Optional[float | Tensor] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.bandwidth = None if bandwidth is None else torch.as_tensor(bandwidth)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        mean, latent_var, total_var, noise_var, Xt = self._posterior_mean_variances(X)
        var = self._select_variance(latent_var, total_var, noise_var)
        std = var.sqrt().clamp_min(self.eps)
        threshold = self.threshold.to(device=mean.device, dtype=mean.dtype)
        bw = (
            std
            if self.bandwidth is None
            else self.bandwidth.to(device=mean.device, dtype=mean.dtype).clamp_min(self.eps)
        )
        z = (mean - threshold) / bw
        score = torch.exp(-0.5 * z.pow(2)) * std
        score = score - self.noise_penalty * noise_var.sqrt()
        return self._finalize_pointwise_score(score, X, Xt, name="qHeteroRegressionICU")


class qHeteroRegressionBoundaryVariance(_HeteroRegressionLevelSetBase):
    """Noise-aware boundary-weighted posterior variance acquisition."""

    def __init__(
        self,
        model: Model,
        *,
        tau: float | Tensor = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.register_buffer("tau", torch.as_tensor(tau))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        mean, latent_var, total_var, noise_var, Xt = self._posterior_mean_variances(X)
        var = self._select_variance(latent_var, total_var, noise_var)
        threshold = self.threshold.to(device=mean.device, dtype=mean.dtype)
        tau = self.tau.to(device=mean.device, dtype=mean.dtype).clamp_min(self.eps)
        boundary_weight = torch.exp(-0.5 * ((mean - threshold) / tau).pow(2))
        score = var * boundary_weight - self.noise_penalty * noise_var
        return self._finalize_pointwise_score(
            score,
            X,
            Xt,
            name="qHeteroRegressionBoundaryVariance",
        )


class qHeteroRegressionProbabilityOfExceedance(_HeteroRegressionLevelSetBase):
    """Noise-aware probability-of-exceedance / feasibility acquisition."""

    def __init__(
        self,
        model: Model,
        *,
        mode: ProbabilityMode = "above",
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
        mean, latent_var, total_var, noise_var, Xt = self._posterior_mean_variances(X)
        var = self._select_variance(latent_var, total_var, noise_var)
        std = var.sqrt().clamp_min(self.eps)
        threshold = self.threshold.to(device=mean.device, dtype=mean.dtype)

        if self.temperature is not None:
            temp = self.temperature.to(device=mean.device, dtype=mean.dtype).clamp_min(self.eps)
            if self.mode == "above":
                score = torch.sigmoid((mean - threshold) / temp)
            elif self.mode == "below":
                score = torch.sigmoid((threshold - mean) / temp)
            else:
                if self.lower is None or self.upper is None:
                    raise ValueError("lower and upper must be provided when mode='interval'.")
                lo = self.lower.to(device=mean.device, dtype=mean.dtype)
                hi = self.upper.to(device=mean.device, dtype=mean.dtype)
                score = torch.sigmoid((mean - lo) / temp) * torch.sigmoid((hi - mean) / temp)
        else:
            if self.mode == "above":
                score = _safe_normal_cdf((mean - threshold) / std)
            elif self.mode == "below":
                score = _safe_normal_cdf((threshold - mean) / std)
            else:
                if self.lower is None or self.upper is None:
                    raise ValueError("lower and upper must be provided when mode='interval'.")
                lo = self.lower.to(device=mean.device, dtype=mean.dtype)
                hi = self.upper.to(device=mean.device, dtype=mean.dtype)
                score = _safe_normal_cdf((hi - mean) / std) - _safe_normal_cdf((lo - mean) / std)

        score = score.clamp_min(0.0) - self.noise_penalty * noise_var
        return self._finalize_pointwise_score(
            score,
            X,
            Xt,
            name="qHeteroRegressionProbabilityOfExceedance",
        )


__all__ = [
    "HeteroRegressionLevelSetScoreObjective",
    "qHeteroRegressionStraddle",
    "qHeteroRegressionJointStraddle",
    "qHeteroRegressionICU",
    "qHeteroRegressionBoundaryVariance",
    "qHeteroRegressionProbabilityOfExceedance",
]
