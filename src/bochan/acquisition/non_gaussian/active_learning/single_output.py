from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from botorch.acquisition.acquisition import AcquisitionFunction, MCSamplerMixin
from botorch.acquisition.objective import PosteriorTransform
from botorch.utils.transforms import (
    average_over_ensemble_models,
    concatenate_pending_points,
    t_batch_mode_transform,
)

LinkFunction = Literal["softplus", "exp"]
QReduction = Literal["mean", "sum", "max", "none"]

__all__ = [
    "LinkFunction",
    "QReduction",
    "qNegIntegratedPosteriorVarianceNoFantasy",
    "qCountPosteriorStd",
    "make_poisson_mean_transform",
    "make_negative_binomial_mean_transform",
    "make_gamma_mean_transform",
    "make_beta_mean_transform",
    "infer_mean_transform_from_model",
]


def _positive_link(
    x: Tensor,
    link_function: LinkFunction = "softplus",
    exp_clip: float = 8.0,
    min_value: float = 1e-8,
) -> Tensor:
    """Map latent values to positive parameters."""
    if link_function == "softplus":
        return F.softplus(x) + min_value
    if link_function == "exp":
        return torch.exp(x.clamp(max=exp_clip)) + min_value
    raise ValueError("link_function must be 'softplus' or 'exp'.")


class qNegIntegratedPosteriorVarianceNoFantasy(AcquisitionFunction):
    """Fantasy-free latent-space proxy for qNIPV.

    This acquisition uses the latent Gaussian posterior returned by
    ``model.posterior`` and approximates the integrated posterior variance after
    observing candidate points ``X`` in latent space.

    It avoids ``model.fantasize`` entirely, making it usable for custom
    non-Gaussian models whose wrappers do not implement fantasize /
    condition_on_observations.

    Important:
        - This is a latent-space proxy.
        - It is not the exact observation-space fantasy update for Poisson /
          Negative Binomial / Gamma / Beta.
        - It is most useful as a practical exploration heuristic.
    """

    def __init__(
        self,
        model,
        mc_points: Tensor,
        posterior_transform: PosteriorTransform | None = None,
        X_pending: Tensor | None = None,
        jitter: float = 1e-6,
        clamp_min_variance: float = 0.0,
    ) -> None:
        super().__init__(model=model)
        self.posterior_transform = posterior_transform
        self.X_pending = X_pending
        self.jitter = float(jitter)
        self.clamp_min_variance = float(clamp_min_variance)
        self.register_buffer("mc_points", mc_points)

    def _expand_mc_points(self, X: Tensor) -> Tensor:
        """Expand ``mc_points`` to match the batch shape of ``X``."""
        Z = self.mc_points.to(X)
        batch_shape = X.shape[:-2]

        if Z.ndim == 2:
            return Z.view(*([1] * len(batch_shape)), *Z.shape).expand(
                *batch_shape, *Z.shape
            )

        if Z.shape[:-2] == batch_shape:
            return Z

        return Z.expand(*batch_shape, Z.shape[-2], Z.shape[-1])

    @concatenate_pending_points
    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        if self.model.num_outputs > 1 and self.posterior_transform is None:
            raise RuntimeError(
                "A posterior_transform is required for multi-output models."
            )

        Z = self._expand_mc_points(X)
        m = Z.shape[-2]
        q = X.shape[-2]

        ZX = torch.cat([Z, X], dim=-2)
        posterior = self.model.posterior(
            ZX,
            posterior_transform=self.posterior_transform,
        )

        if not hasattr(posterior, "distribution"):
            raise RuntimeError(
                "posterior.distribution is required. This acquisition assumes "
                "a GPyTorchPosterior-like object."
            )

        mvn = posterior.distribution
        cov = mvn.covariance_matrix

        Kzz = cov[..., :m, :m]
        Kzx = cov[..., :m, m:]
        Kxz = cov[..., m:, :m]
        Kxx = cov[..., m:, m:]

        eye = torch.eye(q, device=X.device, dtype=X.dtype)
        eye = eye.view(*([1] * len(X.shape[:-2])), q, q)

        sol = torch.linalg.solve(Kxx + self.jitter * eye, Kxz)

        diag_Kzz = torch.diagonal(Kzz, dim1=-2, dim2=-1)
        reduction = (Kzx * sol.transpose(-2, -1)).sum(dim=-1)
        cond_var_diag = diag_Kzz - reduction

        if self.clamp_min_variance > 0.0:
            cond_var_diag = cond_var_diag.clamp_min(self.clamp_min_variance)

        return -cond_var_diag.mean(dim=-1)


def make_poisson_mean_transform(
    link_function: LinkFunction = "softplus",
    exp_clip: float = 8.0,
) -> Callable[[Tensor, Any, Tensor | None], Tensor]:
    """Create a transform from latent samples to Poisson predictive mean."""

    def _transform(samples: Tensor, model: Any, X: Tensor | None = None) -> Tensor:
        latent = samples[..., 0] if samples.shape[-1] == 1 else samples
        return _positive_link(
            latent,
            link_function=link_function,
            exp_clip=exp_clip,
            min_value=1e-8,
        )

    return _transform


def make_negative_binomial_mean_transform(
    likelihood=None,
    link_function: LinkFunction = "softplus",
    exp_clip: float = 8.0,
) -> Callable[[Tensor, Any, Tensor | None], Tensor]:
    """Create a transform from latent samples to Negative Binomial mean."""

    def _transform(samples: Tensor, model: Any, X: Tensor | None = None) -> Tensor:
        lk = likelihood if likelihood is not None else getattr(model, "likelihood", None)
        if lk is None or not hasattr(lk, "probs"):
            raise RuntimeError(
                "A Negative Binomial likelihood with a 'probs' attribute is required."
            )

        latent = samples[..., 0] if samples.shape[-1] == 1 else samples
        positive = _positive_link(
            latent,
            link_function=link_function,
            exp_clip=exp_clip,
            min_value=1e-8,
        )

        probs = torch.clamp(lk.probs.to(positive), 1e-6, 1 - 1e-6)
        while probs.ndim < positive.ndim:
            probs = probs.unsqueeze(0)

        if getattr(lk, "num_failures_param", False):
            return positive * probs / (1 - probs)
        return positive

    return _transform


def make_gamma_mean_transform(
    link_function: LinkFunction = "softplus",
    exp_clip: float = 8.0,
    min_mean: float = 1e-8,
) -> Callable[[Tensor, Any, Tensor | None], Tensor]:
    """Create a transform from latent samples to Gamma predictive mean."""

    def _transform(samples: Tensor, model: Any, X: Tensor | None = None) -> Tensor:
        latent = samples[..., 0] if samples.shape[-1] == 1 else samples

        mm = min_mean
        lk = getattr(model, "likelihood", None)
        if lk is not None and hasattr(lk, "min_mean"):
            mm = float(lk.min_mean)

        return _positive_link(
            latent,
            link_function=link_function,
            exp_clip=exp_clip,
            min_value=mm,
        )

    return _transform


def make_beta_mean_transform() -> Callable[[Tensor, Any, Tensor | None], Tensor]:
    """Create a transform from latent samples to Beta predictive mean."""

    def _transform(samples: Tensor, model: Any, X: Tensor | None = None) -> Tensor:
        latent = samples[..., 0] if samples.shape[-1] == 1 else samples
        return torch.sigmoid(latent)

    return _transform


def infer_mean_transform_from_model(
    model: Any,
    *,
    link_function: LinkFunction = "softplus",
    exp_clip: float = 8.0,
) -> Callable[[Tensor, Any, Tensor | None], Tensor]:
    """Infer a predictive-mean transform from the model / likelihood type."""
    model_name = model.__class__.__name__.lower()
    lk = getattr(model, "likelihood", None)
    lk_name = lk.__class__.__name__.lower() if lk is not None else ""

    if ("beta" in lk_name) or ("beta" in model_name):
        return make_beta_mean_transform()

    if (
        ("negativebinomial" in lk_name)
        or ("negative_binomial" in lk_name)
        or ("negativebinomial" in model_name)
        or ("negative_binomial" in model_name)
    ):
        return make_negative_binomial_mean_transform(
            likelihood=lk,
            link_function=link_function,
            exp_clip=exp_clip,
        )

    if ("gamma" in lk_name) or ("gamma" in model_name):
        return make_gamma_mean_transform(
            link_function=link_function,
            exp_clip=exp_clip,
        )

    if ("poisson" in lk_name) or ("poisson" in model_name):
        return make_poisson_mean_transform(
            link_function=link_function,
            exp_clip=exp_clip,
        )

    raise RuntimeError(
        "Could not infer a mean transform from the model. "
        "Pass transform explicitly."
    )


class qCountPosteriorStd(AcquisitionFunction, MCSamplerMixin):
    """MC exploration acquisition based on std of transformed posterior samples.

    The model is assumed to return a latent posterior. The transform maps
    latent posterior samples to the final predictive quantity of interest, such as:

    - Poisson rate / predictive mean
    - Negative Binomial predictive mean
    - Gamma predictive mean
    - Beta predictive mean

    If ``transform=None``, a suitable transform is inferred from the model.
    """

    def __init__(
        self,
        model,
        transform: Callable[[Tensor, Any, Tensor | None], Tensor] | None = None,
        sampler=None,
        posterior_transform: PosteriorTransform | None = None,
        X_pending: Tensor | None = None,
        q_reduction: QReduction = "mean",
        link_function: LinkFunction = "softplus",
        exp_clip: float = 8.0,
    ) -> None:
        AcquisitionFunction.__init__(self, model=model)
        MCSamplerMixin.__init__(self, sampler=sampler)

        if q_reduction not in {"mean", "sum", "max", "none"}:
            raise ValueError(
                "q_reduction must be one of {'mean', 'sum', 'max', 'none'}."
            )

        if transform is None:
            transform = infer_mean_transform_from_model(
                model=model,
                link_function=link_function,
                exp_clip=exp_clip,
            )

        self.transform = transform
        self.posterior_transform = posterior_transform
        self.X_pending = X_pending
        self.q_reduction = q_reduction
        self.link_function = link_function
        self.exp_clip = float(exp_clip)

    @concatenate_pending_points
    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(
            X,
            posterior_transform=self.posterior_transform,
        )
        samples = self.get_posterior_samples(posterior)
        values = self.transform(samples, self.model, X)

        if values.ndim == 0:
            raise RuntimeError("transform must return at least 1D samples.")

        if values.shape[-1] == 1:
            values = values[..., 0]

        if self.q_reduction == "none":
            if values.shape[-1] != 1:
                raise RuntimeError("q_reduction='none' assumes q=1.")
            reduced = values[..., 0]
        elif self.q_reduction == "mean":
            reduced = values.mean(dim=-1)
        elif self.q_reduction == "sum":
            reduced = values.sum(dim=-1)
        elif self.q_reduction == "max":
            reduced = values.max(dim=-1).values
        else:
            raise RuntimeError("unreachable")

        return reduced.std(dim=0, unbiased=False)
