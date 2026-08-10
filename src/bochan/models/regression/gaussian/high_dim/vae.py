from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms.input import InputTransform
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from torch.nn import functional as F

from bochan.models.components.projected_utils import (
    _apply_input_transform_for_eval,
    _apply_input_transform_for_training,
    _clone_input_transform,
    _ensure_2d_train_Y,
)
from bochan.models.components.vae import VAEProjector

__all__ = ["VAEGaussianGPModel"]


class VAEGaussianGPModel(Model):
    """Gaussian regression GP with a jointly trained VAE input representation.

    Public prediction and acquisition APIs receive raw-space inputs. Internally,
    ``input_transform`` is applied first, the VAE encoder mean is used as the GP
    input, and the decoder is trained to reconstruct the preprojection input.

    The joint objective is

    ``gp_weight * negative_mll + reconstruction_weight * mse + kl_weight * kl``.

    Notes:
        ``make_mll()`` intentionally returns ``None`` so the high-level bochan
        factory selects this model's dedicated ``fit`` routine rather than the
        generic exact-GP SciPy optimizer. ``make_gp_mll()`` exposes the internal
        exact GP marginal log likelihood for diagnostics.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        likelihood: Any | None = None,
        covar_module: Any | None = None,
        mean_module: Any | None = None,
        outcome_transform: Any | None = None,
        input_transform: InputTransform | None = None,
        latent_dim: int = 2,
        hidden_dims: Sequence[int] | None = None,
        activation: str = "silu",
        decoder_output_activation: str = "identity",
        reconstruction_weight: float = 1.0,
        kl_weight: float = 1e-3,
        gp_weight: float = 1.0,
        logvar_min: float = -10.0,
        logvar_max: float = 10.0,
    ) -> None:
        super().__init__()
        if train_X.ndim != 2:
            raise ValueError("train_X must be a 2D tensor with shape [n, d].")
        train_Y = _ensure_2d_train_Y(train_Y)
        if train_Y.shape[-2] != train_X.shape[-2]:
            raise ValueError("train_X and train_Y must contain the same number of observations.")
        if min(reconstruction_weight, kl_weight, gp_weight) < 0.0:
            raise ValueError(
                "reconstruction_weight, kl_weight, and gp_weight must be non-negative."
            )

        self.input_dim_original = int(train_X.shape[-1])
        self.latent_dim = int(latent_dim)
        self.reconstruction_weight = float(reconstruction_weight)
        self.kl_weight = float(kl_weight)
        self.gp_weight = float(gp_weight)
        self.train_Yvar_original = train_Yvar

        self._raw_train_X = train_X.detach().clone()
        self._train_targets = train_Y.detach().clone()
        self.input_transform = _clone_input_transform(input_transform)
        if self.input_transform is not None and hasattr(self.input_transform, "to"):
            self.input_transform = self.input_transform.to(train_X)

        self._preproject_train_X = _apply_input_transform_for_training(
            train_X,
            self.input_transform,
            name=f"{self.__class__.__name__}.input_transform",
        )
        self.vae = VAEProjector(
            input_dim=self.input_dim_original,
            latent_dim=self.latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            decoder_output_activation=decoder_output_activation,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
        ).to(train_X)

        with torch.no_grad():
            latent_train_X = self.vae.transform(self.preproject_train_input).detach().clone()
        self._latent_train_X = latent_train_X
        self.base_model = SingleTaskGP(
            train_X=latent_train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=outcome_transform,
            input_transform=None,
        )
        self.to(train_X)

    def _set_transformed_inputs(self) -> None:
        """Disable BoTorch's automatic transformed-training-input replacement."""
        return None

    @property
    def model(self) -> SingleTaskGP:
        """Return the internal latent-space GP."""
        return self.base_model

    @property
    def likelihood(self):
        """Return the internal GP likelihood."""
        return self.base_model.likelihood

    @property
    def num_outputs(self) -> int:
        """Return the number of GP outputs."""
        return int(self.base_model.num_outputs)

    @property
    def batch_shape(self) -> torch.Size:
        """Return the internal GP batch shape."""
        return self.base_model.batch_shape

    @property
    def train_input_raw(self) -> Tensor:
        """Return raw-space training inputs."""
        return self._raw_train_X

    @property
    def train_inputs(self) -> tuple[Tensor]:
        """Return raw-space training inputs in BoTorch tuple form."""
        return (self.train_input_raw,)

    @property
    def train_inputs_raw(self) -> tuple[Tensor]:
        """Return raw-space training inputs in BoTorch tuple form."""
        return (self.train_input_raw,)

    @property
    def train_targets(self) -> Tensor:
        """Return original-scale training targets."""
        return self._train_targets

    @property
    def preproject_train_input(self) -> Tensor:
        """Return input-transform output used by the VAE."""
        return self._preproject_train_X

    @property
    def latent_train_input(self) -> Tensor:
        """Return the current deterministic GP training representation."""
        return self._latent_train_X

    def _to_preprojection_space(self, X: Tensor) -> Tensor:
        if isinstance(X, tuple):
            X = X[0]
        X = torch.as_tensor(
            X,
            device=self.train_input_raw.device,
            dtype=self.train_input_raw.dtype,
        )
        if X.ndim == 1:
            X = X.unsqueeze(0)
        if X.shape[-1] != self.input_dim_original:
            raise ValueError(
                f"Expected raw input dim {self.input_dim_original}, got {X.shape[-1]}."
            )
        return _apply_input_transform_for_eval(X, self.input_transform)

    def encode(
        self,
        X: Tensor,
        *,
        sample: bool = False,
        return_stats: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        """Encode raw-space inputs into latent space."""
        X_pre = self._to_preprojection_space(X)
        mu, logvar = self.vae.encode(X_pre)
        latent = self.vae.reparameterize(mu, logvar) if sample else mu
        if return_stats:
            return latent, mu, logvar
        return latent

    def transform_inputs(self, X: Tensor) -> Tensor:
        """Map raw-space inputs to the deterministic latent GP space."""
        latent = self.encode(X, sample=False)
        if not torch.is_tensor(latent):
            raise RuntimeError("Unexpected non-tensor latent representation.")
        return latent

    def decode(self, Z: Tensor, *, raw_space: bool = True) -> Tensor:
        """Decode latent points into preprojection or raw input space."""
        Z = torch.as_tensor(
            Z,
            device=self.train_input_raw.device,
            dtype=self.train_input_raw.dtype,
        )
        X_pre = self.vae.decode(Z)
        if not raw_space or self.input_transform is None:
            return X_pre
        untransform = getattr(self.input_transform, "untransform", None)
        if not callable(untransform):
            raise NotImplementedError(
                "raw_space=True requires input_transform.untransform(). "
                "Use raw_space=False to obtain the decoder output in preprojection space."
            )
        return untransform(X_pre)

    def inverse_transform(self, Z: Tensor) -> Tensor:
        """Decode latent values into raw input space when possible."""
        return self.decode(Z, raw_space=True)

    def reconstruct(
        self,
        X: Tensor,
        *,
        sample: bool = False,
        raw_space: bool = True,
    ) -> Tensor:
        """Encode and reconstruct raw-space inputs."""
        latent = self.encode(X, sample=sample)
        if not torch.is_tensor(latent):
            raise RuntimeError("Unexpected non-tensor latent representation.")
        return self.decode(latent, raw_space=raw_space)

    def refresh_latent_train_inputs(self) -> Tensor:
        """Refresh the internal GP training inputs from the current encoder."""
        with torch.no_grad():
            latent = self.vae.transform(self.preproject_train_input).detach().clone()
        self._latent_train_X = latent
        self.base_model.set_train_data(inputs=latent, targets=None, strict=False)
        return latent

    def joint_loss_components(
        self,
        gp_mll: ExactMarginalLogLikelihood,
    ) -> dict[str, Tensor]:
        """Compute differentiable GP, reconstruction, KL, and total losses."""
        X_pre = self.preproject_train_input
        reconstruction, mu, logvar, _ = self.vae(X_pre, sample=True)
        self.base_model.set_train_data(inputs=mu, targets=None, strict=False)
        gp_output = self.base_model.forward(mu)
        gp_loss = -gp_mll(gp_output, self.base_model.train_targets)
        if gp_loss.ndim > 0:
            gp_loss = gp_loss.sum()

        reconstruction_loss = F.mse_loss(reconstruction, X_pre, reduction="mean")
        kl_loss = -0.5 * torch.mean(
            1.0 + logvar - mu.square() - logvar.exp()
        )
        total_loss = (
            self.gp_weight * gp_loss
            + self.reconstruction_weight * reconstruction_loss
            + self.kl_weight * kl_loss
        )
        return {
            "loss": total_loss,
            "gp_loss": gp_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def forward(self, X: Tensor):
        """Return the latent GP distribution for raw-space inputs."""
        latent = self.transform_inputs(X)
        if self.training:
            return self.base_model.forward(latent)
        return self.base_model(latent)

    def posterior(self, X: Tensor, *args: Any, **kwargs: Any):
        """Return a posterior for raw-space candidate inputs."""
        self.eval()
        self.refresh_latent_train_inputs()
        return self.base_model.posterior(self.transform_inputs(X), *args, **kwargs)

    def make_gp_mll(self) -> ExactMarginalLogLikelihood:
        """Build the exact marginal log likelihood for the internal GP."""
        return ExactMarginalLogLikelihood(
            self.base_model.likelihood,
            self.base_model,
        )

    def make_mll(self, **_: Any) -> None:
        """Return ``None`` to select the dedicated joint VAE-GP fitter."""
        return None

    @staticmethod
    def fit(model: VAEGaussianGPModel, **kwargs: Any):
        """Fit ``model`` with the dedicated full-batch joint objective."""
        from bochan.fit.vae import fit_vae_gp

        return fit_vae_gp(model, **kwargs)
