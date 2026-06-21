from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.transforms.input import InputTransform
from gpytorch.mlls import ExactMarginalLogLikelihood

from bochan.models.components.projected import _BaseProjectedMixedModel
from bochan.models.components.projected_utils import (
    _apply_input_transform_for_training,
    _clone_input_transform,
    _ensure_2d_train_Y,
    _prepare_raw_input_transform_for_mixed,
)
from bochan.models.components.vae import VAEProjector

__all__ = ["VAEMixedSingleTaskGP"]


class VAEMixedSingleTaskGP(_BaseProjectedMixedModel):
    """Mixed Gaussian regression GP with a jointly trained VAE projection.

    Only continuous columns are passed through the VAE. Integer-coded
    categorical columns are kept unchanged, appended after the latent continuous
    representation, and modeled by :class:`MixedSingleTaskGP`.

    The joint objective is

    ``gp_weight * negative_mll + reconstruction_weight * mse + kl_weight * kl``.

    The reconstruction MSE is computed only for continuous columns. The public
    prediction and acquisition APIs continue to accept raw mixed-space inputs.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        train_Yvar: Tensor | None = None,
        *,
        category_counts: dict[int, int] | None = None,
        cont_kernel_factory: Any | None = None,
        likelihood: Any | None = None,
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
            raise ValueError(
                "train_X and train_Y must contain the same number of observations."
            )
        if min(reconstruction_weight, kl_weight, gp_weight) < 0.0:
            raise ValueError(
                "reconstruction_weight, kl_weight, and gp_weight must be non-negative."
            )

        self._raw_train_X = train_X.detach().clone()
        self._setup_mixed_dims(
            input_dim=train_X.shape[-1],
            cat_dims=cat_dims,
            category_counts=category_counts,
        )
        self._validate_categorical_values(train_X)

        self.latent_dim = int(latent_dim)
        self.reconstruction_weight = float(reconstruction_weight)
        self.kl_weight = float(kl_weight)
        self.gp_weight = float(gp_weight)
        self.train_Yvar_original = train_Yvar
        self._train_targets = train_Y.detach().clone()
        self._cont_kernel_factory = cont_kernel_factory

        self.input_transform = _prepare_raw_input_transform_for_mixed(
            _clone_input_transform(input_transform),
            input_dim=self.input_dim_original,
            cont_dims=self.cont_dims,
            cat_dims=self.cat_dims,
        )
        if self.input_transform is not None and hasattr(self.input_transform, "to"):
            self.input_transform = self.input_transform.to(train_X)

        self._preproject_train_X = _apply_input_transform_for_training(
            train_X,
            self.input_transform,
            cat_dims=self.cat_dims,
            name=f"{self.__class__.__name__}.input_transform",
        )
        self._validate_categorical_values(self.preproject_train_input)

        self.vae = VAEProjector(
            input_dim=len(self.cont_dims),
            latent_dim=self.latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            decoder_output_activation=decoder_output_activation,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
        ).to(train_X)

        with torch.no_grad():
            latent_train_X = self.vae.transform(
                self.preproject_train_input[..., self.cont_dims]
            ).detach().clone()
            projected_train_X = self._combine_latent_and_categorical(
                self.preproject_train_input,
                latent_train_X,
            ).detach().clone()

        self._latent_train_X = latent_train_X
        self._projected_train_X = projected_train_X
        latent_cat_dims = list(
            range(self.latent_dim, self.latent_dim + len(self.cat_dims))
        )
        self.base_model = MixedSingleTaskGP(
            train_X=projected_train_X,
            train_Y=train_Y,
            cat_dims=latent_cat_dims,
            train_Yvar=train_Yvar,
            cont_kernel_factory=cont_kernel_factory,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
            input_transform=None,
        )
        self.to(train_X)

    @property
    def latent_train_input(self) -> Tensor:
        """Return the VAE latent means for continuous training columns."""
        return self._latent_train_X

    @property
    def latent_cat_dims(self) -> list[int]:
        """Return categorical dimensions in the internal projected space."""
        return list(range(self.latent_dim, self.latent_dim + len(self.cat_dims)))

    def _combine_latent_and_categorical(
        self,
        X_pre: Tensor,
        latent: Tensor,
    ) -> Tensor:
        """Append unchanged categorical columns to latent continuous values."""
        return self._project_continuous_and_concat_categorical(X_pre, latent)

    def _project_preprojected_inputs(self, X: Tensor) -> Tensor:
        latent = self.vae.transform(X[..., self.cont_dims])
        return self._combine_latent_and_categorical(X, latent)

    def encode(
        self,
        X: Tensor,
        *,
        sample: bool = False,
        return_stats: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        """Encode only the continuous columns of raw mixed-space inputs."""
        X_pre = self._to_preprojection_space(X)
        mu, logvar = self.vae.encode(X_pre[..., self.cont_dims])
        latent = self.vae.reparameterize(mu, logvar) if sample else mu
        if return_stats:
            return latent, mu, logvar
        return latent

    def decode_continuous(self, Z: Tensor) -> Tensor:
        """Decode latent values into continuous preprojection-space columns."""
        Z = torch.as_tensor(
            Z,
            device=self.train_input_raw.device,
            dtype=self.train_input_raw.dtype,
        )
        return self.vae.decode(Z)

    def _assemble_preprojection_input(
        self,
        continuous_X: Tensor,
        categorical_X: Tensor,
    ) -> Tensor:
        """Assemble full preprojection inputs in the original column order."""
        if continuous_X.shape[:-1] != categorical_X.shape[:-1]:
            raise ValueError(
                "continuous_X and categorical_X must have matching batch shapes."
            )
        if continuous_X.shape[-1] != len(self.cont_dims):
            raise ValueError(
                f"Expected {len(self.cont_dims)} continuous columns, "
                f"got {continuous_X.shape[-1]}."
            )
        if categorical_X.shape[-1] != len(self.cat_dims):
            raise ValueError(
                f"Expected {len(self.cat_dims)} categorical columns, "
                f"got {categorical_X.shape[-1]}."
            )

        full_shape = (*continuous_X.shape[:-1], self.input_dim_original)
        X_pre = continuous_X.new_empty(full_shape)
        X_pre[..., self.cont_dims] = continuous_X
        X_pre[..., self.cat_dims] = categorical_X.to(continuous_X)
        return X_pre

    def decode(
        self,
        Z: Tensor,
        *,
        categorical_X: Tensor,
        raw_space: bool = True,
    ) -> Tensor:
        """Decode latent values and restore supplied categorical columns.

        Args:
            Z: Latent continuous representation.
            categorical_X: Integer-coded categorical columns in ``self.cat_dims``
                order. Categories are not inferred from ``Z`` because they are not
                encoded by the VAE.
            raw_space: Apply ``input_transform.untransform`` when available.
        """
        continuous_X = self.decode_continuous(Z)
        categorical_X = torch.as_tensor(
            categorical_X,
            device=continuous_X.device,
            dtype=continuous_X.dtype,
        )
        X_pre = self._assemble_preprojection_input(continuous_X, categorical_X)
        self._validate_categorical_values(X_pre)
        if not raw_space or self.input_transform is None:
            return X_pre

        untransform = getattr(self.input_transform, "untransform", None)
        if not callable(untransform):
            raise NotImplementedError(
                "raw_space=True requires input_transform.untransform(). "
                "Use raw_space=False to obtain the reconstructed mixed input "
                "in preprojection space."
            )
        return untransform(X_pre)

    def inverse_transform(
        self,
        Z: Tensor,
        *,
        categorical_X: Tensor,
    ) -> Tensor:
        """Decode latent values into raw mixed-space inputs."""
        return self.decode(Z, categorical_X=categorical_X, raw_space=True)

    def reconstruct(
        self,
        X: Tensor,
        *,
        sample: bool = False,
        raw_space: bool = True,
    ) -> Tensor:
        """Reconstruct continuous columns while preserving categories."""
        X_pre = self._to_preprojection_space(X)
        continuous_X = X_pre[..., self.cont_dims]
        categorical_X = X_pre[..., self.cat_dims]
        mu, logvar = self.vae.encode(continuous_X)
        latent = self.vae.reparameterize(mu, logvar) if sample else mu
        reconstructed_continuous = self.vae.decode(latent)
        reconstructed = self._assemble_preprojection_input(
            reconstructed_continuous,
            categorical_X,
        )
        if not raw_space or self.input_transform is None:
            return reconstructed

        untransform = getattr(self.input_transform, "untransform", None)
        if not callable(untransform):
            raise NotImplementedError(
                "raw_space=True requires input_transform.untransform(). "
                "Use raw_space=False to obtain the reconstruction in "
                "preprojection space."
            )
        return untransform(reconstructed)

    def refresh_latent_train_inputs(self) -> Tensor:
        """Refresh latent and projected training inputs from the current encoder."""
        with torch.no_grad():
            latent = self.vae.transform(
                self.preproject_train_input[..., self.cont_dims]
            ).detach().clone()
            projected = self._combine_latent_and_categorical(
                self.preproject_train_input,
                latent,
            ).detach().clone()
        self._latent_train_X = latent
        self._projected_train_X = projected
        self.base_model.set_train_data(inputs=projected, targets=None, strict=False)
        return projected

    def joint_loss_components(
        self,
        gp_mll: ExactMarginalLogLikelihood,
    ) -> dict[str, Tensor]:
        """Compute GP, continuous reconstruction, KL, and total losses."""
        X_pre = self.preproject_train_input
        continuous_X = X_pre[..., self.cont_dims]
        reconstruction, mu, logvar, _ = self.vae(continuous_X, sample=True)
        projected = self._combine_latent_and_categorical(X_pre, mu)
        self.base_model.set_train_data(inputs=projected, targets=None, strict=False)
        gp_output = self.base_model.forward(projected)
        gp_loss = -gp_mll(gp_output, self.base_model.train_targets)
        if gp_loss.ndim > 0:
            gp_loss = gp_loss.sum()

        reconstruction_loss = F.mse_loss(
            reconstruction,
            continuous_X,
            reduction="mean",
        )
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
        """Return the internal mixed GP distribution for raw inputs."""
        projected = self.transform_inputs(X)
        if self.training:
            return self.base_model.forward(projected)
        return self.base_model(projected)

    def posterior(self, X: Tensor, *args: Any, **kwargs: Any):
        """Return a posterior for raw mixed-space candidate inputs."""
        self.eval()
        self.refresh_latent_train_inputs()
        return self.base_model.posterior(self.transform_inputs(X), *args, **kwargs)

    def make_gp_mll(self) -> ExactMarginalLogLikelihood:
        """Build the exact marginal log likelihood for the internal mixed GP."""
        return ExactMarginalLogLikelihood(
            self.base_model.likelihood,
            self.base_model,
        )

    def make_mll(self, **_: Any) -> None:
        """Return ``None`` to select the dedicated joint VAE-GP fitter."""
        return None

    @staticmethod
    def fit(model: "VAEMixedSingleTaskGP", **kwargs: Any):
        """Fit ``model`` with the dedicated full-batch joint objective."""
        from bochan.fit.vae import fit_vae_gp

        return fit_vae_gp(model, **kwargs)
