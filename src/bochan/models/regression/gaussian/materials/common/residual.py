"""Residual-GP composition for pretrained material predictors.

A residual GP treats a compatible pretrained direct predictor as a deterministic
baseline and learns only the correction to observed targets.  This module is
backend-neutral: concrete MACE / CHGNet / M3GNet adapters can opt in later by
providing an ``nn.Module`` predictor with the standard tensor contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.posteriors.posterior import Posterior
from botorch.posteriors.transformed import TransformedPosterior
from botorch.utils.transforms import normalize_indices
from torch import Tensor, nn

from .pretrained import PretrainedMaterialSpec


class DirectMaterialPredictor(nn.Module, ABC):
    """Contract for deterministic pretrained material-property prediction."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return the number of predicted material properties."""

        raise NotImplementedError

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        """Return predictions with shape ``[*X.shape[:-1], output_dim]``."""

        raise NotImplementedError


def validate_direct_material_predictions(
    X: Tensor,
    predictions: Tensor,
    *,
    output_dim: int,
) -> Tensor:
    """Validate deterministic pretrained prediction shape/device/dtype."""

    if not torch.is_tensor(X):
        raise TypeError("X must be a Tensor.")
    if not torch.is_tensor(predictions):
        raise TypeError("predictions must be a Tensor.")
    if isinstance(output_dim, bool) or not isinstance(output_dim, int) or output_dim <= 0:
        raise ValueError("output_dim must be a positive integer.")
    expected_shape = (*X.shape[:-1], output_dim)
    if predictions.shape != torch.Size(expected_shape):
        raise ValueError(
            "Direct material predictions must preserve X leading dimensions and "
            f"return output_dim values: {tuple(predictions.shape)} != {expected_shape}."
        )
    if predictions.device != X.device:
        raise ValueError("Direct material predictions must be on the same device as X.")
    if predictions.dtype != X.dtype:
        raise ValueError("Direct material predictions must have the same dtype as X.")
    if not torch.isfinite(predictions).all():
        raise FloatingPointError("Direct material predictor produced non-finite values.")
    return predictions


def predict_material_baseline(predictor: DirectMaterialPredictor, X: Tensor) -> Tensor:
    """Evaluate and validate a pretrained deterministic baseline."""

    if not isinstance(predictor, DirectMaterialPredictor):
        raise TypeError("predictor must implement DirectMaterialPredictor.")
    predictions = predictor(X)
    return validate_direct_material_predictions(X, predictions, output_dim=predictor.output_dim)


def compute_material_residual_targets(
    train_X: Tensor,
    train_Y: Tensor,
    predictor: DirectMaterialPredictor,
) -> Tensor:
    """Return ``train_Y - pretrained_prediction`` without masking observations.

    Missing/partial observations are intentionally preserved.  For example, a
    NaN in ``train_Y`` remains NaN in the returned residual tensor so the
    established observation-aware Gaussian path remains the single owner of
    missing-target semantics.
    """

    if not torch.is_tensor(train_Y):
        raise TypeError("train_Y must be a Tensor.")
    baseline = predict_material_baseline(predictor, train_X)
    if train_Y.shape != baseline.shape:
        raise ValueError(
            "train_Y shape must match the pretrained prediction shape: "
            f"{tuple(train_Y.shape)} != {tuple(baseline.shape)}."
        )
    if train_Y.device != baseline.device:
        raise ValueError("train_Y and pretrained predictions must share a device.")
    if train_Y.dtype != baseline.dtype:
        raise ValueError("train_Y and pretrained predictions must share a dtype.")
    return train_Y - baseline


def require_residual_gp_capability(spec: PretrainedMaterialSpec) -> None:
    """Require that one pretrained family is explicitly residual-GP capable."""

    if not isinstance(spec, PretrainedMaterialSpec):
        raise TypeError("spec must be a PretrainedMaterialSpec.")
    spec.capabilities.require_residual_gp()


class ResidualMaterialGPModel(Model):
    """Expose ``pretrained baseline + residual GP`` as one BoTorch model.

    The wrapped ``residual_model`` is fitted against residual targets.  At
    posterior time its samples and mean are shifted by the deterministic
    pretrained prediction while variance is left unchanged.  This preserves the
    epistemic/observation uncertainty learned by the GP and lets standard BoTorch
    acquisition functions consume the corrected property posterior directly.
    """

    def __init__(
        self,
        *,
        predictor: DirectMaterialPredictor,
        residual_model: Model,
        pretrained_spec: PretrainedMaterialSpec | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(predictor, DirectMaterialPredictor):
            raise TypeError("predictor must implement DirectMaterialPredictor.")
        if not isinstance(residual_model, Model):
            raise TypeError("residual_model must be a BoTorch Model.")
        if pretrained_spec is not None:
            require_residual_gp_capability(pretrained_spec)

        residual_outputs = int(residual_model.num_outputs)
        if residual_outputs != predictor.output_dim:
            raise ValueError(
                "predictor.output_dim must match residual_model.num_outputs: "
                f"{predictor.output_dim} != {residual_outputs}."
            )
        self.predictor = predictor
        self.residual_model = residual_model
        self.pretrained_spec = pretrained_spec

    @property
    def num_outputs(self) -> int:
        """Return the number of corrected property outputs."""

        return int(self.residual_model.num_outputs)

    @property
    def batch_shape(self) -> torch.Size:
        """Delegate model batch shape to the residual GP."""

        return self.residual_model.batch_shape

    def baseline(self, X: Tensor) -> Tensor:
        """Return validated deterministic pretrained predictions."""

        return predict_material_baseline(self.predictor, X)

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: PosteriorTransform | None = None,
        **kwargs: Any,
    ) -> Posterior:
        """Return the corrected posterior in the original property scale."""

        baseline = self.baseline(X)
        indices: list[int] | None = None
        if output_indices is not None:
            indices = normalize_indices(indices=list(output_indices), d=self.num_outputs)
            if not indices:
                raise ValueError("output_indices must contain at least one output index.")
            baseline = baseline[..., indices]

        residual_posterior = self.residual_model.posterior(
            X,
            output_indices=indices,
            observation_noise=observation_noise,
            posterior_transform=None,
            **kwargs,
        )
        corrected: Posterior = TransformedPosterior(
            posterior=residual_posterior,
            sample_transform=lambda samples: samples + baseline,
            mean_transform=lambda mean, variance: mean + baseline,
            variance_transform=lambda mean, variance: variance,
        )
        if posterior_transform is not None:
            corrected = posterior_transform(corrected)
        return corrected

    def condition_on_observations(
        self,
        X: Tensor,
        Y: Tensor,
        **kwargs: Any,
    ) -> ResidualMaterialGPModel:
        """Condition the residual model using observations in original scale."""

        residual_Y = compute_material_residual_targets(X, Y, self.predictor)
        conditioned = self.residual_model.condition_on_observations(X=X, Y=residual_Y, **kwargs)
        if not isinstance(conditioned, Model):
            raise TypeError("condition_on_observations must return a BoTorch Model.")
        return ResidualMaterialGPModel(
            predictor=self.predictor,
            residual_model=conditioned,
            pretrained_spec=self.pretrained_spec,
        )


__all__ = [
    "DirectMaterialPredictor",
    "ResidualMaterialGPModel",
    "compute_material_residual_targets",
    "predict_material_baseline",
    "require_residual_gp_capability",
    "validate_direct_material_predictions",
]
