"""Residual Gaussian process over M3GNet pretrained scalar predictions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from gpytorch.likelihoods import Likelihood
from torch import Tensor, nn

from bochan.composition import M3GNetEncoder, MaterialProcessFusion
from bochan.models.regression.gaussian.deep.deepkernel import InputTransformArg, OutcomeTransformArg
from bochan.models.regression.gaussian.deep.m3gnet import M3GNetGPModel
from bochan.models.regression.gaussian.materials.common.residual import (
    DirectMaterialPredictor,
    ResidualMaterialGPModel,
    compute_material_residual_targets,
)

_DEFAULT_MODEL_NAME = "M3GNet-PES-MatPES-PBE-2025.2"


def _validate_structure_bank(structures: Sequence[Any]) -> tuple[Any, ...]:
    if isinstance(structures, (str, bytes)) or not isinstance(structures, Sequence):
        raise TypeError("structures must be a non-empty sequence.")
    resolved = tuple(structures)
    if not resolved:
        raise ValueError("structures must contain at least one structure.")
    return resolved


def _resolve_encoder(
    encoder: M3GNetEncoder | nn.Module | None,
    *,
    model_name: str,
    encoder_output_dim: int | None,
) -> M3GNetEncoder:
    if isinstance(encoder, M3GNetEncoder):
        if encoder_output_dim is not None and encoder_output_dim != encoder.output_dim:
            raise ValueError(
                "encoder_output_dim does not match M3GNetEncoder.output_dim: "
                f"{encoder_output_dim} != {encoder.output_dim}."
            )
        resolved = encoder
    else:
        resolved = M3GNetEncoder(
            encoder=encoder,
            model_name=model_name,
            output_dim=encoder_output_dim,
        )
    for parameter in resolved.parameters():
        parameter.requires_grad_(False)
    resolved.eval()
    return resolved


def _predict_scalar_one(encoder: M3GNetEncoder, structure: Any) -> Tensor:
    """Return the raw pretrained M3GNet graph-level scalar for one structure."""

    graph, state_attr = encoder._prepare_graph(structure)
    prediction = encoder.encoder(g=graph, state_attr=state_attr)
    if not torch.is_tensor(prediction):
        raise TypeError("M3GNet direct forward must return a Tensor scalar prediction.")
    prediction = prediction.reshape(-1)
    if prediction.numel() != 1:
        raise ValueError(
            "M3GNet residual GP currently requires one scalar pretrained prediction per structure; "
            f"got shape {tuple(prediction.shape)}."
        )
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("M3GNet produced a non-finite direct prediction.")
    return prediction.reshape(1, 1)


def _predict_scalar(encoder: M3GNetEncoder, structures: Sequence[Any]) -> Tensor:
    predictions = [_predict_scalar_one(encoder, structure) for structure in structures]
    return torch.cat(predictions, dim=0)


class M3GNetDirectPredictor(DirectMaterialPredictor):
    """Expose frozen M3GNet scalar predictions through structure-indexed inputs.

    Column 0 of ``X`` selects a structure from ``structures``. Additional process
    columns do not affect the pretrained baseline and remain available to the GP
    residual model.
    """

    def __init__(self, encoder: M3GNetEncoder, structures: Sequence[Any]) -> None:
        super().__init__()
        if not isinstance(encoder, M3GNetEncoder):
            raise TypeError("encoder must be an M3GNetEncoder.")
        self.encoder = encoder
        self.structures = _validate_structure_bank(structures)

    @property
    def output_dim(self) -> int:
        return 1

    def forward(self, X: Tensor) -> Tensor:
        if not torch.is_tensor(X):
            raise TypeError("X must be a Tensor.")
        if X.ndim < 2 or X.shape[-1] < 1:
            raise ValueError("X must have shape [..., q, 1 + process_dim].")

        flat_X = X.reshape(-1, X.shape[-1])
        raw_indices = flat_X[:, 0]
        if not torch.isfinite(raw_indices).all():
            raise ValueError("Structure indices must be finite.")
        rounded = raw_indices.round()
        if not torch.equal(raw_indices, rounded):
            raise ValueError("Structure indices must be integer-valued.")
        indices = rounded.to(dtype=torch.long)
        if indices.numel() and (
            int(indices.min().item()) < 0 or int(indices.max().item()) >= len(self.structures)
        ):
            raise ValueError("Structure index is outside the configured structure bank.")

        unique_indices, inverse = torch.unique(indices, sorted=True, return_inverse=True)
        selected = [self.structures[int(index)] for index in unique_indices.detach().cpu().tolist()]
        baseline_unique = _predict_scalar(self.encoder, selected)
        baseline = baseline_unique[inverse.to(device=baseline_unique.device)]
        baseline = baseline.to(device=X.device, dtype=X.dtype)
        return baseline.reshape(*X.shape[:-1], 1)


class M3GNetResidualGPModel(ResidualMaterialGPModel):
    """Correct a frozen M3GNet scalar prediction with an exact GP residual.

    ``train_Y`` must represent the same physical quantity and units as the raw
    scalar output of the selected pretrained M3GNet model. The GP is trained on
    ``train_Y - M3GNet_prediction`` and the public posterior is shifted back to
    the original target scale.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        structures: Sequence[Any],
        encoder: M3GNetEncoder | nn.Module | None = None,
        model_name: str = _DEFAULT_MODEL_NAME,
        encoder_output_dim: int | None = None,
        latent_dim: int = 32,
        fusion: str | MaterialProcessFusion = "concat",
        projection: nn.Module | None = None,
        likelihood: Likelihood | None = None,
        input_transform: InputTransformArg = "DEFAULT",
        outcome_transform: OutcomeTransformArg = "DEFAULT",
    ) -> None:
        resolved_structures = _validate_structure_bank(structures)
        material_encoder = _resolve_encoder(
            encoder,
            model_name=model_name,
            encoder_output_dim=encoder_output_dim,
        )
        predictor = M3GNetDirectPredictor(material_encoder, resolved_structures)
        residual_Y = compute_material_residual_targets(train_X, train_Y, predictor)
        residual_model = M3GNetGPModel(
            train_X=train_X,
            train_Y=residual_Y,
            train_Yvar=train_Yvar,
            structures=resolved_structures,
            encoder=material_encoder,
            latent_dim=latent_dim,
            fusion=fusion,  # type: ignore[arg-type]
            projection=projection,
            likelihood=likelihood,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )
        super().__init__(predictor=predictor, residual_model=residual_model)
        self.structures = resolved_structures
        self.material_encoder = material_encoder


__all__ = ["M3GNetDirectPredictor", "M3GNetResidualGPModel"]
