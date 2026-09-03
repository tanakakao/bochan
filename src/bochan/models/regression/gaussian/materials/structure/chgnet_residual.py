"""Residual Gaussian process over CHGNet pretrained energy predictions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from gpytorch.likelihoods import Likelihood
from torch import Tensor, nn

from bochan.composition import CHGNetEncoder, MaterialProcessFusion
from bochan.composition.encoders.chgnet import Checkpoint
from bochan.models.regression.gaussian.deep.chgnet import CHGNetGPModel
from bochan.models.regression.gaussian.deep.deepkernel import InputTransformArg, OutcomeTransformArg
from bochan.models.regression.gaussian.materials.common.residual import (
    DirectMaterialPredictor,
    ResidualMaterialGPModel,
    compute_material_residual_targets,
)


def _validate_structure_bank(structures: Sequence[Any]) -> tuple[Any, ...]:
    if isinstance(structures, (str, bytes)) or not isinstance(structures, Sequence):
        raise TypeError("structures must be a non-empty sequence.")
    resolved = tuple(structures)
    if not resolved:
        raise ValueError("structures must contain at least one structure.")
    return resolved


def _resolve_encoder(
    encoder: CHGNetEncoder | nn.Module | None,
    *,
    checkpoint: Checkpoint | None,
    model_name: str,
    encoder_output_dim: int | None,
    strict_checkpoint: bool,
) -> CHGNetEncoder:
    if isinstance(encoder, CHGNetEncoder):
        if checkpoint is not None:
            raise ValueError(
                "checkpoint must be omitted when encoder is already a CHGNetEncoder."
            )
        if encoder_output_dim is not None and encoder_output_dim != encoder.output_dim:
            raise ValueError(
                "encoder_output_dim does not match CHGNetEncoder.output_dim: "
                f"{encoder_output_dim} != {encoder.output_dim}."
            )
        resolved = encoder
    else:
        resolved = CHGNetEncoder(
            encoder=encoder,
            model_name=model_name,
            checkpoint=checkpoint,
            output_dim=encoder_output_dim,
            strict_checkpoint=strict_checkpoint,
        )
    for parameter in resolved.parameters():
        parameter.requires_grad_(False)
    resolved.eval()
    return resolved


def _predict_energy(encoder: CHGNetEncoder, structures: Sequence[Any]) -> Tensor:
    """Return CHGNet energy predictions through the encoder's native graph path."""

    reference = encoder._floating_reference()
    if reference is None:
        device = torch.device("cpu")
        dtype = torch.get_default_dtype()
    else:
        device = reference.device
        dtype = reference.dtype

    graphs = [
        encoder._prepare_graph(structure, device=device, dtype=dtype)
        for structure in structures
    ]
    prediction = encoder.encoder(graphs, task="e", return_crystal_feas=False)
    if not isinstance(prediction, Mapping):
        raise TypeError("CHGNet energy forward must return a mapping containing 'e'.")
    energy = prediction.get("e")
    if not torch.is_tensor(energy):
        raise TypeError("CHGNet energy forward did not return Tensor 'e'.")
    if energy.ndim == 1:
        energy = energy.unsqueeze(-1)
    expected = (len(structures), 1)
    if tuple(energy.shape) != expected:
        raise ValueError(
            f"CHGNet energy predictions must have shape {expected}, got {tuple(energy.shape)}."
        )
    if not torch.isfinite(energy).all():
        raise FloatingPointError("CHGNet produced non-finite energy predictions.")
    return energy


class CHGNetDirectEnergyPredictor(DirectMaterialPredictor):
    """Expose frozen CHGNet energy as a structure-indexed direct predictor.

    The first column of ``X`` is interpreted as an integer index into the
    structure bank. Remaining process columns are intentionally ignored by the
    pretrained baseline and are learned by the residual GP correction.
    """

    def __init__(self, encoder: CHGNetEncoder, structures: Sequence[Any]) -> None:
        super().__init__()
        if not isinstance(encoder, CHGNetEncoder):
            raise TypeError("encoder must be a CHGNetEncoder.")
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
        baseline_unique = _predict_energy(self.encoder, selected)
        baseline = baseline_unique[inverse.to(device=baseline_unique.device)]
        baseline = baseline.to(device=X.device, dtype=X.dtype)
        return baseline.reshape(*X.shape[:-1], 1)


class CHGNetResidualGPModel(ResidualMaterialGPModel):
    """Correct frozen CHGNet energy predictions with an exact GP residual.

    ``train_Y`` must represent the same energy quantity and units as the
    selected CHGNet pretrained model. The GP is trained on
    ``train_Y - CHGNet_energy`` while :meth:`posterior` returns the corrected
    energy posterior in the original target scale.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        structures: Sequence[Any],
        encoder: CHGNetEncoder | nn.Module | None = None,
        checkpoint: Checkpoint | None = None,
        model_name: str = "0.3.0",
        encoder_output_dim: int | None = None,
        latent_dim: int = 32,
        fusion: str | MaterialProcessFusion = "concat",
        projection: nn.Module | None = None,
        strict_checkpoint: bool = True,
        likelihood: Likelihood | None = None,
        input_transform: InputTransformArg = "DEFAULT",
        outcome_transform: OutcomeTransformArg = "DEFAULT",
    ) -> None:
        resolved_structures = _validate_structure_bank(structures)
        material_encoder = _resolve_encoder(
            encoder,
            checkpoint=checkpoint,
            model_name=model_name,
            encoder_output_dim=encoder_output_dim,
            strict_checkpoint=strict_checkpoint,
        )
        predictor = CHGNetDirectEnergyPredictor(material_encoder, resolved_structures)
        residual_Y = compute_material_residual_targets(train_X, train_Y, predictor)
        residual_model = CHGNetGPModel(
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


__all__ = ["CHGNetDirectEnergyPredictor", "CHGNetResidualGPModel"]
