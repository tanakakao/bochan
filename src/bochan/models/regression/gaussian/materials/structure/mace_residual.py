"""Residual Gaussian process over MACE pretrained energy predictions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
from gpytorch.likelihoods import Likelihood
from torch import Tensor, nn

from bochan.composition import MACEEncoder, MaterialProcessFusion
from bochan.models.regression.gaussian.deep.deepkernel import InputTransformArg, OutcomeTransformArg
from bochan.models.regression.gaussian.deep.mace import MACEGPModel
from bochan.models.regression.gaussian.materials.common.residual import (
    DirectMaterialPredictor,
    ResidualMaterialGPModel,
    compute_material_residual_targets,
)

_DEFAULT_MODEL_NAME = "medium-mpa-0"
_Pooling = Literal["mean", "sum"]


def _validate_structure_bank(structures: Sequence[Any]) -> tuple[Any, ...]:
    if isinstance(structures, (str, bytes)) or not isinstance(structures, Sequence):
        raise TypeError("structures must be a non-empty sequence.")
    resolved = tuple(structures)
    if not resolved:
        raise ValueError("structures must contain at least one structure.")
    return resolved


def _resolve_encoder(
    encoder: MACEEncoder | nn.Module | None,
    *,
    model_name: str,
    num_layers: int,
    pooling: _Pooling,
    head: str | None,
) -> MACEEncoder:
    if isinstance(encoder, MACEEncoder):
        if head is not None and head != encoder.head:
            raise ValueError(
                f"Requested MACE head {head!r} does not match encoder.head {encoder.head!r}."
            )
        resolved = encoder
    else:
        resolved = MACEEncoder(
            encoder=encoder,
            model_name=model_name,
            num_layers=num_layers,
            pooling=pooling,
            head=head,
        )
    for parameter in resolved.parameters():
        parameter.requires_grad_(False)
    resolved.eval()
    return resolved


def _predict_energy_one(encoder: MACEEncoder, structure: Any) -> Tensor:
    batch = encoder._build_batch(structure)
    output = encoder.encoder(
        batch,
        compute_force=False,
        compute_virials=False,
        compute_stress=False,
    )
    if not isinstance(output, Mapping):
        raise TypeError("Raw MACE forward must return a mapping containing 'energy'.")
    energy = output.get("energy")
    if not torch.is_tensor(energy):
        raise TypeError("Raw MACE forward did not return Tensor 'energy'.")
    energy = energy.reshape(-1)
    if energy.numel() != 1:
        raise ValueError(
            "MACE direct prediction must return exactly one scalar energy per structure."
        )
    if not torch.isfinite(energy).all():
        raise FloatingPointError("MACE produced non-finite energy predictions.")
    return energy.reshape(1, 1)


def _predict_energy(encoder: MACEEncoder, structures: Sequence[Any]) -> Tensor:
    predictions = [_predict_energy_one(encoder, structure) for structure in structures]
    return torch.cat(predictions, dim=0)


class MACEDirectEnergyPredictor(DirectMaterialPredictor):
    """Expose frozen MACE energy as a structure-indexed direct predictor."""

    def __init__(self, encoder: MACEEncoder, structures: Sequence[Any]) -> None:
        super().__init__()
        if not isinstance(encoder, MACEEncoder):
            raise TypeError("encoder must be a MACEEncoder.")
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
        return baseline.to(device=X.device, dtype=X.dtype).reshape(*X.shape[:-1], 1)


class MACEResidualGPModel(ResidualMaterialGPModel):
    """Correct frozen MACE energy predictions with an exact GP residual.

    ``train_Y`` must use the same energy definition and units as the selected
    MACE pretrained model and head. Process columns do not affect the pretrained
    baseline but remain inputs to the GP residual correction.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        structures: Sequence[Any],
        encoder: MACEEncoder | nn.Module | None = None,
        model_name: str = _DEFAULT_MODEL_NAME,
        num_layers: int = -1,
        pooling: _Pooling = "mean",
        head: str | None = None,
        latent_dim: int = 32,
        fusion: Literal["concat"] | MaterialProcessFusion = "concat",
        projection: nn.Module | None = None,
        likelihood: Likelihood | None = None,
        input_transform: InputTransformArg = "DEFAULT",
        outcome_transform: OutcomeTransformArg = "DEFAULT",
    ) -> None:
        resolved_structures = _validate_structure_bank(structures)
        material_encoder = _resolve_encoder(
            encoder,
            model_name=model_name,
            num_layers=num_layers,
            pooling=pooling,
            head=head,
        )
        predictor = MACEDirectEnergyPredictor(material_encoder, resolved_structures)
        residual_Y = compute_material_residual_targets(train_X, train_Y, predictor)
        residual_model = MACEGPModel(
            train_X=train_X,
            train_Y=residual_Y,
            train_Yvar=train_Yvar,
            structures=resolved_structures,
            encoder=material_encoder,
            latent_dim=latent_dim,
            fusion=fusion,
            projection=projection,
            likelihood=likelihood,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )
        super().__init__(predictor=predictor, residual_model=residual_model)
        self.structures = resolved_structures
        self.material_encoder = material_encoder
        self.head = material_encoder.head


__all__ = ["MACEDirectEnergyPredictor", "MACEResidualGPModel"]
