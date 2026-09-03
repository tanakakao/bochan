"""Correlated multi-output residual GPs for pretrained structure models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from gpytorch.likelihoods import Likelihood
from torch import Tensor, nn

from bochan.composition import CHGNetEncoder, M3GNetEncoder, MACEEncoder, MaterialProcessFusion
from bochan.composition.encoders.chgnet import Checkpoint
from bochan.models.regression.gaussian.deep.chgnet_multitask import CHGNetMultiTaskGPModel
from bochan.models.regression.gaussian.deep.deepkernel import InputTransformArg, OutcomeTransformArg
from bochan.models.regression.gaussian.deep.m3gnet_multitask import M3GNetMultiTaskGPModel
from bochan.models.regression.gaussian.deep.mace_multitask import MACEMultiTaskGPModel
from bochan.models.regression.gaussian.materials.common.residual import (
    ResidualMaterialGPModel,
    RoutedDirectMaterialPredictor,
    compute_material_residual_targets,
)

from .chgnet_residual import (
    CHGNetDirectEnergyPredictor,
    _resolve_encoder as _resolve_chgnet_encoder,
    _validate_structure_bank as _validate_chgnet_structures,
)
from .m3gnet_residual import (
    M3GNetDirectPredictor,
    _resolve_encoder as _resolve_m3gnet_encoder,
    _validate_structure_bank as _validate_m3gnet_structures,
)
from .mace_residual import (
    MACEDirectEnergyPredictor,
    _Pooling,
    _resolve_encoder as _resolve_mace_encoder,
    _validate_structure_bank as _validate_mace_structures,
)

_M3GNET_DEFAULT_MODEL_NAME = "M3GNet-PES-MatPES-PBE-2025.2"
_MACE_DEFAULT_MODEL_NAME = "medium-mpa-0"


def _validate_wide_targets(train_Y: Tensor, baseline_output_index: int) -> tuple[int, int]:
    if not isinstance(train_Y, Tensor):
        raise TypeError("train_Y must be a Tensor.")
    if train_Y.ndim != 2 or train_Y.shape[-1] < 2:
        raise ValueError("Multi-output residual GP requires train_Y with shape [n, m], m >= 2.")
    num_outputs = int(train_Y.shape[-1])
    if isinstance(baseline_output_index, bool) or not isinstance(baseline_output_index, int):
        raise TypeError("baseline_output_index must be an integer.")
    resolved_index = (
        baseline_output_index + num_outputs
        if baseline_output_index < 0
        else baseline_output_index
    )
    if resolved_index < 0 or resolved_index >= num_outputs:
        raise ValueError("baseline_output_index is outside the train_Y output range.")
    return num_outputs, resolved_index


class CHGNetMultiTaskResidualGPModel(ResidualMaterialGPModel):
    """Correlated wide-output GP with CHGNet energy routed to one target column."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        structures: Sequence[Any],
        baseline_output_index: int = 0,
        encoder: CHGNetEncoder | nn.Module | None = None,
        checkpoint: Checkpoint | None = None,
        model_name: str = "0.3.0",
        encoder_output_dim: int | None = None,
        latent_dim: int = 32,
        fusion: Literal["concat"] | MaterialProcessFusion = "concat",
        projection: nn.Module | None = None,
        strict_checkpoint: bool = True,
        likelihood: Likelihood | None = None,
        input_transform: InputTransformArg = "DEFAULT",
        outcome_transform: OutcomeTransformArg = "DEFAULT",
    ) -> None:
        num_outputs, resolved_output = _validate_wide_targets(train_Y, baseline_output_index)
        resolved_structures = _validate_chgnet_structures(structures)
        material_encoder = _resolve_chgnet_encoder(
            encoder,
            checkpoint=checkpoint,
            model_name=model_name,
            encoder_output_dim=encoder_output_dim,
            strict_checkpoint=strict_checkpoint,
        )
        scalar_predictor = CHGNetDirectEnergyPredictor(material_encoder, resolved_structures)
        predictor = RoutedDirectMaterialPredictor(
            scalar_predictor,
            output_dim=num_outputs,
            output_index=resolved_output,
        )
        residual_Y = compute_material_residual_targets(train_X, train_Y, predictor)
        residual_model = CHGNetMultiTaskGPModel(
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
        self.baseline_output_index = resolved_output


class M3GNetMultiTaskResidualGPModel(ResidualMaterialGPModel):
    """Correlated wide-output GP with M3GNet scalar baseline on one target column."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        structures: Sequence[Any],
        baseline_output_index: int = 0,
        encoder: M3GNetEncoder | nn.Module | None = None,
        model_name: str = _M3GNET_DEFAULT_MODEL_NAME,
        encoder_output_dim: int | None = None,
        latent_dim: int = 32,
        fusion: Literal["concat"] | MaterialProcessFusion = "concat",
        projection: nn.Module | None = None,
        likelihood: Likelihood | None = None,
        input_transform: InputTransformArg = "DEFAULT",
        outcome_transform: OutcomeTransformArg = "DEFAULT",
    ) -> None:
        num_outputs, resolved_output = _validate_wide_targets(train_Y, baseline_output_index)
        resolved_structures = _validate_m3gnet_structures(structures)
        material_encoder = _resolve_m3gnet_encoder(
            encoder,
            model_name=model_name,
            encoder_output_dim=encoder_output_dim,
        )
        scalar_predictor = M3GNetDirectPredictor(material_encoder, resolved_structures)
        predictor = RoutedDirectMaterialPredictor(
            scalar_predictor,
            output_dim=num_outputs,
            output_index=resolved_output,
        )
        residual_Y = compute_material_residual_targets(train_X, train_Y, predictor)
        residual_model = M3GNetMultiTaskGPModel(
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
        self.baseline_output_index = resolved_output


class MACEMultiTaskResidualGPModel(ResidualMaterialGPModel):
    """Correlated wide-output GP with selected-head MACE energy on one target column."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        structures: Sequence[Any],
        baseline_output_index: int = 0,
        encoder: MACEEncoder | nn.Module | None = None,
        model_name: str = _MACE_DEFAULT_MODEL_NAME,
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
        num_outputs, resolved_output = _validate_wide_targets(train_Y, baseline_output_index)
        resolved_structures = _validate_mace_structures(structures)
        material_encoder = _resolve_mace_encoder(
            encoder,
            model_name=model_name,
            num_layers=num_layers,
            pooling=pooling,
            head=head,
        )
        scalar_predictor = MACEDirectEnergyPredictor(material_encoder, resolved_structures)
        predictor = RoutedDirectMaterialPredictor(
            scalar_predictor,
            output_dim=num_outputs,
            output_index=resolved_output,
        )
        residual_Y = compute_material_residual_targets(train_X, train_Y, predictor)
        residual_model = MACEMultiTaskGPModel(
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
        self.baseline_output_index = resolved_output
        self.head = material_encoder.head


__all__ = [
    "CHGNetMultiTaskResidualGPModel",
    "M3GNetMultiTaskResidualGPModel",
    "MACEMultiTaskResidualGPModel",
]
