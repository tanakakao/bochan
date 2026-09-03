"""Correlated multi-output residual GPs for pretrained structure models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from gpytorch.likelihoods import Likelihood
from torch import Tensor, nn

from bochan.composition import CHGNetEncoder, M3GNetEncoder, MACEEncoder, MaterialProcessFusion
from bochan.composition.encoders.chgnet import Checkpoint
from bochan.models.regression.gaussian.deep.chgnet_multitask import (
    CHGNetMixedMultiTaskGPModel,
    CHGNetMultiTaskGPModel,
)
from bochan.models.regression.gaussian.deep.deepkernel import InputTransformArg, OutcomeTransformArg
from bochan.models.regression.gaussian.deep.m3gnet_multitask import (
    M3GNetMixedMultiTaskGPModel,
    M3GNetMultiTaskGPModel,
)
from bochan.models.regression.gaussian.deep.mace_multitask import (
    MACEMixedMultiTaskGPModel,
    MACEMultiTaskGPModel,
)
from bochan.models.regression.gaussian.materials.common.residual import (
    ResidualMaterialGPModel,
    compute_material_residual_targets,
)
from bochan.models.regression.gaussian.materials.common.residual_multitask import (
    SingleOutputBaselineAdapter,
)

from .chgnet_residual import (
    CHGNetDirectEnergyPredictor,
    _resolve_encoder as _resolve_chgnet_encoder,
)
from .m3gnet_residual import (
    M3GNetDirectPredictor,
    _resolve_encoder as _resolve_m3gnet_encoder,
)
from .mace_residual import (
    MACEDirectEnergyPredictor,
    _Pooling,
    _resolve_encoder as _resolve_mace_encoder,
)

_M3GNET_DEFAULT_MODEL_NAME = "M3GNet-PES-MatPES-PBE-2025.2"
_MACE_DEFAULT_MODEL_NAME = "medium-mpa-0"


def _validate_wide_targets(train_Y: Tensor) -> int:
    if not isinstance(train_Y, Tensor):
        raise TypeError("train_Y must be a Tensor.")
    if train_Y.ndim != 2 or train_Y.shape[-1] < 2:
        raise ValueError("Residual multitask models require train_Y with shape [n, m], m >= 2.")
    return int(train_Y.shape[-1])


def _wide_predictor(scalar_predictor, train_Y: Tensor, pretrained_output_index: int):
    return SingleOutputBaselineAdapter(
        scalar_predictor,
        output_dim=_validate_wide_targets(train_Y),
        output_index=pretrained_output_index,
    )


class CHGNetMultiTaskResidualGPModel(ResidualMaterialGPModel):
    """Correlated wide-target residual GP with CHGNet energy on one output."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        structures: Sequence[Any],
        pretrained_output_index: int = 0,
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
        structures = tuple(structures)
        material_encoder = _resolve_chgnet_encoder(
            encoder,
            checkpoint=checkpoint,
            model_name=model_name,
            encoder_output_dim=encoder_output_dim,
            strict_checkpoint=strict_checkpoint,
        )
        predictor = _wide_predictor(
            CHGNetDirectEnergyPredictor(material_encoder, structures),
            train_Y,
            pretrained_output_index,
        )
        residual_Y = compute_material_residual_targets(train_X, train_Y, predictor)
        residual_model = CHGNetMultiTaskGPModel(
            train_X=train_X,
            train_Y=residual_Y,
            train_Yvar=train_Yvar,
            structures=structures,
            encoder=material_encoder,
            latent_dim=latent_dim,
            fusion=fusion,
            projection=projection,
            likelihood=likelihood,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )
        super().__init__(predictor=predictor, residual_model=residual_model)
        self.material_encoder = material_encoder
        self.structures = structures
        self.pretrained_output_index = predictor.output_index


class CHGNetMixedMultiTaskResidualGPModel(ResidualMaterialGPModel):
    """Correlated CHGNet residual GP with categorical process variables."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        cat_dims: Sequence[int],
        structures: Sequence[Any],
        pretrained_output_index: int = 0,
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
        structures = tuple(structures)
        material_encoder = _resolve_chgnet_encoder(
            encoder,
            checkpoint=checkpoint,
            model_name=model_name,
            encoder_output_dim=encoder_output_dim,
            strict_checkpoint=strict_checkpoint,
        )
        predictor = _wide_predictor(
            CHGNetDirectEnergyPredictor(material_encoder, structures),
            train_Y,
            pretrained_output_index,
        )
        residual_Y = compute_material_residual_targets(train_X, train_Y, predictor)
        residual_model = CHGNetMixedMultiTaskGPModel(
            train_X=train_X,
            train_Y=residual_Y,
            train_Yvar=train_Yvar,
            cat_dims=cat_dims,
            structures=structures,
            encoder=material_encoder,
            latent_dim=latent_dim,
            fusion=fusion,
            projection=projection,
            likelihood=likelihood,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )
        super().__init__(predictor=predictor, residual_model=residual_model)
        self.material_encoder = material_encoder
        self.structures = structures
        self.pretrained_output_index = predictor.output_index


class M3GNetMultiTaskResidualGPModel(ResidualMaterialGPModel):
    """Correlated wide-target residual GP with M3GNet baseline on one output."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        structures: Sequence[Any],
        pretrained_output_index: int = 0,
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
        structures = tuple(structures)
        material_encoder = _resolve_m3gnet_encoder(
            encoder,
            model_name=model_name,
            encoder_output_dim=encoder_output_dim,
        )
        predictor = _wide_predictor(
            M3GNetDirectPredictor(material_encoder, structures),
            train_Y,
            pretrained_output_index,
        )
        residual_Y = compute_material_residual_targets(train_X, train_Y, predictor)
        residual_model = M3GNetMultiTaskGPModel(
            train_X=train_X,
            train_Y=residual_Y,
            train_Yvar=train_Yvar,
            structures=structures,
            encoder=material_encoder,
            latent_dim=latent_dim,
            fusion=fusion,
            projection=projection,
            likelihood=likelihood,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )
        super().__init__(predictor=predictor, residual_model=residual_model)
        self.material_encoder = material_encoder
        self.structures = structures
        self.pretrained_output_index = predictor.output_index


class M3GNetMixedMultiTaskResidualGPModel(ResidualMaterialGPModel):
    """Correlated M3GNet residual GP with categorical process variables."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        cat_dims: Sequence[int],
        structures: Sequence[Any],
        pretrained_output_index: int = 0,
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
        structures = tuple(structures)
        material_encoder = _resolve_m3gnet_encoder(
            encoder,
            model_name=model_name,
            encoder_output_dim=encoder_output_dim,
        )
        predictor = _wide_predictor(
            M3GNetDirectPredictor(material_encoder, structures),
            train_Y,
            pretrained_output_index,
        )
        residual_Y = compute_material_residual_targets(train_X, train_Y, predictor)
        residual_model = M3GNetMixedMultiTaskGPModel(
            train_X=train_X,
            train_Y=residual_Y,
            train_Yvar=train_Yvar,
            cat_dims=cat_dims,
            structures=structures,
            encoder=material_encoder,
            latent_dim=latent_dim,
            fusion=fusion,
            projection=projection,
            likelihood=likelihood,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )
        super().__init__(predictor=predictor, residual_model=residual_model)
        self.material_encoder = material_encoder
        self.structures = structures
        self.pretrained_output_index = predictor.output_index


class MACEMultiTaskResidualGPModel(ResidualMaterialGPModel):
    """Correlated wide-target residual GP with MACE energy on one output."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        structures: Sequence[Any],
        pretrained_output_index: int = 0,
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
        structures = tuple(structures)
        material_encoder = _resolve_mace_encoder(
            encoder,
            model_name=model_name,
            num_layers=num_layers,
            pooling=pooling,
            head=head,
        )
        predictor = _wide_predictor(
            MACEDirectEnergyPredictor(material_encoder, structures),
            train_Y,
            pretrained_output_index,
        )
        residual_Y = compute_material_residual_targets(train_X, train_Y, predictor)
        residual_model = MACEMultiTaskGPModel(
            train_X=train_X,
            train_Y=residual_Y,
            train_Yvar=train_Yvar,
            structures=structures,
            encoder=material_encoder,
            latent_dim=latent_dim,
            fusion=fusion,
            projection=projection,
            likelihood=likelihood,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )
        super().__init__(predictor=predictor, residual_model=residual_model)
        self.material_encoder = material_encoder
        self.structures = structures
        self.pretrained_output_index = predictor.output_index
        self.head = material_encoder.head


class MACEMixedMultiTaskResidualGPModel(ResidualMaterialGPModel):
    """Correlated MACE residual GP with categorical process variables."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        cat_dims: Sequence[int],
        structures: Sequence[Any],
        pretrained_output_index: int = 0,
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
        structures = tuple(structures)
        material_encoder = _resolve_mace_encoder(
            encoder,
            model_name=model_name,
            num_layers=num_layers,
            pooling=pooling,
            head=head,
        )
        predictor = _wide_predictor(
            MACEDirectEnergyPredictor(material_encoder, structures),
            train_Y,
            pretrained_output_index,
        )
        residual_Y = compute_material_residual_targets(train_X, train_Y, predictor)
        residual_model = MACEMixedMultiTaskGPModel(
            train_X=train_X,
            train_Y=residual_Y,
            train_Yvar=train_Yvar,
            cat_dims=cat_dims,
            structures=structures,
            encoder=material_encoder,
            latent_dim=latent_dim,
            fusion=fusion,
            projection=projection,
            likelihood=likelihood,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )
        super().__init__(predictor=predictor, residual_model=residual_model)
        self.material_encoder = material_encoder
        self.structures = structures
        self.pretrained_output_index = predictor.output_index
        self.head = material_encoder.head


__all__ = [
    "CHGNetMultiTaskResidualGPModel",
    "CHGNetMixedMultiTaskResidualGPModel",
    "M3GNetMultiTaskResidualGPModel",
    "M3GNetMixedMultiTaskResidualGPModel",
    "MACEMultiTaskResidualGPModel",
    "MACEMixedMultiTaskResidualGPModel",
]
