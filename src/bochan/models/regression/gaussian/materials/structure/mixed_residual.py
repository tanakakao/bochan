"""Mixed-input residual Gaussian processes for pretrained structure models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from gpytorch.likelihoods import Likelihood
from torch import Tensor, nn

from bochan.composition import CHGNetEncoder, M3GNetEncoder, MACEEncoder, MaterialProcessFusion
from bochan.composition.encoders.chgnet import Checkpoint
from bochan.models.regression.gaussian.deep.chgnet import CHGNetMixedGPModel
from bochan.models.regression.gaussian.deep.deepkernel import InputTransformArg, OutcomeTransformArg
from bochan.models.regression.gaussian.deep.m3gnet import M3GNetMixedGPModel
from bochan.models.regression.gaussian.deep.mace_mixed import MACEMixedGPModel
from bochan.models.regression.gaussian.materials.common.residual import (
    ResidualMaterialGPModel,
    compute_material_residual_targets,
)

from .chgnet_residual import CHGNetDirectEnergyPredictor, _resolve_encoder as _resolve_chgnet_encoder
from .m3gnet_residual import (
    M3GNetDirectPredictor,
    _DEFAULT_MODEL_NAME as _M3GNET_DEFAULT_MODEL_NAME,
    _resolve_encoder as _resolve_m3gnet_encoder,
)
from .mace_residual import (
    MACEDirectEnergyPredictor,
    _DEFAULT_MODEL_NAME as _MACE_DEFAULT_MODEL_NAME,
    _Pooling,
    _resolve_encoder as _resolve_mace_encoder,
)


class CHGNetMixedResidualGPModel(ResidualMaterialGPModel):
    """Correct CHGNet energy with a mixed exact-GP residual."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        train_Yvar: Tensor | None = None,
        *,
        structures: Sequence[Any],
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
        predictor = CHGNetDirectEnergyPredictor(material_encoder, structures)
        residual_Y = compute_material_residual_targets(train_X, train_Y, predictor)
        residual_model = CHGNetMixedGPModel(
            train_X=train_X,
            train_Y=residual_Y,
            cat_dims=cat_dims,
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
        self.structures = structures
        self.material_encoder = material_encoder


class M3GNetMixedResidualGPModel(ResidualMaterialGPModel):
    """Correct M3GNet scalar predictions with a mixed exact-GP residual."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        train_Yvar: Tensor | None = None,
        *,
        structures: Sequence[Any],
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
        predictor = M3GNetDirectPredictor(material_encoder, structures)
        residual_Y = compute_material_residual_targets(train_X, train_Y, predictor)
        residual_model = M3GNetMixedGPModel(
            train_X=train_X,
            train_Y=residual_Y,
            cat_dims=cat_dims,
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
        self.structures = structures
        self.material_encoder = material_encoder


class MACEMixedResidualGPModel(ResidualMaterialGPModel):
    """Correct MACE energy with a mixed exact-GP residual."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        train_Yvar: Tensor | None = None,
        *,
        structures: Sequence[Any],
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
        predictor = MACEDirectEnergyPredictor(material_encoder, structures)
        residual_Y = compute_material_residual_targets(train_X, train_Y, predictor)
        residual_model = MACEMixedGPModel(
            train_X=train_X,
            train_Y=residual_Y,
            cat_dims=cat_dims,
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
        self.structures = structures
        self.material_encoder = material_encoder
        self.head = material_encoder.head


__all__ = [
    "CHGNetMixedResidualGPModel",
    "M3GNetMixedResidualGPModel",
    "MACEMixedResidualGPModel",
]
