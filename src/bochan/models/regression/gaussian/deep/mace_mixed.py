"""Mixed-input Gaussian processes over invariant MACE crystal representations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

from botorch.models.transforms.input import Normalize
from botorch.utils.transforms import normalize_indices
from gpytorch.likelihoods import Likelihood
from torch import Tensor, nn

from bochan.composition import MACEEncoder, MaterialProcessFusion

from .deepkernel import InputTransformArg, OutcomeTransformArg
from .deepkernel_configurable import DeepKernelGaussianMixedGPModel
from .mace import (
    _DEFAULT_MODEL_NAME,
    _configure_dkl_encoder,
    _MACEGPFeatureExtractor,
    _Pooling,
    _resolve_material_encoder,
    _validate_model_inputs,
    _validate_trainable_encoder_layers,
)


def _resolve_mixed_input_transform(
    train_X: Tensor,
    *,
    cat_dims: Sequence[int],
    input_transform: InputTransformArg,
) -> InputTransformArg:
    """Normalize numeric process columns while preserving selectors/categories."""

    if not isinstance(input_transform, str) or input_transform.upper() != "DEFAULT":
        return input_transform
    categorical = set(cat_dims)
    process_dims = [
        index
        for index in range(1, train_X.shape[-1])
        if index not in categorical
    ]
    if not process_dims:
        return None
    return Normalize(d=train_X.shape[-1], indices=process_dims)


class MACEMixedGPModel(DeepKernelGaussianMixedGPModel):
    """Exact mixed GP over frozen invariant MACE crystal representations.

    Column 0 is the discrete structure selector. ``cat_dims`` contains only
    integer-coded categorical process columns. The remaining process columns
    are numeric and are fused with the selected MACE crystal representation.
    """

    _supports_cache_root = False

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
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
        if train_X.ndim != 2:
            raise ValueError("train_X must have shape [n, d].")
        if train_X.shape[-1] < 2:
            raise ValueError(
                "MACEMixedGPModel requires a structure-index column and at least "
                "one categorical process column."
            )
        model_class_name = self.__class__.__name__
        if train_Y.ndim > 1 and train_Y.shape[-1] != 1:
            raise ValueError(
                f"{model_class_name} currently supports single-output train_Y only."
            )
        if train_Yvar is not None:
            raise NotImplementedError(
                f"{model_class_name} does not yet support train_Yvar."
            )
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a non-empty string.")
        if pooling not in {"mean", "sum"}:
            raise ValueError("pooling must be 'mean' or 'sum'.")

        d = train_X.shape[-1]
        normalized_cat_dims = normalize_indices(indices=list(cat_dims), d=d)
        if not normalized_cat_dims:
            raise ValueError(
                "MACEMixedGPModel requires at least one categorical process dimension."
            )
        if 0 in normalized_cat_dims:
            raise ValueError(
                "The structure-index column (feature 0) is handled by MACE and "
                "cannot be included in cat_dims."
            )

        validated_structures = tuple(structures)
        if not validated_structures:
            raise ValueError("structures must contain at least one structure.")
        _validate_model_inputs(
            train_X,
            num_structures=len(validated_structures),
            input_dim=d,
        )

        continuous_dims = sorted(set(range(d)) - set(normalized_cat_dims))
        if not continuous_dims or continuous_dims[0] != 0:
            raise RuntimeError(
                "The MACE mixed continuous branch must retain structure-index feature 0."
            )
        process_dims = [index for index in continuous_dims if index != 0]

        material_encoder = _resolve_material_encoder(
            encoder,
            model_name=model_name,
            num_layers=num_layers,
            pooling=pooling,
            head=head,
        )
        feature_extractor = _MACEGPFeatureExtractor(
            material_encoder=material_encoder,
            structures=validated_structures,
            process_dim=len(process_dims),
            latent_dim=latent_dim,
            fusion=fusion,
            projection=projection,
        ).to(train_X)
        resolved_input_transform = _resolve_mixed_input_transform(
            train_X,
            cat_dims=normalized_cat_dims,
            input_transform=input_transform,
        )

        self._continuous_process_dims = tuple(process_dims)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=normalized_cat_dims,
            train_Yvar=None,
            likelihood=likelihood,
            input_transform=resolved_input_transform,
            outcome_transform=outcome_transform,
            feature_extractor=feature_extractor,
            latent_dim=latent_dim,
        )

        transformed_train_X = cast(tuple[Tensor, ...], self.deepkernel.train_inputs)[0]
        continuous_train_X = transformed_train_X[..., self.deepkernel.ord_dims]
        self.mace_feature_extractor.validate_input(continuous_train_X)

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> MACEMixedGPModel:
        module = super()._apply(fn, recurse=recurse)
        reference = next(
            (
                value
                for value in (*self.parameters(), *self.buffers())
                if value.is_floating_point()
            ),
            None,
        )
        if reference is not None:
            self._model_dtype = reference.dtype
            self._model_device = reference.device
        return cast(MACEMixedGPModel, module)

    @property
    def mace_feature_extractor(self) -> _MACEGPFeatureExtractor:
        return cast(_MACEGPFeatureExtractor, self.deepkernel.feature_extractor)

    @property
    def material_encoder(self) -> MACEEncoder:
        return cast(MACEEncoder, self.mace_feature_extractor.material_encoder)

    @property
    def projection(self) -> nn.Module:
        return self.mace_feature_extractor.projection

    @property
    def fusion(self) -> MaterialProcessFusion:
        return self.mace_feature_extractor.fusion

    @property
    def structures(self) -> tuple[Any, ...]:
        return self.mace_feature_extractor.structures

    @property
    def num_structures(self) -> int:
        return self.mace_feature_extractor.num_structures

    @property
    def process_dim(self) -> int:
        return self.mace_feature_extractor.process_dim

    @property
    def continuous_process_dims(self) -> tuple[int, ...]:
        return self._continuous_process_dims

    @property
    def categorical_process_dim(self) -> int:
        return len(self.cat_dims)

    @property
    def structure_feature_cache_enabled(self) -> bool:
        return self.mace_feature_extractor.material_feature_cache_enabled

    def clear_structure_feature_cache(self) -> None:
        self.mace_feature_extractor.clear_material_feature_cache()


class MACEMixedDKLModel(MACEMixedGPModel):
    """Mixed exact GP that jointly fine-tunes the MACE representation backbone."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
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
        trainable_encoder_layers: int | Literal["all"] = 1,
        likelihood: Likelihood | None = None,
        input_transform: InputTransformArg = "DEFAULT",
        outcome_transform: OutcomeTransformArg = "DEFAULT",
    ) -> None:
        resolved_trainable_layers = _validate_trainable_encoder_layers(
            trainable_encoder_layers
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=cat_dims,
            train_Yvar=train_Yvar,
            structures=structures,
            encoder=encoder,
            model_name=model_name,
            num_layers=num_layers,
            pooling=pooling,
            head=head,
            latent_dim=latent_dim,
            fusion=fusion,
            projection=projection,
            likelihood=likelihood,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )
        training_mode, trainable_modules = _configure_dkl_encoder(
            self.material_encoder,
            resolved_trainable_layers,
        )
        self._trainable_encoder_layers = resolved_trainable_layers
        self.mace_feature_extractor._configure_encoder_training(
            training_mode,
            trainable_modules,
        )

    @property
    def trainable_encoder_layers(self) -> int | Literal["all"]:
        return self._trainable_encoder_layers


__all__ = ["MACEMixedDKLModel", "MACEMixedGPModel"]
