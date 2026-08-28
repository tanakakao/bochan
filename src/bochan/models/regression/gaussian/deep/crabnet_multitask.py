"""Correlated multi-output Gaussian processes over shared CrabNet representations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, cast

from botorch.utils.transforms import normalize_indices
from gpytorch.kernels import MultitaskKernel
from gpytorch.likelihoods import Likelihood
from torch import Tensor, nn

from bochan.composition import CrabNetEncoder, MaterialProcessFusion
from bochan.composition.encoders.crabnet import Checkpoint

from .crabnet import (
    _configure_dkl_encoder,
    _resolve_material_encoder,
    _validate_trainable_encoder_layers,
)
from .crabnet_mixed import (
    CrabNetMixedGPModel,
    _CrabNetMixedContinuousFeatureExtractor,
)
from .crabnet_mixed_dkl import (
    CrabNetMixedDKLModel,
    _CrabNetMixedDKLFeatureExtractor,
    _resolve_category_cardinalities,
    _resolve_category_embedding_dims,
)
from .deepkernel import InputTransformArg, OutcomeTransformArg
from .deepkernel_configurable import (
    DeepKernelGaussianGPModel,
    DeepKernelGaussianMixedGPModel,
)
from .material import (
    CompositionMaterialInputTransform,
    MaterialGPFeatureExtractor,
    _resolve_composition_input_transform,
    _validate_composition_element_ids,
    _validate_composition_model_inputs,
)


class _CorrelatedCrabNetMultiTaskMixin:
    """Expose the common correlated-task diagnostics contract."""

    deepkernel: object
    num_outputs: int

    def _validate_correlated_kernel(self) -> None:
        covar_module = getattr(self.deepkernel, "covar_module", None)
        if not isinstance(covar_module, MultitaskKernel):
            raise RuntimeError(
                f"{self.__class__.__name__} requires a correlated MultitaskKernel."
            )

    @property
    def num_tasks(self) -> int:
        """Return the number of correlated material-property tasks."""

        return int(self.num_outputs)

    @property
    def task_covar_module(self) -> nn.Module:
        """Expose the learned task covariance module for diagnostics."""

        self._validate_correlated_kernel()
        return cast(MultitaskKernel, self.deepkernel.covar_module).task_covar_module

    @property
    def task_kernel(self) -> nn.Module:
        """Alias the task covariance module to the common multitask interface."""

        return self.task_covar_module


def _validate_multitask_targets(
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor | None,
    *,
    model_name: str,
    latent_dim: int,
) -> None:
    """Validate the shared wide-target contract used by CrabNet multitask models."""

    if train_X.ndim != 2:
        raise ValueError("train_X must have shape [n, d].")
    if train_Y.ndim != 2 or train_Y.shape[-1] < 2:
        raise ValueError(
            f"{model_name} requires wide train_Y with shape [n, m] and at least "
            "two target columns."
        )
    if train_Yvar is not None:
        raise NotImplementedError(f"{model_name} does not yet support train_Yvar.")
    if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim <= 0:
        raise ValueError("latent_dim must be a positive integer.")


class CrabNetMultiTaskGPModel(
    _CorrelatedCrabNetMultiTaskMixin,
    DeepKernelGaussianGPModel,
):
    """Correlated multi-output GP with one shared frozen CrabNet encoder.

    ``train_Y`` remains wide with shape ``[n, m]``. Composition and continuous
    process variables are mapped to one shared latent representation and the
    exact GP learns cross-property covariance with ``MultitaskKernel``. This is
    intentionally different from bochan's independent CrabNet ``ModelListGP``
    path, where every output owns a separate encoder, projection, and GP.
    """

    _supports_cache_root = False

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        element_ids: Tensor,
        encoder: CrabNetEncoder | nn.Module | None = None,
        checkpoint: Checkpoint | None = None,
        latent_dim: int = 32,
        fusion: Literal["concat"] | MaterialProcessFusion = "concat",
        projection: nn.Module | None = None,
        strict_checkpoint: bool = True,
        likelihood: Likelihood | None = None,
        input_transform: InputTransformArg = "DEFAULT",
        outcome_transform: OutcomeTransformArg = "DEFAULT",
    ) -> None:
        _validate_multitask_targets(
            train_X,
            train_Y,
            train_Yvar,
            model_name=self.__class__.__name__,
            latent_dim=latent_dim,
        )

        validated_element_ids = _validate_composition_element_ids(element_ids)
        composition_dim = int(validated_element_ids.numel())
        if isinstance(input_transform, CompositionMaterialInputTransform):
            if input_transform.composition_dim != composition_dim:
                raise ValueError(
                    "CompositionMaterialInputTransform.n_components must match element_ids: "
                    f"{input_transform.composition_dim} != {composition_dim}."
                )
            if train_X.shape[-1] != input_transform.input_dim:
                raise ValueError(
                    "train_X width must match CompositionMaterialInputTransform.input_dim: "
                    f"{train_X.shape[-1]} != {input_transform.input_dim}."
                )
            process_dim = input_transform.process_dim
        else:
            if train_X.shape[-1] < composition_dim:
                raise ValueError(
                    "train_X must contain one fraction column for every element_id. "
                    f"Got {train_X.shape[-1]} columns for {composition_dim} elements."
                )
            _validate_composition_model_inputs(
                train_X,
                composition_dim=composition_dim,
                input_dim=train_X.shape[-1],
            )
            process_dim = train_X.shape[-1] - composition_dim

        material_encoder = _resolve_material_encoder(
            encoder,
            checkpoint,
            strict_checkpoint=strict_checkpoint,
        )
        feature_extractor = MaterialGPFeatureExtractor(
            material_encoder=material_encoder,
            element_ids=validated_element_ids,
            process_dim=process_dim,
            latent_dim=latent_dim,
            fusion=fusion,
            projection=projection,
        )
        resolved_input_transform = (
            input_transform
            if isinstance(input_transform, CompositionMaterialInputTransform)
            else _resolve_composition_input_transform(
                train_X,
                composition_dim=composition_dim,
                input_transform=input_transform,
            )
        )

        DeepKernelGaussianGPModel.__init__(
            self,
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=None,
            likelihood=likelihood,
            input_transform=resolved_input_transform,
            outcome_transform=outcome_transform,
            feature_extractor=feature_extractor,
            latent_dim=latent_dim,
        )
        self._validate_correlated_kernel()
        transformed_train_X = cast(tuple[Tensor, ...], self.deepkernel.train_inputs)[0]
        self.material_feature_extractor.validate_input(transformed_train_X)

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> CrabNetMultiTaskGPModel:
        """Move model components while preserving the wrapper dtype/device contract."""

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
        return cast(CrabNetMultiTaskGPModel, module)

    @property
    def material_feature_extractor(self) -> MaterialGPFeatureExtractor:
        """Return the shared material/process feature extractor."""

        return cast(MaterialGPFeatureExtractor, self.deepkernel.feature_extractor)

    @property
    def material_encoder(self) -> CrabNetEncoder:
        """Return the single shared CrabNet encoder."""

        return cast(CrabNetEncoder, self.material_feature_extractor.material_encoder)

    @property
    def projection(self) -> nn.Module:
        """Return the shared trainable latent projection."""

        return self.material_feature_extractor.projection

    @property
    def fusion(self) -> MaterialProcessFusion:
        """Return the shared material/process fusion module."""

        return self.material_feature_extractor.fusion

    @property
    def element_ids(self) -> Tensor:
        """Return the fixed atomic-number vocabulary."""

        return self.material_feature_extractor.element_ids

    @property
    def composition_dim(self) -> int:
        """Return the number of material-fraction components."""

        return self.material_feature_extractor.composition_dim

    @property
    def process_dim(self) -> int:
        """Return the number of continuous process columns."""

        return self.material_feature_extractor.process_dim


class CrabNetMultiTaskDKLModel(CrabNetMultiTaskGPModel):
    """Correlated CrabNet multitask DKL with shared encoder fine-tuning."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        element_ids: Tensor,
        encoder: CrabNetEncoder | nn.Module | None = None,
        checkpoint: Checkpoint | None = None,
        latent_dim: int = 32,
        fusion: Literal["concat"] | MaterialProcessFusion = "concat",
        projection: nn.Module | None = None,
        strict_checkpoint: bool = True,
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
            train_Yvar=train_Yvar,
            element_ids=element_ids,
            encoder=encoder,
            checkpoint=checkpoint,
            latent_dim=latent_dim,
            fusion=fusion,
            projection=projection,
            strict_checkpoint=strict_checkpoint,
            likelihood=likelihood,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )
        training_mode, trainable_modules = _configure_dkl_encoder(
            self.material_encoder,
            resolved_trainable_layers,
        )
        self._trainable_encoder_layers = resolved_trainable_layers
        self.material_feature_extractor._configure_encoder_training(
            training_mode,
            trainable_modules,
        )

    @property
    def trainable_encoder_layers(self) -> int | Literal["all"]:
        """Return the shared encoder fine-tuning policy."""

        return self._trainable_encoder_layers


class CrabNetMixedMultiTaskGPModel(
    _CorrelatedCrabNetMultiTaskMixin,
    CrabNetMixedGPModel,
):
    """Correlated multitask GP with a categorical process kernel.

    Composition and numeric process variables share one frozen CrabNet latent
    representation. Categorical process values remain outside CrabNet and enter
    the same mixed continuous/categorical kernel used by ``CrabNetMixedGPModel``.
    The resulting mixed base kernel is wrapped by ``MultitaskKernel`` so all
    material-property targets share and learn one task covariance matrix.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        train_Yvar: Tensor | None = None,
        *,
        element_ids: Tensor,
        composition_indices: Sequence[int],
        method: str = "ilr",
        reference_index: int | None = None,
        process_bounds: Tensor | None = None,
        component_weights: Tensor | None = None,
        normalize_process: bool = True,
        encoder: CrabNetEncoder | nn.Module | None = None,
        checkpoint: Checkpoint | None = None,
        latent_dim: int = 32,
        fusion: Literal["concat"] | MaterialProcessFusion = "concat",
        projection: nn.Module | None = None,
        strict_checkpoint: bool = True,
        likelihood: Likelihood | None = None,
        outcome_transform: OutcomeTransformArg = "DEFAULT",
    ) -> None:
        _validate_multitask_targets(
            train_X,
            train_Y,
            train_Yvar,
            model_name=self.__class__.__name__,
            latent_dim=latent_dim,
        )

        d = train_X.shape[-1]
        normalized_cat_dims = normalize_indices(indices=list(cat_dims), d=d)
        if not normalized_cat_dims:
            raise ValueError(
                "CrabNetMixedMultiTaskGPModel requires at least one categorical "
                "process dimension."
            )
        continuous_dims = sorted(set(range(d)) - set(normalized_cat_dims))
        raw_composition_indices = [int(index) for index in composition_indices]
        if not raw_composition_indices:
            raise ValueError("composition_indices must not be empty.")
        if any(index in normalized_cat_dims for index in raw_composition_indices):
            raise ValueError(
                "composition_indices must refer only to continuous composition coordinates."
            )
        if min(raw_composition_indices) < 0 or max(raw_composition_indices) >= d:
            raise ValueError("composition_indices must be valid train_X columns.")

        continuous_position = {
            raw_index: index for index, raw_index in enumerate(continuous_dims)
        }
        try:
            continuous_composition_indices = [
                continuous_position[index] for index in raw_composition_indices
            ]
        except KeyError as error:
            raise ValueError(
                "Every composition index must remain in the continuous subset."
            ) from error

        validated_element_ids = _validate_composition_element_ids(element_ids)
        material_encoder = _resolve_material_encoder(
            encoder,
            checkpoint,
            strict_checkpoint=strict_checkpoint,
        )
        feature_extractor = _CrabNetMixedContinuousFeatureExtractor(
            continuous_input_dim=len(continuous_dims),
            composition_indices=continuous_composition_indices,
            element_ids=validated_element_ids,
            method=method,
            reference_index=reference_index,
            process_bounds=process_bounds,
            component_weights=component_weights,
            normalize_process=normalize_process,
            material_encoder=material_encoder,
            latent_dim=latent_dim,
            fusion=fusion,
            projection=projection,
        ).to(train_X)

        self.raw_composition_indices = tuple(raw_composition_indices)
        DeepKernelGaussianMixedGPModel.__init__(
            self,
            train_X=train_X,
            train_Y=train_Y,
            cat_dims=normalized_cat_dims,
            train_Yvar=None,
            likelihood=likelihood,
            input_transform=None,
            outcome_transform=outcome_transform,
            feature_extractor=feature_extractor,
            latent_dim=latent_dim,
        )
        self._validate_correlated_kernel()


class CrabNetMixedMultiTaskDKLModel(
    _CorrelatedCrabNetMultiTaskMixin,
    CrabNetMixedDKLModel,
):
    """Correlated mixed multitask DKL with learned categorical embeddings.

    A single CrabNet encoder, numeric-process representation, category embedding
    set, and neural projection are shared by every target. The shared encoder is
    partially or fully fine-tuned and the final exact GP learns cross-target
    covariance with ``MultitaskKernel``.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        train_Yvar: Tensor | None = None,
        *,
        element_ids: Tensor,
        composition_indices: Sequence[int],
        method: str = "ilr",
        reference_index: int | None = None,
        process_bounds: Tensor | None = None,
        component_weights: Tensor | None = None,
        normalize_process: bool = True,
        encoder: CrabNetEncoder | nn.Module | None = None,
        checkpoint: Checkpoint | None = None,
        latent_dim: int = 32,
        category_cardinalities: Sequence[int] | None = None,
        category_embedding_dims: int | Sequence[int] | None = None,
        projection_hidden_dim: int | None = None,
        projection: nn.Module | None = None,
        strict_checkpoint: bool = True,
        trainable_encoder_layers: int | Literal["all"] = 1,
        likelihood: Likelihood | None = None,
        outcome_transform: OutcomeTransformArg = "DEFAULT",
    ) -> None:
        _validate_multitask_targets(
            train_X,
            train_Y,
            train_Yvar,
            model_name=self.__class__.__name__,
            latent_dim=latent_dim,
        )

        d = train_X.shape[-1]
        normalized_cat_dims = normalize_indices(indices=list(cat_dims), d=d)
        if not normalized_cat_dims:
            raise ValueError(
                "CrabNetMixedMultiTaskDKLModel requires at least one categorical "
                "process dimension."
            )
        raw_composition_indices = [int(index) for index in composition_indices]
        if not raw_composition_indices:
            raise ValueError("composition_indices must not be empty.")
        if len(raw_composition_indices) != len(set(raw_composition_indices)):
            raise ValueError("composition_indices must not contain duplicates.")
        if min(raw_composition_indices) < 0 or max(raw_composition_indices) >= d:
            raise ValueError("composition_indices must be valid train_X columns.")
        if any(index in normalized_cat_dims for index in raw_composition_indices):
            raise ValueError(
                "composition_indices must refer only to continuous composition coordinates."
            )

        cardinalities = _resolve_category_cardinalities(
            train_X,
            normalized_cat_dims,
            category_cardinalities,
        )
        embedding_dims = _resolve_category_embedding_dims(
            cardinalities,
            category_embedding_dims,
        )
        resolved_trainable_layers = _validate_trainable_encoder_layers(
            trainable_encoder_layers
        )
        material_encoder = _resolve_material_encoder(
            encoder,
            checkpoint,
            strict_checkpoint=strict_checkpoint,
        )
        feature_extractor = _CrabNetMixedDKLFeatureExtractor(
            input_dim=d,
            cat_dims=normalized_cat_dims,
            composition_indices=raw_composition_indices,
            element_ids=element_ids,
            method=method,
            reference_index=reference_index,
            process_bounds=process_bounds,
            component_weights=component_weights,
            normalize_process=normalize_process,
            material_encoder=material_encoder,
            category_cardinalities=cardinalities,
            category_embedding_dims=embedding_dims,
            latent_dim=latent_dim,
            projection_hidden_dim=projection_hidden_dim,
            projection=projection,
        ).to(train_X)

        DeepKernelGaussianGPModel.__init__(
            self,
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=None,
            likelihood=likelihood,
            input_transform=None,
            outcome_transform=outcome_transform,
            feature_extractor=feature_extractor,
            latent_dim=latent_dim,
        )
        self.cat_dims = list(normalized_cat_dims)
        self.ord_dims = sorted(set(range(d)) - set(normalized_cat_dims))
        self._ignore_X_dims_scaling_check = list(normalized_cat_dims)
        self.raw_composition_indices = tuple(raw_composition_indices)

        training_mode, trainable_modules = _configure_dkl_encoder(
            self.material_encoder,
            resolved_trainable_layers,
        )
        self._trainable_encoder_layers = resolved_trainable_layers
        self.material_feature_extractor._configure_encoder_training(
            training_mode,
            trainable_modules,
        )
        self._validate_correlated_kernel()


__all__ = [
    "CrabNetMixedMultiTaskDKLModel",
    "CrabNetMixedMultiTaskGPModel",
    "CrabNetMultiTaskDKLModel",
    "CrabNetMultiTaskGPModel",
]
