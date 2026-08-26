"""ALIGNN-backed exact GP for mixed continuous/categorical process inputs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

from botorch.models.transforms.input import Normalize
from botorch.utils.transforms import normalize_indices
from gpytorch.likelihoods import Likelihood
from torch import Tensor, nn

from bochan.composition import ALIGNNEncoder, MaterialProcessFusion
from bochan.composition.encoders.alignn import Checkpoint

from .alignn import (
    _ALIGNNGPFeatureExtractor,
    _configure_dkl_encoder,
    _resolve_material_encoder,
    _validate_model_inputs,
    _validate_structure_bank,
    _validate_trainable_encoder_layers,
)
from .deepkernel import InputTransformArg, OutcomeTransformArg
from .deepkernel_configurable import DeepKernelGaussianMixedGPModel


def _resolve_mixed_input_transform(
    train_X: Tensor,
    *,
    cat_dims: Sequence[int],
    input_transform: InputTransformArg,
) -> InputTransformArg:
    """Normalize numeric process columns while preserving structure/categories."""

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


class ALIGNNMixedGPModel(DeepKernelGaussianMixedGPModel):
    """Exact mixed GP over frozen ALIGNN crystal representations.

    The raw input contract keeps ``structure_index`` at column 0. Columns in
    ``cat_dims`` are integer-coded categorical process variables and are kept
    outside the ALIGNN/process feature extractor. Every remaining column after
    the structure selector is a continuous process variable.

    The continuous branch therefore receives::

        [structure_index, continuous_process_1, ...]

    and maps the selected crystal graph through the pure-PyTorch ALIGNN encoder
    before fusing its representation with the numeric process variables. The
    resulting latent features are handled by bochan's standard mixed GP kernel,
    while categorical process columns are handled by the categorical kernel.
    """

    _supports_cache_root = False

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        train_Yvar: Tensor | None = None,
        *,
        structure_graphs: Sequence[Any],
        encoder: ALIGNNEncoder | nn.Module | None = None,
        checkpoint: Checkpoint | None = None,
        encoder_output_dim: int | None = None,
        encoder_config: object | None = None,
        latent_dim: int = 32,
        fusion: Literal["concat"] | MaterialProcessFusion = "concat",
        projection: nn.Module | None = None,
        strict_checkpoint: bool = True,
        likelihood: Likelihood | None = None,
        input_transform: InputTransformArg = "DEFAULT",
        outcome_transform: OutcomeTransformArg = "DEFAULT",
    ) -> None:
        if train_X.ndim != 2:
            raise ValueError("train_X must have shape [n, d].")
        if train_X.shape[-1] < 2:
            raise ValueError(
                "ALIGNNMixedGPModel requires a structure-index column and at least one categorical process column."
            )
        if train_Y.ndim > 1 and train_Y.shape[-1] != 1:
            raise ValueError(
                "ALIGNNMixedGPModel currently supports single-output train_Y only."
            )
        if train_Yvar is not None:
            raise NotImplementedError(
                "ALIGNNMixedGPModel does not yet support train_Yvar."
            )
        if (
            isinstance(latent_dim, bool)
            or not isinstance(latent_dim, int)
            or latent_dim <= 0
        ):
            raise ValueError("latent_dim must be a positive integer.")

        d = train_X.shape[-1]
        normalized_cat_dims = normalize_indices(indices=list(cat_dims), d=d)
        if not normalized_cat_dims:
            raise ValueError(
                "ALIGNNMixedGPModel requires at least one categorical process dimension."
            )
        if 0 in normalized_cat_dims:
            raise ValueError(
                "The structure-index column (feature 0) is handled by ALIGNN and cannot be included in cat_dims."
            )

        validated_graphs = _validate_structure_bank(structure_graphs)
        _validate_model_inputs(
            train_X,
            num_structures=len(validated_graphs),
            input_dim=d,
        )

        continuous_dims = sorted(set(range(d)) - set(normalized_cat_dims))
        if not continuous_dims or continuous_dims[0] != 0:
            raise RuntimeError(
                "The ALIGNN mixed continuous branch must retain structure-index feature 0."
            )
        process_dims = [index for index in continuous_dims if index != 0]

        material_encoder = _resolve_material_encoder(
            encoder,
            checkpoint,
            output_dim=encoder_output_dim,
            config=encoder_config,
            strict_checkpoint=strict_checkpoint,
        )
        feature_extractor = _ALIGNNGPFeatureExtractor(
            material_encoder=material_encoder,
            structure_graphs=validated_graphs,
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
        self.alignn_feature_extractor.validate_input(continuous_train_X)

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> ALIGNNMixedGPModel:
        """Move all components and refresh the wrapper dtype/device contract."""

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
        return cast(ALIGNNMixedGPModel, module)

    @property
    def alignn_feature_extractor(self) -> _ALIGNNGPFeatureExtractor:
        """Return the ALIGNN/numeric-process extractor used by the mixed GP."""

        return cast(_ALIGNNGPFeatureExtractor, self.deepkernel.feature_extractor)

    @property
    def material_encoder(self) -> ALIGNNEncoder:
        """Return the ALIGNN material encoder."""

        return self.alignn_feature_extractor.material_encoder

    @property
    def projection(self) -> nn.Module:
        """Return the trainable latent projection."""

        return self.alignn_feature_extractor.projection

    @property
    def fusion(self) -> MaterialProcessFusion:
        """Return the structure/numeric-process fusion module."""

        return self.alignn_feature_extractor.fusion

    @property
    def structure_graphs(self) -> tuple[Any, ...]:
        """Return the canonical structure graph bank."""

        return self.alignn_feature_extractor.structure_graphs

    @property
    def num_structures(self) -> int:
        """Return the number of structures in the graph bank."""

        return self.alignn_feature_extractor.num_structures

    @property
    def process_dim(self) -> int:
        """Return the number of continuous process dimensions."""

        return self.alignn_feature_extractor.process_dim

    @property
    def continuous_process_dims(self) -> tuple[int, ...]:
        """Return raw input indices of numeric process columns."""

        return self._continuous_process_dims

    @property
    def categorical_process_dim(self) -> int:
        """Return the number of categorical process dimensions."""

        return len(self.cat_dims)


class ALIGNNMixedDKLModel(ALIGNNMixedGPModel):
    """Mixed exact GP that jointly fine-tunes the ALIGNN structure encoder.

    The mixed covariance remains identical to :class:`ALIGNNMixedGPModel`:
    categorical process columns stay in the categorical kernel while the
    continuous branch contains the discrete structure selector plus numeric
    process variables. DKL only changes the ALIGNN representation policy by
    unfreezing selected graph-convolution blocks or the complete representation
    backbone.

    A positive ``trainable_encoder_layers`` value unfreezes that many final
    graph-convolution blocks from the ordered ALIGNN + GCN stacks. ``"all"``
    fine-tunes the complete representation backbone while keeping the upstream
    scalar property head outside the Bochan DKL representation path.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        cat_dims: Sequence[int],
        train_Yvar: Tensor | None = None,
        *,
        structure_graphs: Sequence[Any],
        encoder: ALIGNNEncoder | nn.Module | None = None,
        checkpoint: Checkpoint | None = None,
        encoder_output_dim: int | None = None,
        encoder_config: object | None = None,
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
            cat_dims=cat_dims,
            train_Yvar=train_Yvar,
            structure_graphs=structure_graphs,
            encoder=encoder,
            checkpoint=checkpoint,
            encoder_output_dim=encoder_output_dim,
            encoder_config=encoder_config,
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
        self.alignn_feature_extractor._configure_encoder_training(
            training_mode,
            trainable_modules,
        )

    @property
    def trainable_encoder_layers(self) -> int | Literal["all"]:
        """Return the configured partial/full ALIGNN fine-tuning policy."""

        return self._trainable_encoder_layers


__all__ = ["ALIGNNMixedDKLModel", "ALIGNNMixedGPModel"]
