"""CrabNet-backed exact GP for mixed continuous/categorical process inputs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, cast

from botorch.utils.transforms import normalize_indices
from gpytorch.likelihoods import Likelihood
from torch import Tensor, nn

from bochan.composition import CrabNetEncoder, MaterialProcessFusion
from bochan.composition.encoders.crabnet import Checkpoint

from .crabnet import (
    CrabNetInputTransform,
    _CrabNetGPFeatureExtractor,
    _resolve_material_encoder,
    _validate_element_ids,
)
from .deepkernel import OutcomeTransformArg
from .deepkernel_configurable import DeepKernelGaussianMixedGPModel


class _CrabNetMixedContinuousFeatureExtractor(nn.Module):
    """Encode composition coordinates plus numeric process features for mixed GP."""

    def __init__(
        self,
        *,
        continuous_input_dim: int,
        composition_indices: Sequence[int],
        element_ids: Tensor,
        method: str,
        reference_index: int | None,
        process_bounds: Tensor | None,
        component_weights: Tensor | None,
        normalize_process: bool,
        material_encoder: CrabNetEncoder,
        latent_dim: int,
        fusion: Literal["concat"] | MaterialProcessFusion,
        projection: nn.Module | None,
    ) -> None:
        super().__init__()
        validated_element_ids = _validate_element_ids(element_ids)
        self.packer = CrabNetInputTransform(
            input_dim=continuous_input_dim,
            composition_indices=composition_indices,
            n_components=int(validated_element_ids.numel()),
            method=method,
            reference_index=reference_index,
            process_bounds=process_bounds,
            component_weights=component_weights,
            normalize_process=normalize_process,
        )
        self.crabnet = _CrabNetGPFeatureExtractor(
            material_encoder=material_encoder,
            element_ids=validated_element_ids,
            process_dim=self.packer.process_dim,
            latent_dim=latent_dim,
            fusion=fusion,
            projection=projection,
        )
        self.output_dim = int(latent_dim)

    def forward(self, X: Tensor) -> Tensor:
        """Return CrabNet latent features from the continuous input subset."""

        return self.crabnet(self.packer(X))


class CrabNetMixedGPModel(DeepKernelGaussianMixedGPModel):
    """Exact mixed GP over frozen CrabNet material representations.

    Raw ``train_X`` keeps the canonical tabular layout: composition coordinates,
    numeric process columns, and integer-coded categorical process columns.
    ``cat_dims`` are kept outside the CrabNet feature extractor and are handled by
    the categorical kernel. The remaining continuous columns are split into
    composition coordinates and numeric process variables; composition
    coordinates are converted back to atomic fractions before CrabNet encoding.

    The resulting covariance follows bochan's standard mixed deep-kernel form:
    a continuous kernel over the CrabNet/numeric latent representation combined
    additively and multiplicatively with a categorical kernel. This preserves
    gradients through composition and numeric process variables while allowing
    ``optimize_acqf_mixed`` to enumerate categorical process combinations.
    """

    _supports_cache_root = False

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
        if train_X.ndim != 2:
            raise ValueError("train_X must have shape [n, d].")
        if train_Y.ndim > 1 and train_Y.shape[-1] != 1:
            raise ValueError("CrabNetMixedGPModel currently supports single-output train_Y only.")
        if train_Yvar is not None:
            raise NotImplementedError("CrabNetMixedGPModel does not yet support train_Yvar.")
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")

        d = train_X.shape[-1]
        normalized_cat_dims = normalize_indices(indices=list(cat_dims), d=d)
        if not normalized_cat_dims:
            raise ValueError("CrabNetMixedGPModel requires at least one categorical process dimension.")
        continuous_dims = sorted(set(range(d)) - set(normalized_cat_dims))
        raw_composition_indices = [int(index) for index in composition_indices]
        if not raw_composition_indices:
            raise ValueError("composition_indices must not be empty.")
        if any(index in normalized_cat_dims for index in raw_composition_indices):
            raise ValueError("composition_indices must refer only to continuous composition coordinates.")
        if min(raw_composition_indices) < 0 or max(raw_composition_indices) >= d:
            raise ValueError("composition_indices must be valid train_X columns.")

        continuous_position = {raw_index: index for index, raw_index in enumerate(continuous_dims)}
        try:
            continuous_composition_indices = [
                continuous_position[index] for index in raw_composition_indices
            ]
        except KeyError as error:
            raise ValueError(
                "Every composition index must remain in the continuous subset."
            ) from error

        validated_element_ids = _validate_element_ids(element_ids)
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
        super().__init__(
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

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> CrabNetMixedGPModel:
        """Move all components and refresh the wrapper dtype/device contract."""

        module = super()._apply(fn, recurse=recurse)
        reference = next(
            (value for value in (*self.parameters(), *self.buffers()) if value.is_floating_point()),
            None,
        )
        if reference is not None:
            self._model_dtype = reference.dtype
            self._model_device = reference.device
        return cast(CrabNetMixedGPModel, module)

    @property
    def mixed_feature_extractor(self) -> _CrabNetMixedContinuousFeatureExtractor:
        """Return the continuous CrabNet feature extractor owned by the mixed GP."""

        return cast(_CrabNetMixedContinuousFeatureExtractor, self.deepkernel.feature_extractor)

    @property
    def crabnet_feature_extractor(self) -> _CrabNetGPFeatureExtractor:
        """Return the packed fraction/numeric-process CrabNet extractor."""

        return self.mixed_feature_extractor.crabnet

    @property
    def material_encoder(self) -> CrabNetEncoder:
        """Return the frozen CrabNet material encoder."""

        return self.crabnet_feature_extractor.material_encoder

    @property
    def projection(self) -> nn.Module:
        """Return the trainable latent projection."""

        return self.crabnet_feature_extractor.projection

    @property
    def fusion(self) -> MaterialProcessFusion:
        """Return the material/numeric-process fusion module."""

        return self.crabnet_feature_extractor.fusion

    @property
    def element_ids(self) -> Tensor:
        """Return the fixed atomic-number vocabulary."""

        return self.crabnet_feature_extractor.element_ids

    @property
    def composition_dim(self) -> int:
        """Return the number of elemental fractions presented to CrabNet."""

        return self.crabnet_feature_extractor.composition_dim

    @property
    def process_dim(self) -> int:
        """Return the number of numeric process dimensions fused with CrabNet."""

        return self.crabnet_feature_extractor.process_dim

    @property
    def categorical_process_dim(self) -> int:
        """Return the number of categorical process dimensions in the mixed kernel."""

        return len(self.cat_dims)


__all__ = ["CrabNetMixedGPModel"]
