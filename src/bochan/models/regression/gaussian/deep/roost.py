"""Roost composition representations with exact Gaussian processes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, cast

from gpytorch.likelihoods import Likelihood
from torch import Tensor, nn

from bochan.composition import MaterialProcessFusion, RoostEncoder
from bochan.composition.encoders.roost import Checkpoint

from .deepkernel import InputTransformArg, OutcomeTransformArg
from .deepkernel_configurable import DeepKernelGaussianGPModel
from .material import (
    CompositionMaterialInputTransform,
    MaterialGPFeatureExtractor,
    _resolve_composition_input_transform,
    _validate_composition_element_ids,
    _validate_composition_model_inputs,
)


def _resolve_material_encoder(
    encoder: RoostEncoder | nn.Module | None,
    checkpoint: Checkpoint | None,
    *,
    output_dim: int | None,
    strict_checkpoint: bool,
) -> RoostEncoder:
    """Build or reuse a Roost adapter and freeze its representation backbone."""

    if isinstance(encoder, RoostEncoder):
        material_encoder = encoder
        if output_dim is not None and output_dim != material_encoder.output_dim:
            raise ValueError(
                "encoder_output_dim does not match RoostEncoder.output_dim: "
                f"{output_dim} != {material_encoder.output_dim}."
            )
        if checkpoint is not None:
            material_encoder.load_checkpoint(checkpoint, strict=strict_checkpoint)
    else:
        material_encoder = RoostEncoder(
            encoder=encoder,
            checkpoint=checkpoint,
            output_dim=output_dim,
            strict_checkpoint=strict_checkpoint,
        )

    for parameter in material_encoder.parameters():
        parameter.requires_grad_(False)
    material_encoder.eval()
    return material_encoder


class RoostGPModel(DeepKernelGaussianGPModel):
    """Exact Gaussian process over frozen Roost composition representations.

    The first ``len(element_ids)`` columns of ``train_X`` contain atomic
    fractions in the same fixed-element order as ``element_ids``. Remaining
    columns are optional continuous process variables. A
    :class:`CompositionMaterialInputTransform` may instead convert canonical
    fraction, CLR, ALR, or ILR coordinates to this packed representation.

    The public :class:`RoostEncoder` supplies the pooled Aviary
    ``DescriptorNetwork`` representation; Aviary prediction heads are not part
    of this model. The Roost encoder is frozen and kept in evaluation mode,
    while the material/process fusion, projection, exact GP, and likelihood
    remain trainable. Encoder forward passes retain autograd with respect to
    composition fractions.

    Args:
        train_X: ``[n, composition_dim + process_dim]`` training inputs.
        train_Y: Single-output targets with shape ``[n]`` or ``[n, 1]``.
        train_Yvar: Reserved for future fixed-noise support and currently
            unsupported.
        element_ids: Atomic-number vocabulary matching the fraction columns.
        encoder: Optional :class:`RoostEncoder` or raw five-tensor Roost
            backbone. Omit it to construct Aviary's default Roost descriptor.
        checkpoint: Optional upstream Aviary or adapter checkpoint.
        encoder_output_dim: Required only when an injected raw backbone does
            not expose a positive integer ``output_dim``.
        latent_dim: Width of the trainable projection passed to the GP kernel.
        fusion: Material/process fusion strategy.
        projection: Optional projection module. A linear projection is used by
            default.
        strict_checkpoint: Require a complete encoder checkpoint state.
        likelihood: Optional GPyTorch likelihood.
        input_transform: ``"DEFAULT"`` normalizes continuous process columns
            only. ``CompositionMaterialInputTransform`` additionally supports
            canonical composition coordinates while preserving autograd.
        outcome_transform: Outcome transform forwarded to the Gaussian deep
            kernel wrapper.
    """

    _supports_cache_root = False

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        element_ids: Tensor,
        encoder: RoostEncoder | nn.Module | None = None,
        checkpoint: Checkpoint | None = None,
        encoder_output_dim: int | None = None,
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
        if train_Y.ndim == 1:
            train_Y = train_Y.unsqueeze(-1)
        elif train_Y.ndim != 2 or train_Y.shape[-1] != 1:
            raise ValueError("RoostGPModel currently supports single-output train_Y only.")
        if train_Yvar is not None:
            raise NotImplementedError("RoostGPModel does not yet support train_Yvar.")
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")

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
            output_dim=encoder_output_dim,
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

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=None,
            likelihood=likelihood,
            input_transform=resolved_input_transform,
            outcome_transform=outcome_transform,
            feature_extractor=feature_extractor,
            latent_dim=latent_dim,
        )

        transformed_train_X = cast(tuple[Tensor, ...], self.deepkernel.train_inputs)[0]
        self.material_feature_extractor.validate_input(transformed_train_X)

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> RoostGPModel:
        """Move every model component and refresh the wrapper dtype/device contract."""

        module = super()._apply(fn, recurse=recurse)
        reference = next(
            (value for value in (*self.parameters(), *self.buffers()) if value.is_floating_point()),
            None,
        )
        if reference is not None:
            self._model_dtype = reference.dtype
            self._model_device = reference.device
        return cast(RoostGPModel, module)

    @property
    def material_feature_extractor(self) -> MaterialGPFeatureExtractor:
        """Return the shared material/process feature extractor."""

        return cast(MaterialGPFeatureExtractor, self.deepkernel.feature_extractor)

    @property
    def material_encoder(self) -> RoostEncoder:
        """Return the frozen Roost material encoder."""

        return cast(RoostEncoder, self.material_feature_extractor.material_encoder)

    @property
    def projection(self) -> nn.Module:
        """Return the trainable latent projection."""

        return self.material_feature_extractor.projection

    @property
    def fusion(self) -> MaterialProcessFusion:
        """Return the material/process fusion module."""

        return self.material_feature_extractor.fusion

    @property
    def element_ids(self) -> Tensor:
        """Return the fixed atomic-number vocabulary buffer."""

        return self.material_feature_extractor.element_ids

    @property
    def composition_dim(self) -> int:
        """Return the number of fraction columns."""

        return self.material_feature_extractor.composition_dim

    @property
    def process_dim(self) -> int:
        """Return the number of continuous process columns."""

        return self.material_feature_extractor.process_dim


__all__ = ["RoostGPModel"]
