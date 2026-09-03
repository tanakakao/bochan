"""CHGNet force/stress baselines corrected by correlated residual Gaussian processes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
from gpytorch.likelihoods import Likelihood
from torch import Tensor, nn

from bochan.composition import CHGNetEncoder, MaterialProcessFusion
from bochan.composition.encoders.chgnet import Checkpoint
from bochan.models.regression.gaussian.deep.chgnet_multitask import CHGNetMultiTaskGPModel
from bochan.models.regression.gaussian.deep.deepkernel import InputTransformArg, OutcomeTransformArg
from bochan.models.regression.gaussian.materials.common.baseline import (
    MaterialBaselineSpec,
    MaterialPropertyContract,
)
from bochan.models.regression.gaussian.materials.common.residual import (
    DirectMaterialPredictor,
    ResidualMaterialGPModel,
    compute_material_residual_targets,
)
from bochan.models.regression.gaussian.materials.common.tensor_target import TensorTargetLayout

from .chgnet_residual import _resolve_encoder, _validate_structure_bank

_MLIPKind = Literal["force", "stress"]
_DEFAULT_MODEL_NAME = "0.3.0"


def _structure_indices(X: Tensor, *, num_structures: int) -> tuple[Tensor, torch.Size]:
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
        int(indices.min().item()) < 0 or int(indices.max().item()) >= num_structures
    ):
        raise ValueError("Structure index is outside the configured structure bank.")
    return indices, X.shape[:-1]


def _infer_num_atoms(structure: Any) -> int | None:
    if isinstance(structure, Mapping):
        for key in ("elements", "atomic_numbers", "species"):
            value = structure.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return len(value)
    try:
        size = len(structure)
    except (TypeError, AttributeError):
        return None
    return int(size) if size > 0 else None


def _resolve_force_num_atoms(structures: Sequence[Any], num_atoms: int | None) -> int:
    if num_atoms is not None:
        if isinstance(num_atoms, bool) or not isinstance(num_atoms, int) or num_atoms <= 0:
            raise ValueError("num_atoms must be a positive integer when provided.")
        expected = num_atoms
    else:
        inferred = [_infer_num_atoms(structure) for structure in structures]
        if any(value is None for value in inferred):
            raise ValueError(
                "Could not infer a fixed atom count from every structure; pass num_atoms explicitly."
            )
        expected = int(inferred[0])
        if any(int(value) != expected for value in inferred if value is not None):
            raise ValueError(
                "Force residual GP currently requires a fixed topology: every structure must "
                "contain the same number of atoms."
            )
    return expected


def _single_observable(value: Any, *, kind: _MLIPKind) -> Tensor:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"CHGNet single-structure {kind} prediction must contain one tensor.")
        value = value[0]
    if not torch.is_tensor(value):
        raise TypeError(f"CHGNet forward did not return Tensor {kind!r}.")
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"CHGNet produced non-finite {kind} predictions.")
    return value


def _raw_chgnet_observable(
    encoder: CHGNetEncoder,
    structure: Any,
    *,
    kind: _MLIPKind,
) -> Tensor:
    reference = encoder._floating_reference()  # noqa: SLF001
    if reference is None:
        device = torch.device("cpu")
        dtype = torch.get_default_dtype()
    else:
        device = reference.device
        dtype = reference.dtype
    graph = encoder._prepare_graph(structure, device=device, dtype=dtype)  # noqa: SLF001
    prediction = encoder.encoder([graph], task="efs", return_crystal_feas=False)
    if not isinstance(prediction, Mapping):
        raise TypeError("CHGNet forward must return a mapping containing force/stress outputs.")
    key = "f" if kind == "force" else "s"
    if key not in prediction:
        raise KeyError(f"CHGNet forward did not return {key!r} for {kind} prediction.")
    return _single_observable(prediction[key], kind=kind)


class _CHGNetDirectTensorPredictor(DirectMaterialPredictor):
    """Shared fixed-layout direct predictor for CHGNet tensor observables."""

    def __init__(
        self,
        encoder: CHGNetEncoder,
        structures: Sequence[Any],
        *,
        layout: TensorTargetLayout,
    ) -> None:
        super().__init__()
        if not isinstance(encoder, CHGNetEncoder):
            raise TypeError("encoder must be a CHGNetEncoder.")
        self.encoder = encoder
        self.structures = _validate_structure_bank(structures)
        self.layout = layout
        self.register_buffer("_cached_baseline", torch.empty(0), persistent=False)

    @property
    def output_dim(self) -> int:
        return self.layout.output_dim

    def _predict_one(self, structure: Any) -> Tensor:
        value = _raw_chgnet_observable(self.encoder, structure, kind=self.layout.kind)
        if self.layout.kind == "force":
            if value.ndim != 2 or tuple(value.shape) != self.layout.tensor_shape:
                raise ValueError(
                    "CHGNet forces must have shape "
                    f"{self.layout.tensor_shape}; got {tuple(value.shape)}."
                )
        else:
            while value.ndim > 2 and value.shape[0] == 1:
                value = value.squeeze(0)
            if tuple(value.shape) != (3, 3):
                raise ValueError(
                    f"CHGNet stress must have shape (3, 3); got {tuple(value.shape)}."
                )
        return self.layout.flatten(value).reshape(1, self.output_dim)

    def _baseline_bank(self) -> Tensor:
        if self._cached_baseline.numel() == 0:
            self._cached_baseline = torch.cat(
                [self._predict_one(structure) for structure in self.structures], dim=0
            ).detach()
        return self._cached_baseline

    def clear_cache(self) -> None:
        """Discard cached pretrained force/stress predictions."""

        self._cached_baseline = self._cached_baseline.new_empty(0)

    def forward(self, X: Tensor) -> Tensor:
        indices, leading_shape = _structure_indices(X, num_structures=len(self.structures))
        baseline_bank = self._baseline_bank()
        selected = baseline_bank[indices.to(device=baseline_bank.device)]
        return selected.to(device=X.device, dtype=X.dtype).reshape(
            *leading_shape,
            self.output_dim,
        )


class CHGNetDirectForcePredictor(_CHGNetDirectTensorPredictor):
    """Frozen CHGNet Cartesian forces flattened in atom-major xyz order."""

    def __init__(
        self,
        encoder: CHGNetEncoder,
        structures: Sequence[Any],
        *,
        num_atoms: int | None = None,
    ) -> None:
        resolved_structures = _validate_structure_bank(structures)
        resolved_num_atoms = _resolve_force_num_atoms(resolved_structures, num_atoms)
        super().__init__(
            encoder,
            resolved_structures,
            layout=TensorTargetLayout.force(resolved_num_atoms),
        )
        self.num_atoms = resolved_num_atoms


class CHGNetDirectStressPredictor(_CHGNetDirectTensorPredictor):
    """Frozen CHGNet full Cartesian stress tensor flattened in row-major order."""

    def __init__(self, encoder: CHGNetEncoder, structures: Sequence[Any]) -> None:
        super().__init__(encoder, structures, layout=TensorTargetLayout.stress())


def _baseline_spec(
    *,
    kind: _MLIPKind,
    model_name: str,
    target_contract: MaterialPropertyContract | None,
) -> MaterialBaselineSpec | None:
    if target_contract is None:
        return None
    if not isinstance(target_contract, MaterialPropertyContract):
        raise TypeError("target_contract must be a MaterialPropertyContract when provided.")
    if target_contract.quantity.casefold() != kind:
        raise ValueError(
            f"CHGNet {kind} residual requires target_contract.quantity={kind!r}; "
            f"got {target_contract.quantity!r}."
        )
    return MaterialBaselineSpec(
        family="chgnet",
        property=target_contract,
        model_name=model_name,
    )


def _flatten_noise(
    layout: TensorTargetLayout,
    train_Yvar: Tensor | None,
    *,
    n: int,
) -> Tensor | None:
    if train_Yvar is None:
        return None
    return layout.flatten(train_Yvar, n=n)


class CHGNetForceResidualGPModel(ResidualMaterialGPModel):
    """Correct fixed-topology CHGNet forces with a correlated wide residual GP."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        structures: Sequence[Any],
        num_atoms: int | None = None,
        encoder: CHGNetEncoder | nn.Module | None = None,
        checkpoint: Checkpoint | None = None,
        model_name: str = _DEFAULT_MODEL_NAME,
        encoder_output_dim: int | None = None,
        latent_dim: int = 32,
        fusion: Literal["concat"] | MaterialProcessFusion = "concat",
        projection: nn.Module | None = None,
        strict_checkpoint: bool = True,
        likelihood: Likelihood | None = None,
        input_transform: InputTransformArg = "DEFAULT",
        outcome_transform: OutcomeTransformArg = "DEFAULT",
        target_contract: MaterialPropertyContract | None = None,
    ) -> None:
        resolved_structures = _validate_structure_bank(structures)
        material_encoder = _resolve_encoder(
            encoder,
            checkpoint=checkpoint,
            model_name=model_name,
            encoder_output_dim=encoder_output_dim,
            strict_checkpoint=strict_checkpoint,
        )
        predictor = CHGNetDirectForcePredictor(
            material_encoder,
            resolved_structures,
            num_atoms=num_atoms,
        )
        flat_Y = predictor.layout.flatten(train_Y, n=train_X.shape[0])
        baseline_spec = _baseline_spec(
            kind="force", model_name=model_name, target_contract=target_contract
        )
        residual_Y = compute_material_residual_targets(
            train_X,
            flat_Y,
            predictor,
            baseline_spec=baseline_spec,
            target_contract=target_contract,
        )
        residual_model = CHGNetMultiTaskGPModel(
            train_X=train_X,
            train_Y=residual_Y,
            train_Yvar=_flatten_noise(predictor.layout, train_Yvar, n=train_X.shape[0]),
            structures=resolved_structures,
            encoder=material_encoder,
            model_name=model_name,
            encoder_output_dim=encoder_output_dim,
            latent_dim=latent_dim,
            fusion=fusion,
            projection=projection,
            strict_checkpoint=strict_checkpoint,
            likelihood=likelihood,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )
        super().__init__(
            predictor=predictor,
            residual_model=residual_model,
            baseline_spec=baseline_spec,
        )
        self.layout = predictor.layout
        self.structures = resolved_structures
        self.material_encoder = material_encoder
        self.num_atoms = predictor.num_atoms

    def unflatten(self, values: Tensor) -> Tensor:
        """Restore ``[..., num_atoms, 3]`` force axes from GP outputs."""

        return self.layout.unflatten(values)


class CHGNetStressResidualGPModel(ResidualMaterialGPModel):
    """Correct full 3x3 CHGNet stress tensors with a correlated residual GP."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        structures: Sequence[Any],
        encoder: CHGNetEncoder | nn.Module | None = None,
        checkpoint: Checkpoint | None = None,
        model_name: str = _DEFAULT_MODEL_NAME,
        encoder_output_dim: int | None = None,
        latent_dim: int = 32,
        fusion: Literal["concat"] | MaterialProcessFusion = "concat",
        projection: nn.Module | None = None,
        strict_checkpoint: bool = True,
        likelihood: Likelihood | None = None,
        input_transform: InputTransformArg = "DEFAULT",
        outcome_transform: OutcomeTransformArg = "DEFAULT",
        target_contract: MaterialPropertyContract | None = None,
    ) -> None:
        resolved_structures = _validate_structure_bank(structures)
        material_encoder = _resolve_encoder(
            encoder,
            checkpoint=checkpoint,
            model_name=model_name,
            encoder_output_dim=encoder_output_dim,
            strict_checkpoint=strict_checkpoint,
        )
        predictor = CHGNetDirectStressPredictor(material_encoder, resolved_structures)
        flat_Y = predictor.layout.flatten(train_Y, n=train_X.shape[0])
        baseline_spec = _baseline_spec(
            kind="stress", model_name=model_name, target_contract=target_contract
        )
        residual_Y = compute_material_residual_targets(
            train_X,
            flat_Y,
            predictor,
            baseline_spec=baseline_spec,
            target_contract=target_contract,
        )
        residual_model = CHGNetMultiTaskGPModel(
            train_X=train_X,
            train_Y=residual_Y,
            train_Yvar=_flatten_noise(predictor.layout, train_Yvar, n=train_X.shape[0]),
            structures=resolved_structures,
            encoder=material_encoder,
            model_name=model_name,
            encoder_output_dim=encoder_output_dim,
            latent_dim=latent_dim,
            fusion=fusion,
            projection=projection,
            strict_checkpoint=strict_checkpoint,
            likelihood=likelihood,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )
        super().__init__(
            predictor=predictor,
            residual_model=residual_model,
            baseline_spec=baseline_spec,
        )
        self.layout = predictor.layout
        self.structures = resolved_structures
        self.material_encoder = material_encoder

    def unflatten(self, values: Tensor) -> Tensor:
        """Restore ``[..., 3, 3]`` stress axes from GP outputs."""

        return self.layout.unflatten(values)


__all__ = [
    "CHGNetDirectForcePredictor",
    "CHGNetDirectStressPredictor",
    "CHGNetForceResidualGPModel",
    "CHGNetStressResidualGPModel",
]
