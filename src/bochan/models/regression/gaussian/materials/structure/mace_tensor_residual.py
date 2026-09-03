"""MACE force/stress baselines corrected by correlated residual Gaussian processes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
from gpytorch.likelihoods import Likelihood
from torch import Tensor, nn

from bochan.composition import MACEEncoder, MaterialProcessFusion
from bochan.models.regression.gaussian.deep.deepkernel import InputTransformArg, OutcomeTransformArg
from bochan.models.regression.gaussian.deep.mace_multitask import MACEMultiTaskGPModel
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

from .mace_residual import _DEFAULT_MODEL_NAME, _Pooling, _resolve_encoder, _validate_structure_bank

_MLIPKind = Literal["force", "stress"]


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


def _raw_mace_observable(
    encoder: MACEEncoder,
    structure: Any,
    *,
    kind: _MLIPKind,
) -> Tensor:
    batch = encoder._build_batch(structure)  # noqa: SLF001
    if kind == "force":
        output = encoder.encoder(
            batch,
            compute_force=True,
            compute_virials=False,
            compute_stress=False,
        )
        key = "forces"
    else:
        output = encoder.encoder(
            batch,
            compute_force=False,
            compute_virials=True,
            compute_stress=True,
        )
        key = "stress"
    if not isinstance(output, Mapping):
        raise TypeError(f"Raw MACE forward must return a mapping containing {key!r}.")
    value = output.get(key)
    if not torch.is_tensor(value):
        raise TypeError(f"Raw MACE forward did not return Tensor {key!r}.")
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"MACE produced non-finite {kind} predictions.")
    return value


class _MACEDirectTensorPredictor(DirectMaterialPredictor):
    """Shared fixed-layout direct predictor for MACE tensor observables."""

    def __init__(
        self,
        encoder: MACEEncoder,
        structures: Sequence[Any],
        *,
        layout: TensorTargetLayout,
    ) -> None:
        super().__init__()
        if not isinstance(encoder, MACEEncoder):
            raise TypeError("encoder must be a MACEEncoder.")
        self.encoder = encoder
        self.structures = _validate_structure_bank(structures)
        self.layout = layout
        self.register_buffer("_cached_baseline", torch.empty(0), persistent=False)

    @property
    def output_dim(self) -> int:
        return self.layout.output_dim

    def _predict_one(self, structure: Any) -> Tensor:
        value = _raw_mace_observable(self.encoder, structure, kind=self.layout.kind)
        if self.layout.kind == "force":
            if value.ndim != 2 or tuple(value.shape) != self.layout.tensor_shape:
                raise ValueError(
                    "MACE forces must have shape "
                    f"{self.layout.tensor_shape}; got {tuple(value.shape)}."
                )
        else:
            while value.ndim > 2 and value.shape[0] == 1:
                value = value.squeeze(0)
            if tuple(value.shape) != (3, 3):
                raise ValueError(
                    f"MACE stress must have shape (3, 3); got {tuple(value.shape)}."
                )
        return self.layout.flatten(value).reshape(1, self.output_dim)

    def _baseline_bank(self) -> Tensor:
        if self._cached_baseline.numel() == 0:
            values = [self._predict_one(structure) for structure in self.structures]
            cached = torch.cat(values, dim=0)
            self._cached_baseline = cached.detach()
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


class MACEDirectForcePredictor(_MACEDirectTensorPredictor):
    """Frozen MACE Cartesian forces flattened in atom-major xyz order."""

    def __init__(
        self,
        encoder: MACEEncoder,
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


class MACEDirectStressPredictor(_MACEDirectTensorPredictor):
    """Frozen MACE full Cartesian stress tensor flattened in row-major order."""

    def __init__(self, encoder: MACEEncoder, structures: Sequence[Any]) -> None:
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
            f"MACE {kind} residual requires target_contract.quantity={kind!r}; "
            f"got {target_contract.quantity!r}."
        )
    return MaterialBaselineSpec(
        family="mace",
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


class MACEForceResidualGPModel(ResidualMaterialGPModel):
    """Correct fixed-topology MACE forces with a correlated wide residual GP.

    ``train_Y`` accepts either ``[n, num_atoms, 3]`` Cartesian forces or the
    equivalent flattened ``[n, 3 * num_atoms]`` representation. Atom ordering
    must be identical between every structure and target row. Ragged atom counts
    are intentionally rejected in this phase.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        structures: Sequence[Any],
        num_atoms: int | None = None,
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
        target_contract: MaterialPropertyContract | None = None,
    ) -> None:
        resolved_structures = _validate_structure_bank(structures)
        material_encoder = _resolve_encoder(
            encoder,
            model_name=model_name,
            num_layers=num_layers,
            pooling=pooling,
            head=head,
        )
        predictor = MACEDirectForcePredictor(
            material_encoder,
            resolved_structures,
            num_atoms=num_atoms,
        )
        flat_Y = predictor.layout.flatten(train_Y, n=train_X.shape[0])
        baseline_spec = _baseline_spec(
            kind="force",
            model_name=model_name,
            target_contract=target_contract,
        )
        residual_Y = compute_material_residual_targets(
            train_X,
            flat_Y,
            predictor,
            baseline_spec=baseline_spec,
            target_contract=target_contract,
        )
        residual_model = MACEMultiTaskGPModel(
            train_X=train_X,
            train_Y=residual_Y,
            train_Yvar=_flatten_noise(predictor.layout, train_Yvar, n=train_X.shape[0]),
            structures=resolved_structures,
            encoder=material_encoder,
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
        super().__init__(
            predictor=predictor,
            residual_model=residual_model,
            baseline_spec=baseline_spec,
        )
        self.layout = predictor.layout
        self.structures = resolved_structures
        self.material_encoder = material_encoder
        self.num_atoms = predictor.num_atoms
        self.head = material_encoder.head

    def unflatten(self, values: Tensor) -> Tensor:
        """Restore ``[..., num_atoms, 3]`` force axes from GP outputs."""

        return self.layout.unflatten(values)


class MACEStressResidualGPModel(ResidualMaterialGPModel):
    """Correct full 3x3 MACE stress tensors with a correlated residual GP.

    ``train_Y`` accepts either ``[n, 3, 3]`` tensors or flattened ``[n, 9]``
    values. The full tensor is preserved; bochan does not silently convert to
    Voigt-6 notation or change stress sign/unit conventions.
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
        target_contract: MaterialPropertyContract | None = None,
    ) -> None:
        resolved_structures = _validate_structure_bank(structures)
        material_encoder = _resolve_encoder(
            encoder,
            model_name=model_name,
            num_layers=num_layers,
            pooling=pooling,
            head=head,
        )
        predictor = MACEDirectStressPredictor(material_encoder, resolved_structures)
        flat_Y = predictor.layout.flatten(train_Y, n=train_X.shape[0])
        baseline_spec = _baseline_spec(
            kind="stress",
            model_name=model_name,
            target_contract=target_contract,
        )
        residual_Y = compute_material_residual_targets(
            train_X,
            flat_Y,
            predictor,
            baseline_spec=baseline_spec,
            target_contract=target_contract,
        )
        residual_model = MACEMultiTaskGPModel(
            train_X=train_X,
            train_Y=residual_Y,
            train_Yvar=_flatten_noise(predictor.layout, train_Yvar, n=train_X.shape[0]),
            structures=resolved_structures,
            encoder=material_encoder,
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
        super().__init__(
            predictor=predictor,
            residual_model=residual_model,
            baseline_spec=baseline_spec,
        )
        self.layout = predictor.layout
        self.structures = resolved_structures
        self.material_encoder = material_encoder
        self.head = material_encoder.head

    def unflatten(self, values: Tensor) -> Tensor:
        """Restore ``[..., 3, 3]`` stress axes from GP outputs."""

        return self.layout.unflatten(values)


__all__ = [
    "MACEDirectForcePredictor",
    "MACEDirectStressPredictor",
    "MACEForceResidualGPModel",
    "MACEStressResidualGPModel",
]
