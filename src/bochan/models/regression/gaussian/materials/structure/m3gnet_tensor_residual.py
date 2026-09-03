"""M3GNet force/stress baselines corrected by correlated residual Gaussian processes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any, Literal

import torch
from gpytorch.likelihoods import Likelihood
from torch import Tensor, nn

from bochan.composition import M3GNetEncoder, MaterialProcessFusion
from bochan.models.regression.gaussian.deep.deepkernel import InputTransformArg, OutcomeTransformArg
from bochan.models.regression.gaussian.deep.m3gnet_multitask import M3GNetMultiTaskGPModel
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
from bochan.structure.adapter import StructureAdapter

from .m3gnet_residual import _resolve_encoder, _validate_structure_bank

_MLIPKind = Literal["force", "stress"]
_DEFAULT_MODEL_NAME = "M3GNet-PES-MatPES-PBE-2025.2"


def _load_potential(model_name: str) -> Any:
    try:
        matgl = import_module("matgl")
    except ImportError as error:
        raise ImportError(
            "M3GNet force/stress support requires matgl>=4.0.3,<5. "
            "Install bochan[materials]."
        ) from error
    load_model = getattr(matgl, "load_model", None)
    if not callable(load_model):
        raise RuntimeError("The installed matgl package does not expose matgl.load_model().")
    potential = load_model(model_name)
    if not callable(getattr(potential, "forward", None)) and not hasattr(potential, "model"):
        raise TypeError("matgl.load_model() did not return a compatible PES potential.")
    return potential


def _load_calculator(potential: Any) -> Any:
    try:
        module = import_module("matgl.ext.ase")
    except ImportError as error:
        raise ImportError(
            "M3GNet force/stress support requires matgl>=4.0.3,<5 with ASE support."
        ) from error
    calculator_class = getattr(module, "PESCalculator", None)
    if not isinstance(calculator_class, type):
        raise RuntimeError("The installed matgl package does not expose PESCalculator.")
    return calculator_class(potential)


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


class _M3GNetDirectTensorPredictor(DirectMaterialPredictor):
    """Shared fixed-layout predictor using MatGL's official ASE calculator."""

    def __init__(
        self,
        structures: Sequence[Any],
        *,
        layout: TensorTargetLayout,
        model_name: str = _DEFAULT_MODEL_NAME,
        potential: Any | None = None,
        calculator: Any | None = None,
        adapter: StructureAdapter | None = None,
    ) -> None:
        super().__init__()
        self.structures = _validate_structure_bank(structures)
        self.layout = layout
        self.model_name = model_name
        self.adapter = StructureAdapter() if adapter is None else adapter
        if not isinstance(self.adapter, StructureAdapter):
            raise TypeError("adapter must be a StructureAdapter.")
        self.potential = _load_potential(model_name) if potential is None and calculator is None else potential
        self.calculator = (
            _load_calculator(self.potential) if calculator is None else calculator
        )
        self.register_buffer("_cached_baseline", torch.empty(0), persistent=False)

    @property
    def output_dim(self) -> int:
        return self.layout.output_dim

    def _predict_one(self, structure: Any) -> Tensor:
        atoms = self.adapter.to_ase(structure).copy()
        atoms.calc = self.calculator
        if self.layout.kind == "force":
            value = torch.as_tensor(atoms.get_forces(), dtype=torch.get_default_dtype())
            if tuple(value.shape) != self.layout.tensor_shape:
                raise ValueError(
                    f"M3GNet forces must have shape {self.layout.tensor_shape}; got {tuple(value.shape)}."
                )
        else:
            value = torch.as_tensor(atoms.get_stress(voigt=False), dtype=torch.get_default_dtype())
            if tuple(value.shape) != (3, 3):
                raise ValueError(f"M3GNet stress must have shape (3, 3); got {tuple(value.shape)}.")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"M3GNet produced non-finite {self.layout.kind} predictions.")
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
        return selected.to(device=X.device, dtype=X.dtype).reshape(*leading_shape, self.output_dim)


class M3GNetDirectForcePredictor(_M3GNetDirectTensorPredictor):
    """Frozen MatGL M3GNet Cartesian forces flattened in atom-major xyz order."""

    def __init__(self, structures: Sequence[Any], *, num_atoms: int | None = None, **kwargs: Any) -> None:
        resolved = _validate_structure_bank(structures)
        resolved_num_atoms = _resolve_force_num_atoms(resolved, num_atoms)
        super().__init__(resolved, layout=TensorTargetLayout.force(resolved_num_atoms), **kwargs)
        self.num_atoms = resolved_num_atoms


class M3GNetDirectStressPredictor(_M3GNetDirectTensorPredictor):
    """Frozen MatGL M3GNet full Cartesian ASE stress tensor flattened row-major."""

    def __init__(self, structures: Sequence[Any], **kwargs: Any) -> None:
        super().__init__(structures, layout=TensorTargetLayout.stress(), **kwargs)


def _baseline_spec(*, kind: _MLIPKind, model_name: str, target_contract: MaterialPropertyContract | None) -> MaterialBaselineSpec | None:
    if target_contract is None:
        return None
    if not isinstance(target_contract, MaterialPropertyContract):
        raise TypeError("target_contract must be a MaterialPropertyContract when provided.")
    if target_contract.quantity.casefold() != kind:
        raise ValueError(
            f"M3GNet {kind} residual requires target_contract.quantity={kind!r}; got {target_contract.quantity!r}."
        )
    return MaterialBaselineSpec(family="m3gnet", property=target_contract, model_name=model_name)


def _flatten_noise(layout: TensorTargetLayout, train_Yvar: Tensor | None, *, n: int) -> Tensor | None:
    return None if train_Yvar is None else layout.flatten(train_Yvar, n=n)


class M3GNetForceResidualGPModel(ResidualMaterialGPModel):
    """Correct fixed-topology M3GNet forces with a correlated residual GP."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        structures: Sequence[Any],
        num_atoms: int | None = None,
        encoder: M3GNetEncoder | nn.Module | None = None,
        model_name: str = _DEFAULT_MODEL_NAME,
        encoder_output_dim: int | None = None,
        latent_dim: int = 32,
        fusion: Literal["concat"] | MaterialProcessFusion = "concat",
        projection: nn.Module | None = None,
        likelihood: Likelihood | None = None,
        input_transform: InputTransformArg = "DEFAULT",
        outcome_transform: OutcomeTransformArg = "DEFAULT",
        target_contract: MaterialPropertyContract | None = None,
        potential: Any | None = None,
        calculator: Any | None = None,
        adapter: StructureAdapter | None = None,
    ) -> None:
        resolved_structures = _validate_structure_bank(structures)
        material_encoder = _resolve_encoder(encoder, model_name=model_name, encoder_output_dim=encoder_output_dim)
        predictor = M3GNetDirectForcePredictor(
            resolved_structures,
            num_atoms=num_atoms,
            model_name=model_name,
            potential=potential,
            calculator=calculator,
            adapter=adapter,
        )
        flat_Y = predictor.layout.flatten(train_Y, n=train_X.shape[0])
        baseline_spec = _baseline_spec(kind="force", model_name=model_name, target_contract=target_contract)
        residual_Y = compute_material_residual_targets(
            train_X, flat_Y, predictor, baseline_spec=baseline_spec, target_contract=target_contract
        )
        residual_model = M3GNetMultiTaskGPModel(
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
            likelihood=likelihood,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )
        super().__init__(predictor=predictor, residual_model=residual_model, baseline_spec=baseline_spec)
        self.layout = predictor.layout
        self.structures = resolved_structures
        self.material_encoder = material_encoder
        self.num_atoms = predictor.num_atoms

    def unflatten(self, values: Tensor) -> Tensor:
        return self.layout.unflatten(values)


class M3GNetStressResidualGPModel(ResidualMaterialGPModel):
    """Correct full 3x3 M3GNet ASE stress tensors with a correlated residual GP."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        *,
        structures: Sequence[Any],
        encoder: M3GNetEncoder | nn.Module | None = None,
        model_name: str = _DEFAULT_MODEL_NAME,
        encoder_output_dim: int | None = None,
        latent_dim: int = 32,
        fusion: Literal["concat"] | MaterialProcessFusion = "concat",
        projection: nn.Module | None = None,
        likelihood: Likelihood | None = None,
        input_transform: InputTransformArg = "DEFAULT",
        outcome_transform: OutcomeTransformArg = "DEFAULT",
        target_contract: MaterialPropertyContract | None = None,
        potential: Any | None = None,
        calculator: Any | None = None,
        adapter: StructureAdapter | None = None,
    ) -> None:
        resolved_structures = _validate_structure_bank(structures)
        material_encoder = _resolve_encoder(encoder, model_name=model_name, encoder_output_dim=encoder_output_dim)
        predictor = M3GNetDirectStressPredictor(
            resolved_structures,
            model_name=model_name,
            potential=potential,
            calculator=calculator,
            adapter=adapter,
        )
        flat_Y = predictor.layout.flatten(train_Y, n=train_X.shape[0])
        baseline_spec = _baseline_spec(kind="stress", model_name=model_name, target_contract=target_contract)
        residual_Y = compute_material_residual_targets(
            train_X, flat_Y, predictor, baseline_spec=baseline_spec, target_contract=target_contract
        )
        residual_model = M3GNetMultiTaskGPModel(
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
            likelihood=likelihood,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )
        super().__init__(predictor=predictor, residual_model=residual_model, baseline_spec=baseline_spec)
        self.layout = predictor.layout
        self.structures = resolved_structures
        self.material_encoder = material_encoder

    def unflatten(self, values: Tensor) -> Tensor:
        return self.layout.unflatten(values)


__all__ = [
    "M3GNetDirectForcePredictor",
    "M3GNetDirectStressPredictor",
    "M3GNetForceResidualGPModel",
    "M3GNetStressResidualGPModel",
]
